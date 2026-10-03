"""Local full WiLoR / AnyHand-WiLoR inference for the isolated CUDA worker.

Inputs are mirrored OpenCV BGR frames. The model, detector and MANO files must
already exist locally; this module never downloads, opens a camera, or sends
desktop input. Detector scores describe whole hands, not individual joints.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time

import numpy as np


_CHAINS = ((0, 1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
           (13, 14, 15, 16), (17, 18, 19, 20))


@contextmanager
def _trusted_detector_load(torch):
    """Old Ultralytics checkpoints store a model object, not a tensor dict.

    Only the pinned, setup-verified detector is loaded inside this scope. The
    old loader does not accept weights_only. Restore torch.load even on error.
    This runs before the single-threaded inference worker accepts any frames.
    """
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def _load_complete_state(model, checkpoint):
    """Reject partial or incompatible inference weights instead of hiding them."""
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("The WiLoR checkpoint has no state_dict.")
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    # Training-only state in the original Lightning model has no inference use.
    allowed_extra = {key for key in unexpected
                     if key == "initialized" or key.startswith("discriminator.")}
    unexpected -= allowed_extra
    incompatible = [key for key in expected.keys() & state.keys()
                    if tuple(getattr(state[key], "shape", ())) != tuple(expected[key].shape)]
    if missing or unexpected or incompatible:
        details = []
        for name, values in (("missing", missing), ("unexpected", unexpected),
                             ("shape mismatch", incompatible)):
            if values:
                details.append(f"{name}: {', '.join(sorted(values)[:8])}")
        raise ValueError("Incompatible WiLoR inference checkpoint (" + "; ".join(details) + ")")
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    return len(expected)


def _legacy_numpy_compat():
    """Support the original MANO pickle's Chumpy objects in this worker only."""
    for name, value in {"bool": bool, "int": int, "float": float, "complex": complex,
                        "object": object, "str": str, "unicode": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _configure_local_detector(model_dir):
    """Disable network integrations before importing pinned Ultralytics 8.1.34."""
    settings_dir = Path(model_dir).resolve().parent / "wilor-settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    # The pinned SettingsManager resets incomplete schemas to online defaults.
    # Supply its complete schema before first import, in our ignored data dir.
    settings = {
        "settings_version": "0.0.4",
        "datasets_dir": str(settings_dir / "datasets"),
        "weights_dir": str(settings_dir / "weights"),
        "runs_dir": str(settings_dir / "runs"),
        "uuid": "intuitionos-local-hand-tracker", "sync": False,
        "api_key": "", "openai_api_key": "",
    }
    settings.update({name: False for name in ("clearml", "comet", "dvc", "hub", "mlflow",
                                            "neptune", "raytune", "tensorboard", "wandb")})
    # JSON is a YAML subset and does not require importing more runtime modules.
    (settings_dir / "settings.yaml").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"


def _project_joints(joints, camera, box, is_right, width, height, crop_scale=2.0):
    """Project MANO metres through WiLoR's estimated camera to image pixels.

    The MANO wrapper already orders its 21 joints as wrist, thumb, index,
    middle, ring and pinky, matching MediaPipe. Left-hand crops are mirrored
    during preprocessing, so both world x and camera x must be restored here.
    """
    joints = np.asarray(joints, dtype=np.float64).copy()
    camera = np.asarray(camera, dtype=np.float64).copy()
    box = np.asarray(box, dtype=np.float64)
    if (joints.shape != (21, 3) or camera.shape != (3,) or box.shape != (4,)
            or not np.isfinite(joints).all() or not np.isfinite(camera).all()
            or not np.isfinite(box).all() or camera[0] <= 1e-6
            or np.any(box[2:] <= box[:2])):
        return None
    direction = 1.0 if is_right else -1.0
    joints[:, 0] *= direction
    camera[1] *= direction
    center = (box[:2] + box[2:]) / 2.0
    size = float(np.max(box[2:] - box[:2]) * crop_scale)
    focal = 5000.0 / 256.0 * max(width, height)
    denominator = size * camera[0]
    translation = np.array([
        2.0 * (center[0] - width / 2.0) / denominator + camera[1],
        2.0 * (center[1] - height / 2.0) / denominator + camera[2],
        2.0 * focal / denominator,
    ])
    transformed = joints + translation
    if np.any(transformed[:, 2] <= 1e-6):
        return None
    pixels = transformed[:, :2] / transformed[:, 2:] * focal + (width / 2.0, height / 2.0)
    # Keep actual relative depth in image-width units. Existing controls use
    # x/y; the preview and future 3D controls need not receive invented zeros.
    depth = (joints[:, 2] - joints[0, 2]) * focal / (transformed[0, 2] * width)
    return pixels, joints, depth


class _HandSelector:
    """Maintain one hand identity while allowing genuinely occluded fingertips.

    WiLoR reconstructs 3D joints and has no per-joint confidence. We use the
    real detector score and broad 3D anatomy/projection checks. Missing frames
    always return an empty result; no old pose can continue moving or clicking.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.wrist = None
        self.scale = None
        self.handedness = None
        self.last_seen = None
        self.lost_at = None

    @staticmethod
    def candidate(pixels, joints, depth, box, score, is_right, width, height):
        pixels = np.asarray(pixels, dtype=float)
        joints = np.asarray(joints, dtype=float)
        depth = np.asarray(depth, dtype=float)
        box = np.asarray(box, dtype=float)
        if (pixels.shape != (21, 2) or joints.shape != (21, 3)
                or depth.shape != (21,) or box.shape != (4,)
                or any(not np.isfinite(item).all() for item in (pixels, joints, depth, box))
                or not math.isfinite(score) or not 0.35 <= score <= 1.0):
            return None
        size = box[2:] - box[:2]
        image_size = np.array([width, height], dtype=float)
        if np.min(size) < 20 or width < 2 or height < 2:
            return None
        if (np.any(pixels < -0.10 * image_size) or np.any(pixels > 1.10 * image_size)
                or np.any(pixels < box[:2] - 0.6 * size)
                or np.any(pixels > box[2:] + 0.6 * size)):
            return None
        # Do not discard bent or edge-on fingers just because a 2D segment
        # becomes short. Anatomical plausibility is checked in reconstructed 3D.
        palm = float(np.linalg.norm(joints[9] - joints[0]))
        spread = float(np.linalg.norm(joints[5] - joints[17]))
        if not (0.01 <= palm <= 0.25 and 0.15 <= spread / palm <= 2.5):
            return None
        for chain in _CHAINS:
            lengths = np.linalg.norm(np.diff(joints[list(chain)], axis=0), axis=1)
            if np.max(lengths) > 2.0 * palm or np.sum(lengths) > 4.5 * palm:
                return None
        if np.linalg.norm(pixels[9] - pixels[0]) < 4:
            return None
        normalized = pixels / image_size
        # Wrist / palm joints must be visible; a small fingertip overrun does
        # not cause a tracking gap when the hand itself remains in view.
        if np.any(normalized[[0, 5, 9, 13, 17]] < -0.025) or np.any(normalized[[0, 5, 9, 13, 17]] > 1.025):
            return None
        return {"points": np.column_stack((normalized, depth)), "wrist": normalized[0],
                "scale": float(np.linalg.norm(size / image_size)),
                "area": float(np.prod(size)), "score": float(score), "right": bool(is_right)}

    def lost(self, now):
        if self.lost_at is None:
            self.lost_at = now
        elif now - self.lost_at >= 0.25:
            self.reset()
        return []

    def select(self, candidates, now):
        candidates = [candidate for candidate in candidates if candidate is not None]
        if not candidates:
            return self.lost(now)
        if self.wrist is not None:
            elapsed = now - self.last_seen
            if elapsed < 0 or elapsed > 0.35:
                self.reset()
                return []
            # Chirality is a tie breaker, not a hard visibility requirement:
            # detector left/right labels can flicker during a palm rotation.
            def cost(hand):
                distance = float(np.linalg.norm(hand["wrist"] - self.wrist))
                return distance + (0.035 if hand["right"] != self.handedness else 0.0)
            selected = min(candidates, key=cost)
            distance = float(np.linalg.norm(selected["wrist"] - self.wrist))
            max_distance = min(0.25, max(0.08, self.scale * 0.25) + elapsed)
            ratio = selected["scale"] / self.scale
            if distance > max_distance or not 0.4 <= ratio <= 2.5:
                return self.lost(now)
        else:
            selected = max(candidates, key=lambda hand: (hand["area"], hand["score"]))
        self.wrist, self.scale = selected["wrist"], selected["scale"]
        self.handedness = selected["right"]
        self.last_seen, self.lost_at = now, None
        return selected["points"].tolist()


class WiLoRModel:
    """Full 32-layer WiLoR transformer, with locally selected checkpoint."""

    def __init__(self, model_dir, device_id=0, checkpoint_name="anyhand_wilor.ckpt", precision="float16"):
        if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")
        if checkpoint_name not in ("anyhand_wilor.ckpt", "wilor_final.ckpt"):
            raise ValueError("Unsupported WiLoR checkpoint name")
        if precision not in ("float32", "float16"):
            raise ValueError("precision must be float32 or float16")
        model_dir = Path(model_dir)
        for name in (checkpoint_name, "detector.pt", "MANO_RIGHT.pkl", "mano_mean_params.npz"):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(f"Hand tracking model is missing: {model_dir / name}")
        import torch
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            raise RuntimeError("WiLoR requires a CUDA-capable NVIDIA GPU and a matching PyTorch build.")
        _configure_local_detector(model_dir)
        from wilor_mini.models.wilor import WiLor
        from ultralytics import YOLO
        _legacy_numpy_compat()
        self._torch = torch
        self._device = torch.device(f"cuda:{device_id}")
        self._dtype = torch.float16 if precision == "float16" else torch.float32
        self._crop_scale = 2.0  # AnyHand's official predictor default.
        self._model = WiLor(mano_model_path=str(model_dir / "MANO_RIGHT.pkl"),
                            mano_mean_path=str(model_dir / "mano_mean_params.npz"),
                            focal_length=5000, image_size=256)
        # Checkpoints are official upstream files whose hashes setup verifies.
        # Lightning stores configuration objects alongside its tensor state.
        checkpoint = torch.load(str(model_dir / checkpoint_name), map_location="cpu", weights_only=False)
        loaded = _load_complete_state(self._model, checkpoint)
        del checkpoint
        self._model.eval().to(device=self._device, dtype=self._dtype)
        with _trusted_detector_load(torch):
            self._detector = YOLO(str(model_dir / "detector.pt")).to(self._device)
        self._selector = _HandSelector()
        self.metadata = {
            "model": "anyhand-wilor" if checkpoint_name.startswith("anyhand") else "wilor-full",
            "model_name": "AnyHand WiLoR Full (CUDA)" if checkpoint_name.startswith("anyhand") else "WiLoR Full (CUDA)",
            "checkpoint": checkpoint_name, "device_id": device_id,
            "gpu_name": torch.cuda.get_device_name(device_id), "providers": ["CUDA"],
            "vram_bytes": int(torch.cuda.get_device_properties(device_id).total_memory),
            "precision": precision, "landmarks": 21, "dimensions": 3,
            "loaded_state_tensors": loaded, "transformer_layers": 32,
        }

    def reset(self):
        self._selector.reset()

    def _detect(self, image):
        result = self._detector(image, conf=0.35, iou=0.3, max_det=4,
                                verbose=False, device=self._device)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=bool)
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
        scores = result.boxes.conf.detach().cpu().numpy().astype(float)
        classes = result.boxes.cls.detach().cpu().numpy().astype(float)
        keep = (np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
                & np.isfinite(classes) & ((boxes[:, 2:] - boxes[:, :2]) >= 20).all(axis=1)
                & (scores >= 0.35) & (scores <= 1.0) & np.isin(classes, [0, 1]))
        return boxes[keep], scores[keep], classes[keep] == 1

    def _reconstruct(self, image, boxes, rights):
        from skimage.filters import gaussian
        from wilor_mini.utils.utils import generate_image_patch_cv2
        import cv2
        torch = self._torch
        patches = []
        for box, right in zip(boxes, rights):
            center = (box[:2] + box[2:]) * 0.5
            size = float(np.max(box[2:] - box[:2]) * self._crop_scale)
            downsampling = size / 256.0 / 2.0
            source = image
            if downsampling > 1.1:
                source = gaussian(image, sigma=(downsampling - 1) / 2,
                                  channel_axis=2, preserve_range=True)
            patch, _ = generate_image_patch_cv2(source, *center, size, size, 256, 256,
                                                not bool(right), 1.0, 0,
                                                border_mode=cv2.BORDER_CONSTANT)
            patches.append(patch)
        tensor = torch.from_numpy(np.stack(patches)).to(device=self._device, dtype=self._dtype)
        with torch.inference_mode():
            output = self._model(tensor)
        # Transfer only joints and camera; meshes are not needed for desktop
        # control. This also synchronizes the completed GPU prediction.
        return (output["pred_keypoints_3d"].detach().float().cpu().numpy(),
                output["pred_cam"].detach().float().cpu().numpy())

    def warmup(self):
        """Initialize both CUDA stages with synthetic input, without a camera."""
        from torchvision.ops import nms

        # A blank image has no detections, so YOLO skips its CUDA NMS kernel.
        # Load that kernel here instead of on the first real hand, which may
        # appear long after the first camera frame and exceed its time budget.
        torch = self._torch
        nms_boxes = torch.tensor([[0, 0, 100, 100], [10, 10, 110, 110]],
                                 device=self._device, dtype=torch.float32)
        nms_scores = torch.tensor([0.9, 0.8], device=self._device, dtype=torch.float32)
        nms(nms_boxes, nms_scores, 0.3)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        self._detect(image)
        # Live prediction reconstructs up to two hands. CUDA initialization is
        # shape-dependent, so prepare both batch sizes before accepting frames.
        boxes = np.array([[200, 100, 440, 380], [40, 80, 180, 300]], dtype=float)
        rights = np.array([True, False])
        for count in (1, 2):
            self._reconstruct(image, boxes[:count], rights[:count])
        self._torch.cuda.synchronize(self._device)
        self.reset()

    def predict(self, bgr_frame):
        if (not isinstance(bgr_frame, np.ndarray) or bgr_frame.dtype != np.uint8
                or bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3
                or min(bgr_frame.shape[:2]) < 2):
            raise ValueError("Expected a uint8 BGR image with three channels.")
        height, width = bgr_frame.shape[:2]
        boxes, scores, rights = self._detect(bgr_frame)
        if len(boxes) == 0:
            return self._selector.lost(time.monotonic())
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        rank = np.argsort(-areas, kind="stable")
        if self._selector.wrist is not None:
            wrist = self._selector.wrist * (width, height)
            inside = ((boxes[:, :2] <= wrist) & (wrist <= boxes[:, 2:])).all(axis=1)
            rank = np.concatenate((rank[inside[rank]], rank[~inside[rank]]))
        rank = rank[:2]  # Bounded latency, retain the previous hand first.
        boxes, scores, rights = boxes[rank], scores[rank], rights[rank]
        joints, cameras = self._reconstruct(bgr_frame, boxes, rights)
        if joints.shape != (len(boxes), 21, 3) or cameras.shape != (len(boxes), 3):
            raise ValueError("WiLoR returned invalid joint or camera dimensions.")
        candidates = []
        for points, camera, box, score, right in zip(joints, cameras, boxes, scores, rights):
            projected = _project_joints(points, camera, box, right, width, height, self._crop_scale)
            if projected is not None:
                pixels, world, depth = projected
                candidates.append(self._selector.candidate(pixels, world, depth, box, score, right, width, height))
        return self._selector.select(candidates, time.monotonic())
