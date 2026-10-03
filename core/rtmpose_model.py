"""Local RTMDet + RTMPose Hand5 inference for the isolated GPU worker.

This module never opens a camera, downloads models, or injects desktop input.
RTMLib supplies the pinned models' preprocessing and SimCC decoding. Model
scores are used as rejection thresholds, not calibrated probabilities.
"""

from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np


_IMPORTANT = (0, 4, 5, 6, 7, 8, 9, 13, 17)
_CHAINS = ((0, 1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
           (13, 14, 15, 16), (17, 18, 19, 20))


def _directml_session(path, device_id, ort=None):
    if ort is None:
        import onnxruntime as ort
    if not Path(path).is_file():
        raise FileNotFoundError(f"Hand tracking model is missing: {path}")
    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("The GPU tracking environment has no DirectML provider.")
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options,
        providers=[("DmlExecutionProvider", {"device_id": device_id})])
    # ONNX Runtime may add CPU for unsupported small operators. A failed GPU
    # initialization must never silently turn the entire tracker into CPU mode.
    if session.get_providers()[0] != "DmlExecutionProvider":
        raise RuntimeError("DirectML could not initialize the selected graphics card.")
    session.disable_fallback()
    return session


def _input_size(session):
    shape = session.get_inputs()[0].shape
    if (len(shape) != 4 or shape[1] != 3
            or not all(isinstance(value, int) and value > 0 for value in shape[2:])):
        raise ValueError("Expected a fixed-size NCHW hand tracking model.")
    return int(shape[3]), int(shape[2])  # width, height


def _local_tools(detector_session, pose_session):
    from rtmlib.tools.object_detection.rtmdet import RTMDet
    from rtmlib.tools.pose_estimation.rtmpose import RTMPose

    # Avoid BaseTool's URL download and provider selection entirely. Only these
    # private instances use our sessions; no global RTMLib settings are changed.
    class LocalDetector(RTMDet):
        def __init__(self, session):
            self.session = session
            width, height = _input_size(session)
            self.model_input_size = (height, width)
            self.mean = (103.5300, 116.2800, 123.6750)
            self.std = (57.3750, 57.1200, 58.3950)
            self.backend = "onnxruntime"
            self.det_mode = "human"  # single hand class at class index zero
            self.nms_thr, self.score_thr = 0.45, 0.40

    class LocalPose(RTMPose):
        def __init__(self, session):
            self.session = session
            self.model_input_size = _input_size(session)
            self.mean = (123.675, 116.28, 103.53)
            self.std = (58.395, 57.12, 57.375)
            self.backend = "onnxruntime"
            self.to_openpose = False

        def preprocess(self, image, bbox):
            # The pinned Hand5 export's pipeline.json specifies to_rgb=true.
            # RTMLib 0.0.16 omits this conversion, despite using RGB means.
            # Detection continues to receive BGR as its own export requires.
            import cv2
            return super().preprocess(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), bbox)

    return LocalDetector(detector_session), LocalPose(pose_session)


class _HandSelector:
    """Reject uncertain hands and retain the nearest wrist across frames.

    A lost identity produces an empty frame before a new hand may be acquired.
    The caller uses that boundary to release buttons and disarm held gestures.
    Brief occlusion retains the old identity for 0.25 seconds, while still
    returning no landmarks. Coordinates are never smoothed here.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.wrist = None
        self.palm = None
        self.last_seen = None
        self.lost_at = None

    @staticmethod
    def _candidate(points, scores, box, width, height):
        points = np.asarray(points, dtype=float)
        scores = np.asarray(scores, dtype=float)
        box = np.asarray(box, dtype=float)
        if (points.shape != (21, 2) or scores.shape != (21,) or box.shape != (4,)
                or not np.isfinite(points).all() or not np.isfinite(scores).all()
                or not np.isfinite(box).all()):
            return None
        if (np.min(scores) < 0.30 or np.min(scores[list(_IMPORTANT)]) < 0.40
                or np.mean(scores) < 0.45):
            return None
        box_size = box[2:] - box[:2]
        if np.min(box_size) < 20:
            return None
        # A partially clipped or wildly extrapolated skeleton is unsafe for
        # finger actions. A small border tolerance permits ordinary model noise.
        image_size = np.array([width, height], dtype=float)
        if (np.any(points < -0.025 * image_size)
                or np.any(points > 1.025 * image_size)
                or np.any(points < box[:2] - 0.35 * box_size)
                or np.any(points > box[2:] + 0.35 * box_size)):
            return None
        palm = float(np.linalg.norm(points[0] - points[9]))
        spread = float(np.linalg.norm(points[5] - points[17]))
        if not (8 <= palm <= min(width, height) * 0.75 and 0.12 <= spread / palm <= 2.5):
            return None
        for chain in _CHAINS:
            lengths = np.linalg.norm(np.diff(points[list(chain)], axis=0), axis=1)
            if np.max(lengths) > 2.0 * palm or np.sum(lengths) > 4.5 * palm:
                return None
        normalized = points / image_size
        return {"points": normalized, "wrist": normalized[0],
                "palm": palm / math.hypot(width, height),
                "area": float(np.prod(box_size)), "score": float(np.mean(scores))}

    def _lost(self, now):
        if self.lost_at is None:
            self.lost_at = now
        elif now - self.lost_at >= 0.25:
            self.reset()
        return []

    def select(self, keypoints, scores, boxes, width, height, now):
        candidates = []
        for points, confidence, box in zip(keypoints, scores, boxes):
            candidate = self._candidate(points, confidence, box, width, height)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return self._lost(now)
        if self.wrist is not None:
            elapsed = now - self.last_seen
            if elapsed < 0 or elapsed > 0.35:
                # Emit a loss boundary after a stalled worker even if another
                # plausible hand is present in the first returning frame.
                self.reset()
                return []
            selected = min(candidates, key=lambda hand: np.linalg.norm(hand["wrist"] - self.wrist))
            distance = float(np.linalg.norm(selected["wrist"] - self.wrist))
            max_distance = min(0.22, max(0.07, self.palm * 0.75) + elapsed * 1.0)
            scale_ratio = selected["palm"] / self.palm
            if distance > max_distance or not 0.55 <= scale_ratio <= 1.8:
                return self._lost(now)
        else:
            # On initial acquisition prefer the largest confidently visible hand.
            selected = max(candidates, key=lambda hand: (hand["area"], hand["score"]))
        self.wrist, self.palm = selected["wrist"], selected["palm"]
        self.last_seen, self.lost_at = now, None
        return [[float(x), float(y), 0.0] for x, y in selected["points"]]


class RTMPoseModel:
    """Load local detector.onnx / pose.onnx and predict mirrored BGR frames."""

    def __init__(self, model_dir, device_id=0):
        if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")
        model_dir = Path(model_dir)
        detector_session = _directml_session(model_dir / "detector.onnx", device_id)
        pose_session = _directml_session(model_dir / "pose.onnx", device_id)
        self.detector, self.pose = _local_tools(detector_session, pose_session)
        self._selector = _HandSelector()
        self.metadata = {
            "model": "rtmpose-hand5", "model_name": "RTMPose Hand5 (GPU)",
            "providers": {"detector": detector_session.get_providers(),
                          "pose": pose_session.get_providers()},
            "device_id": device_id, "landmarks": 21, "dimensions": 2,
        }

    def reset(self):
        self._selector.reset()

    def warmup(self):
        """Compile both GPU graphs without a camera or desktop input."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector(blank)
        self.pose(blank, bboxes=np.array([[200, 100, 440, 380]], dtype=float))
        self.reset()

    def predict(self, bgr_frame):
        if (not isinstance(bgr_frame, np.ndarray) or bgr_frame.dtype != np.uint8
                or bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3
                or min(bgr_frame.shape[:2]) < 2):
            raise ValueError("Expected a uint8 BGR image with three channels.")
        height, width = bgr_frame.shape[:2]
        now = time.monotonic()
        boxes = np.asarray(self.detector(bgr_frame), dtype=float)
        if boxes.size == 0:
            # RTMLib treats empty boxes as a request to infer the entire frame.
            # That behavior hallucinates landmarks on backgrounds; never call it.
            return self._selector.select([], [], [], width, height, now)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("The hand detector returned invalid boxes.")
        boxes = boxes[np.isfinite(boxes).all(axis=1)
                      & ((boxes[:, 2:] - boxes[:, :2]) >= 20).all(axis=1)]
        if len(boxes) == 0:
            return self._selector.select([], [], [], width, height, now)
        # Bound pose work with crowded images. Retain the box containing the
        # previous wrist first, then favor larger hands for initial acquisition.
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        rank = np.argsort(-areas, kind="stable")
        if self._selector.wrist is not None:
            wrist = self._selector.wrist * (width, height)
            inside = np.flatnonzero(((boxes[:, :2] <= wrist) & (wrist <= boxes[:, 2:])).all(axis=1))
            rank = np.concatenate((rank[np.isin(rank, inside)], rank[~np.isin(rank, inside)]))
        boxes = boxes[rank[:2]]
        keypoints, scores = self.pose(bgr_frame, bboxes=boxes)
        return self._selector.select(keypoints, scores, boxes, width, height, now)
