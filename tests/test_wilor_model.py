"""Coordinate, identity, and complete-weight checks without camera or CUDA."""
from types import SimpleNamespace
import json
import os
import sys

import numpy as np
import pytest

from core.wilor_model import (
    WiLoRModel, _HandSelector, _load_complete_state, _project_joints,
    _trusted_detector_load, _configure_local_detector,
)


def world_hand():
    # MANO/OpenPose/MediaPipe wrist-thumb-index-middle-ring-pinky ordering.
    return np.array([
        [0, .04, 0], [-.02, .02, 0], [-.04, .005, 0], [-.06, -.01, 0], [-.075, -.025, 0],
        [-.025, -.02, 0], [-.025, -.055, 0], [-.025, -.075, 0], [-.025, -.09, 0],
        [0, -.025, 0], [0, -.06, 0], [0, -.085, 0], [0, -.10, 0],
        [.025, -.02, 0], [.025, -.05, 0], [.025, -.07, 0], [.025, -.085, 0],
        [.045, -.012, 0], [.045, -.035, 0], [.045, -.05, 0], [.045, -.06, 0],
    ], dtype=float)


def candidate(offset=0.0, *, right=True, score=.9, area_scale=1.0):
    box = np.array([200 + offset, 130, 400 + offset, 350], dtype=float)
    pixels, joints, depth = _project_joints(world_hand(), [.8, 0, 0], box, right, 640, 480)
    result = _HandSelector.candidate(pixels, joints, depth, box, score, right, 640, 480)
    if result is not None:
        result["area"] *= area_scale
    return result


def test_projection_center_is_detection_center_and_order_preserved():
    joints = np.zeros((21, 3))
    joints[8, 0] = .1
    points, _, depth = _project_joints(joints, [1, 0, 0], [100, 100, 300, 300], True, 640, 480)
    np.testing.assert_allclose(points[0], [200, 200])
    assert points[8, 0] > points[0, 0]
    assert np.count_nonzero(points[:, 0] != 200) == 1
    assert np.all(depth == 0)


def test_projection_restores_left_hand_and_camera_horizontal_offset():
    joints = world_hand()
    right = _project_joints(joints, [.8, .01, .02], [100, 100, 300, 300], True, 640, 480)
    left = _project_joints(joints, [.8, .01, .02], [100, 100, 300, 300], False, 640, 480)
    np.testing.assert_allclose(right[0][:, 0] + left[0][:, 0], 400)
    np.testing.assert_allclose(right[0][:, 1], left[0][:, 1])
    np.testing.assert_allclose(right[1][:, 0], -left[1][:, 0])


def test_projection_depth_is_wrist_relative_in_image_width_units():
    joints = world_hand()
    joints[:, 2] = np.linspace(0, .05, 21)
    _, _, depth = _project_joints(joints, [.8, 0, 0], [100, 100, 300, 300], True, 640, 480)
    assert depth[0] == 0
    assert 0 < depth[20] < .1


@pytest.mark.parametrize("camera", [[0, 0, 0], [-1, 0, 0], [np.nan, 0, 0], [1, 2]])
def test_invalid_camera_does_not_produce_control_coordinates(camera):
    assert _project_joints(world_hand(), camera, [100, 100, 300, 300], True, 640, 480) is None


def test_projection_rejects_invalid_joint_and_box_shapes():
    assert _project_joints(world_hand()[:20], [1, 0, 0], [100, 100, 300, 300], True, 640, 480) is None
    assert _project_joints(world_hand(), [1, 0, 0], [300, 100, 100, 300], True, 640, 480) is None


def test_detector_confidence_is_real_whole_hand_score():
    assert candidate(score=.9) is not None
    assert candidate(score=.34) is None
    assert candidate(score=float("nan")) is None
    assert candidate(score=1.1) is None


def test_folded_fingers_survive_plausible_3d_geometry():
    joints = world_hand()
    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        joints[pip] = joints[mcp] + [0, -.025, -.005]
        joints[dip] = joints[pip] + [0, .006, -.018]
        joints[tip] = joints[dip] + [0, .02, .001]
    box = [200, 130, 400, 350]
    projected = _project_joints(joints, [.8, 0, 0], box, True, 640, 480)
    assert _HandSelector.candidate(*projected, box, .8, True, 640, 480) is not None


def test_impossible_3d_finger_rejected_even_if_projected_points_fit_box():
    good = world_hand()
    pixels, _, depth = _project_joints(good, [.8, 0, 0], [200, 130, 400, 350], True, 640, 480)
    good[8, 2] = 3
    assert _HandSelector.candidate(pixels, good, depth, [200, 130, 400, 350], .9, True, 640, 480) is None


def test_nonfinite_or_offscreen_projection_rejected():
    pixels, joints, depth = _project_joints(world_hand(), [.8, 0, 0], [200, 130, 400, 350], True, 640, 480)
    pixels[8, 0] = np.nan
    assert _HandSelector.candidate(pixels, joints, depth, [200, 130, 400, 350], .9, True, 640, 480) is None
    pixels[8, 0] = 800
    assert _HandSelector.candidate(pixels, joints, depth, [200, 130, 400, 350], .9, True, 640, 480) is None


def test_selection_retains_old_hand_when_larger_hand_enters():
    selector = _HandSelector()
    first = selector.select([candidate()], 1)
    second = selector.select([candidate(140, right=False, area_scale=4), candidate(2)], 1.03)
    assert second
    assert abs(second[0][0] - first[0][0]) < .02


def test_handedness_flicker_alone_does_not_drop_hand():
    selector = _HandSelector()
    original = candidate()
    selector.select([original], 1)
    changed = dict(original, right=False)
    assert selector.select([changed], 1.03)


def test_lost_hand_returns_empty_immediately_and_reacquires_after_boundary():
    selector = _HandSelector()
    selector.select([candidate()], 1)
    assert selector.select([], 1.03) == []
    assert selector.wrist is not None
    assert selector.select([], 1.31) == []
    assert selector.wrist is None
    assert selector.select([candidate(180)], 1.34)


def test_abrupt_identity_switch_produces_loss_not_pointer_teleport():
    selector = _HandSelector()
    selector.select([candidate(-150)], 1)
    assert selector.select([candidate(180)], 1.03) == []


@pytest.mark.parametrize("now", [.9, 1.5])
def test_stalled_or_reversed_clock_requires_new_acquisition(now):
    selector = _HandSelector()
    selector.select([candidate()], 1)
    assert selector.select([candidate()], now) == []
    assert selector.wrist is None


def test_reset_clears_identity():
    selector = _HandSelector()
    selector.select([candidate()], 1)
    selector.reset()
    assert selector.wrist is None and selector.handedness is None and selector.last_seen is None


class StateModel:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"backbone.weight": np.zeros((2, 3)), "refine_net.weight": np.zeros((3, 2))}

    def load_state_dict(self, state, *, strict):
        assert strict is True
        self.loaded = state


def test_every_inference_tensor_must_be_loaded_with_only_training_extras_ignored():
    model = StateModel()
    state = {**model.state_dict(), "initialized": np.array(True), "discriminator.weight": np.zeros(4)}
    assert _load_complete_state(model, {"state_dict": state}) == 2
    assert set(model.loaded) == set(model.state_dict())


@pytest.mark.parametrize("change", ["missing", "unexpected", "shape"])
def test_partial_or_incompatible_checkpoint_fails_loudly(change):
    model = StateModel()
    state = model.state_dict()
    if change == "missing":
        del state["refine_net.weight"]
    elif change == "unexpected":
        state["backbone.other"] = np.zeros(3)
    else:
        state["backbone.weight"] = np.zeros((1, 2))
    with pytest.raises(ValueError, match="Incompatible WiLoR"):
        _load_complete_state(model, {"state_dict": state})
    assert model.loaded is None


@pytest.mark.parametrize("checkpoint", [None, {}, {"state_dict": {}}, {"state_dict": []}])
def test_missing_state_dict_rejected(checkpoint):
    with pytest.raises(ValueError, match="no state_dict"):
        _load_complete_state(StateModel(), checkpoint)


def test_detector_loader_compatibility_scope_restored_on_error():
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
    torch = SimpleNamespace(load=load)
    with pytest.raises(RuntimeError):
        with _trusted_detector_load(torch):
            torch.load("pinned-detector.pt")
            raise RuntimeError("load failed")
    assert calls == [{"weights_only": False}]
    assert torch.load is load


def test_detector_offline_configuration_precedes_import(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    monkeypatch.setenv("YOLO_CONFIG_DIR", "unrelated")
    _configure_local_detector(tmp_path / "models")
    assert os.environ["YOLO_AUTOINSTALL"] == "false"
    assert os.environ["YOLO_CONFIG_DIR"] == str(tmp_path / "wilor-settings")
    settings = json.loads((tmp_path / "wilor-settings" / "settings.yaml").read_text())
    assert settings["settings_version"] == "0.0.4"
    assert settings["sync"] is False and settings["hub"] is False
    assert settings["api_key"] == "" and settings["openai_api_key"] == ""
    assert len(settings) == 17  # Complete pinned schema prevents default reset.


def mock_model():
    model = WiLoRModel.__new__(WiLoRModel)
    model._selector = _HandSelector()
    model._crop_scale = 2.0
    return model


@pytest.mark.parametrize("detected_hands", [1, 2, 4])
def test_warmup_prepares_every_live_pose_batch_before_camera_frames(detected_hands, monkeypatch):
    model = mock_model()
    model._selector.select([candidate()], 1)
    model._device = object()
    warmed = set()
    events = []
    warming = True
    float32 = object()

    def tensor(data, *, device, dtype):
        assert device is model._device and dtype is float32
        return SimpleNamespace(values=np.array(data), device=device, dtype=dtype)

    def nms(boxes, scores, threshold):
        assert boxes.device is scores.device is model._device
        assert boxes.dtype is scores.dtype is float32
        assert boxes.values.shape == (2, 4) and scores.values.shape == (2,)
        assert (boxes.values[:, 2:] > boxes.values[:, :2]).all()
        assert (scores.values > 0).all() and threshold == 0.3
        events.append("cuda-nms")

    monkeypatch.setitem(sys.modules, "torchvision.ops", SimpleNamespace(nms=nms))

    def detect(image):
        if warming:
            assert events == ["cuda-nms"], "Blank detector warmup must not skip CUDA NMS"
            assert image.shape == (480, 640, 3) and image.dtype == np.uint8
            assert not image.any()
            events.append("detector")
        return (np.array([[200, 130, 400, 350]] * detected_hands),
                np.array([.9] * detected_hands), np.array([True] * detected_hands))

    def reconstruct(image, boxes, rights):
        count = len(boxes)
        assert len(rights) == count
        if warming:
            warmed.add(count)
            events.append(f"pose-{count}")
        else:
            assert count in warmed, "First live frame reached an unwarmed CUDA batch"
        return np.stack([world_hand()] * count), np.array([[.8, 0, 0]] * count)

    def synchronize(device):
        assert device is model._device
        assert warmed == {1, 2}
        assert model._selector.wrist is not None
        events.append("synchronize")

    model._detect, model._reconstruct = detect, reconstruct
    model._torch = SimpleNamespace(tensor=tensor, float32=float32,
                                   cuda=SimpleNamespace(synchronize=synchronize))
    model.warmup()

    assert events == ["cuda-nms", "detector", "pose-1", "pose-2", "synchronize"]
    assert model._selector.wrist is None and model._selector.last_seen is None
    warming = False
    assert len(model.predict(np.zeros((480, 640, 3), np.uint8))) == 21


def test_blank_detection_never_invokes_pose_or_fabricates_landmarks():
    model = mock_model()
    model._detect = lambda image: (np.empty((0, 4)), np.empty(0), np.empty(0, dtype=bool))
    def forbidden(*args):
        raise AssertionError("Pose cannot run without a detected hand")
    model._reconstruct = forbidden
    assert model.predict(np.zeros((480, 640, 3), np.uint8)) == []


def test_prediction_keeps_bgr_input_and_returns_normalized_21_point_hand():
    model = mock_model()
    image = np.zeros((480, 640, 3), np.uint8)
    image[:, :, 0] = 200
    def detect(frame):
        assert frame is image and frame[0, 0].tolist() == [200, 0, 0]
        return np.array([[200, 130, 400, 350]]), np.array([.9]), np.array([True])
    def reconstruct(frame, boxes, rights):
        assert frame is image and rights.tolist() == [True]
        return world_hand()[None], np.array([[.8, 0, 0]])
    model._detect, model._reconstruct = detect, reconstruct
    points = model.predict(image)
    assert len(points) == 21 and all(len(point) == 3 for point in points)
    assert np.isfinite(points).all()
    assert np.all((np.array(points)[:, :2] >= 0) & (np.array(points)[:, :2] <= 1))


def test_crowded_scene_bounds_pose_work_to_two_hands():
    model = mock_model()
    model._detect = lambda image: (np.array([[20, 10, 100, 200], [200, 130, 400, 350],
                                            [300, 100, 390, 250], [400, 80, 520, 270]]),
                                  np.array([.9] * 4), np.array([True] * 4))
    def reconstruct(frame, boxes, rights):
        assert len(boxes) == 2
        return np.stack([world_hand()] * 2), np.array([[.8, 0, 0]] * 2)
    model._reconstruct = reconstruct
    assert len(model.predict(np.zeros((480, 640, 3), np.uint8))) == 21


@pytest.mark.parametrize("image", [np.zeros((20, 20, 3)), np.zeros((20, 20), np.uint8),
                                   np.zeros((0, 20, 3), np.uint8), None])
def test_bad_input_rejected_before_gpu(image):
    with pytest.raises(ValueError, match="uint8 BGR"):
        mock_model().predict(image)


def test_invalid_reconstruction_shape_rejected():
    model = mock_model()
    model._detect = lambda image: (np.array([[200, 130, 400, 350]]), np.array([.9]), np.array([True]))
    model._reconstruct = lambda *args: (np.zeros((1, 20, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="invalid joint"):
        model.predict(np.zeros((480, 640, 3), np.uint8))


@pytest.mark.parametrize("device_id", [-1, True, "0"])
def test_invalid_device_rejected_without_importing_cuda(tmp_path, device_id):
    with pytest.raises(ValueError, match="device_id"):
        WiLoRModel(tmp_path, device_id)


def test_missing_assets_fail_without_downloading(tmp_path):
    with pytest.raises(FileNotFoundError, match="anyhand_wilor.ckpt"):
        WiLoRModel(tmp_path)


def test_unknown_checkpoint_name_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        WiLoRModel(tmp_path, checkpoint_name="../untrusted.ckpt")


@pytest.mark.parametrize("precision", ["int8", "bfloat16", "auto", None, True])
def test_unsupported_precision_rejected_before_model_load(tmp_path, precision):
    with pytest.raises(ValueError, match="precision must be"):
        WiLoRModel(tmp_path, precision=precision)
