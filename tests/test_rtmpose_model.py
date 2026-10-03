"""GPU tracker contracts; no model downloads, cameras, or native input."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from core.rtmpose_model import RTMPoseModel, _HandSelector, _directml_session, _input_size, _local_tools


def sample(dx=0, dy=0, scale=1):
    points = np.array([
        [320, 350], [290, 325], [265, 300], [245, 280], [220, 255],
        [285, 260], [280, 215], [280, 180], [280, 145],
        [320, 250], [320, 205], [320, 165], [320, 130],
        [350, 255], [355, 215], [360, 180], [360, 145],
        [380, 270], [390, 240], [395, 210], [400, 180],
    ], dtype=float)
    points = (points - [320, 350]) * scale + [320 + dx, 350 + dy]
    box = np.r_[points.min(axis=0) - 15, points.max(axis=0) + 15]
    return points, np.full(21, 0.8), box


def select(selector, samples, now=0):
    if not samples:
        return selector.select([], [], [], 640, 480, now)
    points, scores, boxes = zip(*samples)
    return selector.select(points, scores, boxes, 640, 480, now)


def fake_model(detector, pose):
    model = RTMPoseModel.__new__(RTMPoseModel)
    model.detector, model.pose = detector, pose
    model._selector = _HandSelector()
    return model


def test_confident_landmarks_keep_order_and_have_no_invented_depth():
    result = select(_HandSelector(), [sample()])
    assert np.asarray(result)[:, :2] == pytest.approx(sample()[0] / (640, 480))
    assert all(point[2] == 0 for point in result)


@pytest.mark.parametrize("joint", range(21))
def test_any_uncertain_finger_joint_rejects_whole_hand(joint):
    points, scores, box = sample()
    scores[joint] = 0.29
    assert select(_HandSelector(), [(points, scores, box)]) == []


@pytest.mark.parametrize("joint", [0, 4, 5, 6, 7, 8, 9, 13, 17])
def test_control_joints_have_stricter_threshold(joint):
    points, scores, box = sample()
    scores[joint] = 0.39
    assert select(_HandSelector(), [(points, scores, box)]) == []


@pytest.mark.parametrize("kind", ["nan", "infinite_score", "outside", "collapsed", "long_bone", "tiny_box"])
def test_implausible_geometry_cannot_become_a_gesture(kind):
    points, scores, box = sample()
    if kind == "nan":
        points[8, 0] = np.nan
    elif kind == "infinite_score":
        scores[8] = np.inf
    elif kind == "outside":
        points[8, 0] = 800
    elif kind == "collapsed":
        points[9] = points[0]
    elif kind == "long_bone":
        points[8, 1] = 465
    else:
        box[2] = box[0] + 10
    assert select(_HandSelector(), [(points, scores, box)]) == []


def test_bending_distal_joints_is_allowed_without_temporal_smoothing():
    selector = _HandSelector()
    select(selector, [sample()])
    points, scores, box = sample(dx=5)
    points[7], points[8] = [286, 235], [288, 255]
    result = select(selector, [(points, scores, box)], now=0.033)
    assert np.asarray(result)[:, :2] == pytest.approx(points / (640, 480))


def test_second_larger_hand_does_not_take_over_existing_wrist():
    selector = _HandSelector()
    select(selector, [sample(dx=-125, scale=0.7)])
    original = sample(dx=-120, scale=0.7)
    other = sample(dx=110, scale=1.1)
    result = select(selector, [other, original], now=0.033)
    assert result[0][0] == pytest.approx(original[0][0, 0] / 640)


def test_lost_identity_emits_empty_frames_before_acquiring_other_hand():
    selector = _HandSelector()
    select(selector, [sample(dx=-130, scale=0.7)])
    other = sample(dx=130, scale=0.7)
    assert select(selector, [other], now=0.033) == []
    assert select(selector, [other], now=0.10) == []
    assert select(selector, [other], now=0.29) == []
    assert select(selector, [other], now=0.32)


def test_short_occlusion_preserves_identity_but_emits_loss_immediately():
    selector = _HandSelector()
    select(selector, [sample()])
    assert select(selector, [], now=0.033) == []
    result = select(selector, [sample(dx=8)], now=0.067)
    assert result[0][0] == pytest.approx(328 / 640)


@pytest.mark.parametrize("now", [0.5, -0.1])
def test_time_discontinuity_requires_a_loss_boundary(now):
    selector = _HandSelector()
    select(selector, [sample()], now=0)
    assert select(selector, [sample()], now=now) == []
    assert select(selector, [sample()], now=now + 0.033)


def test_no_detection_never_calls_pose_on_whole_frame():
    def forbidden(*args, **kwargs):
        pytest.fail("Pose must not run when no hand is detected")
    model = fake_model(lambda _: np.empty((0, 4)), forbidden)
    assert model.predict(np.zeros((480, 640, 3), np.uint8)) == []


def test_invalid_detector_boxes_do_not_trigger_pose():
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid detector boxes must not trigger pose inference")
    model = fake_model(lambda _: [[0, 0, 1, 1], [0, np.nan, 80, 80]], forbidden)
    assert model.predict(np.zeros((480, 640, 3), np.uint8)) == []


def test_prediction_preserves_bgr_and_bounds_pose_work():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[0, 0] = [10, 20, 30]
    examples = [sample(dx=-150, scale=0.5), sample(), sample(dx=150, scale=0.6)]
    all_boxes = np.array([item[2] for item in examples])
    seen = []

    def pose(image, bboxes):
        assert image is frame
        assert image[0, 0].tolist() == [10, 20, 30]
        seen.append(bboxes)
        indices = [next(i for i, box in enumerate(all_boxes) if np.array_equal(box, candidate))
                   for candidate in bboxes]
        return np.array([examples[i][0] for i in indices]), np.array([examples[i][1] for i in indices])

    model = fake_model(lambda _: all_boxes, pose)
    result = model.predict(frame)
    assert len(seen[0]) == 2
    assert result[0][0] == pytest.approx(0.5)


@pytest.mark.parametrize("frame", [None, np.zeros((5, 5)), np.zeros((5, 5, 4), np.uint8), np.zeros((5, 5, 3))])
def test_invalid_frame_rejected(frame):
    model = fake_model(None, None)
    with pytest.raises(ValueError, match="uint8 BGR"):
        model.predict(frame)


def test_warmup_explicitly_exercises_pose_without_acquiring_a_hand():
    calls = []
    model = fake_model(lambda image: calls.append(("detector", image.shape)),
                       lambda image, bboxes: calls.append(("pose", bboxes.tolist())))
    select(model._selector, [sample()])
    model.warmup()
    assert calls == [("detector", (480, 640, 3)), ("pose", [[200, 100, 440, 380]])]
    assert model._selector.wrist is None


def fake_ort(providers=None, actual=None):
    seen = {}
    session = SimpleNamespace(get_providers=lambda: actual or ["DmlExecutionProvider", "CPUExecutionProvider"],
                              disable_fallback=lambda: seen.update(disabled=True))

    def create(path, **kwargs):
        seen.update(path=path, **kwargs)
        return session

    ort = SimpleNamespace(
        SessionOptions=SimpleNamespace,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        get_available_providers=lambda: providers or ["DmlExecutionProvider"],
        InferenceSession=create)
    return ort, seen


def test_directml_uses_requested_card_and_required_session_options(tmp_path):
    model = tmp_path / "model.onnx"
    model.touch()
    ort, seen = fake_ort()
    _directml_session(model, 2, ort)
    assert seen["providers"] == [("DmlExecutionProvider", {"device_id": 2})]
    assert seen["sess_options"].execution_mode == "sequential"
    assert seen["sess_options"].enable_mem_pattern is False
    assert seen["disabled"] is True


def test_gpu_initialization_cannot_silently_fall_back_to_cpu(tmp_path):
    model = tmp_path / "model.onnx"
    model.touch()
    ort, _ = fake_ort(actual=["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="could not initialize"):
        _directml_session(model, 0, ort)


def test_missing_provider_and_model_are_actionable_errors(tmp_path):
    model = tmp_path / "model.onnx"
    ort, _ = fake_ort(providers=["CPUExecutionProvider"])
    with pytest.raises(FileNotFoundError, match="model is missing"):
        _directml_session(model, 0, ort)
    model.touch()
    with pytest.raises(RuntimeError, match="no DirectML"):
        _directml_session(model, 0, ort)


def test_model_input_size_requires_fixed_nchw():
    assert _input_size(SimpleNamespace(get_inputs=lambda: [SimpleNamespace(shape=[1, 3, 256, 192])])) == (192, 256)
    with pytest.raises(ValueError, match="fixed-size NCHW"):
        _input_size(SimpleNamespace(get_inputs=lambda: [SimpleNamespace(shape=[1, 3, "height", "width"])]))


def test_pose_pipeline_converts_bgr_to_rgb_but_detector_preserves_bgr(monkeypatch):
    class FakeTool:
        def preprocess(self, image, bbox=None):
            return image, bbox

    monkeypatch.setitem(sys.modules, "rtmlib.tools.object_detection.rtmdet", SimpleNamespace(RTMDet=FakeTool))
    monkeypatch.setitem(sys.modules, "rtmlib.tools.pose_estimation.rtmpose", SimpleNamespace(RTMPose=FakeTool))
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(shape=[1, 3, 256, 256])])
    detector, pose = _local_tools(session, session)
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    box = [0, 0, 2, 1]
    converted, returned_box = pose.preprocess(image, box)
    assert converted.tolist() == [[[30, 20, 10], [60, 50, 40]]]
    assert returned_box is box
    assert detector.preprocess(image)[0].tolist() == [[[10, 20, 30], [40, 50, 60]]]
    assert image.tolist() == [[[10, 20, 30], [40, 50, 60]]]
