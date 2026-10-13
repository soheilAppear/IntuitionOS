"""One-second causal waveform learning and stream boundaries; no devices."""

import copy
import importlib
import json

import numpy as np
import pytest

from core.eeg_fast_blink import (FastCnnBlinkDecoder, FEATURE_PROFILE, FORMAT, HOP_SAMPLES,
                                PARAMETER_COUNT, _augmentation, _logits, _scores, _torch_network,
                                preprocess_waveform)


CHANNELS = [{"num": i, "name": name} for i, name in enumerate(("O1", "O2", "T3", "T4"))]
TRAIN_TRIALS = 16


def waveform(seed, positive=False, *, kind=None, dc=True, ramp=True):
    rng = np.random.default_rng(seed)
    t = np.arange(250) / 250
    values = rng.normal(0, 1.8e-6, (250, 4))
    values += 2e-6 * np.sin(2 * np.pi * rng.uniform(8, 12) * t[:, None] + np.arange(4))
    gains = np.array([.75, .85, 1, .9]) * rng.uniform(.8, 1.2, 4)
    if positive:
        second = rng.uniform(.6, .86)
        centers, width = (second - rng.uniform(.18, .4), second), rng.uniform(.035, .055)
    else:
        kind = kind or ("quiet", "single", "movement", "separated", "muscle")[seed % 5]
        centers = ([rng.uniform(.2, .86)] if kind in ("single", "movement") else
                   [rng.uniform(.05, .12), rng.uniform(.82, .9)] if kind == "separated" else [])
        width = rng.uniform(.14, .22) if kind == "movement" else rng.uniform(.035, .055)
        if kind == "muscle":
            values += rng.normal(0, 15e-6, values.shape) * np.exp(-.5 * ((t - .5) / .10) ** 2)[:, None]
    for center in centers:
        values += rng.uniform(110e-6, 210e-6) * np.exp(-.5 * ((t - center) / width) ** 2)[:, None] * gains
    offset, drift = rng.uniform(-.03, .03, 4), rng.uniform(-60e-6, 60e-6, 4)
    if dc:
        values += offset
    if ramp:
        values += t[:, None] * drift
    return values


def packet_signal(values, *, start=100, counter=0, session="fast-test"):
    return {"session_id": session, "nominal_hz": 250, "units": "V", "channels": copy.deepcopy(CHANNELS),
            "samples": [{"counter": (counter + i) % 2**32, "samples": row.tolist(),
                         "estimated_monotonic": start + i / 250,
                         "host_received_monotonic": start + i / 250 + .01}
                        for i, row in enumerate(values)]}


def signal(index=0, label="rest", *, kind=None):
    return packet_signal(waveform(index, label == "select", kind=kind),
                         start=100 + index * 2, counter=index * 500)


def add(decoder, index, label, *, phase="train", kind=None):
    result = decoder.add_trial(label, signal(index, label, kind=kind), phase=phase,
                               trial_id=index, start=100 + index * 2, end=101 + index * 2)
    assert result["ok"], result.get("error")
    return result


@pytest.fixture(scope="module")
def trained_decoder():
    pytest.importorskip("torch")
    decoder = FastCnnBlinkDecoder()
    for index in range(TRAIN_TRIALS):
        add(decoder, index, "select" if index % 2 else "rest")
    result = decoder.train()
    assert result["ok"], result.get("error")
    return decoder


def clone(decoder):
    return FastCnnBlinkDecoder.from_state(decoder.export_state())


def continuous(*, centers=(), seconds=8, counter=0, kind=None):
    rng = np.random.default_rng(5)
    t = np.arange(seconds * 250) / 250
    values = rng.normal(0, 2e-6, (len(t), 4))
    if centers:
        values += sum(180e-6 * np.exp(-.5 * ((t - center) / .05) ** 2) for center in centers)[:, None] * np.array([.75, .85, 1, .9])
    if kind == "movement":
        values += 180e-6 * np.exp(-.5 * ((t - 3) / .2) ** 2)[:, None] * np.array([.75, .85, 1, .9])
    if kind == "muscle":
        values += rng.normal(0, 15e-6, values.shape) * np.exp(-.5 * ((t - 3) / .10) ** 2)[:, None]
    return packet_signal(values, counter=counter)


def run_stream(decoder, stream, *, start=250, end=None):
    end = len(stream["samples"]) if end is None else end
    results = []
    for stop in range(start, end + 1, HOP_SAMPLES):
        snapshot = {**stream, "samples": stream["samples"][max(0, stop - 1250):stop]}
        now = snapshot["samples"][-1]["host_received_monotonic"] + .01
        results.append(decoder.update(snapshot, now=now))
    return results


def test_preprocessing_is_one_second_and_removes_dc_and_linear_drift():
    a = preprocess_waveform(waveform(1, True, dc=False, ramp=False))
    b = preprocess_waveform(waveform(1, True))
    assert a.shape == (4, 125)
    np.testing.assert_allclose(a, b, atol=2e-5)
    with pytest.raises(ValueError):
        preprocess_waveform(np.zeros((1250, 4)))


def test_torch_numpy_endpoint_parity_and_small_architecture():
    torch = pytest.importorskip("torch")
    torch.manual_seed(93)
    model = _torch_network(torch).eval()
    values = np.random.default_rng(18).normal(0, 1, (3, 4, 125)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(values)).numpy()
    weights = {name: tensor.detach().numpy() for name, tensor in model.state_dict().items()}
    assert sum(t.size for t in weights.values()) == PARAMETER_COUNT == 5270
    np.testing.assert_allclose(_logits(values, weights), expected, atol=2e-6, rtol=2e-5)


def test_observed_development_trials_validate_without_changing_frozen_weights(trained_decoder):
    decoder = clone(trained_decoder)
    before = copy.deepcopy(decoder._model)
    for index in range(16, 32):
        add(decoder, index, "select" if index % 2 else "rest", phase="validate")
    status = decoder.status()
    assert status["passed_validation"], status["evaluation"]
    assert status["evaluation"]["rest_false_activations"] == 0
    assert decoder._model == before
    assert status["window_seconds"] == status["trial_seconds"] == 1.0
    assert status["prediction_interval_seconds"] == .04
    assert status["validation_scope"] == "held_out_one_second_trials"
    assert "not measured physical blink-completion latency" in status["disclaimer"]


def test_independent_final_trials_reject_insufficient_generalization_without_retuning(trained_decoder):
    # Seeds 0..15 train, 16..31 are development regressions. This separate
    # seed range was not used to choose architecture, augmentation, or thresholds.
    # Synthetic results do not estimate this headset user's accuracy.
    decoder = clone(trained_decoder)
    before = decoder.export_state()["model"]
    for index in range(1000, 1016):
        add(decoder, index, "select" if index % 2 else "rest", phase="validate")
    status = decoder.status()
    # This final assessment exposed insufficient positive recall (5/8), even
    # though all controls were rejected. Preserve that failure: a passing
    # development example must not silently authorize an unproven live model.
    assert status["evaluation"]["confusion"]["select"] == {"rest": 3, "select": 5, "uncertain": 0}
    assert status["evaluation"]["rest_false_activations"] == 0
    assert not status["passed_validation"] and not status["arm_eligible"]
    assert not decoder.train()["ok"]  # Held-out evidence cannot refit this model.
    assert decoder.export_state()["model"] == before


def test_control_augmentation_translates_without_compressing_wide_pairs():
    class FixedRng:
        def integers(self, *_):
            return 0
        def uniform(self, _low, _high, shape):
            return np.ones(shape)
        def normal(self, _mean, _sd, shape):
            return np.zeros(shape)
    values = np.zeros((4, 125), dtype=np.float32)
    values[:, [12, 108]] = 1
    result = _augmentation(values, FixedRng(), command=False)
    # The two original peaks stay 96 samples apart; controls cannot become a
    # positive-looking fast pair through time compression during training.
    assert np.array_equal(np.sort(np.argsort(result[0])[-2:]), [12, 108])


def test_bad_heldout_labels_are_retained_and_cannot_tune_model(trained_decoder):
    decoder = clone(trained_decoder)
    before = decoder.export_state()["model"]
    for index in range(16, 24):
        result = decoder.add_trial("rest", signal(index, "select"), phase="validate", trial_id=index,
                                   start=100 + index * 2, end=101 + index * 2)
        assert result["ok"]
    assert decoder.status()["evaluation"]["rest_false_activations"] > 0
    assert decoder.export_state()["model"] == before and not decoder.status()["passed_validation"]


def test_archive_roundtrip_is_inert_and_distinct_from_original_cnn(trained_decoder):
    state = trained_decoder.export_state()
    assert state["format"] == FORMAT and state["feature_profile"] == FEATURE_PROFILE
    restored = FastCnnBlinkDecoder.from_state(json.loads(json.dumps(state, allow_nan=False)))
    assert restored.export_state() == state
    original = copy.deepcopy(state)
    original["format"] = "intuitionos.eeg-cnn-blink-model"
    with pytest.raises(ValueError):
        FastCnnBlinkDecoder.from_state(original)


@pytest.mark.parametrize("fault", ["shape", "nan", "huge", "architecture", "overlap", "threshold", "duration", "profile"])
def test_archive_rejects_malformed_or_inconsistent_state(trained_decoder, fault):
    state = trained_decoder.export_state()
    if fault == "shape":
        state["model"]["weights"]["conv4.bias"].pop()
    elif fault == "nan":
        state["model"]["weights"]["conv1.bias"][0] = float("nan")
    elif fault == "huge":
        state["training"][0]["waveform"][0][0] = 100
    elif fault == "architecture":
        state["model"]["architecture"] = "untrusted-module"
    elif fault == "overlap":
        state["training"][1]["counter_start"] = state["training"][0]["counter_start"]
    elif fault == "threshold":
        state["model"]["threshold"] = .98
    elif fault == "duration":
        state["training"][0]["end"] += 4
    else:
        state["feature_profile"] = "cnn_blink_waveform_v1"
    with pytest.raises(ValueError):
        FastCnnBlinkDecoder.from_state(state)


def test_trial_rejects_incomplete_old_duration_reused_packets_and_bad_quality():
    decoder = FastCnnBlinkDecoder()
    add(decoder, 0, "rest")
    assert not decoder.add_trial("select", signal(), trial_id=1, start=100, end=101)["ok"]
    assert not decoder.add_trial("rest", signal(1), trial_id=1, start=102, end=107)["ok"]
    short = signal(1)
    short["samples"].pop()
    assert not decoder.add_trial("rest", short, trial_id=1, start=102, end=103)["ok"]
    bad = signal(1)
    bad["samples"][100]["samples"][0] += .02
    assert not decoder.add_trial("rest", bad, trial_id=1, start=102, end=103)["ok"]


@pytest.mark.parametrize("centers", [(3, 3.3), (3, 3.3, 4.5, 4.8), (3, 3.3, 5, 5.3)])
def test_causal_pair_events_finish_after_second_peak_without_duplicate_clicks(trained_decoder, centers):
    decoder = clone(trained_decoder)
    results = run_stream(decoder, continuous(centers=centers))
    assert all(p["valid"] for p in results)
    events = [p for p in results if p["event_id"]]
    assert len(events) == len(centers) // 2, [(p["event_at"], p["confidence"]) for p in events]
    for event, second_peak in zip(events, centers[1::2]):
        # Known synthetic waveform peak, not a measured human reopening time.
        assert 0 < event["event_at"] - (100 + second_peak) < .8
        assert event["eligible_for_event"] and not event["ready_for_blink"]
    assert len({p["event_id"] for p in events}) == len(events)


@pytest.mark.parametrize("centers,kind", [((3,), None), ((3, 3.9), None), ((), None),
                                         ((), "movement"), ((), "muscle")])
def test_single_blink_slow_pair_and_artifact_controls_do_not_click(trained_decoder, centers, kind):
    assert not any(p["event_id"] for p in run_stream(clone(trained_decoder), continuous(centers=centers, kind=kind)))


def test_full_adapter_cache_starts_with_only_fresh_one_second_context(trained_decoder):
    decoder, stream = clone(trained_decoder), continuous()
    first = run_stream(decoder, stream, start=1250, end=1250)[0]
    assert first["valid"] and not first["ready"]
    results = run_stream(decoder, stream, start=1260, end=1500)
    assert all(p["valid"] for p in results) and results[-1]["ready"]
    assert decoder.status()["fresh_context_seconds"] == 1
    decoder.reset_stream()
    again = run_stream(decoder, stream, start=1500, end=1540)
    assert all(p["valid"] and not p["ready"] and p["event_id"] is None for p in again)


def test_old_outside_window_noise_does_not_extend_quality_gate_to_five_seconds(trained_decoder):
    decoder, stream = clone(trained_decoder), continuous()
    stream["samples"][100]["samples"][0] = .03
    assert run_stream(decoder, stream, start=1250, end=1510)[-1]["ready"]


@pytest.mark.parametrize("fault", ["stale", "gap", "session", "units", "backwards", "nan", "flat", "clip", "oversize"])
def test_stream_faults_clear_history_without_replacing_frozen_model(trained_decoder, fault):
    decoder, stream = clone(trained_decoder), continuous()
    run_stream(decoder, stream, end=650)
    snapshot = {**stream, "samples": copy.deepcopy(stream["samples"][:660])}
    now = snapshot["samples"][-1]["host_received_monotonic"] + .01
    if fault == "stale":
        now += 1
    elif fault == "gap":
        snapshot["samples"].pop(-2)
    elif fault == "session":
        snapshot["session_id"] = "different"
    elif fault == "units":
        snapshot["units"] = "uV"
    elif fault == "backwards":
        for row in snapshot["samples"]:
            row["counter"] = (row["counter"] - 200) % 2**32
    elif fault == "nan":
        snapshot["samples"][-2]["samples"][0] = float("nan")
    elif fault == "flat":
        for row in snapshot["samples"]:
            row["samples"][0] = 0
    elif fault == "clip":
        snapshot["samples"][-1]["samples"][0] += .02
    else:
        snapshot["samples"] *= 2
    result = decoder.update(snapshot, now=now)
    assert not result["valid"] and not result["ready"] and result["event_id"] is None
    assert decoder.status()["trained"] and decoder.status()["fresh_context_seconds"] == 0


def test_counter_wrap_and_repeated_snapshot_never_replay_events(trained_decoder):
    decoder, stream = clone(trained_decoder), continuous(centers=(3, 3.3), counter=2**32 - 500)
    events = [p for p in run_stream(decoder, stream) if p["event_id"]]
    assert len(events) == 1
    repeated = run_stream(decoder, stream, start=2000, end=2000)[0]
    assert repeated["valid"] and not repeated["event_id"]


def test_amplitude_pause_requires_fresh_context_and_rest_before_a_new_event(trained_decoder, monkeypatch):
    module = importlib.import_module("core.eeg_fast_blink")
    decoder, stream = clone(trained_decoder), continuous()
    stream["samples"][655]["samples"][0] += .02
    phase = {"score": .1}
    monkeypatch.setattr(module, "_scores", lambda *_: np.array([phase["score"]]))
    before = decoder.export_state()
    run_stream(decoder, stream, end=630)
    phase["score"] = .99
    assert not any(p["event_id"] for p in run_stream(decoder, stream, start=640, end=650))
    assert decoder._select_count == 2  # An unfinished command must be discarded.
    fault = run_stream(decoder, stream, start=660, end=660)[0]
    assert (fault["fault_code"], fault["fault_category"], fault["recoverable"]) == (
        "signal_amplitude", "signal_quality", True)
    assert not fault["valid"] and not fault["ready"] and not fault["event_id"]
    assert fault["fault_details"]["channel"] == "O1"
    assert fault["fault_details"]["peak_to_peak_v"] > fault["fault_details"]["limit_v"] == .010
    assert decoder._counter is None and decoder._paused_transport_boundary[0] == 659
    assert decoder._select_count == 0 and decoder.status()["fresh_context_seconds"] == 0
    # The offending packet remains in the newest second through stop=900.
    pauses = run_stream(decoder, stream, start=670, end=900)
    assert all(p["recoverable"] and not p["event_id"] for p in pauses)
    clean = run_stream(decoder, stream, start=910, end=910)[0]
    assert clean["valid"] and not clean["ready"] and not clean["recoverable"]
    assert decoder.status()["fresh_context_seconds"] == 0
    warmup = run_stream(decoder, stream, start=920, end=1150)
    assert all(p["valid"] and not p["ready"] and not p["event_id"] for p in warmup)
    high = run_stream(decoder, stream, start=1160, end=1200)
    assert all(p["ready"] and not p["event_id"] and not p["ready_for_blink"] for p in high)
    phase["score"] = .1
    assert run_stream(decoder, stream, start=1210, end=1230)[-1]["ready_for_blink"]
    phase["score"] = .99
    assert sum(bool(p["event_id"]) for p in run_stream(decoder, stream, start=1240, end=1260)) == 1
    assert decoder.export_state() == before


@pytest.mark.parametrize("already_paused", [False, True])
@pytest.mark.parametrize("fault,code", [
    ("stale", "buffer_stale"), ("gap", "packet_discontinuity"),
    ("boundary_gap", "packet_discontinuity"), ("backwards", "counter_backwards"),
    ("delivery", "delivery_paused"), ("session", "context_changed"),
    ("units", "context_invalid"), ("nan", "packet_invalid"),
    ("receipt_nan", "packet_invalid"), ("flat", "signal_flatline"),
    ("identical", "signal_identical"), ("plateau", "signal_plateau"),
    ("same_channel_plateau", "signal_plateau"),
])
def test_hard_faults_dominate_amplitude_in_every_channel_and_between_paused_buffers(
        trained_decoder, already_paused, fault, code):
    decoder, stream = clone(trained_decoder), continuous()
    run_stream(decoder, stream, end=650)
    stop = 660
    if already_paused:
        first = {**stream, "samples": copy.deepcopy(stream["samples"][:660])}
        first["samples"][-1]["samples"][0] += .02
        assert decoder.update(first, now=first["samples"][-1]["host_received_monotonic"] + .01)["recoverable"]
        stop = 670
    snapshot = {**stream, "samples": copy.deepcopy(stream["samples"][:stop])}
    snapshot["samples"][-1]["samples"][0] += .02
    now = snapshot["samples"][-1]["host_received_monotonic"] + .01
    if fault == "stale":
        now += 1
    elif fault == "gap":
        snapshot["samples"].pop(-2)
    elif fault in ("boundary_gap", "backwards"):
        for row in snapshot["samples"]:
            row["counter"] = (row["counter"] + (1000 if fault == "boundary_gap" else -200)) % 2**32
    elif fault == "delivery":
        snapshot["samples"][-10]["estimated_monotonic"] += 1
    elif fault == "session":
        snapshot["session_id"] = "other"
    elif fault == "units":
        snapshot["units"] = "uV"
    elif fault == "nan":
        snapshot["samples"][-2]["samples"][1] = float("nan")
    elif fault == "receipt_nan":
        snapshot["samples"][-2]["host_received_monotonic"] = float("nan")
    elif fault == "flat":
        for row in snapshot["samples"]:
            row["samples"][1] = 0
    elif fault == "identical":
        for row in snapshot["samples"][-40:-15]:
            row["samples"][1] = 1e-6
    else:
        for row in snapshot["samples"][-20:-15]:
            row["samples"][0 if fault == "same_channel_plateau" else 1] = .03
    result = decoder.update(snapshot, now=now)
    assert result["fault_code"] == code
    assert not result["recoverable"] and not result["valid"] and not result["event_id"]
    assert decoder._counter is None and decoder._paused_transport_boundary is None
    assert decoder.status()["trained"]


def test_large_absolute_dc_does_not_trigger_amplitude_pause(trained_decoder):
    decoder, stream = clone(trained_decoder), continuous()
    for row in stream["samples"]:
        row["samples"] = [v + .03 for v in row["samples"]]
    results = run_stream(decoder, stream, end=650)
    assert all(p["valid"] and not p["recoverable"] and p["fault_code"] is None for p in results)


def test_unknown_fault_remains_hard_and_explicit_reset_clears_pause_boundary(trained_decoder, monkeypatch):
    module = importlib.import_module("core.eeg_fast_blink")
    decoder, stream = clone(trained_decoder), continuous()
    snapshot = {**stream, "samples": copy.deepcopy(stream["samples"][:250])}
    snapshot["samples"][-1]["samples"][0] += .02
    assert decoder.update(snapshot, now=101.01)["recoverable"]
    decoder.reset_stream()
    assert decoder._paused_transport_boundary is None
    def unexpected(*_):
        raise ValueError("unclassified input failure")
    monkeypatch.setattr(module, "_quality", unexpected)
    result = run_stream(decoder, stream, start=250, end=250)[0]
    assert result["fault_code"] == "input_invalid" and not result["recoverable"]


def test_finger_cancel_anchors_next_batch_and_excludes_old_evidence(trained_decoder, monkeypatch):
    module = importlib.import_module("core.eeg_fast_blink")
    decoder, stream = clone(trained_decoder), continuous()
    phase = {"score": .1}
    monkeypatch.setattr(module, "_scores", lambda *_: np.array([phase["score"]]))
    run_stream(decoder, stream, end=650)
    before, history = decoder.export_state(), decoder._live_rows
    decoder.cancel_pending()
    assert decoder.status()["suppressed_evidence_remaining_seconds"] == 1
    assert decoder._live_rows is history and len(history) == 250
    phase["score"] = .99
    assert not any(p["event_id"] for p in run_stream(decoder, stream, start=660, end=850))
    assert decoder._event_watermark == 659
    phase["score"] = .1
    run_stream(decoder, stream, start=860, end=900)
    assert not decoder.status()["ready_for_blink"]
    phase["score"] = .99
    assert not any(p["event_id"] for p in run_stream(decoder, stream, start=910, end=950))
    phase["score"] = .1
    run_stream(decoder, stream, start=960, end=980)
    assert decoder.status()["ready_for_blink"]
    phase["score"] = .99
    events = run_stream(decoder, stream, start=990, end=1010)
    assert sum(bool(p["event_id"]) for p in events) == 1
    assert decoder.export_state() == before


def test_optional_torch_failure_is_actionable_and_never_requires_startup_import(monkeypatch):
    import builtins
    decoder = FastCnnBlinkDecoder()
    for index in range(16):
        add(decoder, index, "select" if index % 2 else "rest")
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("test optional package absent")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    result = decoder.train()
    assert not result["ok"] and "pip install torch" in result["error"]
