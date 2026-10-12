"""Synthetic blink waveforms only; no headset, files, camera or desktop input."""

import copy
import json

import numpy as np
import pytest

from core.eeg_blink import BlinkDecoder, FEATURE_PROFILE, VERSION, _aligned_similarity, _noise, _pulse_features, _rows


CHANNELS = [{"num": i, "name": name} for i, name in enumerate(("O1", "O2", "T3", "T4"))]


def signal(index=0, label="rest", *, session="blink-session", kind=None, count=1250, spacing=.55):
    rng = np.random.default_rng(100 + index)
    t = np.arange(count) / 250
    values = rng.normal(0, 2e-6, (count, 4))
    values += 3e-6 * np.sin(2 * np.pi * 10 * t[:, None] + np.arange(4)[None, :])
    centers = (1.5, 1.5 + spacing) if label == "select" else ()
    if kind == "single":
        centers = (1.5,)
    elif kind == "slow":
        centers = (1.5, 3.0)
    elif kind == "triple":
        centers = (1.5, 2.05, 2.60)
    gains = np.array([0.7, 0.8, 1.0, 0.9]) * (1 + (index % 7 - 3) * .012)
    for center in centers:
        pulse = 240e-6 * np.exp(-0.5 * ((t - center) / .045)**2)
        values += pulse[:, None] * gains
    if kind == "jaw":
        values += ((t > 1.4) & (t < 2.5))[:, None] * np.sin(2 * np.pi * 60 * t[:, None]) * np.array([1, -.8, .6, -.9]) * 150e-6
    if kind == "drift":
        values += (np.clip(t - 1, 0, 2) * 1e-3)[:, None] * np.array([1, -.2, .4, .7])
    start = 100 + index * 6
    return {"session_id": session, "nominal_hz": 250, "channels": copy.deepcopy(CHANNELS),
            "samples": [{"counter": (index * 1500 + n) % 2**32, "estimated_monotonic": start + n / 250,
                         "samples": row.tolist()} for n, row in enumerate(values)]}


def add(decoder, index, label, *, phase="train", kind=None):
    stream = signal(index, label, kind=kind)
    return decoder.add_trial(label, stream, phase=phase, trial_id=index, start=100 + index * 6, end=105 + index * 6)


def trained_decoder(*, validate=False):
    decoder = BlinkDecoder()
    for i in range(16):
        label = "rest" if i % 2 == 0 else "select"
        result = add(decoder, i, label, kind="single" if label == "rest" and i % 4 == 0 else None)
        assert result["ok"], result.get("error")
    result = decoder.train()
    assert result["ok"], result.get("error")
    if validate:
        for i in range(16, 32):
            label = "rest" if i % 2 == 0 else "select"
            assert add(decoder, i, label, phase="validate")["ok"]
    return decoder


def playback(decoder, stream, *, batch=25):
    results = []
    # Initial half-second prefix is ignored; the following half-second warms the causal filter.
    decoder.update({**stream, "samples": stream["samples"][:125]})
    for end in range(125 + batch, len(stream["samples"]) + 1, batch):
        results.append(decoder.update({**stream, "samples": stream["samples"][max(0, end - 1250):end]}))
    return results


def events(results):
    return [r for r in results if r["event_id"] is not None]


def test_personalized_training_validation_and_json_roundtrip():
    decoder = trained_decoder(validate=True)
    assert decoder.status()["passed_validation"]
    state = json.loads(json.dumps(decoder.export_state(), allow_nan=False))
    restored = BlinkDecoder.from_state(state)
    assert restored.export_state() == state
    assert restored.status() == decoder.status()
    assert '"samples":' not in json.dumps(state)


@pytest.mark.parametrize("kind,expected", [(None, 1), ("single", 0), ("slow", 0), ("jaw", 0), ("drift", 0)])
def test_only_completed_learned_double_blinks_emit_once(kind, expected):
    decoder = trained_decoder()
    stream = signal(40, "select" if kind in (None, "triple", "slow", "single") else "rest", kind=kind)
    results = playback(decoder, stream)
    assert len(events(results)) == expected, [(r["reason"], r["event_id"]) for r in results]
    if expected:
        event = events(results)[0]
        assert event["label"] == "select" and event["valid"] and event["ready"]
        assert event["event_at"] > stream["samples"][round(2.05 * 250)]["estimated_monotonic"]
    for _ in range(4):
        assert decoder.update(stream)["event_id"] is None


def test_three_blinks_cannot_reuse_a_pulse_to_emit_two_commands():
    decoder = trained_decoder()
    assert len(events(playback(decoder, signal(40, "select", kind="triple")))) <= 1


def test_reset_and_binding_never_replay_buffer_or_alter_original_evidence():
    decoder = trained_decoder(validate=True)
    original = decoder.export_state()
    stream = signal(40, "select", session="new-session")
    assert not decoder.update(stream)["valid"]
    assert decoder.bind_prediction_context(stream) == {"ok": True}
    assert decoder.update(stream)["event_id"] is None
    decoder.reset_stream()
    assert decoder.update(stream)["event_id"] is None
    assert decoder.export_state() == original
    assert decoder.status()["inference_context_bound"]
    assert not add(decoder, 50, "rest")["ok"]
    altered = copy.deepcopy(stream)
    altered["channels"][0]["name"] = "different"
    with pytest.raises(ValueError, match="layout"):
        decoder.bind_prediction_context(altered)
    assert decoder.export_state() == original


@pytest.mark.parametrize("fault", ["gap", "nan", "amplitude", "clip", "flat", "session", "rate", "channels"])
def test_raw_faults_cancel_and_cannot_be_reinterpreted_as_blinks(fault):
    decoder = trained_decoder()
    stream = signal(40, "select")
    decoder.update({**stream, "samples": stream["samples"][:250]})
    damaged = {**stream, "samples": copy.deepcopy(stream["samples"][:700])}
    if fault == "gap":
        del damaged["samples"][450]
    elif fault == "nan":
        damaged["samples"][-10]["samples"][0] = float("nan")
    elif fault == "amplitude":
        damaged["samples"][-10]["samples"][0] = .02
    elif fault == "clip":
        for row in damaged["samples"][-10:]:
            row["samples"][0] = .005
    elif fault == "flat":
        for row in damaged["samples"]:
            row["samples"][0] = 0.0
    elif fault == "session":
        damaged["session_id"] = "other"
    elif fault == "rate":
        damaged["nominal_hz"] = 500
    else:
        damaged["channels"] = copy.deepcopy(CHANNELS)
        damaged["channels"][1]["name"] = "different"
    result = decoder.update(damaged)
    assert not result["valid"] and result["event_id"] is None
    assert not result["first_blink_pending"]


def test_training_rejects_overlap_duplicate_incomplete_and_invalid_positive():
    decoder = BlinkDecoder()
    assert add(decoder, 0, "rest")["ok"]
    assert not add(decoder, 0, "rest")["ok"]
    assert not add(decoder, 1, "select", kind="single")["ok"]
    stream = signal(2)
    assert not decoder.add_trial("rest", stream, trial_id=2, start=112, end=115)["ok"]
    assert not decoder.train()["ok"]


def test_no_command_matching_double_blink_blocks_training():
    decoder = BlinkDecoder()
    for i in range(16):
        stream = signal(i, "select")
        assert decoder.add_trial("rest" if i % 2 == 0 else "select", stream, trial_id=i,
                                 start=100 + i * 6, end=105 + i * 6)["ok"]
    assert not decoder.train()["ok"]
    assert not decoder.status()["trained"]


def test_failed_validation_survives_save_load_without_becoming_eligible():
    decoder = trained_decoder()
    for i in range(16, 32):
        label = "rest" if i % 2 == 0 else "select"
        stream = signal(i, "select")  # Every no-command trial is deliberately a false command.
        assert decoder.add_trial(label, stream, phase="validate", trial_id=i, start=100 + i * 6, end=105 + i * 6)["ok"]
    assert not decoder.status()["passed_validation"]
    restored = BlinkDecoder.from_state(decoder.export_state())
    assert restored.status() == decoder.status()
    assert not restored.status()["arm_eligible"]


@pytest.mark.parametrize("change", [
    lambda s: s.update(feature_profile="click_4_30hz_v1"),
    lambda s: s["model"]["templates"][0].__setitem__(0, .95),
    lambda s: s["training"][0]["noise"].__setitem__(0, 0),
    lambda s: s["training"][0].update(id=s["training"][1]["id"]),
    lambda s: s["validation"][0].update(event_count=1, predicted="select"),
])
def test_corrupt_archives_are_rejected(change):
    state = trained_decoder(validate=True).export_state()
    change(state)
    with pytest.raises((ValueError, TypeError)):
        BlinkDecoder.from_state(state)


def test_stream_batch_partition_does_not_change_event_identity_or_time():
    small, large = trained_decoder(), trained_decoder()
    stream = signal(40, "select")
    # Both cover the same waveform with different packet delivery batches.
    a = events(playback(small, stream, batch=5))
    b = events(playback(large, stream, batch=25))
    assert len(a) == len(b) == 1
    assert a[0]["event_id"] == b[0]["event_id"]
    assert a[0]["event_at"] == b[0]["event_at"]


def test_packet_counter_wrap_is_contiguous_and_event_identity_is_stable():
    decoder = trained_decoder()
    stream = signal(40, "select")
    for i, row in enumerate(stream["samples"]):
        row["counter"] = (2**32 - 400 + i) % 2**32
    assert len(events(playback(decoder, stream))) == 1


def test_delayed_backlog_cannot_release_an_old_gesture():
    decoder = trained_decoder()
    stream = signal(40, "select")
    decoder.update({**stream, "samples": stream["samples"][:250]})
    result = decoder.update(stream)
    assert not result["valid"] and "backlog" in result["reason"]
    assert result["event_id"] is None
    assert decoder.update(stream)["event_id"] is None


def test_wrong_polarity_waveform_is_not_a_learned_blink():
    decoder = trained_decoder()
    stream = signal(40, "select")
    for row in stream["samples"]:
        row["samples"] = [-v for v in row["samples"]]
    assert not events(playback(decoder, stream))


def test_first_pulse_is_not_carried_across_stream_reset():
    decoder = trained_decoder()
    stream = signal(40, "select")
    decoder.update({**stream, "samples": stream["samples"][:125]})
    for end in range(150, 501, 25):
        assert decoder.update({**stream, "samples": stream["samples"][:end]})["event_id"] is None
    decoder.reset_stream()
    results = [decoder.update({**stream, "samples": stream["samples"][:end]}) for end in range(525, 1251, 25)]
    assert not events(results)


@pytest.mark.parametrize("value", [10**500, float("nan"), True, [], "0.8"])
def test_malformed_model_threshold_is_rejected_as_value_error(value):
    state = trained_decoder().export_state()
    state["model"]["threshold"] = value
    with pytest.raises(ValueError):
        BlinkDecoder.from_state(state)


@pytest.mark.parametrize("spacing", [.2, .3, .4, .5, .6, .8])
def test_quick_and_slow_double_blinks_train_validate_and_emit_once(spacing):
    decoder = BlinkDecoder()
    for i in range(32):
        label = "rest" if i % 2 == 0 else "select"
        stream = signal(i, label, spacing=spacing, kind="single" if label == "rest" else None)
        result = decoder.add_trial(label, stream, trial_id=i, phase="train" if i < 16 else "validate",
                                   start=100 + i * 6, end=105 + i * 6)
        assert result["ok"], result.get("error")
        if i == 15:
            result = decoder.train()
            assert result["ok"], result.get("error")
    assert decoder.status()["passed_validation"], decoder.status()["evaluation"]
    assert len(events(playback(decoder, signal(40, "select", spacing=spacing)))) == 1
    decoder.reset_stream()
    assert not events(playback(decoder, signal(41, "rest", kind="single")))


def test_v1_archive_retains_exact_original_features_model_and_behavior():
    legacy = BlinkDecoder()
    legacy._version = 1  # Reproduce the explicitly supported original archive path.
    for i in range(32):
        label = "rest" if i % 2 == 0 else "select"
        assert add(legacy, i, label, phase="train" if i < 16 else "validate")["ok"]
        if i == 15:
            assert legacy.train()["ok"]
    before = legacy.export_state()
    restored = BlinkDecoder.from_state(copy.deepcopy(before))
    assert before["version"] == restored.status()["detector_version"] == 1
    assert restored.export_state() == before
    assert restored.status() == legacy.status()
    assert len(events(playback(restored, signal(40, "select")))) == 1
    restored.reset_stream()
    assert not events(playback(restored, signal(41, "select", spacing=.3)))


def test_retained_legacy_untrained_trials_can_train_with_new_quick_pairs():
    decoder = BlinkDecoder()
    decoder._version = 1
    for i in range(11):  # Preserve the current user's eight no-command and three positive trial shape contract.
        label = "rest" if i < 8 else "select"
        assert add(decoder, i, label)["ok"]
    retained = copy.deepcopy(decoder._training)
    decoder._version = 2  # A new unfrozen classifier consumes the unchanged compatible waveform features.
    for i in range(11, 16):
        stream = signal(i, "select", spacing=(.2, .3, .4, .5, .6)[i - 11])
        result = decoder.add_trial("select", stream, trial_id=i, start=100 + i * 6, end=105 + i * 6)
        assert result["ok"], result.get("error")
    assert decoder._training[:11] == retained
    trained = decoder.train()
    assert trained["ok"], trained.get("error")
    state = decoder.export_state()
    assert state["version"] == 2 and state["training"][:11] == retained
    assert BlinkDecoder.from_state(state).export_state() == state
    for i, spacing in enumerate((.2, .3, .4, .5, .6, .8)):
        decoder.reset_stream()
        assert len(events(playback(decoder, signal(40 + i, "select", spacing=spacing)))) == 1


@pytest.mark.parametrize("spacing", [.05, .10, .14, 1.0, 1.3])
def test_outside_pair_timing_cannot_become_a_double_blink(spacing):
    decoder = trained_decoder()
    assert not events(playback(decoder, signal(40, "select", spacing=spacing)))


def test_splitting_does_not_change_any_isolated_pulse_feature():
    stream = signal(40, "rest", kind="single")
    scale = _noise(_rows(stream["samples"]))
    assert _pulse_features(stream["samples"], scale, split=False) == _pulse_features(stream["samples"], scale, split=True)


def test_partial_legacy_calibration_draft_roundtrips_and_upgrades_explicitly():
    legacy = BlinkDecoder()
    legacy._version = 1
    for i in range(11):
        assert add(legacy, i, "rest" if i < 8 else "select")["ok"]
    state = legacy.export_draft()
    preserved = BlinkDecoder.from_calibration_state(json.loads(json.dumps(state)))
    assert preserved.export_draft() == state and preserved.status()["detector_version"] == 1
    upgraded = BlinkDecoder.from_calibration_state(state, upgrade=True)
    assert upgraded.status()["detector_version"] == VERSION
    assert upgraded.status()["train_counts"] == {"rest": 8, "select": 3}
    assert not upgraded.status()["trained"] and not upgraded.status()["arm_eligible"]
    assert upgraded._training == legacy._training and upgraded._context == legacy._context
    assert upgraded.update(signal(40, "select"))["event_id"] is None
    with pytest.raises(ValueError):
        BlinkDecoder.from_state(state)  # A draft cannot masquerade as an actionable trained model.
    for i in range(11, 16):
        stream = signal(i, "select", spacing=.3)
        assert upgraded.add_trial("select", stream, trial_id=i, start=100 + i * 6, end=105 + i * 6)["ok"]
    assert upgraded.train()["ok"]
    assert legacy.export_draft() == state  # The caller and original live calibration were never mutated.


@pytest.mark.parametrize("change", [
    lambda s: s.update(model={}),
    lambda s: s.update(validation=[{}]),
    lambda s: s["training"][0].update(noise=[0, 0, 0, 0]),
    lambda s: s["training"][1].update(counter_start=s["training"][0]["counter_start"]),
    lambda s: s["training"][0].update(id=s["training"][1]["id"]),
    lambda s: s["training"][0]["pulses"].append({}),
])
def test_invalid_partial_calibration_cannot_replace_retained_progress(change):
    decoder = BlinkDecoder()
    assert add(decoder, 0, "rest")["ok"]
    assert add(decoder, 1, "select")["ok"]
    state = decoder.export_draft()
    original = copy.deepcopy(state)
    change(state)
    with pytest.raises(ValueError):
        BlinkDecoder.from_calibration_state(state, upgrade=True)
    assert decoder.export_draft() == original


def resumed_draft():
    legacy = BlinkDecoder()
    legacy._version = 1
    for i in range(8):
        assert add(legacy, i, "rest")["ok"]
    original = legacy.export_draft()
    decoder = BlinkDecoder.from_calibration_state(original, upgrade=True)
    assert decoder.resume_calibration(signal(session="resumed-session")) == {"ok": True}
    return decoder, original


def test_resumed_eight_rest_trials_keep_provenance_then_train_and_validate_new_acquisition():
    decoder, original = resumed_draft()
    draft = decoder.export_draft()
    assert draft["context"] == original["context"]
    assert draft["training"] == original["training"]
    assert draft["calibration_context"]["session_id"] == "resumed-session"
    decoder = BlinkDecoder.from_calibration_state(json.loads(json.dumps(draft)))
    assert decoder.export_draft() == draft
    for i in range(8):
        # New acquisition deliberately reuses both the old counter values and host times.
        stream = signal(i, "select", session="resumed-session", spacing=.3)
        result = decoder.add_trial("select", stream, trial_id=f"resumed-{i}", start=100 + i * 6, end=105 + i * 6)
        assert result["ok"], result.get("error")
    assert decoder._training[:8] == original["training"]
    assert all(t["session_id"] == "resumed-session" for t in decoder._training[8:])
    assert decoder.train()["ok"]
    assert not decoder.status()["arm_eligible"]
    for i in range(8, 24):
        label = "rest" if i % 2 == 0 else "select"
        stream = signal(i, label, session="resumed-session", spacing=.3)
        result = decoder.add_trial(label, stream, phase="validate", trial_id=f"held-out-{i}",
                                   start=100 + i * 6, end=105 + i * 6)
        assert result["ok"], result.get("error")
    assert decoder.status()["passed_validation"]
    state = decoder.export_state()
    assert state["context"] == original["context"] and state["training"][:8] == original["training"]
    restored = BlinkDecoder.from_state(json.loads(json.dumps(state)))
    assert restored.export_state() == state and restored.status() == decoder.status()
    assert len(events(playback(restored, signal(40, "select", session="resumed-session", spacing=.3)))) == 1
    assert not restored.update(signal(41, "select"))["valid"]
    with pytest.raises(ValueError, match="untrained"):
        restored.resume_calibration(signal(session="third-session"))
    assert restored.export_state() == state


@pytest.mark.parametrize("change", [
    lambda s: s["channels"][0].update(name="different-electrode"),
    lambda s: s.update(nominal_hz=500),
    lambda s: s.update(feature_profile="different-feature-profile"),
])
def test_resume_rejects_montage_rate_and_feature_changes_without_altering_draft(change):
    decoder = BlinkDecoder()
    assert add(decoder, 0, "rest")["ok"]
    original = decoder.export_draft()
    candidate = signal(session="new-session")
    change(candidate)
    with pytest.raises(ValueError):
        decoder.resume_calibration(candidate)
    assert decoder.export_draft() == original


def test_resume_requires_explicit_upgrade_and_cannot_silently_rebind_or_reuse_current_packets():
    decoder, original = resumed_draft()
    legacy = BlinkDecoder.from_calibration_state(original)
    with pytest.raises(ValueError, match="upgrade"):
        legacy.resume_calibration(signal(session="resumed-session"))
    assert legacy.export_draft() == original
    before = decoder.export_draft()
    assert decoder.resume_calibration(signal(session="resumed-session"))["ok"]
    assert decoder.export_draft() == before
    with pytest.raises(ValueError, match="already resumed"):
        decoder.resume_calibration(signal(session="third-session"))
    assert decoder.export_draft() == before
    assert not decoder.add_trial("rest", signal(20), trial_id="old-acquisition", start=220, end=225)["ok"]
    stream = signal(0, "select", session="resumed-session", spacing=.3)
    assert decoder.add_trial("select", stream, trial_id="fresh", start=100, end=105)["ok"]
    before = decoder.export_draft()
    assert not decoder.add_trial("select", stream, trial_id="reused", start=100, end=105)["ok"]
    assert decoder.export_draft() == before


@pytest.mark.parametrize("change", [
    lambda s: s.update(version=1),
    lambda s: s["calibration_context"].update(session_id=s["context"]["session_id"]),
    lambda s: s["calibration_context"]["channels"][0].update(name="different"),
    lambda s: s["training"][0].update(session_id="unknown-acquisition"),
    lambda s: s.update(calibration_context=None),
])
def test_resumed_archive_rejects_invalid_acquisition_provenance(change):
    decoder, _ = resumed_draft()
    state = decoder.export_draft()
    before = copy.deepcopy(state)
    change(state)
    with pytest.raises(ValueError):
        BlinkDecoder.from_calibration_state(state)
    assert decoder.export_draft() == before


def test_failed_validation_cannot_resume_or_become_an_untrained_draft():
    decoder = trained_decoder()
    for i in range(16, 32):
        label = "rest" if i % 2 == 0 else "select"
        stream = signal(i, "select")
        assert decoder.add_trial(label, stream, phase="validate", trial_id=i,
                                 start=100 + i * 6, end=105 + i * 6)["ok"]
    assert decoder.status()["state"] == "validation_failed"
    before = decoder.export_state()
    with pytest.raises(ValueError, match="untrained"):
        decoder.resume_calibration(signal(session="new-session"))
    with pytest.raises(ValueError):
        decoder.export_draft()
    assert decoder.export_state() == before and not decoder.status()["arm_eligible"]


def untrained_with_atypical_commands(count):
    """Keep valid accepted feature records, with deliberately nonrepeating command attempts."""
    decoder = BlinkDecoder()
    for i in range(16):
        assert add(decoder, i, "rest" if i % 2 == 0 else "select")["ok"]
    for number, trial in enumerate([t for t in decoder._training if t["label"] == "select"][:count]):
        for pulse in trial["pulses"]:
            if number == 0:
                shape = -np.asarray(pulse["shape"])  # An opposite-polarity attempt must not teach a second command.
            else:
                rng = np.random.default_rng(991 + number)
                shape = rng.normal(size=96)  # Unrelated multichannel activity is not repeated by other trials.
                shape /= np.linalg.norm(shape)
            pulse["shape"] = shape.tolist()
            pulse["amplitude"] = [v * 20 for v in pulse["amplitude"]]  # Excluded trials must not widen accepted amplitudes.
    return decoder


@pytest.mark.parametrize("atypical", [1, 2])
def test_robust_fit_keeps_one_or_two_atypical_trials_without_teaching_their_patterns(atypical):
    decoder = untrained_with_atypical_commands(atypical)
    before = decoder.export_draft()
    result = decoder.train()
    assert result["ok"], result.get("error")
    assert decoder.status()["training_fit"] == {"supported_select_trials": 8 - atypical,
                                                "total_select_trials": 8, "excluded_select_trials": atypical}
    assert decoder._training == before["training"]
    assert not decoder.status()["passed_validation"] and not decoder.status()["arm_eligible"]
    unsupported = [t for t in decoder._training if t["label"] == "select"][:atypical]
    assert not any(decoder._score(p)[0] for t in unsupported for p in t["pulses"])
    assert decoder._model["amplitude_max"] < min(np.linalg.norm(p["amplitude"]) for t in unsupported for p in t["pulses"])
    state = decoder.export_state()
    restored = BlinkDecoder.from_state(json.loads(json.dumps(state)))
    assert restored.export_state() == state and restored.status() == decoder.status()
    assert len(events(playback(restored, signal(40, "select")))) == 1
    restored.reset_stream()
    assert not events(playback(restored, signal(41, "rest", kind="single")))


def test_three_atypical_commands_do_not_supply_the_required_three_quarters_consensus():
    decoder = untrained_with_atypical_commands(3)
    before = decoder.export_draft()
    result = decoder.train()
    assert not result["ok"] and "three quarters" in result["error"]
    assert decoder.export_draft() == before and not decoder.status()["trained"]


def test_consensus_is_whole_trial_and_never_pairs_a_supported_half_with_an_outlier():
    decoder = untrained_with_atypical_commands(0)
    positives = [t for t in decoder._training if t["label"] == "select"]
    for number, trial in enumerate(positives[:3]):
        rng = np.random.default_rng(9871 + number)
        shape = rng.normal(size=96)
        shape /= np.linalg.norm(shape)
        trial["pulses"][0]["shape"] = shape.tolist()  # The second pulse still matches; the entire trial must nevertheless be excluded.
    before = decoder.export_draft()
    assert not decoder.train()["ok"]
    assert decoder.export_draft() == before


def test_temporal_alignment_is_bounded_preserves_polarity_and_penalizes_cropped_energy():
    x = np.arange(24)
    base = (np.exp(-.5 * ((x - 10) / .7)**2)[:, None] * np.array([1, .8, .7, .9])).ravel()
    base /= np.linalg.norm(base)
    nearby = np.roll(base.reshape(24, 4), 2, axis=0).ravel()
    distant = np.roll(base.reshape(24, 4), 6, axis=0).ravel()
    assert np.dot(base, nearby) < .65
    assert _aligned_similarity([base], [nearby])[0, 0] > .99
    assert _aligned_similarity([base], [-nearby])[0, 0] < .10
    assert _aligned_similarity([base], [distant])[0, 0] < .10
    cropped = nearby.copy()
    cropped[:4] = 3  # A matching central fragment cannot hide a much larger unrelated edge artifact.
    cropped /= np.linalg.norm(cropped)
    assert _aligned_similarity([base], [cropped])[0, 0] < .20


def test_temporally_shifted_no_command_pairs_still_block_a_robust_model():
    decoder = untrained_with_atypical_commands(1)
    rest = next(t for t in decoder._training if t["label"] == "rest")
    source = [t for t in decoder._training if t["label"] == "select"][2]
    # Create a negative command collision from retained pulse features; their timing remains one deliberate pair.
    rest["pulses"] = copy.deepcopy(source["pulses"])
    for pulse in rest["pulses"]:
        shape = np.asarray(pulse["shape"]).reshape(24, 4)
        shifted = np.zeros_like(shape)
        shifted[2:] = shape[:-2]
        shifted /= np.linalg.norm(shifted)
        pulse["shape"] = shifted.ravel().tolist()
    before = copy.deepcopy(decoder._training)
    result = decoder.train()
    assert not result["ok"] and "No-command" in result["error"]
    assert decoder._training == before and not decoder.status()["trained"]


def test_version_two_frozen_model_keeps_its_original_matching_and_archive():
    decoder = BlinkDecoder()
    decoder._version = 2
    for i in range(32):
        assert add(decoder, i, "rest" if i % 2 == 0 else "select", phase="train" if i < 16 else "validate")["ok"]
        if i == 15:
            assert decoder.train()["ok"]
    state = decoder.export_state()
    restored = BlinkDecoder.from_state(copy.deepcopy(state))
    assert state["version"] == 2 and restored.export_state() == state
    assert restored.status() == decoder.status() and restored.status()["training_fit"] is None
    pulse = copy.deepcopy(decoder._training[1]["pulses"][0])
    assert restored._score(pulse) == decoder._score(pulse)
    assert "template_trial_indices" not in restored._model


def test_extra_training_stops_at_the_savable_eighty_trial_limit_without_losing_progress():
    decoder = BlinkDecoder()
    for i in range(80):
        assert add(decoder, i, "rest")["ok"]
    before = decoder.export_draft()
    assert len(before["training"]) == 80
    assert BlinkDecoder.from_calibration_state(before).export_draft() == before
    result = add(decoder, 80, "select")
    assert not result["ok"] and "80 accepted trials" in result["error"]
    assert decoder.export_draft() == before


@pytest.mark.parametrize("key", ["threshold", "amplitude_min", "amplitude_max", "duration_min", "duration_max"])
def test_archive_rejects_exact_numeric_strings_in_scalar_model_parameters(key):
    decoder = trained_decoder()
    original = decoder.export_state()
    damaged = copy.deepcopy(original)
    damaged["model"][key] = str(damaged["model"][key])  # Previously coerced for comparison, then installed as a runtime-breaking string.
    with pytest.raises(ValueError, match="parameters"):
        BlinkDecoder.from_state(damaged)
    assert decoder.export_state() == original
    assert decoder._score(decoder._training[1]["pulses"][0])[0]


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("field", ["noise", "waveform"])
def test_archive_rejects_numeric_strings_inside_noise_and_template_arrays(version, field):
    decoder = BlinkDecoder()
    decoder._version = version
    for i in range(16):
        assert add(decoder, i, "rest" if i % 2 == 0 else "select")["ok"]
    assert decoder.train()["ok"]
    state = decoder.export_state()
    values = state["model"]["noise"] if field == "noise" else (state["model"]["template"] if version == 1 else state["model"]["templates"][0])
    values[0] = str(values[0])
    with pytest.raises(ValueError, match="parameters"):
        BlinkDecoder.from_state(state)


@pytest.mark.parametrize("change", [
    lambda value: str(value),
    lambda value: float(value),
    lambda value: value + 1e-13,
    lambda value: True,
    lambda value: value + 1,
])
def test_archive_requires_exact_integer_prototype_provenance(change):
    decoder = trained_decoder()
    state = decoder.export_state()
    before = copy.deepcopy(state)
    state["model"]["template_trial_indices"][0] = change(state["model"]["template_trial_indices"][0])
    with pytest.raises(ValueError, match="parameters"):
        BlinkDecoder.from_state(state)
    assert decoder.export_state() == before
    assert decoder.status()["training_fit"] == {"supported_select_trials": 8, "total_select_trials": 8, "excluded_select_trials": 0}


def test_exact_model_provenance_survives_numeric_parameter_roundoff_without_rewriting_archive():
    state = trained_decoder().export_state()
    state["model"]["threshold"] += 1e-14  # Keep the pre-existing tolerance for harmless floating-point serialization differences.
    restored = BlinkDecoder.from_state(state)
    assert restored.export_state() == state
    assert restored.status()["training_fit"]["supported_select_trials"] == 8
    assert restored._score(restored._training[1]["pulses"][0])[0]
