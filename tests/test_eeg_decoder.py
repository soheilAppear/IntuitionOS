"""Synthetic EEG only: no device, webcam, file recording, or desktop actions."""

import copy
import json

import numpy as np
import pytest

from core.eeg_decoder import EegDecoder, LABELS, extract_features


CHANNELS = [{"num": 0, "name": "T3"}, {"num": 1, "name": "T4"}]


def packets(frequency=10, *, seed=1, start=100.0, counter=0, count=500, rate=250):
    rng = np.random.default_rng(seed)
    t = np.arange(count) / rate
    data = np.column_stack([20e-6 * np.sin(2 * np.pi * frequency * t),
                            12e-6 * np.cos(2 * np.pi * frequency * t)])
    data += rng.normal(0, 0.5e-6, data.shape)
    return [{"counter": (counter + index) % 2**32, "samples": values.tolist(),
             "estimated_monotonic": start + index / rate,
             "host_received_monotonic": start + index / rate}
            for index, values in enumerate(data)]


def extracted(label="left", index=0, *, jitter=0.0, session="session-1"):
    centers = {"left": [-20, -30, -20, -30], "right": [-20, -30, -30, -20],
               "rest": [-22, -22, -30, -30]}
    values = [value + jitter * (position + 1) / 4 for position, value in enumerate(centers[label])]
    return {"ok": True, "features": values,
            "context": {"session_id": session, "nominal_hz": 250.0,
                        "channels": [{"num": 0, "name": "T3"}]},
            "window": {"start": 100 + index * 3, "end": 102 + index * 3,
                       "counter_start": index * 750, "counter_end": index * 750 + 499,
                       "n_samples": 500, "window_id": str(index)}}


def trained_decoder():
    decoder = EegDecoder()
    for index in range(24):
        label = LABELS[index % 3]
        trial = extracted(label, index, jitter=(index // 3 - 3.5) * 0.04)
        assert decoder.add_trial(label, trial, trial_id=index)["ok"]
    assert decoder.train()["ok"]
    return decoder


def validate(decoder, rest_as=None):
    for index in range(24, 48):
        label = LABELS[index % 3]
        trial = extracted(rest_as if label == "rest" and rest_as else label, index, jitter=0.015)
        assert decoder.add_trial(label, trial, phase="validate", trial_id=index)["ok"]
    return decoder.status()


def test_full_rate_features_match_known_band_and_ignore_camera_fields():
    rows = packets()
    result = extract_features(rows, CHANNELS, 250, {"session_id": "fake"})
    assert result["ok"]
    assert result["diagnostics"]["n_samples"] == 500
    assert len(result["features"]) == 8
    for channel in range(2):
        _, theta, alpha, beta = result["features"][channel * 4:(channel + 1) * 4]
        assert alpha > theta + 5
        assert alpha > beta + 5
    for row in rows:
        row.update(camera_direction="right", video_features=[999])
    assert extract_features(rows, CHANNELS, 250, {"session_id": "fake"}) == result


def test_channel_number_mapping_is_canonical_not_input_order():
    a = extract_features(packets(), CHANNELS, 250)
    b = extract_features(packets(), list(reversed(CHANNELS)), 250)
    assert a == b


def test_linear_detrending_removes_dc_and_slope():
    original = packets()
    shifted = copy.deepcopy(original)
    for index, row in enumerate(shifted):
        row["samples"] = [value + 100e-6 + index * 2e-7 for value in row["samples"]]
    a = extract_features(original, CHANNELS, 250)
    b = extract_features(shifted, CHANNELS, 250)
    assert a["ok"] and b["ok"]
    np.testing.assert_allclose(a["features"], b["features"], atol=1e-9)


@pytest.mark.parametrize("offsets", [(0.020, -0.020), (-0.050, 0.050)])
def test_large_per_channel_dc_offsets_preserve_features(offsets):
    original = packets()
    shifted = copy.deepcopy(original)
    for row in shifted:
        row["samples"] = [value + offset for value, offset in zip(row["samples"], offsets)]
    baseline = extract_features(original, CHANNELS, 250)
    result = extract_features(shifted, CHANNELS, 250)
    assert baseline["ok"] and result["ok"]
    np.testing.assert_allclose(result["features"], baseline["features"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(result["diagnostics"]["peak_to_peak_v"],
                               baseline["diagnostics"]["peak_to_peak_v"], rtol=0, atol=1e-15)


@pytest.mark.parametrize("fault", ["spike", "linear_drift"])
def test_excess_raw_excursions_are_rejected_before_detrending_despite_offset(fault):
    rows = packets()
    for index, row in enumerate(rows):
        row["samples"] = [row["samples"][0] + 0.020, row["samples"][1] - 0.030]
        if fault == "linear_drift":
            row["samples"][1] += index * 0.040 / (len(rows) - 1)
    if fault == "spike":
        rows[200]["samples"][1] += 0.020
    result = extract_features(rows, CHANNELS, 250)
    assert not result["ok"]
    assert "peak-to-peak" in result["reason"]
    assert "10 mV engineering limit" in result["reason"]
    assert "T4 [1]:" in result["reason"]
    assert "T3 [0]" not in result["reason"]


def test_flat_channels_at_nonzero_baselines_report_names_without_amplitude_error():
    rows = packets()
    for row in rows:
        row["samples"] = [0.020, -0.030]
    result = extract_features(rows, CHANNELS, 250)
    assert not result["ok"]
    assert "flat" in result["reason"]
    assert "T3 [0]" in result["reason"] and "T4 [1]" in result["reason"]
    assert "10 mV" not in result["reason"]


@pytest.mark.parametrize("fault,reason", [("flat_segment", "flatline"), ("plateau", "clipping"),
                                         ("nonfinite", "Nonfinite")])
def test_quality_rejections_remain_active_with_large_dc_offsets(fault, reason):
    rows = packets()
    for row in rows:
        row["samples"] = [row["samples"][0] + 0.020, row["samples"][1] - 0.030]
    if fault == "nonfinite":
        rows[200]["samples"][1] = float("nan")
    else:
        for row in rows[100:150]:
            row["samples"][1] = -0.030 if fault == "flat_segment" else -0.0299
    result = extract_features(rows, CHANNELS, 250)
    assert not result["ok"]
    assert reason in result["reason"]
    if fault != "nonfinite":
        assert "T4 [1]" in result["reason"]


def test_counter_wrap_and_small_host_batch_jitter_are_accepted():
    rows = packets(counter=2**32 - 250)
    for index, row in enumerate(rows):
        row["estimated_monotonic"] += 0.020 if (index // 25) % 2 else 0
    result = extract_features(rows, CHANNELS, 250)
    assert result["ok"]
    assert result["window"]["counter_start"] == 2**32 - 250
    assert result["window"]["counter_end"] == 249


@pytest.mark.parametrize("fault", ["gap", "duplicate", "nan", "channel_count", "flat", "flat_segment", "linear", "amplitude", "plateau", "timing"])
def test_signal_integrity_rejections(fault):
    rows = packets()
    if fault == "gap":
        rows[200]["counter"] += 1
    elif fault == "duplicate":
        rows[200]["counter"] -= 1
    elif fault == "nan":
        rows[200]["samples"][0] = float("nan")
    elif fault == "channel_count":
        rows[200]["samples"].append(0)
    elif fault == "flat":
        for row in rows:
            row["samples"][0] = 1e-6
    elif fault == "linear":
        for index, row in enumerate(rows):
            row["samples"][0] = index * 1e-7
    elif fault == "flat_segment":
        for row in rows[100:150]:
            row["samples"][0] = 0.0
    elif fault == "amplitude":
        rows[200]["samples"][0] = 0.02
    elif fault == "plateau":
        for row in rows[100:110]:
            row["samples"][0] = 100e-6
    elif fault == "timing":
        rows[-1]["estimated_monotonic"] += 1
    assert not extract_features(rows, CHANNELS, 250)["ok"]


@pytest.mark.parametrize("rate,channels", [(None, CHANNELS), (30, CHANNELS), (True, CHANNELS),
                                          (250, []), (250, [{"num": 1, "name": "T3"}]),
                                          (250, [{"num": 0, "name": "T3"}] * 2)])
def test_invalid_sample_rate_and_channel_identity(rate, channels):
    assert not extract_features(packets(), channels, rate)["ok"]


def test_display_decimation_is_rejected_and_newest_complete_window_selected():
    rows = packets(count=750)
    assert not extract_features(rows[::3], CHANNELS, 250)["ok"]
    result = extract_features(rows, CHANNELS, 250)
    assert result["ok"]
    assert result["window"]["counter_start"] == 250
    assert result["window"]["counter_end"] == 749


def test_training_requires_each_class_and_explicit_freeze():
    decoder = EegDecoder()
    assert not decoder.train()["ok"]
    for index in range(24):
        assert decoder.add_trial("left", extracted("left", index), trial_id=index)["ok"]
    assert not decoder.train()["ok"]
    assert decoder.status()["train_counts"] == {"left": 24, "right": 0, "rest": 0}
    assert not decoder.predict(extracted())["valid"]
    assert not decoder.add_trial("rest", extracted("rest", 25), "validate", "v")["ok"]


def test_known_three_class_prediction_and_no_arming_before_validation():
    decoder = trained_decoder()
    assert decoder.status()["trained"]
    assert not decoder.status()["arm_eligible"]
    for label in LABELS:
        result = decoder.predict(extracted(label, 24))
        assert result["valid"] and not result["ood"]
        assert result["label"] == label
        assert result["confidence"] >= 0.8
        assert "not probabilities" in result["score_note"]


def test_independent_validation_passes_with_honest_per_class_metrics():
    result = validate(trained_decoder())
    assert result["arm_eligible"] and result["passed_validation"]
    assert result["state"] == "validated"
    metrics = result["evaluation"]
    assert metrics["balanced_accuracy"] == 1
    assert metrics["rest_false_activation_rate"] == 0
    assert metrics["rest_trials"] == 8
    for label in LABELS:
        assert metrics["per_class"][label]["precision"] == 1
        assert metrics["per_class"][label]["recall"] == 1
        assert metrics["confusion_matrix"][label][label] == 8
    assert "not guarantee" in metrics["reason"]


def test_rest_false_activations_fail_and_validation_cannot_be_cherry_picked():
    decoder = trained_decoder()
    result = validate(decoder, rest_as="left")
    assert not result["arm_eligible"]
    assert result["state"] == "validation_failed"
    assert result["evaluation"]["rest_false_activations"] == 8
    assert result["evaluation"]["rest_false_activation_rate"] == 1
    assert result["evaluation"]["per_class"]["left"]["precision"] == 0.5
    assert not decoder.add_trial("rest", extracted("rest", 48), "validate", 48)["ok"]
    assert decoder.status()["evaluation"] == result["evaluation"]


def test_frozen_training_and_validation_never_change_parameters():
    decoder = trained_decoder()
    before = {key: value.copy() for key, value in decoder._model.items()}
    assert not decoder.train()["ok"]
    assert not decoder.add_trial("left", extracted("left", 24), trial_id=24)["ok"]
    validate(decoder, rest_as="left")
    for key, expected in before.items():
        np.testing.assert_array_equal(decoder._model[key], expected)


def test_duplicate_ids_and_overlapping_windows_are_rejected_across_phases():
    decoder = trained_decoder()
    assert not decoder.add_trial("left", extracted("left", 24), "validate", 0)["ok"]
    overlap = extracted("left", 25)
    overlap["window"].update(counter_start=10, counter_end=509)
    assert "counters overlap" in decoder.add_trial("left", overlap, "validate", "overlap")["error"]
    shifted_time = extracted("left", 24)
    shifted_time["window"].update(start=169.0, end=171.0)
    assert "must not overlap" in decoder.add_trial("left", shifted_time, "validate", "time")["error"]
    assert decoder.status()["validation_counts"] == dict.fromkeys(LABELS, 0)


def test_packet_overlap_across_uint32_wrap_is_rejected():
    decoder = EegDecoder()
    first = extracted(index=0)
    first["window"].update(counter_start=2**32 - 250, counter_end=249)
    assert decoder.add_trial("left", first, trial_id=0)["ok"]
    second = extracted(index=1)
    second["window"].update(counter_start=0, counter_end=499)
    assert not decoder.add_trial("left", second, trial_id=1)["ok"]


def test_context_mismatch_and_bad_features_fail_closed():
    decoder = trained_decoder()
    for fault in ("session", "channels", "rate", "nan", "shape", "window"):
        trial = extracted(index=24)
        if fault == "session":
            trial["context"]["session_id"] = "new-session"
        elif fault == "channels":
            trial["context"]["channels"][0]["name"] = "T4"
        elif fault == "rate":
            trial["context"]["nominal_hz"] = 125
        elif fault == "nan":
            trial["features"][0] = float("nan")
        elif fault == "shape":
            trial["features"].append(1)
        else:
            trial["window"] = "invalid"
        prediction = decoder.predict(trial)
        assert not prediction["valid"] and prediction["label"] is None
        assert not decoder.add_trial("left", trial, "validate", fault)["ok"]


def test_ambiguous_scores_abstain_and_do_not_count_as_success():
    decoder = trained_decoder()
    ambiguous = extracted(index=24)
    ambiguous["features"] = [-20, -30, -25, -25]
    result = decoder.predict(ambiguous)
    assert result["label"] is None
    assert result["confidence"] < 0.8
    assert decoder.add_trial("left", ambiguous, "validate", 24)["ok"]
    evaluation = decoder.status()["evaluation"]
    assert evaluation["abstentions"] == 1
    assert evaluation["per_class"]["left"]["recall"] == 0
    assert evaluation["confusion_matrix"]["left"]["uncertain"] == 1


def test_out_of_distribution_rejection_and_heldout_uses_identical_rule():
    decoder = trained_decoder()
    outlier = extracted(index=24)
    outlier["features"] = [-80, -80, -20, -80]
    prediction = decoder.predict(outlier)
    assert prediction["ood"] and not prediction["valid"] and prediction["label"] is None
    assert decoder.add_trial("left", outlier, "validate", 24)["ok"]
    assert decoder._validation[0]["prediction"] == prediction
    assert decoder.status()["evaluation"]["abstentions"] == 1


def test_predict_has_no_validation_side_effects_or_raw_feature_export():
    decoder = trained_decoder()
    before = decoder.status()
    for _ in range(10):
        decoder.predict(extracted(index=24))
    assert decoder.status() == before
    serialized = json.dumps(before, allow_nan=False)
    assert '"features"' not in serialized
    assert '"samples"' not in serialized
    assert "session-1" not in serialized


def test_storage_is_bounded_and_reset_discards_model_and_trials():
    decoder = EegDecoder()
    for index in range(96):
        assert decoder.add_trial(LABELS[index % 3], extracted(LABELS[index % 3], index), trial_id=index)["ok"]
    assert not decoder.add_trial("left", extracted(index=96), trial_id=96)["ok"]
    assert decoder.train()["ok"]
    for index in range(96, 120):
        assert decoder.add_trial(LABELS[index % 3], extracted(LABELS[index % 3], index), "validate", index)["ok"]
    assert decoder.status()["total"] == 120
    assert not decoder.add_trial("left", extracted(index=120), "validate", 120)["ok"]
    result = decoder.reset()
    assert result["total"] == 0 and not result["trained"] and not result["arm_eligible"]
    assert decoder._model is None and not decoder._windows and not decoder._trial_ids
    assert decoder.add_trial("left", extracted(index=0, session="new-session"), trial_id=0)["ok"]


def test_caller_mutation_cannot_alter_retained_training_features():
    decoder = EegDecoder()
    trial = extracted()
    assert decoder.add_trial("left", trial, trial_id=0)["ok"]
    trial["features"][0] = 99
    trial["context"]["channels"][0]["name"] = "Changed"
    assert decoder._training[0]["features"][0] == -20
    assert decoder.status()["context"]["channels"][0]["name"] == "T3"


def test_end_to_end_synthetic_frequency_classes_with_separate_trials():
    decoder = EegDecoder()
    frequencies = {"left": 6, "right": 10, "rest": 20}
    for index in range(48):
        if index == 24:
            assert decoder.train()["ok"]
        label = LABELS[index % 3]
        rows = packets(frequencies[label], seed=index + 1, start=100 + index * 3, counter=index * 750)
        result = extract_features(rows, CHANNELS, 250, {"session_id": "synthetic"})
        assert result["ok"]
        assert decoder.add_trial(label, result, "train" if index < 24 else "validate", index)["ok"]
    status = decoder.status()
    assert status["arm_eligible"]
    assert status["evaluation"]["accuracy"] == 1


def test_indistinguishable_training_classes_abstain_without_ood():
    decoder = EegDecoder()
    for index in range(24):
        trial = extracted("rest", index, jitter=(index // 3 - 3.5) * 0.04)
        assert decoder.add_trial(LABELS[index % 3], trial, trial_id=index)["ok"]
    assert decoder.train()["ok"]
    result = decoder.predict(extracted("rest", 24))
    assert result["valid"] and not result["ood"]
    assert result["label"] is None
    assert result["confidence"] == pytest.approx(1 / 3)
    assert "Uncertain" in result["reason"]
