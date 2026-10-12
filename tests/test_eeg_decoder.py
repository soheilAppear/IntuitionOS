"""Synthetic EEG only: no device, webcam, file recording, or desktop actions."""

import copy
import json

import numpy as np
import pytest

# Import the versioned click profile so tests exercise the public processing contract.
from core.eeg_decoder import CLICK_FEATURE_PROFILE, EegDecoder, LABELS, extract_features


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


def click_extracted(label="rest", index=0, *, session="click-session"):
    """Build independent synthetic click windows without reading a device."""
    # Reuse separated synthetic vectors while mapping their labels to the binary task.
    trial = extracted("left" if label == "select" else "rest", index, jitter=(index % 8 - 3.5) * 0.015,
                      session=session)
    # Tag the actual transform expected by the binary decoder before it binds a session.
    trial["context"]["feature_profile"] = CLICK_FEATURE_PROFILE
    # Return a fresh dictionary on every call so fault injection cannot affect later trials.
    return trial


def trained_click_decoder():
    """Freeze a binary model using exactly eight chronological trials per class."""
    # Both labels and processing are explicit; the legacy constructor remains unchanged.
    decoder = EegDecoder(labels=("rest", "select"), feature_profile=CLICK_FEATURE_PROFILE)
    # Alternate classes so counts and window order match a practical guided session.
    for index in range(16):
        # Select each configured class in its public immutable order.
        label = decoder.labels[index % 2]
        # Each vector has a unique identity and nonoverlapping packet counters.
        assert decoder.add_trial(label, click_extracted(label, index), trial_id=index)["ok"]
    # Freeze once; every following trial must be held out from parameter fitting.
    assert decoder.train()["ok"]
    # Return the real decoder, keeping validation and fault tests independent.
    return decoder


@pytest.mark.parametrize("labels", [None, "rest", {}, ("rest",), ("rest", "rest"),
                                  ("left", "right"), ("rest", "unknown"),
                                  ("rest", "select", "left", "right"), ("rest", [])])
def test_decoder_rejects_invalid_label_configuration(labels):
    # Invalid classes must fail before calibration can construct an unsafe model.
    with pytest.raises(ValueError, match="labels"):
        # Exercise the public constructor rather than reaching into private state.
        EegDecoder(labels=labels)


def test_click_configuration_is_immutable_and_survives_reset():
    # Start with mutable caller-owned configuration to test defensive copying.
    labels = ["rest", "select"]
    # Bind both the two-class order and the processing profile at construction.
    decoder = EegDecoder(labels=labels, feature_profile=CLICK_FEATURE_PROFILE)
    # A caller modifying its original list must not relabel the model.
    labels[1] = "left"
    # The public configuration is an immutable tuple in its original order.
    assert decoder.labels == ("rest", "select")
    # Returned status lists must also be independent from internal configuration.
    decoder.status()["labels"][1] = "right"
    # Reassignment through the public API must not silently alter a frozen model later.
    with pytest.raises(AttributeError):
        # Assignment intentionally tests the read-only label property.
        decoder.labels = ("rest", "left")
    # Processing must be equally immutable, even before the first trial.
    with pytest.raises(AttributeError):
        # Assignment intentionally tests the read-only profile property.
        decoder.feature_profile = None
    # Reset clears data, not the user's selected calibration protocol.
    status = decoder.reset()
    # No discarded caller or status mutation may leak into the reset configuration.
    assert status["labels"] == ["rest", "select"]
    # The processing version remains explicit after reset.
    assert status["feature_profile"] == CLICK_FEATURE_PROFILE
    # The displayed feature description must match what the click extractor computes.
    assert "[4,30)" in status["feature_processing"]


@pytest.mark.parametrize("profile", ["unknown", True, {}, []])
def test_unknown_processing_profiles_fail_closed(profile):
    # Invalid decoder configuration should surface immediately as a configuration error.
    with pytest.raises(ValueError, match="feature profile"):
        # Do not silently default to variance when the requested transform is unsupported.
        EegDecoder(feature_profile=profile)
    # Extraction uses the existing rejection result instead of raising during a preview.
    result = extract_features(packets(), CHANNELS, 250, feature_profile=profile)
    # The caller must be able to suppress inference using the ordinary error contract.
    assert not result["ok"] and "feature profile" in result["reason"]


def test_click_power_excludes_line_bin_without_hiding_raw_diagnostics():
    # Use real synthetic full-rate samples so this test checks actual spectral processing.
    clean_rows = packets()
    # Add strong line interference without crossing the unchanged raw amplitude limit.
    noisy_rows = copy.deepcopy(clean_rows)
    # A 60 Hz sinusoid should affect raw RMS far more than integrated 4--30 Hz power.
    for index, row in enumerate(noisy_rows):
        # Keep each original EEG sample and add the same controlled electrical component.
        row["samples"] = [value + 500e-6 * np.sin(2 * np.pi * 60 * index / 250) for value in row["samples"]]
    # Legacy extraction must retain the exact old context when no profile is selected.
    legacy = extract_features(noisy_rows, CHANNELS, 250, {"session_id": "line-test"})
    # Explicit None must remain fully equivalent to the default call.
    assert legacy == extract_features(noisy_rows, CHANNELS, 250, {"session_id": "line-test"}, feature_profile=None)
    # An omitted version key distinguishes the backwards-compatible legacy context.
    assert "feature_profile" not in legacy["context"]
    # Process both recordings with the identical versioned click transform.
    click_clean = extract_features(clean_rows, CHANNELS, 250, {"session_id": "line-test"}, feature_profile=CLICK_FEATURE_PROFILE)
    # No external filter or interpolated sample is involved in this comparison.
    click_noisy = extract_features(noisy_rows, CHANNELS, 250, {"session_id": "line-test"}, feature_profile=CLICK_FEATURE_PROFILE)
    # All three valid sample windows should satisfy unchanged engineering gates.
    assert legacy["ok"] and click_clean["ok"] and click_noisy["ok"]
    # The returned context must bind the click transform for downstream model checks.
    assert click_noisy["context"]["feature_profile"] == CLICK_FEATURE_PROFILE
    # Raw RMS and excursions still reveal interference; the click transform does not conceal it.
    assert click_noisy["diagnostics"] == legacy["diagnostics"]
    # Recover linear powers to verify that feature zero covers exactly the three adjacent bands.
    powers = np.exp(np.asarray(click_noisy["features"]).reshape(-1, 4))
    # Integrated 4--30 Hz power must equal the sum of the existing theta, alpha, and beta bins.
    np.testing.assert_allclose(powers[:, 0], powers[:, 1:].sum(axis=1), rtol=1e-12)
    # A finite Hann window permits tiny leakage; require negligible change in total band power.
    np.testing.assert_allclose(np.asarray(click_clean["features"])[::4],
                               np.asarray(click_noisy["features"])[::4], rtol=0, atol=0.02)
    # The legacy broadband feature should still expose the deliberately much larger line power.
    assert all(np.asarray(legacy["features"])[::4] > np.asarray(click_noisy["features"])[::4] + 4)


@pytest.mark.parametrize("fault", ["spike", "flat", "plateau", "gap"])
def test_click_profile_preserves_raw_quality_rejections(fault):
    # Start from finite contiguous full-rate synthetic EEG before injecting one fault.
    rows = packets()
    # Excess raw amplitude must be rejected even if band selection would attenuate it.
    if fault == "spike":
        # A 20 mV spike exceeds the unchanged 10 mV peak-to-peak limit.
        rows[200]["samples"][0] += 0.020
    # A constant channel cannot become a usable selection signal through band processing.
    elif fault == "flat":
        # Replace every sample of one channel with a constant baseline.
        for row in rows:
            # Keep the other channel valid to test per-channel enforcement.
            row["samples"][0] = 0.001
    # Repeated extrema remain a possible-clipping rejection before spectral extraction.
    elif fault == "plateau":
        # Ten repeated extremes exceed the existing minimum plateau length.
        for row in rows[100:110]:
            # This amplitude stays below the raw-excursion ceiling to isolate the plateau gate.
            row["samples"][0] = 100e-6
    # Discontinuous counters must never be repaired by the new feature profile.
    else:
        # Introduce one duplicate/gap pair without changing timestamps or values.
        rows[200]["counter"] += 1
    # Both profiles should reject the same unchanged raw acquisition defect.
    assert not extract_features(rows, CHANNELS, 250, feature_profile=CLICK_FEATURE_PROFILE)["ok"]


def test_binary_validation_uses_two_classes_and_keeps_frozen_parameters():
    # Train with the real binary model and keep an exact copy of every fitted parameter.
    decoder = trained_click_decoder()
    # A trained model alone must never authorize a click.
    assert not decoder.status()["arm_eligible"]
    # Copy arrays because validation must not update normalization, radii, or centroids.
    frozen = {key: value.copy() for key, value in decoder._model.items()}
    # New validation windows follow training chronologically and share no sample counters.
    for index in range(16, 32):
        # Alternate rest/select to obtain exactly eight held-out trials for each.
        label = decoder.labels[index % 2]
        # Synthetic separated classes should be evaluated by the same public inference method.
        assert decoder.add_trial(label, click_extracted(label, index), phase="validate", trial_id=index)["ok"]
    # Inspect public metrics rather than assuming three-class array positions.
    status = decoder.status()
    # Both classes must pass all unchanged gates before the service can consider arming.
    assert status["arm_eligible"] and status["passed_validation"]
    # The majority-class baseline for a balanced binary holdout is one half.
    assert status["evaluation"]["baseline_accuracy"] == 0.5
    # A successful binary classifier should not have its recall divided by three.
    assert status["evaluation"]["balanced_accuracy"] == 1.0
    # Rest errors are counted using select rather than nonexistent direction columns.
    assert status["evaluation"]["rest_false_activations"] == 0
    # The reported unit must disclose selection predictions, not desktop swipes.
    assert "Accepted select predictions" in status["evaluation"]["rest_false_activation_unit"]
    # No training or holdout sample may change a frozen parameter.
    for key, expected in frozen.items():
        # Exact equality catches accidental retraining during validation.
        np.testing.assert_array_equal(decoder._model[key], expected)
    # New training is forbidden until an explicit reset clears the model.
    assert not decoder.add_trial("select", click_extracted("select", 32), trial_id=32)["ok"]
    # Completed class validation cannot be extended until a favorable result appears.
    assert not decoder.add_trial("rest", click_extracted("rest", 32), phase="validate", trial_id=32)["ok"]


def test_one_false_select_in_eight_rest_trials_blocks_binary_arming():
    # Fit clean separated classes before introducing one held-out false activation.
    decoder = trained_click_decoder()
    # Validation uses a fixed chronological sequence, including the deliberately wrong trial.
    for index in range(16, 32):
        # The real cue remains rest for even indices, including the first validation trial.
        label = decoder.labels[index % 2]
        # The first rest sample deliberately resembles select; all later trials are ordinary.
        observed_label = "select" if index == 16 else label
        # Retain the incorrect prediction rather than discarding or relabeling the trial.
        assert decoder.add_trial(label, click_extracted(observed_label, index), phase="validate", trial_id=index)["ok"]
    # One false select out of eight rest trials exceeds the unchanged ten-percent limit.
    status = decoder.status()
    # A seemingly high overall accuracy must not override the rest false-activation gate.
    assert status["evaluation"]["accuracy"] == 15 / 16
    # Report the exact observed false-selection count and denominator.
    assert status["evaluation"]["rest_false_activation_rate"] == 1 / 8
    # This safety gate remains false even when the other per-class thresholds pass.
    assert not status["evaluation"]["gates"]["rest_false_activation_rate"]
    # The binary model must stay unavailable to the click dispatcher.
    assert not status["arm_eligible"] and status["state"] == "validation_failed"


@pytest.mark.parametrize("profile", [None, "unknown"])
def test_click_profile_binding_rejects_missing_or_changed_processing(profile):
    # Check the constructor binding even before any acquisition has been learned.
    fresh = EegDecoder(labels=("rest", "select"), feature_profile=CLICK_FEATURE_PROFILE)
    # Start from an otherwise valid sample window and replace only its processing identity.
    trial = click_extracted()
    # Missing and unknown version keys must both be distinguished from click processing.
    if profile is None:
        # A legacy vector is not safe merely because it has the same width.
        trial["context"].pop("feature_profile")
    # Use an unsupported version to simulate processing changes after an app update.
    else:
        # Changing the version must force an explicit matching decoder configuration.
        trial["context"]["feature_profile"] = profile
    # No mismatched first trial may bind the decoder to the wrong transform.
    assert "feature profile" in fresh.add_trial("rest", trial, trial_id=0)["error"]
    # The same invariant must apply to prediction after a valid model has been frozen.
    decoder = trained_click_decoder()
    # A mismatched transform must abstain rather than return any desktop action label.
    prediction = decoder.predict(trial)
    # The rejection is distinct from ordinary statistical uncertainty.
    assert not prediction["valid"] and prediction["label"] is None
    # Preserve the actionable reason for the UI and control guard.
    assert "feature profile" in prediction["reason"]


def test_binary_ood_session_and_overlap_guards_remain_active():
    # Start each fault from a correctly frozen binary click model.
    decoder = trained_click_decoder()
    # An extreme feature vector must not be converted into a high-confidence click.
    outlier = click_extracted("select", 16)
    # Finite in-contract numbers isolate the training-distance gate from format validation.
    outlier["features"] = [-80, -80, -20, -80]
    # Use the production predictor so OOD rejection is shared with held-out validation.
    prediction = decoder.predict(outlier)
    # A large normalized score cannot override a failed distance check.
    assert prediction["ood"] and not prediction["valid"] and prediction["label"] is None
    # A new acquisition must not reuse features from the previous physical fit/session.
    changed_session = click_extracted("select", 16, session="replacement-session")
    # Prediction must fail closed after the session identity changes.
    assert not decoder.predict(changed_session)["valid"]
    # Chronologically new timestamps cannot disguise reused training sample counters.
    overlap = click_extracted("rest", 16)
    # Preserve a valid 500-sample shape while overlapping the first training window.
    overlap["window"].update(counter_start=10, counter_end=509)
    # Held-out validation must reject the reused window instead of overstating accuracy.
    assert "counters overlap" in decoder.add_trial("rest", overlap, phase="validate", trial_id=16)["error"]
    # The legacy decoder must also reject click-profile vectors before binding any session.
    assert "feature profile" in EegDecoder().add_trial("rest", click_extracted(), trial_id=0)["error"]


def test_binary_click_processing_end_to_end_with_four_channels_and_line_noise():
    # Use the same four channel names as the connected BrainBit2, without using real samples.
    channels = [{"num": index, "name": name} for index, name in enumerate(("O2", "T3", "T4", "O1"))]
    # Both calibration phases and later predictions must use the same fixed click transform.
    decoder = EegDecoder(labels=("rest", "select"), feature_profile=CLICK_FEATURE_PROFILE)
    # Build sixteen training plus sixteen held-out trials with disjoint counters and times.
    for index in range(32):
        # Freeze before the first held-out window; validation cannot change feature selection.
        if index == 16:
            # The minimum remains eight per class, even though there are only two classes.
            assert decoder.train()["ok"]
        # Alternate synthetic rest/select labels rather than correlating labels with elapsed time.
        label = decoder.labels[index % 2]
        # Two different known frequencies provide a controlled separability test, not a physiological claim.
        rows = packets(20 if label == "rest" else 10, seed=index + 1,
                       start=100 + index * 3, counter=index * 750)
        # Expand to four finite channels and add line noise equally across the two classes.
        for sample_index, row in enumerate(rows):
            # Vary line amplitude independently of the class cue to avoid encoding the label in interference.
            line = (100 + 20 * (index // 2 % 3)) * 1e-6 * np.sin(2 * np.pi * 60 * sample_index / 250)
            # Duplicate known spectral content with channel-specific amplitudes for a full four-channel vector.
            row["samples"] = [row["samples"][0] + line, row["samples"][1] + line,
                              row["samples"][0] * 0.7 + line, row["samples"][1] * 1.2 + line]
        # Run real raw integrity checks, detrending, spectral integration, and version tagging.
        features = extract_features(rows, channels, 250, {"session_id": "synthetic-four-channel"},
                                    feature_profile=CLICK_FEATURE_PROFILE)
        # Fail immediately if raw validation or the new processing contract rejects this valid fixture.
        assert features["ok"]
        # Feed the resulting full metadata into the production training/validation path.
        assert decoder.add_trial(label, features, phase="train" if index < 16 else "validate", trial_id=index)["ok"]
    # A real four-channel extraction pipeline must separate the known synthetic classes in held-out windows.
    assert decoder.status()["arm_eligible"]
    # Scores and validation columns remain binary instead of silently creating a third class.
    assert set(decoder.status()["evaluation"]["confusion_matrix"]) == {"rest", "select"}
