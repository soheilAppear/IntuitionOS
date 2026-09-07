"""Calibration turns a stated confidence into a claim that can be checked, so the
tests are mostly about the check being right when the system is wrong."""

import pytest

from core.calibration import (
    DEFAULT_THRESHOLDS,
    Calibrator,
    CalibrationStore,
    load_thresholds,
    observations,
    reliability,
)
from core.capabilities import capabilities, gate, set_safe_mode
from core.episodes import Episode


def shown(conf, accepted, action="x"):
    return Episode(action=action, predicted=action, predicted_conf=conf,
                   accepted_prediction=accepted)


def unshown(conf=0.9, action="x"):
    return Episode(action=action, predicted=action, predicted_conf=conf,
                   accepted_prediction=None)


# ── Only shown predictions count ────────────────────────────────────────────


def test_predictions_never_shown_are_excluded():
    """Counting a hint the user never saw as a miss would drag every bin down."""
    eps = [shown(0.8, 1), shown(0.8, 0), unshown(0.8), unshown(0.9)]
    assert observations(eps) == [(0.8, 1), (0.8, 0)]


def test_an_episode_without_a_confidence_is_excluded():
    assert observations([Episode(action="x", accepted_prediction=1)]) == []


def test_an_empty_log_reports_nothing_rather_than_perfection():
    r = reliability([])
    assert r.n == 0
    assert r.ece == 0.0
    assert "nothing" in r.table().lower()


# ── The reliability curve ───────────────────────────────────────────────────


def test_a_perfectly_calibrated_system_has_near_zero_error():
    # The 0.7-0.8 bin states its midpoint, 0.75, so a perfectly calibrated
    # system is right exactly 75 times in 100.
    eps = [shown(0.75, 1) for _ in range(75)] + [shown(0.75, 0) for _ in range(25)]

    r = reliability(eps)
    assert r.n == 100
    assert r.ece < 0.01, f"ECE was {r.ece:.3f} for a system right 75% of the time at 0.75"


def test_overconfidence_is_detected_and_reported():
    """The brief's example: if the model says 0.8 and is right 40% of the time,
    that must be visible."""
    eps = [shown(0.85, 1) for _ in range(40)] + [shown(0.85, 0) for _ in range(60)]
    r = reliability(eps)

    assert r.ece > 0.3
    hot_bin = next(b for b in r.bins if b.count)
    assert hot_bin.observed == pytest.approx(0.4)
    assert hot_bin.gap < 0, "an overconfident bin must show a negative gap"
    assert "Poorly calibrated" in r.table()


def test_the_table_reports_expected_calibration_error():
    r = reliability([shown(0.9, 0) for _ in range(40)])
    text = r.table()
    assert "Expected calibration error" in text
    assert "Maximum calibration error" in text
    assert f"{r.ece:.3f}" in text


def test_maximum_error_catches_a_rare_but_badly_wrong_bin():
    """ECE weights by frequency, so a small confidently-wrong corner can hide in
    it. MCE is what surfaces that."""
    eps = [shown(0.55, 1) for _ in range(96)] + [shown(0.55, 0) for _ in range(80)]
    eps += [shown(0.95, 0) for _ in range(4)]
    r = reliability(eps)
    assert r.mce > r.ece


def test_brier_score_is_reported():
    r = reliability([shown(1.0, 1) for _ in range(40)])
    assert r.brier == pytest.approx(0.0)


# ── Isotonic fitting ────────────────────────────────────────────────────────


def test_an_unfitted_calibrator_is_the_identity():
    c = Calibrator()
    for v in (0.0, 0.25, 0.5, 0.99, 1.0):
        assert c.calibrate(v) == pytest.approx(v)


def test_it_refuses_to_fit_on_too_little_data():
    """A three-sample curve is noise wearing a curve's clothes."""
    c = Calibrator().fit([shown(0.9, 0), shown(0.9, 1), shown(0.8, 1)])
    assert not c.is_fitted
    assert c.calibrate(0.9) == pytest.approx(0.9)


def test_fitting_pulls_an_overconfident_estimate_down():
    eps = [shown(0.9, 1) for _ in range(30)] + [shown(0.9, 0) for _ in range(70)]
    c = Calibrator().fit(eps)

    assert c.is_fitted
    corrected = c.calibrate(0.9)
    assert corrected < 0.5, f"0.9 was right 30% of the time but calibrated to {corrected:.2f}"


def test_fitting_pushes_an_underconfident_estimate_up():
    eps = [shown(0.3, 1) for _ in range(90)] + [shown(0.3, 0) for _ in range(10)]
    c = Calibrator().fit(eps)
    assert c.calibrate(0.3) > 0.7


def test_the_fitted_map_is_monotonic():
    """More confident must never come out less probable. This is the property
    isotonic regression is chosen for."""
    eps = []
    for conf, rate in ((0.2, 0.5), (0.4, 0.3), (0.6, 0.8), (0.8, 0.6), (0.95, 0.9)):
        hits = int(20 * rate)
        eps += [shown(conf, 1) for _ in range(hits)]
        eps += [shown(conf, 0) for _ in range(20 - hits)]

    c = Calibrator().fit(eps)
    xs = [i / 50 for i in range(51)]
    ys = [c.calibrate(x) for x in xs]
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:])), "the calibrated map decreased"


def test_calibrated_output_stays_in_range():
    eps = [shown(0.5, 1) for _ in range(50)]
    c = Calibrator().fit(eps)
    for x in (-1.0, 0.0, 0.5, 1.0, 2.0):
        assert 0.0 <= c.calibrate(x) <= 1.0


def test_calibration_measurably_improves_the_error_it_was_fitted_on():
    eps = [shown(0.9, 1) for _ in range(20)] + [shown(0.9, 0) for _ in range(80)]
    before = reliability(eps).ece

    c = Calibrator().fit(eps)
    recalibrated = [shown(c.calibrate(e.predicted_conf), e.accepted_prediction) for e in eps]
    after = reliability(recalibrated).ece

    assert after < before
    assert after < 0.1


def test_a_calibrator_round_trips_through_a_dict():
    eps = [shown(0.9, 1) for _ in range(20)] + [shown(0.9, 0) for _ in range(30)]
    c = Calibrator().fit(eps)
    revived = Calibrator.from_dict(c.to_dict())
    assert revived.calibrate(0.9) == pytest.approx(c.calibrate(0.9))


def test_it_persists(memory):
    store = CalibrationStore(memory)
    eps = [shown(0.9, 1) for _ in range(20)] + [shown(0.9, 0) for _ in range(30)]
    store.save(Calibrator().fit(eps))

    revived = CalibrationStore(memory).load()
    assert revived.is_fitted
    assert revived.calibrate(0.9) < 0.6


def test_a_corrupt_stored_curve_falls_back_to_the_identity(memory):
    store = CalibrationStore(memory)
    memory.execute("INSERT INTO calibration_state (id, updated_ts, state_json) VALUES (1,?,?)",
                   (0.0, "{not json"))
    c = store.load()
    assert not c.is_fitted
    assert c.calibrate(0.7) == pytest.approx(0.7)


# ── Cost-gated thresholds ───────────────────────────────────────────────────


def test_defaults_encode_the_asymmetry_between_prewarm_and_reveal():
    t = load_thresholds(None)
    assert t["free"] < t["reveal"] < t["auto_execute"]
    assert t["irreversible"] is None


def test_config_overrides_are_honoured():
    t = load_thresholds({"free": 0.1, "reveal": 0.5, "auto_execute": 0.99})
    assert t["free"] == 0.1
    assert t["reveal"] == 0.5
    assert t["auto_execute"] == 0.99


def test_a_configured_irreversible_threshold_is_refused():
    """Putting a number here would mean some confidence buys an unrepeatable
    action, which is the one thing the gate must never allow."""
    t = load_thresholds({"irreversible": 0.99})
    assert t["irreversible"] is None


def test_reveal_can_never_drop_below_prewarm():
    t = load_thresholds({"free": 0.8, "reveal": 0.2})
    assert t["reveal"] == 0.8


def test_unknown_keys_are_ignored():
    t = load_thresholds({"nonsense": 1.0})
    assert "nonsense" not in t
    assert t == load_thresholds(None)


# ── The gate honours the policy ─────────────────────────────────────────────


def test_an_irreversible_capability_is_never_auto_executed_at_any_confidence(project):
    """The acceptance criterion, checked against the real manifest and the real
    threshold policy rather than a hand-built capability."""
    set_safe_mode(False)
    thresholds = load_thresholds({"free": 0.0, "auto_execute": 0.0})

    for name, args in (("run_local", {"cmd": "echo hi"}), ("delete_task", {"task_id": 1})):
        cap = capabilities.get(name)
        for conf in (0.5, 0.95, 0.999, 1.0):
            d = gate(cap, args, confidence=conf, actor="model", thresholds=thresholds)
            assert d.verdict == "confirm", f"{name} auto-executed at {conf}"


def test_a_reversible_capability_needs_the_auto_execute_threshold(project):
    thresholds = load_thresholds({"auto_execute": 0.95})
    cap = capabilities.get("write_file")
    args = {"path": "a.txt", "text": "x"}

    assert gate(cap, args, confidence=0.94, actor="model", thresholds=thresholds).verdict == "confirm"
    assert gate(cap, args, confidence=0.96, actor="model", thresholds=thresholds).verdict == "allow"


def test_a_free_capability_only_needs_the_free_threshold(project):
    thresholds = load_thresholds({"free": 0.30})
    cap = capabilities.get("list_dir")

    assert gate(cap, {"path": "."}, confidence=0.2, actor="anticipator",
                thresholds=thresholds).verdict == "confirm"
    assert gate(cap, {"path": "."}, confidence=0.4, actor="anticipator",
                thresholds=thresholds).verdict == "allow"


def test_a_user_typed_command_is_not_subject_to_confidence_thresholds(project):
    """Thresholds gate guesses. A human who typed it has supplied the certainty."""
    thresholds = load_thresholds({"auto_execute": 0.99})
    cap = capabilities.get("write_file")
    d = gate(cap, {"path": "a.txt", "text": "x"}, confidence=0.0, actor="user",
             thresholds=thresholds)
    assert d.verdict == "allow"
