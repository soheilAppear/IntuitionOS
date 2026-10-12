"""Counter-timed calibration with simulated BLE jitter; never opens hardware."""

import copy

import pytest

from test_eeg_blink_control import blink  # Reuse the fake detector and fake hand/mouse service fixture.


def begin(h):
    h.blink.trained = h.blink.passed = h.blink.eligible = False
    h.service._decoder_status = h.blink.status()
    assert h.service.eeg_trial("select", "train").get("ok")
    return h.clock.now, h.service._eeg_pending["blink_start_counter"]


def batch(h, start, watermark, first, count=25, delay=0.0, missing=None):
    rows = [{"counter": (watermark + n + 1) & 0xFFFFFFFF,
             "estimated_monotonic": start + n / 250 + delay,
             "host_received_monotonic": start + (first + count) / 250 + delay,
             "samples": [1e-6, -2e-6]} for n in range(first, first + count) if n != missing]
    h.eeg.rows = (h.eeg.rows + rows)[-1250:]  # Match the acquisition cache's overlapping snapshots.
    at = start + (first + count) / 250 + delay
    h.tick(add_rows=False, step=max(0, at - h.clock.now))
    return rows


@pytest.mark.parametrize("drift", [0.0, 0.008, 0.016])
def test_complete_counter_window_survives_callback_clock_drift(blink, drift):
    h = blink
    start, watermark = begin(h)
    original = []
    for index in range(50):
        original.extend(batch(h, start, watermark, index * 25, delay=(index + 1) * drift))
        if index == 24:
            h.tick(add_rows=False, step=0)  # A repeated buffer must not duplicate accepted packets.
    assert h.control()["pending"] is None
    assert len(h.blink.trials) == 1, h.control()["last_result"]
    trial = h.blink.trials[0]
    rows = trial["signal"]["samples"]
    assert len(rows) == len({r["counter"] for r in rows}) == 1250
    assert rows[0]["counter"] == (watermark + 1) & 0xFFFFFFFF
    assert rows[-1]["counter"] == (watermark + 1250) & 0xFFFFFFFF
    assert rows[0]["estimated_monotonic"] == start
    assert rows[-1]["estimated_monotonic"] == pytest.approx(start + 4.996)
    assert h.eeg.rows[-1]["estimated_monotonic"] == original[-1]["estimated_monotonic"]
    assert [r["samples"] for r in rows] == [r["samples"] for r in original]
    assert h.mouse.clicks == 0


def test_late_final_batch_keeps_cue_pending_and_reports_sample_progress(blink):
    h = blink
    start, watermark = begin(h)
    for index in range(49):
        batch(h, start, watermark, index * 25)
    h.tick(add_rows=False, step=0.15)
    pending = h.control()["pending"]
    assert pending["remaining_seconds"] == 0
    assert pending["samples_collected"] == 1225
    assert pending["samples_required"] == 1250
    assert "finishing" in pending["cue"].lower()
    assert not h.blink.trials
    batch(h, start, watermark, 1225, delay=0.3)
    assert len(h.blink.trials) == 1


def test_true_missing_counter_is_rejected_and_previous_trials_are_preserved(blink):
    h = blink
    h.blink.trials.append({"label": "rest", "phase": "train", "retained": True})
    previous = copy.deepcopy(h.blink.trials)
    start, watermark = begin(h)
    for index in range(50):
        batch(h, start, watermark, index * 25, missing=610)
    assert h.blink.trials == previous
    assert "packet gap" in h.control()["last_result"].lower()
    assert h.control()["pending"] is None


def test_collection_handles_counter_wrap_without_reusing_previous_cue(blink):
    h = blink
    h.eeg.rows = []
    h.service._last_valid_counter = 2**32 - 20
    start, watermark = begin(h)
    for index in range(50):
        batch(h, start, watermark, index * 25)
    first = h.blink.trials[0]
    start2, watermark2 = begin(h)
    for index in range(50):
        batch(h, start2, watermark2, index * 25)
    second = h.blink.trials[1]
    a = {r["counter"] for r in first["signal"]["samples"]}
    b = {r["counter"] for r in second["signal"]["samples"]}
    assert not a & b
    assert len(a) == len(b) == 1250


def test_delivery_timeout_retains_accepted_trials_and_never_fits_partial_data(blink):
    h = blink
    start, watermark = begin(h)
    for index in range(40):
        batch(h, start, watermark, index * 25)
    signal = h.eeg.acquisition_signal_window()
    h.clock.now = start + 5 + h.service.BLINK_COLLECTION_GRACE
    h.service._process_blink(signal, h.clock.now, h.service._generation)
    assert h.control()["pending"] is None
    assert "1000/1250" in h.control()["last_result"]
    assert not h.blink.trials


def test_complete_burst_delivered_after_deadline_is_rejected(blink):
    h = blink
    start, watermark = begin(h)
    for index in range(40):
        batch(h, start, watermark, index * 25)
    batch(h, start, watermark, 1000, count=250, delay=2.0)
    assert h.control()["pending"] is None
    assert "timed out" in h.control()["last_result"]
    assert not h.blink.trials
