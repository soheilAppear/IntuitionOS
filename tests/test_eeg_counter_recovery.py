"""Acquisition reset recovery uses fake packets, models, hands, and mouse input."""

import copy

import pytest

from test_eeg_blink_control import arm_ready, blink


def publish(h, counter, *, gap=1, invalid=False):
    row = {"counter": counter, "estimated_monotonic": h.clock.now,
           "host_received_monotonic": h.clock.now, "samples": [1e-6, -2e-6]}
    if invalid:
        row["samples"][0] = None
    h.eeg.rows.append(row)
    h.tick(add_rows=False, gap=gap)


@pytest.mark.parametrize("retain_old_rows", [False, True])
def test_backward_counter_reset_recovers_waveform_only_after_new_progress(blink, retain_old_rows):
    h = blink
    # The display buffer may retain pre-reset high counters alongside current low counters.
    if retain_old_rows:
        h.eeg.acquisition_snapshot = lambda: {**copy.deepcopy(h.eeg.acq), "samples": copy.deepcopy(h.eeg.rows)}
    assert h.service._last_valid_counter > 100
    assert h.service.snapshot()["eeg"]["waveform"]
    publish(h, 0)
    assert h.service._last_valid_counter == 0
    assert not h.service._eeg_signal_healthy
    assert h.service.snapshot()["eeg"]["waveform"] == []
    h.tick(8, add_rows=False, gap=1)  # A cache reporting zero age cannot finish recovery by itself.
    assert not h.service._eeg_signal_healthy
    assert h.service.snapshot()["eeg"]["waveform"] == []
    publish(h, 1)
    assert h.service._last_valid_counter == 1
    assert h.service._eeg_signal_healthy
    assert h.service.snapshot()["eeg"]["waveform"][-1]["counter"] == 1
    publish(h, 2)
    assert h.service._last_valid_counter == 2
    assert h.service._eeg_signal_healthy


def test_malformed_gap_tail_keeps_recovery_pending_without_using_historical_rows(blink):
    h = blink
    h.eeg.acquisition_snapshot = lambda: {**copy.deepcopy(h.eeg.acq), "samples": copy.deepcopy(h.eeg.rows)}
    publish(h, 0, invalid=True)
    assert h.service._eeg_counter_reanchor
    assert not h.service._eeg_signal_healthy
    h.tick(8, add_rows=False, gap=1)
    assert h.service._eeg_counter_reanchor
    assert h.service.snapshot()["eeg"]["waveform"] == []
    publish(h, 1)
    assert not h.service._eeg_counter_reanchor
    assert not h.service._eeg_signal_healthy  # First finite tail only establishes the new watermark.
    publish(h, 2)
    assert h.service._eeg_signal_healthy
    assert h.service.snapshot()["eeg"]["waveform"][-1]["counter"] == 2


def test_counter_reset_cancels_old_blink_event_preserves_calibration_and_requires_new_pair(blink):
    h = blink
    arm_ready(h)
    h.blink.trials.append({"label": "rest", "phase": "train", "preserved": True})
    original = copy.deepcopy(h.blink.trials)
    resets = h.blink.stream_resets
    h.blink.event("pre-reset-pair", h.clock.now)
    publish(h, 0)
    assert h.control()["armed"] and h.mouse.clicks == 0
    publish(h, 1)
    assert h.blink.stream_resets > resets
    assert h.blink.watermarks[-1] == 1
    assert h.mouse.clicks == 0 and h.blink.trials == original
    h.blink.event("post-reset-pair", h.clock.now)
    publish(h, 2)
    assert h.mouse.clicks == 1
    assert h.blink.trials == original


def test_normal_uint32_wrap_advances_without_spurious_integrity_reset(blink):
    h = blink
    h.service._last_valid_counter = 2**32 - 2
    publish(h, 2**32 - 1, gap=0)
    assert h.service._eeg_signal_healthy
    publish(h, 0, gap=0)
    assert h.service._eeg_signal_healthy
    assert h.service._last_valid_counter == 0


def test_backward_counter_without_adapter_gap_cannot_refresh_or_replay_data(blink):
    h = blink
    previous = h.service._last_valid_counter
    publish(h, 0, gap=0)
    h.tick(8, add_rows=False, gap=0)
    assert h.service._last_valid_counter == previous
    assert not h.service._eeg_signal_healthy
    assert h.service.snapshot()["eeg"]["waveform"] == []
