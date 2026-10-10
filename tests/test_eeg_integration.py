"""Real EEG feature/classifier integration with synthetic sensors and a mock actuator.

Unlike the service boundary tests, these exercise the mixed feature windows
that naturally occur between confident rest and a confident direction.
"""

import math

import numpy as np
import pytest

from core import multimodal
from core.eeg_decoder import EegDecoder, LABELS, extract_features
from test_eeg_control import Harness


FREQUENCIES = {"left": 6, "right": 10, "rest": 20}


def sample(counter, label, rng):
    angle = 2 * math.pi * FREQUENCIES[label] * counter / 250
    return [20e-6 * math.sin(angle) + rng.normal(0, 0.5e-6),
            12e-6 * math.cos(angle) + rng.normal(0, 0.5e-6)]


class RealDecoderHarness(Harness):
    def __init__(self, monkeypatch):
        super().__init__(monkeypatch)
        monkeypatch.setattr(multimodal, "extract_features", extract_features)
        self.decoder = EegDecoder()
        self.service._decoder = self.decoder
        self.signal_label = "rest"
        self.rng = np.random.default_rng(900)
        self.raw_valid = True
        for index in range(48):
            if index == 24:
                assert self.decoder.train()["ok"]
            label = LABELS[index % 3]
            rng = np.random.default_rng(index + 1)
            rows = [{"counter": 100000 + index * 750 + offset,
                     "estimated_monotonic": 1000 + index * 3 + offset / 250,
                     "samples": sample(offset, label, rng)} for offset in range(500)]
            features = extract_features(rows, self.eeg.acq["channels"], 250,
                                        {"session_id": self.eeg.acq["session_id"]})
            assert features["ok"]
            assert self.decoder.add_trial(label, features,
                                          "train" if index < 24 else "validate", index)["ok"]
        assert self.decoder.status()["arm_eligible"]
        self.service._decoder_status = self.decoder.status()

    def tick(self, count=1, *, add_rows=True, **kwargs):
        for _ in range(count):
            if add_rows:
                target = round((self.clock.now + 0.125 - self.started_at) * 250)
                while self.eeg.counter < target:
                    counter = self.eeg.counter
                    self.eeg.rows.append({"counter": counter,
                                          "estimated_monotonic": self.started_at + counter / 250,
                                          "host_received_monotonic": self.clock.now + 0.125,
                                          "samples": sample(counter, self.signal_label, self.rng)
                                          if self.raw_valid else [0.0, 0.0]})
                    self.eeg.counter += 1
                self.eeg.rows = self.eeg.rows[-1250:]
            super().tick(add_rows=False, **kwargs)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_real_classifier_rest_transition_can_reach_one_direction_without_camera(monkeypatch, direction):
    h = RealDecoderHarness(monkeypatch)
    h.warm()
    h.arm()
    h.tick(12)
    assert h.control()["neutral_ready"]
    h.signal_label = direction
    saw_uncertain = False
    for _ in range(28):
        previous_count = len(h.actions)
        h.tick(tracked=False)
        prediction = h.control().get("prediction") or {}
        if prediction.get("label") is None:
            saw_uncertain = True
            assert len(h.actions) == previous_count
        assert h.control()["armed"]
    assert saw_uncertain  # The test actually traversed mixed rest/direction windows.
    assert h.actions == [(("os_switch_desktop", {"direction": direction}), {"actor": "gesture"})]
    assert h.camera.starts == 0 and h.camera.reads == 0
    h.tick(20, tracked=False)
    assert len(h.actions) == 1  # Sustaining one class cannot repeat navigation.


def test_real_classifier_fresh_repeated_poll_preserves_armed_rest_without_action(monkeypatch):
    h = RealDecoderHarness(monkeypatch)
    h.warm()
    h.arm()
    h.tick(12)
    assert h.control()["neutral_ready"]
    h.tick(add_rows=False, eeg_age=0.125)
    assert h.control()["armed"]
    assert h.control()["neutral_ready"]
    assert h.actions == []


def test_real_feature_quality_failure_disarms_instead_of_using_transition_grace(monkeypatch):
    h = RealDecoderHarness(monkeypatch)
    h.warm()
    h.arm()
    h.tick(12)
    h.raw_valid = False
    # A sustained exactly constant segment must reject feature quality;
    # it must not be treated as ordinary model uncertainty for three seconds.
    h.tick(2)
    assert not h.control()["armed"]
    assert h.actions == []
