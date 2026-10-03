"""Click audio never blocks capture, queues a backlog, or plays during tests."""
import io
import struct
import wave

import pytest

from core.hand_feedback import HandClickSound, click_wave


class Workers:
    def __init__(self):
        self.pending = []

    def __call__(self, *, target, args, daemon, name):
        assert daemon and name == "hand-click-sound"
        pending = self.pending

        class Worker:
            def start(self):
                pending.append(lambda: target(*args))
        return Worker()

    def run(self):
        self.pending.pop(0)()


def test_click_wave_is_short_quiet_valid_pcm():
    with wave.open(io.BytesIO(click_wave()), "rb") as sound:
        assert (sound.getnchannels(), sound.getsampwidth(), sound.getframerate()) == (1, 2, 32000)
        assert sound.getnframes() / sound.getframerate() == pytest.approx(0.035)
        samples = struct.unpack(f"<{sound.getnframes()}h", sound.readframes(sound.getnframes()))
    assert 0 < max(abs(value) for value in samples) < 0.23 * 32767
    assert abs(samples[-1]) < 20


def test_click_sound_is_nonblocking_deduplicated_and_has_no_backlog():
    played, workers = [], Workers()
    sound = HandClickSound(player=played.append, thread_factory=workers)
    assert sound.play("one") is True
    assert played == []
    assert sound.play("one") is False
    assert sound.play("two") is False  # Busy ticks are dropped, never queued.
    assert len(workers.pending) == 1
    workers.run()
    assert played == [click_wave()]
    assert sound.play("two") is False
    assert sound.play("three") is True
    workers.run()
    assert len(played) == 2


@pytest.mark.parametrize("action", ["mute", "stop", "close"])
def test_muting_or_stopping_cancels_a_tick_not_yet_started(action):
    played, workers = [], Workers()
    sound = HandClickSound(player=played.append, thread_factory=workers)
    sound.play("old")
    if action == "mute":
        sound.set_enabled(False)
    elif action == "close":
        sound.close()
    else:
        sound.cancel_pending()
    workers.run()
    assert played == []
    sound.set_enabled(True)
    assert sound.play("old") is False
    assert sound.play("new") is True
    workers.run()
    assert len(played) == 1


def test_audio_failure_is_reported_once_and_can_be_retried_after_toggle():
    workers, reports, attempts = Workers(), [], []

    def player(_data):
        attempts.append(True)
        raise OSError("speaker unavailable")

    sound = HandClickSound(player=player, thread_factory=workers, on_status=reports.append)
    sound.play("one")
    workers.run()
    assert "speaker unavailable" in sound.status()["error"]
    assert reports[-1] == sound.status()
    assert sound.play("two") is False
    assert attempts == [True]
    sound.set_enabled(False)
    sound.set_enabled(True)
    assert sound.status()["error"] is None
    assert sound.play("three") is True


@pytest.mark.parametrize("value", [None, 1, "true", [], {}])
def test_sound_setting_requires_boolean(value):
    with pytest.raises(ValueError):
        HandClickSound(enabled=value)
    with pytest.raises(ValueError):
        HandClickSound(player=lambda _: None).set_enabled(value)


def test_thread_start_failure_never_escapes_to_pointer_control():
    def broken(**kwargs):
        raise RuntimeError("could not create worker")
    sound = HandClickSound(player=lambda _: None, thread_factory=broken)
    assert sound.play("one") is False
    assert "could not create worker" in sound.status()["error"]
