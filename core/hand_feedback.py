"""Brief local click feedback, isolated from camera and pointer timing."""

import io
import math
import random
import struct
import sys
import threading
import wave
from collections import deque
from functools import lru_cache


@lru_cache(maxsize=1)
def click_wave():
    """An original 35 ms soft mechanical tick, generated entirely in memory."""
    rate, duration = 32000, 0.035
    noise = random.Random(7432)
    samples = []
    for index in range(round(rate * duration)):
        t = index / rate
        envelope = (1 - math.exp(-t / 0.0004)) * math.exp(-t / 0.005)
        value = 0.22 * envelope * (
            0.65 * noise.uniform(-1, 1) + 0.35 * math.sin(2 * math.pi * 2100 * t))
        samples.append(round(value * 32767))
    output = io.BytesIO()
    with wave.open(output, "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(rate)
        sound.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return output.getvalue()


def _play_local_wave(data):
    import winsound
    # SND_MEMORY is synchronous. A separate short-lived worker keeps it off the
    # camera thread; there is no dependency on browser focus or autoplay policy.
    winsound.PlaySound(data, winsound.SND_MEMORY | winsound.SND_NODEFAULT)


class HandClickSound:
    def __init__(self, enabled=True, *, player=None, on_status=None, thread_factory=threading.Thread):
        if type(enabled) is not bool:
            raise ValueError("Click sound enabled must be a boolean")
        self._enabled = enabled
        self._available = player is not None or sys.platform == "win32"
        self._player = player or _play_local_wave
        self._on_status = on_status
        self._thread_factory = thread_factory
        self._lock = threading.RLock()
        self._seen = deque(maxlen=128)
        self._busy = False
        self._generation = 0
        self._error = None

    def status(self):
        with self._lock:
            return {"enabled": self._enabled, "available": self._available, "error": self._error}

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("Click sound enabled must be a boolean")
        with self._lock:
            self._enabled = enabled
            self._error = None
            self._generation += 1
        return self.status()

    def cancel_pending(self):
        with self._lock:
            self._generation += 1

    def close(self):
        self.set_enabled(False)

    def play(self, event_id):
        """Schedule at most one tick, never queue sounds or block tracking."""
        with self._lock:
            if not isinstance(event_id, str) or not event_id or event_id in self._seen:
                return False
            self._seen.append(event_id)
            if not self._enabled or not self._available or self._busy or self._error:
                return False
            self._busy = True
            generation = self._generation
            try:
                self._thread_factory(target=self._run, args=(generation,),
                                     daemon=True, name="hand-click-sound").start()
            except Exception as error:
                self._busy = False
                self._error = f"Click sound could not start: {error}"
                self._report()
                return False
            return True

    def _run(self, generation):
        failed = False
        try:
            with self._lock:
                if not self._enabled or generation != self._generation:
                    return
            self._player(click_wave())
        except Exception as error:
            with self._lock:
                self._error = f"Click sound unavailable: {error}"
            failed = True
        finally:
            with self._lock:
                self._busy = False
            if failed:
                self._report()

    def _report(self):
        if self._on_status:
            try:
                self._on_status(self.status())
            except Exception:
                pass
