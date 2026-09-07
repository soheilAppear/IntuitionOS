"""The anticipator: speculative work done while the user is still typing.

Three things were wrong with the original beyond its fake confidence numbers, and
all three are the kind that only show up after the thing has been running a while:

  * The cache was an unbounded dict that was never evicted, so it grew for the
    lifetime of the process (Appendix A #8).
  * It was keyed on exact text and never invalidated, so a prewarmed `ls` was
    served after the directory had changed underneath it (Appendix A #9).
  * The worker busy-polled every 50 ms forever, waking forty times a second to
    discover that nothing had been typed (Appendix A #10).

It now takes a Predictor rather than a raw callable, so what gets prewarmed is
whatever the system has actually learned the user tends to do — and the reveal
decision (whether to *show* a hint) is separate from the prewarm decision, because
being wrong costs differently in each case.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Optional


class TTLCache:
    """Bounded, expiring cache. Both bounds matter: the size bound stops the leak,
    the TTL stops a stale directory listing being served as fresh."""

    def __init__(self, maxsize: int = 64, ttl_s: float = 20.0):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key, value):
        with self._lock:
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

    def get(self, key):
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            stamped, value = entry
            if time.monotonic() - stamped > self.ttl_s:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def clear(self):
        with self._lock:
            self._items.clear()

    def __len__(self):
        with self._lock:
            return len(self._items)


class Anticipator:
    def __init__(self, prewarm_fn, predictor=None, predict_fn=None, enabled=True,
                 debounce_ms=180, match_threshold=0.6, reveal_threshold=None,
                 context_fn=None, cache_size=64, cache_ttl_s=20.0, thresholds=None):
        # Either a Predictor (preferred) or a bare callable, so an eval harness or
        # a test can drive it with something trivial.
        self.predictor = predictor
        self.predict_fn = predict_fn
        self.prewarm_fn = prewarm_fn
        self.context_fn = context_fn
        self.enabled = enabled
        self.debounce_ms = debounce_ms

        # Two thresholds, not one, and the asymmetry is the point. Prewarming
        # wrongly costs a few milliseconds of a background thread. Revealing
        # wrongly costs the user the attention to notice and dismiss a bad
        # suggestion, which is a real cost the original design did not model.
        thresholds = thresholds or {}
        self.match_threshold = float(thresholds.get("free", match_threshold))
        self.reveal_threshold = float(
            reveal_threshold if reveal_threshold is not None
            else thresholds.get("reveal", max(match_threshold, 0.7))
        )

        self._buffer = ""
        self._cache = TTLCache(maxsize=cache_size, ttl_s=cache_ttl_s)
        self._last_prediction: Optional[list] = None
        self._stop = False
        self._thread = None
        self._last_change = 0.0
        # Signalled by update_buffer, waited on by the worker. This is what
        # replaces the 50 ms spin.
        self._wake = threading.Condition()
        self._dirty = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, name="anticipator", daemon=True)
        self._thread.start()

    def stop(self):
        with self._wake:
            self._stop = True
            self._wake.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # ── Input ────────────────────────────────────────────────────────────

    def update_buffer(self, text: str):
        with self._wake:
            self._buffer = text
            self._last_change = time.monotonic()
            self._dirty = True
            self._wake.notify()

    def invalidate(self):
        """Drop everything prewarmed. Called when the world may have moved —
        a write, a directory change, a resumed session."""
        self._cache.clear()

    def try_serve(self, text: str):
        return self._cache.get(text)

    def last_predictions(self) -> list:
        return list(self._last_prediction or [])

    # ── Prediction ───────────────────────────────────────────────────────

    def predict(self, buf: str) -> list:
        """Ranked candidates for the current buffer, as Prediction objects."""
        if self.predictor is not None:
            ctx = self.context_fn() if self.context_fn else None
            return self.predictor.predict(buf, ctx)
        if self.predict_fn is not None:
            out = self.predict_fn(buf)
            return out if isinstance(out, list) else ([out] if out else [])
        return []

    def should_reveal(self, prediction) -> bool:
        """Whether a hint is worth the user's attention, as distinct from whether
        it is worth a background thread's cycles."""
        return _confidence(prediction) >= self.reveal_threshold

    # ── Worker ───────────────────────────────────────────────────────────

    def _run(self):
        debounce_s = self.debounce_ms / 1000.0
        while True:
            with self._wake:
                # Sleep until there is something to do, rather than waking forty
                # times a second to find out there is not.
                while not self._dirty and not self._stop:
                    self._wake.wait(timeout=1.0)
                if self._stop:
                    return
                buf = self._buffer
                since_change = time.monotonic() - self._last_change
                if since_change < debounce_s:
                    # Still typing. Wait out the rest of the debounce; another
                    # keystroke will re-signal and reset it.
                    self._wake.wait(timeout=debounce_s - since_change)
                    continue
                self._dirty = False

            if not buf.strip():
                continue

            try:
                predictions = self.predict(buf)
            except Exception:
                continue
            if not predictions:
                continue

            self._last_prediction = predictions
            top = predictions[0]
            if _confidence(top) < self.match_threshold:
                continue

            try:
                result = self.prewarm_fn(top)
            except Exception:
                # A failed speculation is not an error the user should ever see.
                continue
            if result:
                key, value = result
                self._cache.put(key, value)


def _confidence(prediction) -> float:
    if prediction is None:
        return 0.0
    if isinstance(prediction, dict):
        return float(prediction.get("confidence", 0.0))
    return float(getattr(prediction, "confidence", 0.0))
