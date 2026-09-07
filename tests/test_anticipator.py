"""The anticipator's bugs were all of the kind that only appear after hours of
running, so they get tests that make them appear in milliseconds."""

import threading
import time

import pytest

from core.anticipator import Anticipator, TTLCache
from core.predictor import Prediction


# ── The cache (Appendix A #8 and #9) ────────────────────────────────────────


def test_the_cache_is_bounded():
    """It used to be a plain dict that grew for the lifetime of the process."""
    cache = TTLCache(maxsize=4, ttl_s=60)
    for i in range(100):
        cache.put(f"key{i}", i)
    assert len(cache) == 4


def test_the_cache_evicts_the_least_recently_used():
    cache = TTLCache(maxsize=2, ttl_s=60)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")          # touch a, so b becomes the coldest
    cache.put("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_a_stale_entry_expires_rather_than_being_served():
    """The specific failure: a prewarmed `ls` served after the directory changed."""
    cache = TTLCache(maxsize=8, ttl_s=0.05)
    cache.put("ls", ["old listing"])
    assert cache.get("ls") == ["old listing"]
    time.sleep(0.08)
    assert cache.get("ls") is None


def test_invalidate_drops_everything():
    ant = Anticipator(prewarm_fn=lambda p: None, enabled=False)
    ant._cache.put("ls", "x")
    ant.invalidate()
    assert ant.try_serve("ls") is None


# ── The worker (Appendix A #10) ─────────────────────────────────────────────


def test_the_worker_sleeps_instead_of_spinning():
    """It used to wake every 50 ms forever. With nothing typed it should do
    essentially no work at all."""
    predictions = []

    def predict(buf):
        predictions.append(buf)
        return [Prediction(buf, 0.9)]

    ant = Anticipator(prewarm_fn=lambda p: None, predict_fn=predict, debounce_ms=10)
    ant.start()
    try:
        time.sleep(0.3)
        assert predictions == [], "the worker predicted with an empty buffer"
    finally:
        ant.stop()


def test_typing_wakes_the_worker_promptly():
    done = threading.Event()

    def prewarm(prediction):
        done.set()
        return (prediction.action, {"reply": "warmed"})

    ant = Anticipator(prewarm_fn=prewarm, predict_fn=lambda b: [Prediction(b, 0.95)],
                      debounce_ms=10, match_threshold=0.5)
    ant.start()
    try:
        ant.update_buffer("ls")
        assert done.wait(timeout=2.0), "the worker never woke"
        assert ant.try_serve("ls") == {"reply": "warmed"}
    finally:
        ant.stop()


def test_stop_actually_joins_the_thread():
    ant = Anticipator(prewarm_fn=lambda p: None, predict_fn=lambda b: [], debounce_ms=10)
    ant.start()
    ant.stop()
    time.sleep(0.05)
    assert not ant._thread.is_alive()


def test_a_prewarm_that_raises_is_swallowed():
    """A failed speculation is not something the user should ever see."""
    def exploding(_prediction):
        raise RuntimeError("disk gone")

    ant = Anticipator(prewarm_fn=exploding, predict_fn=lambda b: [Prediction(b, 0.9)],
                      debounce_ms=10, match_threshold=0.5)
    ant.start()
    try:
        ant.update_buffer("ls")
        time.sleep(0.2)
        assert ant.try_serve("ls") is None
    finally:
        ant.stop()


# ── Thresholds ──────────────────────────────────────────────────────────────


def test_prewarm_and_reveal_thresholds_are_separate():
    """The asymmetry is the point: being wrong costs cycles in one case and the
    user's attention in the other."""
    ant = Anticipator(prewarm_fn=lambda p: None, enabled=False,
                      thresholds={"free": 0.30, "reveal": 0.70})
    assert ant.match_threshold == pytest.approx(0.30)
    assert ant.reveal_threshold == pytest.approx(0.70)

    assert not ant.should_reveal(Prediction("ls", 0.5))
    assert ant.should_reveal(Prediction("ls", 0.8))


def test_a_low_confidence_prediction_is_not_prewarmed():
    warmed = []
    ant = Anticipator(prewarm_fn=lambda p: warmed.append(p) or (p.action, {}),
                      predict_fn=lambda b: [Prediction(b, 0.1)],
                      debounce_ms=10, thresholds={"free": 0.5, "reveal": 0.9})
    ant.start()
    try:
        ant.update_buffer("maybe")
        time.sleep(0.15)
        assert warmed == []
    finally:
        ant.stop()


# ── Driving it with a real Predictor ────────────────────────────────────────


def test_it_prewarms_what_the_predictor_learned(project):
    from core.context import Context
    from core.episodes import Episode
    from core.predictor import Predictor

    def context():
        return Context(ts=time.time(), cwd=str(project), hour_of_day=10,
                       recent_commands=[{"text": "git commit", "exit": 0}])

    predictor = Predictor(min_episodes=3)
    for _ in range(10):
        predictor.update(Episode(action="pytest", context=context(), keystroke_prefix="pytest"))

    warmed = []

    def prewarm(prediction):
        warmed.append(prediction.action)
        return (prediction.action, {"reply": "ok"})

    ant = Anticipator(prewarm_fn=prewarm, predictor=predictor, context_fn=context,
                      debounce_ms=10, thresholds={"free": 0.1, "reveal": 0.7})
    ant.start()
    try:
        ant.update_buffer("py")
        deadline = time.time() + 2.0
        while not warmed and time.time() < deadline:
            time.sleep(0.01)
        assert warmed == ["pytest"]
    finally:
        ant.stop()
