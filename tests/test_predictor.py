"""The predictor replaces four hardcoded confidence literals. The tests that
matter are the two ends: that a cold install behaves exactly as before, and that
a warm one has demonstrably learned something."""

import time

import pytest

from core.context import Context
from core.episodes import Episode, EpisodeLog
from core.predictor import (
    DEFAULT_HALF_LIFE_S,
    FrequencyModel,
    Predictor,
    PredictorStore,
    features,
    heuristic_predictions,
)


def ctx(cwd="/proj", branch="main", hour=10, dirty=False, prev=None, exit_code=0, gap=1.0):
    return Context(
        ts=time.time(), cwd=cwd, git_branch=branch, git_dirty=dirty,
        hour_of_day=hour, day_of_week=2, last_exit_code=exit_code, idle_gap_s=gap,
        recent_commands=([{"text": prev, "exit": exit_code}] if prev else []),
    )


def ep(action, context=None, prefix=None, ts=None):
    return Episode(action=action, context=context or ctx(),
                   keystroke_prefix=prefix if prefix is not None else action,
                   ts=ts or time.time())


# ── Cold start: no regression on a fresh install ────────────────────────────


def test_an_empty_log_reproduces_the_previous_heuristics_exactly():
    """The literals the old inline predict() returned, preserved."""
    p = Predictor()
    assert p.predict("tree")[0].confidence == pytest.approx(0.9)
    assert p.predict("ls")[0].confidence == pytest.approx(0.9)
    assert p.predict("read file x.txt")[0].confidence == pytest.approx(0.85)
    assert p.predict("anything else")[0].confidence == pytest.approx(0.65)


def test_cold_start_predictions_are_marked_as_heuristic():
    assert Predictor().predict("ls")[0].source == "heuristic"


def test_empty_input_predicts_nothing():
    assert Predictor().predict("") == []
    assert Predictor().predict("   ") == []


def test_the_fallback_holds_until_min_episodes():
    p = Predictor(min_episodes=50)
    for _ in range(49):
        p.update(ep("pytest"))
    assert p.predict("ls")[0].source == "heuristic", "49 episodes is not enough to trust"
    p.update(ep("pytest"))
    assert p.seen == 50


# ── The minimum viable demonstration that it learns ─────────────────────────


def test_it_learns_that_pytest_follows_git_commit():
    """The brief's acceptance criterion, verbatim: given a log where the user
    always types pytest after git commit, pytest must rank first in that context."""
    p = Predictor(min_episodes=5)
    for _ in range(20):
        p.update(ep("git commit -m wip", ctx(prev="git add -A")))
        p.update(ep("pytest", ctx(prev="git commit -m wip")))

    ranked = p.predict("p", ctx(prev="git commit -m wip"))
    assert ranked, "the predictor produced nothing in a context it has seen 20 times"
    assert ranked[0].action == "pytest"
    assert ranked[0].source == "learned"


def test_the_same_prefix_predicts_differently_in_different_contexts():
    """If context did not change the answer, the context sensor would be dead
    weight and this would be a prefix table."""
    p = Predictor(min_episodes=5)
    for _ in range(15):
        p.update(ep("git push", ctx(cwd="/work", prev="git commit")))
        p.update(ep("git diff", ctx(cwd="/notes", prev="git commit")))

    at_work = p.predict("git", ctx(cwd="/work", prev="git commit"))[0].action
    at_notes = p.predict("git", ctx(cwd="/notes", prev="git commit"))[0].action
    assert at_work == "git push"
    assert at_notes == "git diff"


def test_a_prefix_narrows_the_candidates():
    p = Predictor(min_episodes=3)
    for _ in range(10):
        p.update(ep("git status", ctx()))
        p.update(ep("pytest -x", ctx()))

    top = p.predict("git", ctx())[0]
    assert top.action == "git status"


def test_predictions_come_back_ranked_and_bounded():
    p = Predictor(min_episodes=3)
    for i in range(10):
        for name in ("alpha", "beta", "gamma", "delta"):
            p.update(ep(name, ctx()))
    out = p.predict("a", ctx(), k=2)
    assert len(out) <= 2
    assert out == sorted(out, key=lambda x: -x.confidence)


def test_every_prediction_carries_a_reason():
    """A HUD that shows a hunch has to be able to justify it."""
    p = Predictor(min_episodes=3)
    for _ in range(10):
        p.update(ep("pytest", ctx(prev="git commit")))
    assert p.predict("py", ctx(prev="git commit"))[0].why


# ── Recency ─────────────────────────────────────────────────────────────────


def test_recent_behaviour_outweighs_old_behaviour():
    """A habit you dropped a month ago should not beat one you have this week."""
    now = time.time()
    old = now - 60 * 24 * 3600
    p = Predictor(min_episodes=3)
    for _ in range(30):
        p.update(ep("make build", ctx(prev="git pull"), ts=old))
    for _ in range(5):
        p.update(ep("just build", ctx(prev="git pull"), ts=now))

    assert p.predict("bu", ctx(prev="git pull"))[0].action == "just build"


def test_decay_halves_the_weight_over_one_half_life():
    m = FrequencyModel(half_life_s=100.0)
    t0 = 1_000_000.0
    m.observe("x", ctx(), prefix="x", ts=t0)
    fresh = m.score("x", ctx(), now=t0)[0][1]
    later = m.score("x", ctx(), now=t0 + 100.0)[0][1]
    assert later == pytest.approx(fresh * 0.5, rel=1e-6)


# ── update() is actually called ─────────────────────────────────────────────


def test_update_is_called_for_every_episode(monkeypatch):
    p = Predictor(min_episodes=1)
    seen = []
    real = p.update
    monkeypatch.setattr(p, "update", lambda e: (seen.append(e.action), real(e))[1])

    p.fit([ep("a"), ep("b"), ep("c")])
    assert seen == ["a", "b", "c"]
    assert p.seen == 3


def test_an_episode_with_no_action_is_ignored():
    p = Predictor(min_episodes=1)
    p.update(Episode(action=""))
    assert p.seen == 0


# ── Persistence ─────────────────────────────────────────────────────────────


def test_state_survives_a_restart(memory):
    store = PredictorStore(memory)
    p = Predictor(store=store, min_episodes=3)
    for _ in range(10):
        p.update(ep("pytest", ctx(prev="git commit")))
    p.save()

    revived = Predictor(store=PredictorStore(memory), min_episodes=3)
    assert revived.seen == 10
    assert revived.predict("py", ctx(prev="git commit"))[0].action == "pytest"


def test_a_corrupt_state_blob_falls_back_rather_than_crashing(memory):
    store = PredictorStore(memory)
    memory.execute("INSERT INTO predictor_state (id, updated_ts, state_json) VALUES (1,?,?)",
                   (time.time(), "{not json"))
    p = Predictor(store=store)
    assert p.seen == 0
    assert p.predict("ls")[0].source == "heuristic"


def test_loading_from_an_empty_store_is_a_cold_start(memory):
    p = Predictor(store=PredictorStore(memory))
    assert p.seen == 0


def test_state_round_trips_through_a_dict():
    p = Predictor(min_episodes=3)
    for _ in range(10):
        p.update(ep("pytest", ctx(prev="git commit")))
    revived = Predictor(min_episodes=3)
    revived.load_dict(p.to_dict())

    assert revived.seen == p.seen
    assert revived.predict("py", ctx(prev="git commit"))[0].action == "pytest"


def test_it_can_be_fitted_straight_from_the_episode_log(memory, project):
    log = EpisodeLog(memory)
    for _ in range(10):
        log.record("pytest", ctx(prev="git commit"))

    p = Predictor(min_episodes=3).fit(log.all())
    assert p.seen == 10
    assert p.predict("py", ctx(prev="git commit"))[0].action == "pytest"


# ── Features ────────────────────────────────────────────────────────────────


def test_features_capture_the_signals_the_brief_asked_for():
    f = features("/read config.yaml", ctx(branch="feature", dirty=True, exit_code=1, prev="pytest"))
    assert f["slash"] == 1.0
    assert f["git:dirty"] == 1.0
    assert f["branch:feature"] == 1.0
    assert f["exit:fail"] == 1.0
    assert f["prev:pytest"] == 1.0
    assert any(k.startswith("ng3:") for k in f)
    assert any(k.startswith("len:") for k in f)


def test_features_tolerate_a_missing_context():
    f = features("ls", None)
    assert f["bias"] == 1.0
    assert not any(k.startswith("branch:") for k in f)


def test_a_successful_and_a_failed_previous_command_look_different():
    ok = features("pytest", ctx(exit_code=0))
    bad = features("pytest", ctx(exit_code=1))
    assert "exit:ok" in ok and "exit:ok" not in bad
    assert "exit:fail" in bad


# ── The heuristics in isolation ─────────────────────────────────────────────


def test_heuristics_are_unchanged_from_the_original_if_chain():
    assert heuristic_predictions("")== []
    assert heuristic_predictions("tree src")[0].intent == "tree"
    assert heuristic_predictions("ls")[0].intent == "ls"
    assert heuristic_predictions("read file a.txt")[0].intent == "read_file"
    assert heuristic_predictions("hello there")[0].intent == "plan"
