"""Consolidation has to work with the model and without it, and the rules it
produces have to be something a person can read, disagree with, and delete."""

import json
import time

import pytest

from core.calibration import CalibrationStore, Calibrator
from core.consolidation import (
    Candidate,
    RuleStore,
    consolidate,
    find_candidates,
    judge,
    prune,
    render_rules,
)
from core.context import Context
from core.episodes import Episode
from core.predictor import Predictor


def ctx(cwd="/proj", branch="main", hour=10, dirty=False, prev=None):
    return Context(ts=time.time(), cwd=cwd, git_branch=branch, git_dirty=dirty,
                   hour_of_day=hour, day_of_week=2,
                   recent_commands=([{"text": prev, "exit": 0}] if prev else []))


def ep(action, **kw):
    return Episode(action=action, context=ctx(**kw), keystroke_prefix=action, ts=time.time())


def planted_log(n=10):
    """A log with one habit planted in it: pytest always follows git commit."""
    log = []
    for _ in range(n):
        log.append(ep("git add -A"))
        log.append(ep("git commit -m wip", prev="git add -A"))
        log.append(ep("pytest", prev="git commit -m wip"))
    return log


class StubLLM:
    def __init__(self, response=None, explode=False):
        self.response = response
        self.explode = explode
        self.calls = []

    def chat(self, messages, on_token=None):
        self.calls.append(messages)
        if self.explode:
            raise RuntimeError("Cannot reach Ollama")
        return self.response or json.dumps(
            {"genuine": True, "name": "test after commit",
             "description": "You run the test suite right after committing."}
        )


# ── Finding the planted pattern ─────────────────────────────────────────────


def test_a_planted_pattern_is_discovered():
    found = find_candidates(planted_log(), min_support=4)
    sequential = [c for c in found if c.pattern.get("previous") == "git commit -m wip"]
    assert sequential, "the planted habit was not found"
    assert sequential[0].action == "pytest"
    assert sequential[0].support == 10


def test_noise_below_the_support_threshold_is_not_promoted():
    log = [ep("something unusual", prev="git commit")] * 2
    assert find_candidates(log, min_support=4) == []


def test_a_pattern_the_user_contradicts_half_the_time_is_rejected():
    """Support alone is not enough: a pattern needs to beat its alternatives."""
    log = []
    for _ in range(6):
        log.append(ep("pytest", prev="git commit"))
        log.append(ep("git push", prev="git commit"))
        log.append(ep("make", prev="git commit"))
    found = find_candidates(log, min_support=4, min_confidence=0.5)
    assert found == [], "a 1-in-3 pattern was promoted on volume alone"


def test_commands_about_the_system_itself_are_not_learned():
    """Without this, /dream promptly discovers that you often run /dream."""
    log = [ep("/dream", prev="/rules") for _ in range(20)]
    assert find_candidates(log, min_support=4) == []


def test_candidates_are_ordered_strongest_first():
    log = planted_log(10)
    log += [ep("make docs", prev="make clean") for _ in range(4)]
    log += [ep("make build", prev="make clean") for _ in range(4)]
    found = find_candidates(log, min_support=4)
    confidences = [c.confidence for c in found]
    assert confidences == sorted(confidences, reverse=True)


def test_an_empty_log_yields_nothing():
    assert find_candidates([]) == []


# ── /dream end to end ───────────────────────────────────────────────────────


def test_dream_discovers_the_pattern_and_creates_a_rule(memory):
    rules = RuleStore(memory)
    report = consolidate(planted_log(), rules, llm=StubLLM())

    assert report.promoted, report.summary()
    assert any(r["action"] == "pytest" for r in rules.all())
    assert report.used_model


def test_the_rule_carries_the_models_description(memory):
    rules = RuleStore(memory)
    consolidate(planted_log(), rules, llm=StubLLM())
    rule = next(r for r in rules.all() if r["action"] == "pytest")
    assert rule["description"] == "You run the test suite right after committing."


def test_consolidation_runs_without_a_model_and_does_not_crash(memory):
    """Ollama may be down, the user may have pulled no model, and the eval
    harness runs headless."""
    rules = RuleStore(memory)
    report = consolidate(planted_log(), rules, llm=None)

    assert report.promoted
    assert not report.used_model
    rule = next(r for r in rules.all() if r["action"] == "pytest")
    assert "pytest" in rule["description"]
    assert "No model available" in report.summary()


def test_a_model_that_raises_falls_back_rather_than_failing(memory):
    rules = RuleStore(memory)
    report = consolidate(planted_log(), rules, llm=StubLLM(explode=True))
    assert report.promoted
    assert not report.used_model


def test_a_model_emitting_junk_falls_back_to_the_statistical_description(memory):
    rules = RuleStore(memory)
    report = consolidate(planted_log(), rules, llm=StubLLM(response="I'm not sure, honestly"))
    assert report.promoted
    # The planted log contains several equally strong patterns, so this checks
    # the fallback wrote a real sentence for each rather than which came first.
    assert all(r["description"].startswith("You run ") for r in report.promoted)
    assert any("pytest" in r["description"] for r in report.promoted)


def test_the_model_can_veto_a_coincidence(memory):
    rules = RuleStore(memory)
    veto = StubLLM(response=json.dumps(
        {"genuine": False, "name": "n", "description": "Looks like an accident of a short log."}
    ))
    report = consolidate(planted_log(), rules, llm=veto)

    assert report.promoted == []
    assert report.rejected
    assert rules.all() == []


def test_running_twice_refreshes_rather_than_duplicates(memory):
    """Consolidation runs over an overlapping window, so a second /dream must not
    add another copy of the same habit."""
    rules = RuleStore(memory)
    consolidate(planted_log(), rules, llm=StubLLM())
    before = len(rules.all())
    consolidate(planted_log(), rules, llm=StubLLM())
    assert len(rules.all()) == before


def test_an_empty_log_reports_that_plainly(memory):
    report = consolidate([], RuleStore(memory), llm=StubLLM())
    assert "empty" in report.summary().lower()


def test_a_log_with_no_habits_says_so(memory):
    log = [ep(f"unique command {i}") for i in range(20)]
    report = consolidate(log, RuleStore(memory), llm=StubLLM())
    assert "No new habits" in report.summary()


# ── /rules ──────────────────────────────────────────────────────────────────


def test_rules_lists_descriptions_support_and_hit_rates(memory):
    rules = RuleStore(memory)
    consolidate(planted_log(), rules, llm=StubLLM())
    text = render_rules(rules.all())

    assert "You run the test suite right after committing." in text
    assert "seen 10x" in text
    assert "pytest" in text

    # A rule nobody has been shown yet reports the confidence it was mined with
    # and says plainly that it is untested. It used to claim "hit rate 50%",
    # which was not a measurement of anything — no code ever wrote that column.
    assert "recently held 100%" in text
    assert "not yet tested on you" in text
    assert "hit rate" not in text

    # Once it has actually been put in front of the user, the measured number
    # replaces it.
    rule_id = rules.all()[0]["id"]
    rules.record_outcome(rule_id, hit=True)
    assert "hit rate" in render_rules(rules.all())


def test_rules_is_helpful_when_empty():
    text = render_rules([])
    assert "/dream" in text


def test_a_rule_can_be_deleted(memory):
    rules = RuleStore(memory)
    consolidate(planted_log(), rules, llm=StubLLM())
    rule_id = rules.all()[0]["id"]

    assert rules.delete(rule_id) is True
    assert all(r["id"] != rule_id for r in rules.all())
    assert rules.delete(rule_id) is False, "deleting twice must not claim success"


# ── Rules influence predictions ─────────────────────────────────────────────


def test_a_rule_reaches_the_predictor(memory):
    rules = RuleStore(memory)
    consolidate(planted_log(), rules, llm=StubLLM())

    p = Predictor(min_episodes=1, rules=rules)
    p.update(ep("anything"))
    ranked = p.predict("py", ctx(prev="git commit -m wip"))

    assert ranked
    assert ranked[0].action == "pytest"
    assert ranked[0].source == "rule"
    assert ranked[0].why == "You run the test suite right after committing."


def test_deleting_a_rule_removes_its_influence(memory):
    """The acceptance criterion: disagreeing with the system has to actually
    change what it does."""
    rules = RuleStore(memory)
    consolidate(planted_log(), rules, llm=StubLLM())
    p = Predictor(min_episodes=1, rules=rules)
    p.update(ep("anything"))

    assert p.predict("py", ctx(prev="git commit -m wip"))[0].source == "rule"

    for r in rules.all():
        rules.delete(r["id"])
    after = p.predict("py", ctx(prev="git commit -m wip"))
    assert all(pred.source != "rule" for pred in after)


def test_a_rule_does_not_fire_in_the_wrong_situation(memory):
    rules = RuleStore(memory)
    consolidate(planted_log(), rules, llm=StubLLM())

    assert rules.match("py", ctx(prev="git commit -m wip"))
    assert rules.match("py", ctx(prev="something else")) == []
    assert rules.match("py", ctx(cwd="/elsewhere", prev="git commit -m wip")) == []


def test_a_prefix_that_cannot_match_filters_the_rule_out(memory):
    rules = RuleStore(memory)
    consolidate(planted_log(), rules, llm=StubLLM())
    assert rules.match("git", ctx(prev="git commit -m wip")) == []


def test_a_broken_rule_store_does_not_break_prediction():
    class Exploding:
        def match(self, prefix, ctx):
            raise RuntimeError("db gone")

    p = Predictor(min_episodes=1, rules=Exploding())
    p.update(ep("pytest", prev="git commit"))
    assert p.predict("py", ctx(prev="git commit")) is not None


# ── Hit rates and pruning ───────────────────────────────────────────────────


def test_hit_rate_moves_with_outcomes(memory):
    rules = RuleStore(memory)
    rule_id = rules.add({"kind": "sequential", "previous": "x"}, "y", 10, "desc", hit_rate=0.5)

    for _ in range(10):
        rules.record_outcome(rule_id, hit=True)
    assert rules.get(rule_id)["hit_rate"] > 0.8

    for _ in range(20):
        rules.record_outcome(rule_id, hit=False)
    assert rules.get(rule_id)["hit_rate"] < 0.2


def test_a_decayed_rule_is_retired(memory):
    rules = RuleStore(memory)
    rule_id = rules.add({"kind": "sequential", "previous": "x"}, "y", 20, "desc", hit_rate=0.5)
    for _ in range(30):
        rules.record_outcome(rule_id, hit=False)

    retired = prune(rules)
    assert [r["id"] for r in retired] == [rule_id]
    assert rules.all(active_only=True) == []
    assert rules.get(rule_id) is not None, "retired, not deleted — /rules should still explain it"


def test_a_rule_that_has_never_fired_is_not_pruned(memory):
    """It has not had a chance to be wrong yet."""
    rules = RuleStore(memory)
    rules.add({"kind": "sequential", "previous": "x"}, "y", 50, "desc", hit_rate=0.0)
    assert prune(rules) == []


def test_a_retired_rule_stops_influencing_predictions(memory):
    rules = RuleStore(memory)
    rule_id = rules.add({"kind": "sequential", "cwd": "/proj", "previous": "git commit"},
                        "pytest", 20, "desc", hit_rate=0.5)
    assert rules.match("py", ctx(prev="git commit"))

    for _ in range(30):
        rules.record_outcome(rule_id, hit=False)
    prune(rules)
    assert rules.match("py", ctx(prev="git commit")) == []


# ── Calibration refit ───────────────────────────────────────────────────────


def test_consolidation_refits_the_calibrator(memory):
    """The brief puts the refit here, not on every prediction: a curve that moves
    under the user mid-session is worse than one a few hours stale."""
    log = [Episode(action="x", predicted="x", predicted_conf=0.9,
                   accepted_prediction=1 if i < 20 else 0) for i in range(100)]
    calibrator = Calibrator()
    store = CalibrationStore(memory)

    report = consolidate(log, RuleStore(memory), llm=None,
                         calibrator=calibrator, calibration_store=store)

    assert report.calibration_refit
    assert calibrator.is_fitted
    assert calibrator.calibrate(0.9) < 0.5, "0.9 was right 20% of the time"
    assert store.load().is_fitted, "the refit curve must survive a restart"


def test_consolidation_without_a_calibrator_is_fine(memory):
    report = consolidate(planted_log(), RuleStore(memory), llm=None)
    assert not report.calibration_refit


# ── /dream must make the system better, not worse ───────────────────────────
#
# Every test below covers a way the consolidation feature was actively harmful:
# running it removed hints, resurrected abandoned habits, asserted contradictory
# beliefs, or froze the interface. They exist because the suite was fully green
# while all of that was true.


def _habit_log(action, prev, n, ts):
    return [Episode(action=action, keystroke_prefix=action, ts=ts + i,
                    context=ctx(prev=prev)) for i in range(n)]


def test_dream_does_not_downgrade_a_prediction_it_just_certified(memory):
    """The regression that made /dream counterproductive.

    A promoted rule reported `hit_rate`, which nothing ever wrote, so it claimed
    a flat 0.5 — below the default reveal threshold of 0.70. Running /dream on a
    habit the predictor already knew therefore *removed* the ghost hint for it.
    """
    now = time.time()
    log = _habit_log("pytest -q", "git commit", 60, now)
    rules = RuleStore(memory)

    before = Predictor(min_episodes=10)
    for e in log:
        before.update(e)
    top_before = before.predict("pyt", ctx(prev="git commit"))[0]
    assert top_before.confidence >= 0.7

    consolidate(log, rules, llm=None)

    after = Predictor(min_episodes=10, rules=rules)
    for e in log:
        after.update(e)
    top_after = after.predict("pyt", ctx(prev="git commit"))[0]

    assert top_after.action == "pytest -q"
    assert top_after.confidence >= top_before.confidence, (
        "consolidating a habit must not lower the confidence in it"
    )


def test_a_rule_never_claims_a_hit_rate_nobody_measured(memory):
    rules = RuleStore(memory)
    consolidate(planted_log(), rules, llm=None)
    rule = rules.all()[0]

    assert rule["hit_rate"] is None, "untested rules have no measured accuracy"
    assert rule["confidence"] == pytest.approx(1.0)
    assert rule["fired"] == 0


def test_the_same_habit_does_not_fill_every_hint_slot(memory):
    """Sequential and situational mining routinely agree, and both rules used to
    be returned, so two of the three suggestions were the identical command."""
    rules = RuleStore(memory)
    log = _habit_log("pytest -q", "git commit", 60, time.time())
    consolidate(log, rules, llm=None)

    p = Predictor(min_episodes=10, rules=rules)
    for e in log:
        p.update(e)
    actions = [pred.action for pred in p.predict("pyt", ctx(prev="git commit"))]
    assert len(actions) == len(set(actions)), actions


def test_an_abandoned_habit_does_not_outvote_the_current_one(memory):
    """Mining counted the whole window flat while the predictor decays by
    recency, so a workflow dropped months ago was promoted and then masked the
    correct answer permanently."""
    now = time.time()
    log = _habit_log("npm run build", "git pull", 200, now - 120 * 86400)
    log += _habit_log("npm run dev", "git pull", 20, now - 2 * 86400)

    rules = RuleStore(memory)
    consolidate(log, rules, llm=None)

    promoted = {r["action"] for r in rules.all(active_only=True)}
    assert promoted == {"npm run dev"}, promoted

    p = Predictor(min_episodes=10, rules=rules)
    for e in log:
        p.update(e)
    assert p.predict("npm", ctx(prev="git pull"))[0].action == "npm run dev"


def test_a_belief_the_log_no_longer_supports_is_retired(memory):
    now = time.time()
    rules = RuleStore(memory)
    consolidate(_habit_log("npm run build", "git pull", 60, now - 200 * 86400),
                rules, llm=None)
    assert {r["action"] for r in rules.all(active_only=True)} == {"npm run build"}

    consolidate(_habit_log("npm run dev", "git pull", 60, now), rules, llm=None)

    assert {r["action"] for r in rules.all(active_only=True)} == {"npm run dev"}
    retired = [r for r in rules.all(active_only=False) if not r["active"]]
    assert retired, "superseded beliefs are deactivated, not deleted"
    assert "(retired)" in render_rules(rules.all(active_only=False))


def test_an_even_split_does_not_promote_two_contradictory_rules(memory):
    """`support / total >= 0.5` let a coin flip qualify, so /rules asserted both
    halves of the same situation as things the system believed."""
    now = time.time()
    log = []
    for i in range(12):
        log.append(Episode(action="pytest -q", keystroke_prefix="pytest -q",
                           ts=now + i * 2, context=ctx()))
        log.append(Episode(action="git commit", keystroke_prefix="git commit",
                           ts=now + i * 2 + 1, context=ctx()))

    rules = RuleStore(memory)
    consolidate(log, rules, llm=None)

    situational = [r for r in rules.all() if r["pattern"].get("kind") == "situational"]
    assert len(situational) <= 1, [r["description"] for r in situational]


def test_an_unreachable_model_is_only_tried_once(memory):
    """Every candidate got its own doomed request, so a dead Ollama cost one
    connection timeout per pattern and froze the interface for the sum of them."""
    llm = StubLLM(explode=True)
    log = []
    now = time.time()
    for name in ("alpha", "beta", "gamma", "delta", "epsilon"):
        log += _habit_log(f"run {name}", f"before {name}", 8, now)

    report = consolidate(log, RuleStore(memory), llm=llm)

    assert report.candidates > 1, "need several candidates for this to mean anything"
    assert len(llm.calls) == 1, f"one failure should end model use, got {len(llm.calls)}"
    assert not report.used_model
    assert report.promoted, "the statistical fallback still produces rules"


def test_a_rule_is_measured_once_it_has_been_shown(memory):
    """`record_outcome` had no caller outside the tests, so `hit_rate` stayed at
    its default forever and `prune` could never retire anything."""
    rules = RuleStore(memory)
    consolidate(planted_log(), rules, llm=None)
    rule_id = rules.all()[0]["id"]

    for _ in range(12):
        rules.record_outcome(rule_id, hit=False)

    rule = rules.get(rule_id)
    assert rule["fired"] == 12
    assert rule["hit_rate"] < 0.25
    assert rule["last_fired_ts"] is not None
    assert [r["id"] for r in prune(rules)] == [rule_id]
