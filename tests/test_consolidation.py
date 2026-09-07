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
    assert "hit rate" in text
    assert "pytest" in text


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
