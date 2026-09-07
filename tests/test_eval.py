"""The harness has to be trustworthy before its numbers are worth quoting, so the
tests here are mostly about it being unable to flatter itself."""

import json

import pytest

from core.calibration import load_thresholds
from core.predictor import Predictor
from eval import synthetic
from eval.metrics import Result, compare, render_table, replay
from eval.run import HeuristicBaseline, main


# ── The synthetic log ───────────────────────────────────────────────────────


def test_generation_is_deterministic():
    a = synthetic.generate(n=100, seed=3)
    b = synthetic.generate(n=100, seed=3)
    assert [e.action for e in a] == [e.action for e in b]


def test_a_different_seed_gives_a_different_log():
    a = synthetic.generate(n=100, seed=1)
    b = synthetic.generate(n=100, seed=2)
    assert [e.action for e in a] != [e.action for e in b]


def test_the_planted_habits_are_actually_present():
    episodes = synthetic.generate(n=1000, seed=5, noise=0.25)
    pairs = [(episodes[i - 1].action, episodes[i].action) for i in range(1, len(episodes))]

    # Every planted chain must actually occur, and must usually be followed.
    for trigger, follows in synthetic.SEQUENCES:
        occasions = [nxt for prev, nxt in pairs if prev == trigger]
        assert len(occasions) >= 10, f"{trigger!r} barely occurs in the log"
        rate = occasions.count(follows) / len(occasions)
        assert rate > 0.5, f"{trigger!r} -> {follows!r} held only {rate:.0%} of the time"


def test_noise_keeps_the_ceiling_below_perfect():
    """A log where every habit holds every time makes any predictor look
    flawless, which would make the metrics meaningless."""
    episodes = synthetic.generate(n=600, seed=5, noise=0.4)
    pairs = [(episodes[i - 1].action, episodes[i].action) for i in range(1, len(episodes))]
    triggered = [nxt for prev, nxt in pairs if prev == "git commit -m wip"]
    assert len(set(triggered)) > 1, "the planted habit was never once interrupted"


def test_episodes_are_chronological():
    episodes = synthetic.generate(n=200, seed=4)
    timestamps = [e.ts for e in episodes]
    assert timestamps == sorted(timestamps)


def test_the_split_is_chronological_not_random():
    """A random split lets the model learn from the user's future, which
    flatters every metric."""
    episodes = synthetic.generate(n=100, seed=4)
    train, test = synthetic.split(episodes, 0.7)
    assert len(train) == 70
    assert max(e.ts for e in train) <= min(e.ts for e in test)


# ── Replay ──────────────────────────────────────────────────────────────────


def test_the_predictor_is_never_scored_on_data_it_has_seen(monkeypatch):
    """update() must happen after predict(), or every number is inflated."""
    order = []

    class Spy(Predictor):
        def predict(self, prefix, ctx=None, k=3):
            order.append("predict")
            return super().predict(prefix, ctx, k)

        def update(self, episode):
            order.append("update")
            return super().update(episode)

    episodes = synthetic.generate(n=6, seed=1)
    replay(Spy(min_episodes=1), episodes, load_thresholds(None), "spy")

    assert order[:2] == ["predict", "update"]
    assert order == ["predict", "update"] * 6


def test_the_learned_predictor_beats_the_baseline_on_the_planted_log():
    """The headline claim. If this fails, Phase 4 did not earn its place."""
    episodes = synthetic.generate(n=800, seed=7, noise=0.25)
    train, test = synthetic.split(episodes)
    thresholds = load_thresholds(None)

    baseline = replay(HeuristicBaseline(), test, thresholds, "baseline", warm_with=train)
    learned = replay(Predictor(min_episodes=20), test, thresholds, "learned", warm_with=train)

    assert learned.top1_accuracy > baseline.top1_accuracy + 0.10, (
        f"learned {learned.top1_accuracy:.1%} vs baseline {baseline.top1_accuracy:.1%}"
    )


def test_the_baseline_really_does_not_learn():
    """It is the pre-Phase-4 behaviour, so if it improved with data the
    comparison would be measuring the wrong thing."""
    episodes = synthetic.generate(n=400, seed=2)
    train, test = synthetic.split(episodes)
    thresholds = load_thresholds(None)

    cold = replay(HeuristicBaseline(), test, thresholds, "cold")
    warm = replay(HeuristicBaseline(), test, thresholds, "warm", warm_with=train)
    assert cold.top1_accuracy == pytest.approx(warm.top1_accuracy)


def test_an_empty_log_produces_zeroes_not_a_crash():
    r = replay(Predictor(), [], load_thresholds(None), "empty")
    assert r.n == 0
    assert r.top1_accuracy == 0.0
    assert r.prewarm_hit_rate == 0.0
    assert r.false_reveal_rate == 0.0
    assert r.median_latency_saved_ms == 0.0


# ── Metric definitions ──────────────────────────────────────────────────────


def test_wasted_prewarm_is_the_complement_of_the_hit_rate():
    r = Result(name="x", n=10, prewarms=10, prewarm_hits=3)
    assert r.prewarm_hit_rate == pytest.approx(0.3)
    assert r.wasted_prewarm_rate == pytest.approx(0.7)


def test_false_reveal_rate_counts_hints_shown_and_ignored():
    r = Result(name="x", n=10, reveals=8, reveal_hits=6)
    assert r.false_reveal_rate == pytest.approx(0.25)


def test_a_predictor_that_never_reveals_has_no_false_reveals():
    """Not 100%. Showing nothing wastes no attention."""
    r = Result(name="x", n=10, reveals=0, reveal_hits=0)
    assert r.false_reveal_rate == 0.0


def test_coverage_separates_being_quiet_from_being_wrong():
    r = Result(name="x", n=10, predictions_made=4, top1=4)
    assert r.coverage == pytest.approx(0.4)
    assert r.top1_accuracy == pytest.approx(0.4)


def test_the_table_renders_every_metric_side_by_side():
    episodes = synthetic.generate(n=120, seed=3)
    train, test = synthetic.split(episodes)
    thresholds = load_thresholds(None)
    results = [
        replay(HeuristicBaseline(), test, thresholds, "baseline", warm_with=train),
        replay(Predictor(min_episodes=5), test, thresholds, "learned", warm_with=train),
    ]
    table = render_table(results)

    for label in ("Top-1 accuracy", "Top-3 accuracy", "Prewarm hit rate",
                  "Wasted prewarm rate", "False reveal rate",
                  "Expected calibration error", "Median latency saved"):
        assert label in table
    assert "baseline" in table and "learned" in table


def test_compare_reports_the_direction_honestly():
    better = compare(Result("b", n=10, top1=2), Result("l", n=10, top1=8))
    worse = compare(Result("b", n=10, top1=8), Result("l", n=10, top1=2))
    same = compare(Result("b", n=10, top1=5), Result("l", n=10, top1=5))

    assert "better than" in better
    assert "worse than" in worse
    assert "no better than" in same


# ── The tool loop ───────────────────────────────────────────────────────────


def test_every_scripted_plan_behaves_as_intended(memory, project):
    """Measured by running the real loop, not inferred from the episode log."""
    from core.actions import actions
    from eval import plans

    result = plans.measure(memory, actions, str(project))
    assert result["failures"] == [], result["failures"]
    assert result["rate"] == 1.0


def test_the_irreversible_plan_is_parked_rather_than_run(memory, project):
    from core.actions import actions
    from eval import plans

    only = [p for p in plans.standard_plans() if p.expect == "parked"]
    assert only, "the plan set no longer covers an irreversible action"
    assert plans.measure(memory, actions, str(project), plans=only)["rate"] == 1.0


# ── The command line ────────────────────────────────────────────────────────


def test_it_runs_on_a_fresh_clone(capsys):
    """The acceptance criterion: python -m eval.run prints a metrics table."""
    assert main(["--n", "300", "--min-episodes", "10"]) == 0
    out = capsys.readouterr().out
    assert "Top-1 accuracy" in out
    assert "baseline" in out and "learned" in out
    assert "Tool-loop success rate" in out


def test_json_output_is_machine_readable(capsys):
    assert main(["--n", "200", "--min-episodes", "10", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert {r["name"] for r in payload["results"]} == {"baseline", "learned", "calibrated"}
    assert payload["tool_loop"]["rate"] == 1.0
    for r in payload["results"]:
        assert 0.0 <= r["top1"] <= 1.0


def test_consolidation_can_be_reported_too(capsys):
    assert main(["--n", "300", "--min-episodes", "10", "--consolidate"]) == 0
    assert "Consolidation found" in capsys.readouterr().out


def test_an_empty_real_database_is_refused_with_a_message(tmp_path, capsys):
    from core.memory import Memory

    db = tmp_path / "empty.db"
    Memory(str(db)).close()

    assert main(["--db", str(db)]) == 1
    assert "Use the system for a while" in capsys.readouterr().err


def test_the_ci_gate_passes_on_the_current_code(capsys):
    """eval/check.py is what fails the build if prediction regresses, so it has
    to be green here or it is not guarding anything."""
    from eval.check import run

    assert run(n=600) == []


def test_the_ci_gate_would_fail_a_regressed_predictor(monkeypatch, capsys):
    """And it has to be able to say no. Swap the learned predictor for one that
    learns nothing and the gate must fire."""
    import eval.check as check

    monkeypatch.setattr(check, "Predictor", lambda **kw: check.HeuristicBaseline())
    failures = check.run(n=400)

    assert failures
    assert "regressed" in failures[0]
