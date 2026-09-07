"""The evaluation harness.

    python -m eval.run                 # synthetic log, baseline vs learned
    python -m eval.run --db data/intuition.db   # your own recorded episodes
    python -m eval.run --json          # machine-readable, for CI

Before this existed there was one test and no measurement of the thing the
project claims to do, which meant no phase could be shown to have helped. The
baseline column is the important one: it is the original prefix heuristics
replayed over the same log, so the comparison answers whether replacing them was
worth the code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from core.calibration import Calibrator, load_thresholds
from core.consolidation import RuleStore, consolidate
from core.predictor import Predictor
from eval import synthetic
from eval.metrics import compare, render_table, replay


class HeuristicBaseline:
    """The original four-branch if-chain, wrapped in the Predictor interface.

    It learns nothing, which is the point — it is what the system did before
    Phase 4, and everything is measured against it.
    """

    def __init__(self):
        self.seen = 0

    def predict(self, prefix, ctx=None, k=3):
        from core.predictor import heuristic_predictions
        return heuristic_predictions(prefix)[:k]

    def update(self, episode):
        self.seen += 1


def _tool_loop_success() -> dict:
    """Run the real Brain against scripted model output and report the rate.

    Measured rather than inferred: eval/plans.py drives the actual loop through
    the failure modes a local model produces, in a throwaway directory so the
    write and the path-jail plans have something real to act on.
    """
    import shutil
    import tempfile
    from pathlib import Path

    from core.actions import actions, set_memory
    from core.memory import Memory
    from eval import plans

    cwd = Path.cwd()
    tmp = tempfile.mkdtemp(prefix="intuition-eval-")
    memory = None
    try:
        os.chdir(tmp)
        memory = Memory(str(Path(tmp) / "eval.db"))
        set_memory(memory)
        return plans.measure(memory, actions, tmp)
    finally:
        os.chdir(cwd)
        if memory is not None:
            memory.close()
        shutil.rmtree(tmp, ignore_errors=True)


def load_episodes(db_path: str) -> list:
    from core.episodes import EpisodeLog
    from core.memory import Memory

    log = EpisodeLog(Memory(db_path))
    return log.all()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate IntuitionOS prediction.")
    parser.add_argument("--db", help="evaluate a real episode log instead of a synthetic one")
    parser.add_argument("--n", type=int, default=800, help="synthetic episodes to generate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise", type=float, default=0.25,
                        help="probability a planted habit is not followed")
    parser.add_argument("--train", type=float, default=0.7, help="chronological train fraction")
    parser.add_argument("--min-episodes", type=int, default=20,
                        help="episodes before the learned predictor stops falling back")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--consolidate", action="store_true",
                        help="also run consolidation and report the rules found")
    args = parser.parse_args(argv)

    if args.db:
        episodes = load_episodes(args.db)
        source = f"{args.db} ({len(episodes)} episodes)"
        if len(episodes) < 20:
            print(f"Only {len(episodes)} episodes in {args.db}. Use the system for a "
                  f"while, or run without --db for the synthetic log.", file=sys.stderr)
            return 1
    else:
        episodes = synthetic.generate(n=args.n, seed=args.seed, noise=args.noise)
        source = (f"synthetic ({len(episodes)} episodes, seed {args.seed}, "
                  f"noise {args.noise:.0%})")

    train, test = synthetic.split(episodes, train_fraction=args.train)
    thresholds = load_thresholds(None)

    baseline = replay(HeuristicBaseline(), test, thresholds, "baseline", warm_with=train)
    learned = replay(
        Predictor(min_episodes=args.min_episodes), test, thresholds, "learned", warm_with=train
    )

    # And the same predictor with its confidence calibrated on the training half,
    # to show whether calibration bought anything beyond honesty.
    calibrated_predictor = Predictor(min_episodes=args.min_episodes)
    warm = replay(Predictor(min_episodes=args.min_episodes), train, thresholds, "warm")
    calibrator = Calibrator().fit(warm.scored)
    calibrated_predictor.calibrator = calibrator
    calibrated = replay(calibrated_predictor, test, thresholds, "calibrated", warm_with=train)

    results = [baseline, learned, calibrated]
    loop = _tool_loop_success()

    rules_found = None
    if args.consolidate:
        from core.memory import Memory

        store = RuleStore(Memory(":memory:"))
        report = consolidate(train, store, llm=None)
        rules_found = [r["description"] for r in store.all()]

    if args.json:
        payload = {
            "source": source,
            "train": len(train),
            "test": len(test),
            "results": [r.as_dict() for r in results],
            "tool_loop": loop,
            "calibrator_fitted": calibrator.is_fitted,
        }
        if rules_found is not None:
            payload["rules"] = rules_found
        print(json.dumps(payload, indent=2))
        return 0

    print()
    print("IntuitionOS evaluation")
    print(f"Source: {source}")
    print(f"Train: {len(train)} episodes   Test: {len(test)} episodes (chronological split)")
    print()
    print(render_table(results))
    print()
    print(compare(baseline, learned))
    if calibrated.ece < learned.ece - 1e-9:
        print(f"Calibration reduced expected calibration error "
              f"{learned.ece:.3f} -> {calibrated.ece:.3f}.")
    elif calibrator.is_fitted:
        print(f"Calibration did not reduce ECE on this split "
              f"({learned.ece:.3f} -> {calibrated.ece:.3f}).")
    else:
        print("Calibration was not fitted: too few revealed predictions in the training half.")

    print(f"Tool-loop success rate: {loop['rate']:.1%} "
          f"({loop['succeeded']}/{loop['plans']} scripted model plans behaved as intended, "
          f"{loop['parse_recoveries']} needed a parse recovery)")
    for failure in loop["failures"]:
        print(f"  FAILED  {failure}")

    print()
    print("Latency saved is an estimate, not a measurement: the replay does not "
          "execute the prewarmed actions.")

    if rules_found is not None:
        print()
        print(f"Consolidation found {len(rules_found)} rule(s):")
        for description in rules_found[:10]:
            print(f"  - {description}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
