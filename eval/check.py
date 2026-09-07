"""CI gate: fail the build if the system got worse at the thing it claims to do.

    python -m eval.check

The point of building a harness is to be able to say no. Without something like
this, a change that made prediction worse than the pre-Phase-4 if-chain would
pass CI silently and the README would go on quoting a number that no longer held.

Thresholds are deliberately loose. This is a regression gate, not a target: it
should fire when something is broken, not when a run lands a couple of points
below its best.
"""

from __future__ import annotations

import sys

from core.calibration import Calibrator, load_thresholds
from core.predictor import Predictor
from eval import synthetic
from eval.metrics import replay
from eval.run import HeuristicBaseline, _tool_loop_success

# The learned predictor must beat the heuristic baseline by at least this much
# absolute top-1 accuracy on the planted log. Measured margins are far larger;
# this catches a break, not a wobble.
MIN_MARGIN = 0.10

# Every scripted plan must behave as intended. Anything less means a failure mode
# a local model actually exhibits is no longer handled.
MIN_TOOL_LOOP = 1.0


def run(seed: int = 7, n: int = 800) -> list:
    """Returns a list of failure descriptions; empty means healthy."""
    episodes = synthetic.generate(n=n, seed=seed, noise=0.25)
    train, test = synthetic.split(episodes)
    thresholds = load_thresholds(None)

    baseline = replay(HeuristicBaseline(), test, thresholds, "baseline", warm_with=train)
    learned = replay(Predictor(min_episodes=20), test, thresholds, "learned", warm_with=train)
    loop = _tool_loop_success()

    print(f"baseline top-1        {baseline.top1_accuracy:.1%}")
    print(f"learned  top-1        {learned.top1_accuracy:.1%}")
    print(f"learned  top-3        {learned.top3_accuracy:.1%}")
    print(f"false reveal rate     {learned.false_reveal_rate:.1%}")
    print(f"expected cal. error   {learned.ece:.3f}")
    print(f"tool-loop success     {loop['rate']:.1%} ({loop['succeeded']}/{loop['plans']})")

    failures = []
    margin = learned.top1_accuracy - baseline.top1_accuracy
    if margin < MIN_MARGIN:
        failures.append(
            f"the learned predictor beats the baseline by only {margin:+.1%} "
            f"(needs {MIN_MARGIN:.0%}); Phase 4 has regressed"
        )
    if loop["rate"] < MIN_TOOL_LOOP:
        failures.append(f"tool-loop success fell to {loop['rate']:.1%}")
        failures.extend(f"  {f}" for f in loop["failures"])
    return failures


def main(argv=None) -> int:
    failures = run()
    if not failures:
        print("\nOK: prediction and the tool loop are both healthy.")
        return 0
    print()
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
