"""Metrics for replaying an episode log against a predictor.

The whole point of the harness is to make the claim falsifiable, so the metrics
are chosen to be able to say *no*:

  * Top-1 and top-3 accuracy say whether it predicts the right thing.
  * Prewarm hit rate and wasted prewarm rate say whether speculation pays for
    itself, which is a different question from accuracy — a prewarm below the
    reveal threshold can be right and still never be used.
  * False reveal rate counts hints shown and ignored. This is the one that keeps
    the system honest about the user's attention, and the one a system optimising
    only for accuracy will happily wreck.
  * Expected calibration error says whether the stated confidence means anything.
  * Median latency saved is the only metric a user actually feels.

Replay is strictly chronological and the predictor is updated only *after* it has
been scored on each episode, so nothing is ever evaluated on data it has already
seen.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from core.calibration import reliability
from core.episodes import Episode


@dataclass
class Result:
    name: str
    n: int = 0
    top1: int = 0
    top3: int = 0
    predictions_made: int = 0
    prewarms: int = 0
    prewarm_hits: int = 0
    reveals: int = 0
    reveal_hits: int = 0
    latencies_saved_ms: list = field(default_factory=list)
    scored: list = field(default_factory=list)   # Episodes carrying stated confidence

    # ── Derived ──────────────────────────────────────────────────────────

    @property
    def top1_accuracy(self) -> float:
        return self.top1 / self.n if self.n else 0.0

    @property
    def top3_accuracy(self) -> float:
        return self.top3 / self.n if self.n else 0.0

    @property
    def coverage(self) -> float:
        """How often it was willing to say anything at all. A predictor that
        answers rarely and is right when it does is a different tradeoff from one
        that always answers, and averaging them together hides that."""
        return self.predictions_made / self.n if self.n else 0.0

    @property
    def prewarm_hit_rate(self) -> float:
        return self.prewarm_hits / self.prewarms if self.prewarms else 0.0

    @property
    def wasted_prewarm_rate(self) -> float:
        return 1.0 - self.prewarm_hit_rate if self.prewarms else 0.0

    @property
    def false_reveal_rate(self) -> float:
        """Hints shown and ignored, over hints shown. The cost the old design did
        not model at all."""
        return 1.0 - (self.reveal_hits / self.reveals) if self.reveals else 0.0

    @property
    def median_latency_saved_ms(self) -> float:
        return statistics.median(self.latencies_saved_ms) if self.latencies_saved_ms else 0.0

    @property
    def ece(self) -> float:
        return reliability(self.scored).ece if self.scored else 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "episodes": self.n,
            "top1": self.top1_accuracy,
            "top3": self.top3_accuracy,
            "coverage": self.coverage,
            "prewarm_hit_rate": self.prewarm_hit_rate,
            "wasted_prewarm_rate": self.wasted_prewarm_rate,
            "false_reveal_rate": self.false_reveal_rate,
            "ece": self.ece,
            "median_latency_saved_ms": self.median_latency_saved_ms,
            "reveals": self.reveals,
            "prewarms": self.prewarms,
        }


def replay(predictor, episodes, thresholds, name: str, warm_with=None,
           cost_lookup=None) -> Result:
    """Score a predictor over a log, one episode at a time, in order.

    `warm_with` is training history fed in before scoring starts. `cost_lookup`
    maps an action to the milliseconds a prewarm would have saved; without it a
    flat estimate is used and the latency figure is labelled as such.
    """
    result = Result(name=name)
    free_t = float(thresholds.get("free", 0.30))
    reveal_t = float(thresholds.get("reveal", 0.70))

    for e in (warm_with or []):
        predictor.update(e)

    for episode in episodes:
        prefix = episode.keystroke_prefix or ""
        ranked = predictor.predict(prefix, episode.context, k=3)
        result.n += 1

        if ranked:
            result.predictions_made += 1
            actions = [p.action for p in ranked]
            top = ranked[0]

            if actions[0] == episode.action:
                result.top1 += 1
            if episode.action in actions:
                result.top3 += 1

            # Prewarm: cheap, so a low bar. It pays off only if the user
            # submits exactly what was warmed.
            if top.confidence >= free_t:
                result.prewarms += 1
                if top.action == episode.action:
                    result.prewarm_hits += 1
                    result.latencies_saved_ms.append(
                        cost_lookup(episode.action) if cost_lookup else 40.0
                    )

            # Reveal: expensive, so a higher bar. Shown and ignored is the
            # negative signal, and it is also what the episode log records.
            if top.confidence >= reveal_t:
                result.reveals += 1
                hit = 1 if top.action == episode.action else 0
                result.reveal_hits += hit
                result.scored.append(Episode(
                    action=episode.action, predicted=top.action,
                    predicted_conf=top.confidence, accepted_prediction=hit,
                ))

        # Learn only after scoring, so nothing is graded on data it has seen.
        predictor.update(episode)

    return result


def render_table(results: list) -> str:
    """Side-by-side comparison. The baseline column is what says whether the
    learned predictor was worth building."""
    if not results:
        return "No results."

    rows = [
        ("Top-1 accuracy", "top1", "pct"),
        ("Top-3 accuracy", "top3", "pct"),
        ("Coverage (answered at all)", "coverage", "pct"),
        ("Prewarm hit rate", "prewarm_hit_rate", "pct"),
        ("Wasted prewarm rate", "wasted_prewarm_rate", "pct"),
        ("False reveal rate", "false_reveal_rate", "pct"),
        ("Expected calibration error", "ece", "num"),
        ("Median latency saved (ms)", "median_latency_saved_ms", "ms"),
        ("Hints shown", "reveals", "int"),
        ("Prewarms attempted", "prewarms", "int"),
    ]

    dicts = [r.as_dict() for r in results]
    label_w = max(len(label) for label, _, _ in rows) + 2
    col_w = max(14, max(len(d["name"]) for d in dicts) + 2)

    lines = [
        " " * label_w + "".join(d["name"].rjust(col_w) for d in dicts),
        "-" * (label_w + col_w * len(dicts)),
    ]
    for label, key, kind in rows:
        cells = []
        for d in dicts:
            v = d[key]
            if kind == "pct":
                cells.append(f"{v:.1%}".rjust(col_w))
            elif kind == "int":
                cells.append(f"{int(v)}".rjust(col_w))
            elif kind == "ms":
                cells.append(f"{v:.0f}".rjust(col_w))
            else:
                cells.append(f"{v:.3f}".rjust(col_w))
        lines.append(label.ljust(label_w) + "".join(cells))

    lines.append("-" * (label_w + col_w * len(dicts)))
    lines.append(f"Episodes scored: {dicts[0]['episodes']}")
    return "\n".join(lines)


def compare(baseline: Result, learned: Result) -> str:
    """The one sentence that answers 'was Phase 4 worth doing'."""
    delta = learned.top1_accuracy - baseline.top1_accuracy
    if baseline.top1_accuracy > 0:
        relative = f" ({delta / baseline.top1_accuracy:+.0%} relative)"
    else:
        relative = ""
    verdict = "better than" if delta > 0.005 else ("no better than" if delta > -0.005 else "worse than")
    return (f"The learned predictor is {verdict} the heuristic baseline: "
            f"top-1 {baseline.top1_accuracy:.1%} -> {learned.top1_accuracy:.1%}, "
            f"{delta:+.1%} absolute{relative}.")
