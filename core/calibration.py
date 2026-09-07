"""Calibration: making a stated confidence mean what it says.

For a system whose thesis is that hunches vary in strength, the strength was
doing no work. One threshold governed everything, and the numbers it compared
against were arbitrary. This module measures whether they are true and then fixes
them.

The measurement is a reliability curve. Bin predictions by the confidence they
stated, and ask what fraction in each bin actually turned out right. A perfectly
calibrated system puts 80% of its 0.8 predictions on target. If it says 0.8 and
is right 40% of the time, that has to be visible rather than inferred — hence
`/calibration`, which prints the table.

The fix is isotonic regression: fit a monotonic map from stated confidence to
observed frequency. Monotonic matters — more confident must never come out less
probable — and isotonic is the right tool because it assumes only the ordering,
not a shape. Platt scaling would impose a sigmoid this data has no reason to have.

Only episodes where a prediction was actually *shown* carry signal. A hint the
user never saw says nothing about whether they would have taken it, and counting
those as failures would drag every bin toward zero.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

DEFAULT_BINS = 10
# Below this, a fitted curve is fitting noise, and the identity map is honest.
MIN_SAMPLES_TO_FIT = 30


@dataclass
class Bin:
    low: float
    high: float
    count: int
    hits: int

    @property
    def observed(self) -> float:
        return self.hits / self.count if self.count else 0.0

    @property
    def stated(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def gap(self) -> float:
        return self.observed - self.stated


@dataclass
class Reliability:
    bins: list
    n: int
    ece: float          # expected calibration error
    mce: float          # maximum calibration error, the worst single bin
    brier: Optional[float] = None

    def table(self) -> str:
        """The human-readable form, printed by /calibration."""
        if not self.n:
            return ("No predictions have been shown to you yet, so there is nothing "
                    "to calibrate against.")
        lines = [
            f"Reliability over {self.n} shown prediction(s)",
            "",
            "  stated      shown   taken   observed    gap",
            "  " + "-" * 44,
        ]
        for b in self.bins:
            if not b.count:
                continue
            lines.append(
                f"  {b.low:.1f}-{b.high:.1f}   {b.count:6d}  {b.hits:6d}   "
                f"{b.observed:8.2f}  {b.gap:+6.2f}"
            )
        lines += [
            "  " + "-" * 44,
            f"  Expected calibration error (ECE): {self.ece:.3f}",
            f"  Maximum calibration error (MCE):  {self.mce:.3f}",
        ]
        if self.brier is not None:
            lines.append(f"  Brier score:                      {self.brier:.3f}")
        lines += ["", _verdict(self.ece)]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n": self.n, "ece": self.ece, "mce": self.mce, "brier": self.brier,
            "bins": [{"low": b.low, "high": b.high, "count": b.count,
                      "hits": b.hits, "observed": b.observed} for b in self.bins],
        }


def _verdict(ece: float) -> str:
    if ece < 0.05:
        return "  Well calibrated: stated confidence tracks what actually happens."
    if ece < 0.15:
        return "  Roughly calibrated. Usable, with some drift."
    return ("  Poorly calibrated: stated confidence does not match reality. The "
            "thresholds in config.yaml are being applied to numbers that do not "
            "mean what they say.")


def observations(episodes) -> list:
    """(stated_confidence, was_right) for episodes where a hint was shown.

    `accepted_prediction` is NULL when nothing was shown, and those rows are
    dropped here rather than counted as misses.
    """
    out = []
    for e in episodes:
        accepted = getattr(e, "accepted_prediction", None)
        if accepted is None:
            continue
        conf = getattr(e, "predicted_conf", None)
        if conf is None:
            continue
        out.append((float(conf), 1 if accepted else 0))
    return out


def reliability(episodes, bins: int = DEFAULT_BINS) -> Reliability:
    """Bin by stated confidence and compare against what actually happened."""
    obs = observations(episodes)
    edges = [i / bins for i in range(bins + 1)]
    buckets = [Bin(edges[i], edges[i + 1], 0, 0) for i in range(bins)]

    for conf, hit in obs:
        idx = min(int(max(0.0, min(1.0, conf)) * bins), bins - 1)
        buckets[idx].count += 1
        buckets[idx].hits += hit

    n = len(obs)
    if not n:
        return Reliability(bins=buckets, n=0, ece=0.0, mce=0.0)

    # ECE weights each bin by how often it is used; MCE reports the worst bin
    # regardless of how rare it is, which is what catches a confidently wrong
    # corner that the average would hide.
    ece = sum(b.count / n * abs(b.gap) for b in buckets if b.count)
    mce = max((abs(b.gap) for b in buckets if b.count), default=0.0)
    brier = sum((conf - hit) ** 2 for conf, hit in obs) / n
    return Reliability(bins=buckets, n=n, ece=ece, mce=mce, brier=brier)


# ── Isotonic regression ──────────────────────────────────────────────────────


class Calibrator:
    """Maps a stated confidence onto an observed frequency, monotonically.

    Until it has been fitted on enough data it is the identity, so an
    uncalibrated system states exactly what its predictor computed rather than
    something a three-sample fit invented.
    """

    def __init__(self, xs: Optional[list] = None, ys: Optional[list] = None,
                 n: int = 0, fitted_ts: Optional[float] = None):
        self.xs = list(xs or [])
        self.ys = list(ys or [])
        self.n = n
        self.fitted_ts = fitted_ts

    @property
    def is_fitted(self) -> bool:
        return len(self.xs) >= 2

    def calibrate(self, confidence: float) -> float:
        conf = max(0.0, min(1.0, float(confidence)))
        if not self.is_fitted:
            return conf
        return _interpolate(self.xs, self.ys, conf)

    def fit(self, episodes, min_samples: int = MIN_SAMPLES_TO_FIT) -> "Calibrator":
        obs = observations(episodes)
        if len(obs) < min_samples:
            # Not enough to fit; stay the identity rather than fitting noise.
            self.xs, self.ys, self.n = [], [], len(obs)
            return self
        obs.sort(key=lambda t: t[0])
        xs = [c for c, _ in obs]
        ys = _pool_adjacent_violators([h for _, h in obs])
        self.xs, self.ys = _anchor(*_dedupe(xs, ys))
        self.n = len(obs)
        self.fitted_ts = time.time()
        return self

    def to_dict(self) -> dict:
        return {"xs": self.xs, "ys": self.ys, "n": self.n, "fitted_ts": self.fitted_ts}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        d = d or {}
        return cls(xs=d.get("xs"), ys=d.get("ys"), n=int(d.get("n", 0)),
                   fitted_ts=d.get("fitted_ts"))


def _pool_adjacent_violators(ys: list) -> list:
    """Pool Adjacent Violators: the standard isotonic fit.

    Walk left to right; wherever the running mean would decrease, merge the
    offending blocks and use their common mean. What comes out is the closest
    non-decreasing sequence in least squares.
    """
    # Each block is [sum, count].
    blocks: list = []
    for y in ys:
        blocks.append([float(y), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out = []
    for total, count in blocks:
        out.extend([total / count] * count)
    return out


def _dedupe(xs: list, ys: list) -> tuple:
    """Collapse repeated x values, keeping the last fitted y for each."""
    out_x: list = []
    out_y: list = []
    for x, y in zip(xs, ys):
        if out_x and abs(x - out_x[-1]) < 1e-12:
            out_y[-1] = y
        else:
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def _anchor(xs: list, ys: list) -> tuple:
    """Pin the ends of the curve so it covers [0, 1] and can be interpolated.

    Two problems this solves. A predictor that has only ever stated one
    confidence — which is exactly what a heuristic fallback does — collapses to a
    single point after deduping, and a one-point curve cannot be interpolated at
    all. And beyond the range actually observed, some assumption has to be made.

    Both anchors take the conservative side. At the bottom, confidence 0 maps to
    0: something never predicted was never right. At the top the last observed
    rate is extended flat rather than reaching for 1.0, because claiming
    certainty at a confidence never tested is the exact failure calibration is
    supposed to catch.
    """
    if not xs:
        return xs, ys
    xs, ys = list(xs), list(ys)
    if xs[0] > 0.0:
        xs.insert(0, 0.0)
        ys.insert(0, 0.0)
    if xs[-1] < 1.0:
        xs.append(1.0)
        ys.append(ys[-1])
    return xs, ys


def _interpolate(xs: list, ys: list, x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0:
        return ys[hi]
    t = (x - xs[lo]) / span
    return ys[lo] + t * (ys[hi] - ys[lo])


# ── Cost-gated thresholds ────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "free": 0.30,          # prewarm; being wrong costs cycles only
    "reveal": 0.70,        # show a ghost hint; being wrong costs user attention
    "auto_execute": 0.95,  # act without asking; reversible capabilities only
    "irreversible": None,  # never auto-execute, at any confidence
}


def load_thresholds(cfg: Optional[dict]) -> dict:
    """Merge configured thresholds over the defaults, keeping the invariants.

    Two of them are structural rather than preferences. `irreversible` is always
    None — a number there would mean some confidence buys an unrepeatable action,
    which is the one thing the gate must never allow. And `reveal` is never below
    `free`, because the asymmetry between them is the whole design: prewarming
    wrongly wastes milliseconds, revealing wrongly spends the user's attention.
    """
    merged = dict(DEFAULT_THRESHOLDS)
    for key, value in (cfg or {}).items():
        if key in merged:
            merged[key] = value
    merged["irreversible"] = None
    if merged.get("reveal") is not None and merged.get("free") is not None:
        merged["reveal"] = max(float(merged["reveal"]), float(merged["free"]))
    return merged


class CalibrationStore:
    """The fitted curve, kept beside the predictor state."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS calibration_state (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      updated_ts REAL,
      state_json TEXT
    )
    """

    def __init__(self, memory):
        self.mem = memory
        self.mem.executescript(self.SCHEMA)

    def save(self, calibrator: Calibrator) -> None:
        self.mem.execute(
            "INSERT INTO calibration_state (id, updated_ts, state_json) VALUES (1,?,?)"
            " ON CONFLICT(id) DO UPDATE SET updated_ts=excluded.updated_ts,"
            " state_json=excluded.state_json",
            (time.time(), json.dumps(calibrator.to_dict())),
        )

    def load(self) -> Calibrator:
        rows = self.mem.query("SELECT state_json FROM calibration_state WHERE id=1")
        if not rows or not rows[0][0]:
            return Calibrator()
        try:
            return Calibrator.from_dict(json.loads(rows[0][0]))
        except ValueError:
            return Calibrator()
