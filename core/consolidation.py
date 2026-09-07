"""Consolidation: the slow system writing rules for the fast one.

`/dream` used to call plan_dryrun("reflection"), which returned
["Reply to the user"]. It was a stub with an evocative name.

What it should be is the one place the language model genuinely earns its cost.
The predictor is fast and statistical but cannot say *why*; the model is slow and
expensive but can look at a cluster of episodes and judge whether it is a habit
or a coincidence, and name it in a sentence. So the model runs offline, on idle
or on demand, and what it produces is a rule — after which the fast path consults
the rule directly, with no model in the query path at all.

That last property is the whole design. A language model with a multi-second
response time cannot sit inside a keystroke. It can write the policy that does.

The user-facing payoff is `/rules`: plain sentences describing what the system
believes about your habits, each deletable. That transparency is the difference
between a system that feels intuitive and one that feels like it is watching you.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .predictor import context_bucket, last_command

SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (
  id INTEGER PRIMARY KEY,
  created_ts REAL,
  context_pattern_json TEXT,
  action TEXT,
  support INTEGER,        -- episodes supporting it
  hit_rate REAL,          -- rolling accuracy since promotion
  description TEXT,       -- LLM-authored, human-readable
  last_fired_ts REAL,
  active INTEGER
);
CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(active);
"""

# A pattern seen fewer times than this is an anecdote, not a habit.
DEFAULT_MIN_SUPPORT = 4
# And one that is contradicted more often than not is not a pattern either.
DEFAULT_MIN_CONFIDENCE = 0.5
# Once promoted, a rule that keeps missing is retired rather than left to rot.
DEFAULT_PRUNE_BELOW = 0.25
DEFAULT_PRUNE_AFTER = 8  # observations before a rule can be pruned on hit rate


@dataclass
class Candidate:
    """A recurring (situation, action) pair found in the log, before judgement."""
    pattern: dict
    action: str
    support: int
    total: int

    @property
    def confidence(self) -> float:
        return self.support / self.total if self.total else 0.0

    def describe(self) -> str:
        """The fallback description, used when no model is available.

        Deliberately plain rather than evocative: an honest fallback beats an
        empty string, and it must not be mistaken for the model's judgement.
        """
        where = self.pattern.get("cwd") or "anywhere"
        prev = self.pattern.get("previous")
        when = _hour_phrase(self.pattern.get("hour_bucket"))
        parts = [f"You run {self.action!r}"]
        if prev:
            parts.append(f"after {prev!r}")
        if where != "anywhere":
            parts.append(f"in {where}")
        if when:
            parts.append(when)
        return " ".join(parts) + f" ({self.support} of {self.total} times)."


def _hour_phrase(bucket) -> str:
    return {0: "at night", 1: "in the morning", 2: "in the afternoon",
            3: "in the evening"}.get(bucket, "")


# ── Finding candidates ───────────────────────────────────────────────────────


def find_candidates(episodes, min_support: int = DEFAULT_MIN_SUPPORT,
                    min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> list:
    """Cluster the log by context similarity and return recurring pairs.

    Two cues are mined, and they answer different questions. The sequential one
    ("what follows X here") is the strongest signal a shell has — habits are
    chains. The situational one ("what you do in this directory at this hour")
    catches the routines that are not triggered by the previous command.

    Support is counted per (pattern -> action); confidence is that count over
    every episode matching the pattern at all, so a pattern the user contradicts
    half the time does not get promoted on volume alone.
    """
    sequential: dict = defaultdict(lambda: defaultdict(int))
    situational: dict = defaultdict(lambda: defaultdict(int))

    for e in episodes:
        action = (getattr(e, "action", "") or "").strip()
        if not action or _is_meta(action):
            continue
        ctx = getattr(e, "context", None)
        bucket = context_bucket(ctx)
        cwd, branch, hour, dirty = bucket

        prev = last_command(ctx)
        if prev and not _is_meta(prev):
            sequential[(cwd, branch, prev)][action] += 1
        situational[(cwd, branch, hour, dirty)][action] += 1

    candidates = []
    for (cwd, branch, prev), counts in sequential.items():
        total = sum(counts.values())
        for action, support in counts.items():
            if support >= min_support and support / total >= min_confidence:
                candidates.append(Candidate(
                    pattern={"kind": "sequential", "cwd": cwd, "branch": branch, "previous": prev},
                    action=action, support=support, total=total))

    for (cwd, branch, hour, dirty), counts in situational.items():
        total = sum(counts.values())
        for action, support in counts.items():
            if support >= min_support and support / total >= min_confidence:
                candidates.append(Candidate(
                    pattern={"kind": "situational", "cwd": cwd, "branch": branch,
                             "hour_bucket": hour, "git_dirty": bool(dirty)},
                    action=action, support=support, total=total))

    # Strongest first, so a capped consolidation run spends the model's time on
    # the patterns most likely to be real.
    candidates.sort(key=lambda c: (-c.confidence, -c.support))
    return candidates


def _is_meta(action: str) -> bool:
    """Commands about the system itself are not habits worth learning.

    Without this, /dream promptly discovers that you often run /dream.
    """
    a = action.strip().lower()
    return a.startswith(("/dream", "/rules", "/journal", "/episodes", "/calibration",
                         "/forget", "/undo", "/help", "/config", "/capabilities",
                         "/thresholds", "/reload"))


# ── Judgement ────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """You are reviewing a habit a shell assistant thinks it has noticed about its user.

Pattern: {pattern}
Action: {action}
Seen {support} times out of {total} occasions matching this situation.

Decide whether this is a genuine habit worth acting on, or a coincidence.
Reply with EXACTLY ONE JSON object and nothing else:

{{"genuine": true, "name": "short name", "description": "one plain sentence a user would recognise"}}

Be sceptical. A pattern that could easily be an accident of a short log is not a
habit. Describe it in the second person, as something the user does."""


@dataclass
class Judgement:
    genuine: bool
    name: str
    description: str
    from_model: bool


def judge(candidate: Candidate, llm=None, logger=None) -> Judgement:
    """Ask the model whether a pattern is real, and to name it.

    Consolidation must work without a model. Ollama may be down, the user may
    have no model pulled, and the eval harness runs headless — so a failure here
    falls back to the statistical description rather than dropping the rule or
    raising.
    """
    if llm is None:
        return Judgement(True, candidate.action, candidate.describe(), from_model=False)

    prompt = _JUDGE_PROMPT.format(
        pattern=json.dumps(candidate.pattern), action=candidate.action,
        support=candidate.support, total=candidate.total,
    )
    try:
        raw = llm.chat([{"role": "user", "content": prompt}])
    except Exception as e:
        if logger:
            logger(f"consolidation: model unavailable, using fallback description ({e})")
        return Judgement(True, candidate.action, candidate.describe(), from_model=False)

    from .brain import extract_json_object

    obj = extract_json_object(raw or "")
    if not isinstance(obj, dict):
        return Judgement(True, candidate.action, candidate.describe(), from_model=False)

    description = str(obj.get("description") or "").strip() or candidate.describe()
    return Judgement(
        genuine=bool(obj.get("genuine", True)),
        name=str(obj.get("name") or candidate.action).strip(),
        description=description,
        from_model=True,
    )


# ── The rule store ───────────────────────────────────────────────────────────


class RuleStore:
    def __init__(self, memory):
        self.mem = memory
        self.mem.executescript(SCHEMA)

    def add(self, pattern: dict, action: str, support: int, description: str,
            hit_rate: float = 0.5) -> int:
        return self.mem.insert(
            "INSERT INTO rules (created_ts, context_pattern_json, action, support,"
            " hit_rate, description, last_fired_ts, active) VALUES (?,?,?,?,?,?,NULL,1)",
            (time.time(), json.dumps(pattern, sort_keys=True), action, support,
             hit_rate, description),
        )

    def upsert(self, pattern: dict, action: str, support: int, description: str) -> int:
        """Refresh an existing rule rather than accumulating duplicates.

        Consolidation is run repeatedly over an overlapping window, so without
        this every /dream would add another copy of the same habit.
        """
        key = json.dumps(pattern, sort_keys=True)
        rows = self.mem.query(
            "SELECT id FROM rules WHERE context_pattern_json=? AND action=?", (key, action)
        )
        if rows:
            rule_id = rows[0][0]
            self.mem.execute(
                "UPDATE rules SET support=?, description=?, active=1 WHERE id=?",
                (support, description, rule_id),
            )
            return rule_id
        return self.add(pattern, action, support, description)

    def all(self, active_only: bool = True) -> list:
        sql = ("SELECT id, created_ts, context_pattern_json, action, support, hit_rate,"
               " description, last_fired_ts, active FROM rules")
        if active_only:
            sql += " WHERE active=1"
        sql += " ORDER BY support DESC, id ASC"
        return [_row_to_rule(r) for r in self.mem.query(sql)]

    def get(self, rule_id: int) -> Optional[dict]:
        rows = self.mem.query(
            "SELECT id, created_ts, context_pattern_json, action, support, hit_rate,"
            " description, last_fired_ts, active FROM rules WHERE id=?", (rule_id,)
        )
        return _row_to_rule(rows[0]) if rows else None

    def delete(self, rule_id: int) -> bool:
        if not self.get(rule_id):
            return False
        self.mem.execute("DELETE FROM rules WHERE id=?", (rule_id,))
        return True

    def deactivate(self, rule_id: int) -> None:
        self.mem.execute("UPDATE rules SET active=0 WHERE id=?", (rule_id,))

    def record_outcome(self, rule_id: int, hit: bool, alpha: float = 0.2) -> None:
        """Update a rule's rolling hit rate with an exponential moving average."""
        rule = self.get(rule_id)
        if not rule:
            return
        current = rule["hit_rate"] if rule["hit_rate"] is not None else 0.5
        updated = (1 - alpha) * current + alpha * (1.0 if hit else 0.0)
        self.mem.execute(
            "UPDATE rules SET hit_rate=?, last_fired_ts=? WHERE id=?",
            (updated, time.time(), rule_id),
        )

    def match(self, prefix: str, ctx) -> list:
        """Active rules that fire in this situation, best first.

        This runs on the query path, so it does exactly one indexed read and a
        dictionary comparison. No model, no scoring pass.
        """
        prefix = (prefix or "").strip().lower()
        cwd, branch, hour, dirty = context_bucket(ctx)
        prev = last_command(ctx)

        out = []
        for rule in self.all(active_only=True):
            pattern = rule["pattern"]
            if pattern.get("cwd") and pattern["cwd"] != cwd:
                continue
            if pattern.get("branch") and pattern["branch"] != branch:
                continue
            if pattern.get("kind") == "sequential":
                if not prev or pattern.get("previous") != prev:
                    continue
            else:
                if pattern.get("hour_bucket") is not None and pattern["hour_bucket"] != hour:
                    continue
                if pattern.get("git_dirty") is not None and bool(pattern["git_dirty"]) != bool(dirty):
                    continue
            if prefix and not rule["action"].lower().startswith(prefix):
                continue
            out.append(rule)

        out.sort(key=lambda r: -(r["hit_rate"] or 0.0))
        return out


def _row_to_rule(r) -> dict:
    try:
        pattern = json.loads(r[2]) if r[2] else {}
    except ValueError:
        pattern = {}
    return {
        "id": r[0], "created_ts": r[1], "pattern": pattern, "action": r[3],
        "support": r[4], "hit_rate": r[5], "description": r[6],
        "last_fired_ts": r[7], "active": bool(r[8]),
    }


# ── The run ──────────────────────────────────────────────────────────────────


@dataclass
class ConsolidationReport:
    examined: int
    candidates: int
    promoted: list
    rejected: list
    pruned: list
    used_model: bool
    calibration_refit: bool = False

    def summary(self) -> str:
        if not self.examined:
            return ("Nothing to consolidate yet — the episode log is empty. Use the "
                    "system for a while and try again.")
        lines = [f"Consolidated {self.examined} episode(s), {self.candidates} candidate pattern(s)."]
        if self.promoted:
            lines.append("")
            lines.append("Promoted:")
            lines += [f"  #{r['id']}  {r['description']}" for r in self.promoted]
        if self.rejected:
            lines.append("")
            lines.append(f"Judged coincidental: {len(self.rejected)}")
        if self.pruned:
            lines.append("")
            lines.append(f"Retired {len(self.pruned)} rule(s) whose hit rate had decayed.")
        if not self.promoted and not self.pruned:
            lines.append("")
            lines.append("No new habits found. Nothing recurred often enough to be worth a rule.")
        if not self.used_model:
            lines.append("")
            lines.append("(No model available — descriptions are statistical rather than written.)")
        return "\n".join(lines)


def consolidate(episodes, rules: RuleStore, llm=None, *, min_support: int = DEFAULT_MIN_SUPPORT,
                min_confidence: float = DEFAULT_MIN_CONFIDENCE, max_candidates: int = 12,
                calibrator=None, calibration_store=None, logger=None) -> ConsolidationReport:
    """One consolidation pass: find, judge, promote, prune, refit."""
    episode_list = list(episodes)
    candidates = find_candidates(episode_list, min_support=min_support,
                                 min_confidence=min_confidence)[:max_candidates]

    promoted, rejected = [], []
    used_model = False

    for candidate in candidates:
        verdict = judge(candidate, llm=llm, logger=logger)
        used_model = used_model or verdict.from_model
        if not verdict.genuine:
            rejected.append(candidate)
            continue
        rule_id = rules.upsert(candidate.pattern, candidate.action,
                               candidate.support, verdict.description)
        promoted.append(rules.get(rule_id))

    pruned = prune(rules)

    refit = False
    if calibrator is not None:
        # Refit here rather than on every prediction: this is the offline pass,
        # and a curve that moves under the user mid-session is worse than one
        # that is a few hours stale.
        try:
            calibrator.fit([e for e in episode_list if e.accepted_prediction is not None])
            if calibration_store is not None:
                calibration_store.save(calibrator)
            refit = True
        except Exception as e:
            if logger:
                logger(f"consolidation: calibration refit failed ({e})")

    return ConsolidationReport(
        examined=len(episode_list), candidates=len(candidates), promoted=promoted,
        rejected=rejected, pruned=pruned, used_model=used_model, calibration_refit=refit,
    )


def prune(rules: RuleStore, below: float = DEFAULT_PRUNE_BELOW,
          after: int = DEFAULT_PRUNE_AFTER) -> list:
    """Retire rules whose recent hit rate has decayed.

    Deactivated rather than deleted, so `/rules --all` can still show what the
    system used to believe and why it stopped.
    """
    retired = []
    for rule in rules.all(active_only=True):
        if rule["last_fired_ts"] is None:
            continue  # never had a chance to be wrong
        if (rule["support"] or 0) < after:
            continue
        if (rule["hit_rate"] or 0.0) < below:
            rules.deactivate(rule["id"])
            retired.append(rule)
    return retired


def render_rules(rule_list: list) -> str:
    """`/rules`: what the system believes about you, in plain sentences."""
    if not rule_list:
        return ("No rules yet. Run /dream after using the system for a while and it "
                "will tell you what habits it thinks it has noticed.")
    lines = ["What I think I have noticed about how you work:", ""]
    for r in rule_list:
        state = "" if r["active"] else "  (retired)"
        rate = f"{r['hit_rate']:.0%}" if r["hit_rate"] is not None else "untested"
        lines.append(f"  #{r['id']}{state}  {r['description']}")
        lines.append(f"        seen {r['support']}x · hit rate {rate} · predicts {r['action']!r}")
    lines += ["", "Delete one you disagree with: /rules delete <id>"]
    return "\n".join(lines)
