"""Capability manifest and the single gate every action dispatch flows through.

Before this module existed there were three different action paths with three
different safety stories: ``run_local`` checked safe mode and a (buggy) cwd
prefix, ``write_file`` checked nothing at all, and ``hw_call`` splatted
unvalidated user input straight into a driver method. Nothing anywhere recorded
what an action would cost if it turned out to be wrong.

That is the thing a probabilistic component needs in order to be allowed to act.
A manifest entry states, for one action, three facts that a confidence number
cannot supply on its own:

  * how bad it is to be wrong (``reversibility``),
  * whether a human must say yes first (``requires_confirmation``),
  * where in the filesystem it is allowed to look (``path_scope``).

Scope note, stated plainly because the README used to overclaim it: this is a
policy layer, not a sandbox. It governs the four *actors* that drive the system
(the user, the anticipator, the model, the scheduler). It does not defend
against arbitrary Python running in-process, which can always import the
underlying function directly. What it does give you is that no actor reaches a
side effect without a declared cost, a validated argument set, and a journal row.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from jsonschema import Draft7Validator

# ── Vocabulary ───────────────────────────────────────────────────────────────

Reversibility = Literal["free", "reversible", "irreversible"]
# free         : costs only cycles if wrong (a prewarm, a read, a status query)
# reversible   : the world can return to its prior state (a file write, an LED
#                change, a volume change). If we can perform that restoration
#                ourselves, the entry also carries `undo` and `capture_undo` and
#                the journal can replay it. If only the human can undo it
#                (unlocking a locked screen, closing an app we launched), the
#                classification still holds and `undo` is simply absent.
# irreversible : cannot be put back by anyone (a delete, a killed process, a
#                shell command, a shutdown). Never auto-executed, at any
#                confidence.

Actor = Literal["user", "anticipator", "model", "scheduler"]

Verdict = Literal["allow", "confirm", "deny"]

PathScope = Literal["project", "home", "system"]


@dataclass(frozen=True)
class Capability:
    """One registered action plus the facts the gate needs to judge it."""

    name: str
    fn: Callable
    arg_schema: dict  # JSON Schema, validated before dispatch
    reversibility: Reversibility
    est_cost_ms: int  # rough, used for prewarm budgeting
    requires_confirmation: bool  # if True, never auto-executed regardless of confidence
    summary: str = ""  # one line, shown to the model and in /journal
    undo: Optional[Callable] = None  # payload -> anything; reverses one call
    # An undo payload is assembled from up to two hooks, because different
    # actions keep their reversal key in different places. Overwriting a file
    # needs the contents captured *before* it runs; creating a task needs the row
    # id that only exists *after* it runs. Both hooks are optional, their outputs
    # are merged, and a capability with `undo` must supply at least one.
    capture_undo: Optional[Callable] = None  # args -> dict, runs before fn
    capture_undo_result: Optional[Callable] = None  # (args, result) -> dict, after fn
    path_scope: Optional[PathScope] = None
    path_args: tuple = ()  # names of args holding paths, jailed to path_scope
    extra_validate: Optional[Callable] = None  # args -> error string or None
    # "Confirmation depends on the driver": some capabilities are cheap for most
    # arguments and serious for a few, so the decision has to see the arguments.
    dynamic_confirm: Optional[Callable] = None  # args -> bool

    def __post_init__(self):
        if self.path_args and not self.path_scope:
            raise ValueError(f"{self.name}: path_args declared without a path_scope")
        if self.undo and not (self.capture_undo or self.capture_undo_result):
            raise ValueError(f"{self.name}: undo declared with no way to capture its payload")
        if self.undo and self.reversibility != "reversible":
            raise ValueError(f"{self.name}: undo only makes sense for reversible actions")

    def needs_confirmation(self, args: dict) -> bool:
        if self.requires_confirmation:
            return True
        return bool(self.dynamic_confirm and self.dynamic_confirm(args))


@dataclass(frozen=True)
class GateDecision:
    verdict: Verdict
    reason: str
    args: dict = field(default_factory=dict)  # normalized (paths resolved)

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"


# ── Safe mode ────────────────────────────────────────────────────────────────
#
# Appendix A #4: safe mode used to live in os.environ, which meant every child
# process inherited it and could read it, and any child could in principle have
# been told a different story than the parent believed. It is now process-local
# state. INTUITION_SAFE is still honoured, but only once, as the initial value.

_safe_mode = os.environ.get("INTUITION_SAFE", "1") != "0"
_safe_lock = threading.Lock()


def is_safe_mode() -> bool:
    with _safe_lock:
        return _safe_mode


def set_safe_mode(on: bool) -> bool:
    global _safe_mode
    with _safe_lock:
        _safe_mode = bool(on)
        return _safe_mode


# ── Path jailing ─────────────────────────────────────────────────────────────


def scope_root(scope: Optional[PathScope]) -> Optional[Path]:
    """The directory a scope confines paths to, or None for an unconfined scope."""
    if scope == "project":
        return Path(os.getcwd()).resolve()
    if scope == "home":
        return Path.home().resolve()
    return None  # "system" and None are not confined by path


def jail_path(value: str, scope: Optional[PathScope]) -> tuple[Optional[str], Optional[str]]:
    """Resolve `value` and confine it to `scope`.

    Returns (resolved_path, None) on success or (None, reason) on refusal.

    This is the fix for Appendix A #1. The old check was
    ``target.startswith(base_dir)``, which is a string test, not a path test: with
    a base of ``/home/u/proj`` the sibling directory ``/home/u/proj-evil`` shares
    the prefix and passed. Resolving both sides and asking ``is_relative_to``
    compares path components instead of characters, so a sibling can no longer
    masquerade as a child.
    """
    root = scope_root(scope)
    try:
        resolved = Path(value).resolve()
    except (OSError, ValueError) as e:
        return None, f"unresolvable path {value!r}: {e}"
    if root is None:
        return str(resolved), None
    if resolved == root or resolved.is_relative_to(root):
        return str(resolved), None
    return None, f"path {value!r} resolves outside the {scope} scope ({root})"


# ── The registry ─────────────────────────────────────────────────────────────


class CapabilityRegistry:
    """Every capability the system can perform, keyed by name."""

    def __init__(self):
        self._caps: dict[str, Capability] = {}

    def register(self, cap: Capability) -> Capability:
        self._caps[cap.name] = cap
        return cap

    def get(self, name: str) -> Optional[Capability]:
        return self._caps.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._caps

    def names(self) -> list[str]:
        return sorted(self._caps)

    def all(self) -> list[Capability]:
        return [self._caps[n] for n in self.names()]

    def manifest(self) -> list[dict]:
        """Machine-readable manifest. Phase 2 generates the model's tool list and
        config/planner_schema.json from this, so the prompt cannot drift out of
        sync with what the gate will actually permit."""
        return [
            {
                "name": c.name,
                "summary": c.summary,
                "args": c.arg_schema,
                "reversibility": c.reversibility,
                "requires_confirmation": c.requires_confirmation,
                "path_scope": c.path_scope,
                "undoable": c.undo is not None,
                "est_cost_ms": c.est_cost_ms,
            }
            for c in self.all()
        ]


capabilities = CapabilityRegistry()


# ── The gate ─────────────────────────────────────────────────────────────────


def gate(
    cap: Capability,
    args: dict,
    *,
    confidence: float,
    actor: Actor,
    thresholds: Optional[dict] = None,
) -> GateDecision:
    """Judge one proposed action. Pure: decides, never executes, never logs.

    `thresholds` is the cost-gated policy from Phase 5 (config.yaml). When it is
    omitted the gate still enforces every structural rule below; confidence only
    ever *restricts*, it can never unlock something the manifest forbids.
    """
    args = dict(args or {})

    # Rule 1 — validate against the declared schema before anything else.
    # This is what closes the hw_call(**kwargs) hole: an argument the driver
    # never declared is now rejected here, before it reaches the driver.
    errors = sorted(Draft7Validator(cap.arg_schema).iter_errors(args), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.path) or "(root)"
        return GateDecision("deny", f"invalid arguments for {cap.name} at {where}: {e.message}")

    if cap.extra_validate:
        problem = cap.extra_validate(args)
        if problem:
            return GateDecision("deny", f"invalid arguments for {cap.name}: {problem}")

    # Rule 2 — resolve and jail every declared path argument.
    for arg_name in cap.path_args:
        if arg_name not in args or args[arg_name] is None:
            continue
        resolved, problem = jail_path(str(args[arg_name]), cap.path_scope)
        if problem:
            return GateDecision("deny", problem)
        args[arg_name] = resolved

    # Rule 3 — the anticipator acts speculatively on a guess the user has not
    # confirmed and may never make. It gets exactly one privilege: actions whose
    # only cost is cycles. This is the whole reason prewarming is safe to do.
    if actor == "anticipator" and cap.reversibility != "free":
        return GateDecision(
            "deny",
            f"anticipator may only run free capabilities; {cap.name} is {cap.reversibility}",
        )

    # Rule 3b — the scheduler fires unattended, with nobody present to answer a
    # prompt. It may not reach for anything that cannot be taken back.
    if actor == "scheduler" and cap.reversibility == "irreversible":
        return GateDecision(
            "deny", f"scheduled payloads may not invoke irreversible capability {cap.name}"
        )

    # Rule 4 — Safe Mode is a prerequisite for anything irreversible, and it is
    # checked before the confirmation rule rather than after. The order is the
    # point: asking "are you sure?" about an action that Safe Mode is going to
    # refuse anyway teaches the user that confirmation prompts are noise. Off is
    # necessary; confirmation is what makes it sufficient.
    if cap.reversibility == "irreversible" and is_safe_mode():
        return GateDecision(
            "deny", f"{cap.name} is irreversible and Safe Mode is ON — use /safe off first"
        )

    # Rule 5 — an explicit confirmation requirement outranks any confidence.
    if cap.needs_confirmation(args):
        return GateDecision("confirm", f"{cap.name} requires confirmation", args)

    # Rule 6 — and irreversible always needs a human, whether or not the manifest
    # entry remembered to say so.
    if cap.reversibility == "irreversible":
        return GateDecision("confirm", f"{cap.name} is irreversible and needs confirmation", args)

    # Rule 7 — cost-gated confidence thresholds (Phase 5). A capability that got
    # this far is free or reversible; the only question left is whether the
    # actor is sure enough to skip asking.
    if actor in ("model", "anticipator") and thresholds:
        floor = thresholds.get("free" if cap.reversibility == "free" else "auto_execute")
        if floor is not None and confidence < float(floor):
            return GateDecision(
                "confirm",
                f"confidence {confidence:.2f} is below the {cap.reversibility} "
                f"threshold {float(floor):.2f}",
                args,
            )

    return GateDecision("allow", "ok", args)


# ── Pending confirmations ────────────────────────────────────────────────────


@dataclass
class Pending:
    token: str
    capability: str
    args: dict
    actor: Actor
    confidence: float
    reason: str
    created: float


class ConfirmationStore:
    """Actions parked awaiting a human yes.

    A CONFIRM verdict has to survive the round trip out to the HUD or the REPL
    and back, so the arguments the gate already validated are held here rather
    than re-parsed from whatever comes back over the wire. Entries expire, so an
    approval clicked twenty minutes later does not fire a stale command.
    """

    def __init__(self, ttl_s: float = 300.0):
        self.ttl_s = ttl_s
        self._items: dict[str, Pending] = {}
        self._lock = threading.Lock()

    def put(self, capability: str, args: dict, actor: Actor, confidence: float, reason: str) -> Pending:
        token = secrets.token_urlsafe(8)
        p = Pending(token, capability, dict(args), actor, confidence, reason, time.time())
        with self._lock:
            self._evict()
            self._items[token] = p
        return p

    def take(self, token: str) -> Optional[Pending]:
        """Fetch and remove. A token is good for exactly one execution."""
        with self._lock:
            self._evict()
            return self._items.pop(token, None)

    def _evict(self):
        cutoff = time.time() - self.ttl_s
        for tok in [t for t, p in self._items.items() if p.created < cutoff]:
            del self._items[tok]


pending_confirmations = ConfirmationStore()
