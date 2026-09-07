"""Tool-loop success rate, measured by running the loop.

The brief defines this as the fraction of model plans that complete without a
gate denial or a parse failure. Inferring it from the episode log would be a
proxy for a proxy, so instead the real Brain is run against scripted model
output covering what a local model actually emits: clean JSON, JSON wrapped in
markdown fences, JSON with preamble, trailing commas, prose with no JSON at all,
a hallucinated tool name, a path outside the jail, and an irreversible action.

Each plan below is labelled with what *should* happen, so the harness reports not
just the rate but which failure modes are still failing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from core.brain import Brain
from core.capabilities import capabilities, set_safe_mode


def _tool(name, **args):
    return json.dumps({"thought": f"I will {name}", "tool": name, "args": args})


def _reply(text):
    return json.dumps({"thought": "done", "reply": text})


@dataclass
class Plan:
    name: str
    turns: list
    expect: str  # "completes" | "parked" | "denied"


def standard_plans(target_file: str = "probe.txt") -> list:
    """Representative plans, with the shape of real local-model output."""
    return [
        Plan("clean single tool call",
             [_tool("list_dir", path="."), _reply("Listed the directory.")],
             "completes"),

        Plan("two chained tool calls",
             [_tool("write_file", path=target_file, text="hello"),
              _tool("read_file", path=target_file),
              _reply("Wrote and read it back.")],
             "completes"),

        Plan("wrapped in markdown fences",
             ['```json\n' + _tool("list_dir", path=".") + '\n```', _reply("Done.")],
             "completes"),

        Plan("preamble before the JSON",
             ["Sure, here's my plan:\n" + _tool("hw_list"), _reply("No devices.")],
             "completes"),

        Plan("trailing comma",
             ['{"thought": "t", "tool": "hw_list", "args": {},}', _reply("Done.")],
             "completes"),

        Plan("prose with no JSON, then recovers",
             ["I think I should look at the directory.", _tool("list_dir", path="."),
              _reply("Listed.")],
             "completes"),

        Plan("hallucinated tool name, then recovers",
             [_tool("summarise_everything"), _tool("list_dir", path="."), _reply("Listed.")],
             "completes"),

        Plan("reply with no tool call at all",
             [_reply("You do not need a tool for that.")],
             "completes"),

        Plan("path outside the project jail",
             [_tool("write_file", path="../escape.txt", text="x"),
              _reply("I could not write outside the project.")],
             "completes"),

        Plan("irreversible action",
             [_tool("run_local", cmd="echo hi")],
             "parked"),

        Plan("never stops calling tools",
             [_tool("list_dir", path=".")] * 40,
             "denied"),

        Plan("cannot emit JSON at all",
             ["no.", "still no.", "nope."],
             "denied"),
    ]


class ScriptedLLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0

    def chat(self, messages, on_token=None):
        self.calls += 1
        if not self.turns:
            return _reply("out of script")
        return self.turns.pop(0)


def measure(memory, dispatcher, project_dir, plans=None) -> dict:
    """Run every plan and report how many completed as intended.

    A plan "succeeds" when it does what it was supposed to do — which for the
    irreversible one means being parked for confirmation, and for the runaway one
    means being stopped by the iteration bound. A loop that ran the shell command
    would be a failure, not a success.
    """
    plans = plans or standard_plans()
    was_safe = None
    try:
        from core.capabilities import is_safe_mode
        was_safe = is_safe_mode()
        set_safe_mode(False)

        succeeded = 0
        failures = []
        parse_recoveries = 0
        gate_denials = 0

        for plan in plans:
            llm = ScriptedLLM(plan.turns)
            brain = Brain(llm, memory, "SYSTEM", {}, dispatcher=dispatcher,
                          registry=capabilities, max_iters=6, budget_ms=5000,
                          history_turns=0)
            try:
                out = brain.step(f"[eval] {plan.name}")
            except Exception as e:
                failures.append(f"{plan.name}: raised {type(e).__name__}: {e}")
                continue

            if out.get("protocol_error"):
                parse_recoveries += 1

            if plan.expect == "parked":
                ok = bool(out.get("needs_confirmation"))
            elif plan.expect == "denied":
                ok = bool(out.get("exhausted") or out.get("protocol_error"))
            else:
                ok = bool(out.get("reply")) and not out.get("needs_confirmation")

            if ok:
                succeeded += 1
            else:
                failures.append(f"{plan.name}: expected {plan.expect}, got {_describe(out)}")

        return {
            "plans": len(plans),
            "succeeded": succeeded,
            "rate": succeeded / len(plans) if plans else None,
            "failures": failures,
            "parse_recoveries": parse_recoveries,
            "gate_denials": gate_denials,
        }
    finally:
        if was_safe is not None:
            set_safe_mode(was_safe)


def _describe(out: dict) -> str:
    if out.get("needs_confirmation"):
        return "parked for confirmation"
    if out.get("exhausted"):
        return f"exhausted ({out['exhausted']})"
    if out.get("protocol_error"):
        return "protocol error"
    if out.get("error"):
        return f"error ({out['error']})"
    return "completed"
