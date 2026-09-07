"""Brain: the propose–validate–execute–observe loop.

The old Brain sent one message to the model and returned its prose. It never
called an action. Meanwhile every action the system could take was a slash
command a human typed. The two halves never touched, which meant the project's
actual thesis — that a system can act on intuition — had no surface to be tested
on. This module joins them.

The model proposes. `core.capabilities.gate` validates. `core.actions` executes.
The result comes back as an observation and the loop goes round again, bounded by
both an iteration count and a wall clock, because a 20B model on consumer
hardware will otherwise spend minutes discovering that it is stuck.

The model is never given a way to approve its own confirmation. When the gate
parks an action the loop suspends and hands the question outward; it resumes only
when a human has answered.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from .capabilities import capabilities
from .llm import LLMError
from .retrieval import estimate_tokens, render_notes

# ── Wire format ──────────────────────────────────────────────────────────────
#
# Local models through Ollama are inconsistent about native function calling, so
# the protocol is a plain JSON object the model emits and we parse:
#
#   {"thought": "...", "tool": "read_file", "args": {"path": "config/config.yaml"}}
#   {"thought": "...", "reply": "..."}
#
# Parsing is defensive by design. Small local models wrap JSON in markdown fences,
# add a sentence of preamble, emit trailing commas, and occasionally produce two
# objects. None of that is a reason to fall back to regexing the model's prose —
# it is a reason to extract carefully and, on failure, tell the model precisely
# what was wrong and let it try again.

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class ToolCall:
    thought: str = ""
    tool: Optional[str] = None
    args: dict = field(default_factory=dict)
    reply: Optional[str] = None

    @property
    def is_reply(self) -> bool:
        return self.tool is None and self.reply is not None


def extract_json_object(text: str) -> Optional[dict]:
    """Pull the first balanced JSON object out of whatever the model produced.

    Returns None rather than raising, because "the model did not emit JSON" is an
    ordinary event the loop recovers from, not an exception.
    """
    if not text:
        return None

    candidates = []
    for fenced in _FENCE.findall(text):
        candidates.append(fenced.strip())
    candidates.append(text)

    for candidate in candidates:
        obj = _first_balanced_object(candidate)
        if obj is not None:
            return obj
    return None


def _first_balanced_object(text: str) -> Optional[dict]:
    """Scan for the first {...} that parses, tracking string state so a brace
    inside a string value does not end the object early."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start:i + 1]
                    try:
                        parsed = json.loads(blob)
                    except ValueError:
                        try:
                            parsed = json.loads(_strip_trailing_commas(blob))
                        except ValueError:
                            break  # try the next opening brace
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def _strip_trailing_commas(blob: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", blob)


def parse_tool_call(text: str) -> tuple[Optional[ToolCall], Optional[str]]:
    """Parse one model turn. Returns (call, None) or (None, complaint).

    The complaint is written to be fed straight back to the model, so it says what
    was expected rather than what a Python traceback would say.
    """
    obj = extract_json_object(text)
    if obj is None:
        return None, (
            "Your reply contained no JSON object. Reply with exactly one JSON "
            'object, either {"thought": "...", "tool": "...", "args": {...}} '
            'or {"thought": "...", "reply": "..."}.'
        )

    thought = str(obj.get("thought", "") or "")
    tool = obj.get("tool")
    reply = obj.get("reply")

    if tool is None and reply is None:
        return None, (
            'Your JSON object had neither "tool" nor "reply". Include one of them.'
        )
    if tool is not None and not isinstance(tool, str):
        return None, '"tool" must be a string naming one capability.'
    if tool is not None:
        args = obj.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None, '"args" must be a JSON object mapping argument names to values.'
        return ToolCall(thought=thought, tool=tool, args=args), None

    return ToolCall(thought=thought, reply=str(reply)), None


# ── Prompt assembly ──────────────────────────────────────────────────────────


def render_capabilities(manifest: list[dict]) -> str:
    """The tool list the model sees, generated from the manifest.

    Generated rather than hand-maintained on purpose: a prompt that lists a tool
    the gate will refuse, or omits one it would allow, trains the model to
    mistrust its own instructions.
    """
    lines = []
    for c in manifest:
        props = (c.get("args") or {}).get("properties", {}) or {}
        required = set((c.get("args") or {}).get("required", []) or [])
        params = ", ".join(
            f"{name}{'' if name in required else '?'}: {_type_name(spec)}"
            for name, spec in props.items()
        )
        flags = [c["reversibility"]]
        if c["requires_confirmation"]:
            flags.append("needs confirmation")
        if c.get("path_scope"):
            flags.append(f"paths confined to {c['path_scope']}")
        lines.append(f"- {c['name']}({params})  [{'; '.join(flags)}]  {c['summary']}")
    return "\n".join(lines)


def _type_name(spec: dict) -> str:
    if "enum" in spec:
        return "|".join(str(v) for v in spec["enum"])
    t = spec.get("type", "any")
    if isinstance(t, list):
        return "|".join(x for x in t if x != "null")
    return t


# ── The loop ─────────────────────────────────────────────────────────────────


@dataclass
class _Suspended:
    """A loop parked mid-flight waiting for a human to answer a confirmation."""
    messages: list
    iters_left: int
    deadline: float
    trace: list
    confirm_token: str
    capability: str


class Brain:
    def __init__(self, llm, memory, system_prompt, planner_schema, logger=None,
                 dispatcher=None, registry=None, max_iters: int = 5, budget_ms: int = 8000,
                 history_turns: int = 6, retriever=None, retrieve_k: int = 4,
                 prompt_budget_tokens: int = 2400):
        # Save collaborators
        self.llm = llm
        self.mem = memory
        self.system_prompt = system_prompt
        self.schema = planner_schema
        self.log = logger or (lambda s: None)
        # Injected so tests can drive the loop without touching the real registry.
        if dispatcher is None:
            from .actions import actions as _actions
            dispatcher = _actions
        self.dispatcher = dispatcher
        self.registry = registry or capabilities
        self.max_iters = max_iters
        self.budget_ms = budget_ms
        self.history_turns = history_turns
        self.retriever = retriever
        self.retrieve_k = retrieve_k
        # A ceiling on everything that is not the system prompt itself. Without
        # it a long session or a large note database silently pushes the tool
        # protocol out of the front of the context window, which fails in the
        # most confusing way available: the model simply stops using tools.
        self.prompt_budget_tokens = prompt_budget_tokens
        self._suspended: dict[str, _Suspended] = {}

    # ── Prompt ───────────────────────────────────────────────────────────

    def build_system_prompt(self, context=None, notes=None) -> str:
        parts = [self.system_prompt, "", "AVAILABLE TOOLS", render_capabilities(self.registry.manifest())]
        if context is not None:
            parts += ["", "CURRENT SITUATION", _render_context(context)]
        if notes:
            parts += ["", render_notes(notes)]
        return "\n".join(parts)

    def retrieve(self, user_text: str, context=None) -> list:
        """Notes worth putting in front of the model, chosen by the situation
        as well as by the words.

        This is what makes the README's claim about /save true. It is also
        cue-driven: a note surfaces because it matches where you are and what
        you just did, not only because you happened to type a word from it.
        """
        if not self.retriever:
            return []
        try:
            return self.retriever.retrieve(user_text, context, k=self.retrieve_k)
        except Exception as e:
            self.log(f"brain: retrieval failed ({e})")
            return []

    def _trim(self, messages: list) -> list:
        """Drop the oldest turns until the history fits its share of the budget.

        Oldest first, because the exchange the user is still in the middle of is
        the one that must survive.
        """
        budget = max(0, self.prompt_budget_tokens)
        kept: list = []
        used = 0
        for msg in reversed(messages):
            cost = estimate_tokens(msg.get("content", ""))
            if used + cost > budget:
                break
            kept.append(msg)
            used += cost
        kept.reverse()
        return kept

    def _history(self) -> list:
        """The last few conversational turns, so the model has continuity.

        Previously the prompt was [system, user] and nothing else, which is why
        the assistant had no memory of the sentence before.
        """
        if not self.mem or self.history_turns <= 0:
            return []
        try:
            rows = self.mem.recent(limit=self.history_turns * 2)
        except Exception:
            return []
        msgs = []
        for _id, _ts, role, text, _tags in reversed(rows):
            if role in ("user", "assistant") and text:
                msgs.append({"role": role, "content": text[:2000]})
        return msgs[-(self.history_turns * 2):]

    # ── Entry points ─────────────────────────────────────────────────────

    def step(self, user_text: str, context=None, max_iters: Optional[int] = None,
             budget_ms: Optional[int] = None, on_token=None) -> dict:
        """Run the loop until the model replies, or the budget runs out."""
        max_iters = self.max_iters if max_iters is None else max_iters
        budget_ms = self.budget_ms if budget_ms is None else budget_ms

        notes = self.retrieve(user_text, context)
        messages = [{"role": "system", "content": self.build_system_prompt(context, notes)}]
        messages += self._trim(self._history())
        messages.append({"role": "user", "content": user_text})

        self.mem.add("user", user_text)
        deadline = time.monotonic() + budget_ms / 1000.0
        return self._run(messages, max_iters, deadline, trace=[], on_token=on_token)

    def resume(self, resume_token: str, granted: bool, on_token=None) -> dict:
        """Continue a loop that was suspended awaiting confirmation."""
        state = self._suspended.pop(resume_token, None)
        if state is None:
            return {"plan": [], "reply": "That confirmation is no longer pending.", "error": "expired"}

        result = self.dispatcher.confirm(state.confirm_token, granted=granted)
        observation = (
            f"The user declined to run {state.capability}."
            if not granted else _observation(state.capability, result)
        )
        state.trace.append(f"{state.capability}: {'declined' if not granted else 'confirmed'}")
        state.messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
        return self._run(state.messages, state.iters_left, state.deadline, state.trace, on_token=on_token)

    # ── The loop proper ──────────────────────────────────────────────────

    def _run(self, messages, iters_left, deadline, trace, on_token=None) -> dict:
        retried_parse = False

        while True:
            if iters_left <= 0:
                return self._give_up(messages, trace, "reached the tool-call limit")
            # >= not >, so a zero budget spends nothing. time.monotonic() is
            # coarse on Windows (~15 ms), and a strict > let two LLM calls through
            # before the clock had visibly moved.
            if time.monotonic() >= deadline:
                return self._give_up(messages, trace, "ran out of time")
            iters_left -= 1

            try:
                raw = self.llm.chat(messages, on_token=on_token)
            except LLMError as e:
                self.log(f"brain: {e}")
                return {"plan": trace, "reply": str(e), "error": "llm"}

            messages.append({"role": "assistant", "content": raw})
            call, complaint = parse_tool_call(raw)

            if call is None:
                # One structured correction, then take the prose at face value
                # rather than looping forever against a model that cannot comply.
                if retried_parse:
                    self.log(f"brain: giving up on tool protocol after retry: {complaint}")
                    reply = _plain_text(raw)
                    self.mem.add("assistant", reply)
                    return {"plan": trace, "reply": reply, "protocol_error": complaint}
                retried_parse = True
                messages.append({"role": "user", "content": f"FORMAT ERROR: {complaint}"})
                continue

            retried_parse = False

            if call.is_reply:
                self.mem.add("assistant", call.reply)
                return {"plan": trace, "reply": call.reply, "thought": call.thought}

            cap = self.registry.get(call.tool)
            if cap is None:
                known = ", ".join(self.registry.names())
                messages.append({"role": "user", "content":
                                 f"OBSERVATION: there is no tool named {call.tool!r}. Available: {known}"})
                continue

            trace.append(f"{call.tool}({_brief_args(call.args)})")
            result = self.dispatcher.dispatch(call.tool, call.args, actor="model", confidence=1.0)

            if isinstance(result, dict) and result.get("needs_confirmation"):
                # Suspend. The model does not get to answer its own question.
                token = secrets.token_urlsafe(8)
                self._suspended[token] = _Suspended(
                    messages=messages, iters_left=iters_left, deadline=deadline,
                    trace=trace, confirm_token=result["token"], capability=call.tool,
                )
                return {
                    "plan": trace,
                    "reply": "",
                    "needs_confirmation": True,
                    "resume_token": token,
                    "confirm_token": result["token"],
                    "capability": call.tool,
                    "args": result.get("args", {}),
                    "reason": result.get("reason", ""),
                    "reversibility": result.get("reversibility", ""),
                }

            messages.append({"role": "user", "content":
                             f"OBSERVATION: {_observation(call.tool, result)}"})

    def _give_up(self, messages, trace, why: str) -> dict:
        """Out of iterations or out of time: say so instead of inventing a result."""
        reply = f"I stopped after {len(trace)} tool call(s) because I {why}."
        if trace:
            reply += " So far: " + "; ".join(trace) + "."
        self.log(f"brain: {why} after {len(trace)} calls")
        self.mem.add("assistant", reply)
        return {"plan": trace, "reply": reply, "exhausted": why}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _observation(tool: str, result) -> str:
    """What the model is told came back. Truncated, because a list_tree of a
    node_modules directory will otherwise eat the whole context window."""
    try:
        text = json.dumps(result, default=str)
    except Exception:
        text = str(result)
    if len(text) > 4000:
        text = text[:4000] + f"… (truncated, {len(text)} chars total)"
    return f"{tool} returned {text}"


def _brief_args(args: dict) -> str:
    return ", ".join(f"{k}={str(v)[:40]}" for k, v in (args or {}).items())


def _plain_text(raw: str) -> str:
    """Strip a failed JSON attempt down to something worth showing a human."""
    stripped = _FENCE.sub("", raw).strip()
    return stripped or raw.strip()


def _render_context(context) -> str:
    if isinstance(context, dict):
        items = context.items()
    elif hasattr(context, "__dict__"):
        items = vars(context).items()
    else:
        return str(context)
    return "\n".join(f"{k}: {v}" for k, v in items if v not in (None, "", [], {}))
