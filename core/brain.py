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
from datetime import datetime
import os
import platform
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
    bounds = [f"{key} {spec[key]}" for key in ("minimum", "maximum", "minLength", "maxLength")
              if key in spec]
    return t + (f" ({', '.join(bounds)})" if bounds else "")


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
    signature: tuple
    suspended_at: float
    has_action_evidence: bool = False
    retried_completion: bool = False
    seen: dict = field(default_factory=dict)


class Brain:
    def __init__(self, llm, memory, system_prompt, planner_schema, logger=None,
                 dispatcher=None, registry=None, max_iters: int = 5, budget_ms: int = 8000,
                 history_turns: int = 6, retriever=None, retrieve_k: int = 4,
                 prompt_budget_tokens: int = 2400, offer_safe_mode_confirmation: bool = False):
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
        # Only an interface that displays the explicit permission-change copy
        # opts in. Model output cannot choose this setting or approve a token.
        self.offer_safe_mode_confirmation = offer_safe_mode_confirmation
        self._suspended: dict[str, _Suspended] = {}

    # ── Prompt ───────────────────────────────────────────────────────────

    def build_system_prompt(self, context=None, notes=None) -> str:
        parts = [self.system_prompt, "", "AVAILABLE TOOLS", render_capabilities(self.registry.manifest())]
        # Give the model the computer's actual clock, not only an epoch or a
        # numeric weekday it must guess how to interpret. Refresh each turn.
        now = datetime.now().astimezone()
        parts += ["", "CURRENT COMPUTER (fresh system readings)",
                  f"Local date and time: {now.isoformat(timespec='seconds')} ({now.strftime('%A, %Z')})",
                  f"Operating system: {platform.system()}",
                  f"Project directory: {os.getcwd()}",
                  f"Generated files directory: {os.path.join(os.getcwd(), 'CreatedFolder')}"]
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

    def resume(self, resume_token: str, granted: bool, on_token=None,
               on_safe_mode_change=None, allow_safe_mode_change=True) -> dict:
        """Continue a loop that was suspended awaiting confirmation."""
        state = self._suspended.pop(resume_token, None)
        if state is None:
            return {"plan": [], "reply": "That confirmation is no longer pending.", "error": "expired"}

        confirm_options = {}
        if on_safe_mode_change is not None:
            confirm_options["on_safe_mode_change"] = on_safe_mode_change
        if allow_safe_mode_change is not True:
            confirm_options["allow_safe_mode_change"] = allow_safe_mode_change
        resumed_at = time.monotonic()
        result = self.dispatcher.confirm(state.confirm_token, granted=granted, **confirm_options)
        observation = (
            f"The user declined to run {state.capability}."
            if not granted else _observation(state.capability, result)
        )
        # Human deliberation has its own confirmation-token expiry. It must
        # not consume the model's remaining work budget.
        deadline = state.deadline + max(0.0, resumed_at - state.suspended_at)
        state.seen[state.signature] = result if granted else {
            "cancelled": True, "reason": f"The user declined to run {state.capability}."
        }
        state.trace.append(f"{state.capability}: {'declined' if not granted else 'confirmed'}")
        state.messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
        if granted and _successful_action(self.registry.get(state.capability), result):
            state.has_action_evidence = True
        return self._run(
            state.messages, state.iters_left, deadline, state.trace, on_token=on_token,
            has_action_evidence=state.has_action_evidence,
            retried_completion=state.retried_completion, seen=state.seen,
        )

    # ── The loop proper ──────────────────────────────────────────────────

    def _run(self, messages, iters_left, deadline, trace, on_token=None, *,
             has_action_evidence=False, retried_completion=False, seen=None) -> dict:
        retried_parse = False
        # What has already been dispatched this turn, so an identical call is
        # answered from the first result instead of performed again.
        seen = {} if seen is None else seen
        repeats = 0

        while True:
            # >= not >, so a zero budget spends nothing. time.monotonic() is
            # coarse on Windows (~15 ms), and a strict > let two LLM calls through
            # before the clock had visibly moved.
            if time.monotonic() >= deadline:
                return self._give_up(trace, seen, "ran out of time")
            if iters_left <= 0:
                return self._finish(messages, trace, seen, deadline,
                                    "reached the model-step limit", on_token,
                                    has_action_evidence=has_action_evidence)
            iters_left -= 1

            try:
                chat = getattr(self.llm, "chat_json", self.llm.chat)
                raw = chat(messages, on_token=on_token)
            except LLMError as e:
                self.log(f"brain: {e}")
                return self._give_up(trace, seen, str(e), error="llm")

            messages.append({"role": "assistant", "content": raw})
            call, complaint = parse_tool_call(raw)

            # JSON can still fabricate completed work. Check explicit success
            # claims before displaying or saving a final answer. Past history,
            # reads and refusals do not prove this request performed a change.
            # This bounded backstop does not semantically verify every clause.
            candidate_reply = (
                call.reply if call is not None and call.is_reply
                else _plain_text(raw) if call is None and retried_parse else None
            )
            if (candidate_reply is not None and not has_action_evidence
                    and _claims_action_completed(candidate_reply)):
                if retried_completion:
                    reply = "I couldn't verify that the requested action was completed. No successful action was recorded for this request."
                    self.log("brain: refused unsupported action-completion claim")
                    self.mem.add("assistant", reply)
                    return {"plan": trace, "reply": reply, "error": "unverified_action"}
                retried_completion = True
                messages.append({"role": "user", "content":
                    "ACTION EVIDENCE ERROR: Your reply claimed a completed action, but no "
                    "successful action was observed in this request. Previous conversation "
                    "is not proof of a new action. Only if the user asked for an action, "
                    "perform that requested action with an available tool, then check its "
                    "result before reporting success. Do not perform new actions if the "
                    "user only asked a question about prior work or existing information; "
                    "answer as historical or read-only and describe the observed evidence. "
                    "Quoted text or descriptions of what code would do are not actions "
                    "you performed. Respect any refusal "
                    "or declined permission; if you cannot complete the action, say so "
                    "honestly instead of claiming it happened."})
                continue

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

            # A slow model can finish a proposal after the deadline. Receiving
            # that proposal is not permission to start another action late.
            if time.monotonic() >= deadline:
                return self._give_up(trace, seen, "ran out of time")

            cap = self.registry.get(call.tool)
            if cap is None:
                known = ", ".join(self.registry.names())
                messages.append({"role": "user", "content":
                                 f"OBSERVATION: there is no tool named {call.tool!r}. Available: {known}"})
                continue

            # An identical call, already made this turn, is answered from the
            # first result rather than performed again. A model that cannot get
            # what it wants from a tool tends to try the same call repeatedly —
            # harmless for a read, but "how is the weather" opened one browser
            # tab per iteration until the loop hit its limit, because opening a
            # page can never return the page's contents for it to read.
            signature = (call.tool, json.dumps(call.args or {}, sort_keys=True, default=str))
            if signature in seen:
                repeats += 1
                if repeats >= 2:
                    return self._finish(messages, trace, seen, deadline,
                                        "repeated the same tool without progress", on_token,
                                        has_action_evidence=has_action_evidence)
                messages.append({"role": "user", "content":
                                 f"OBSERVATION: {call.tool} was already called with these "
                                 f"exact arguments in this turn and was not run again. Its "
                                 f"result was: {_observation(call.tool, seen[signature])}. Repeating it will not "
                                 f"produce a different result — use what you have to answer "
                                 f"the user, or try a different tool or different arguments."})
                continue

            repeats = 0
            trace.append(f"{call.tool}({_brief_args(call.args)})")
            dispatch_options = {}
            if self.offer_safe_mode_confirmation:
                dispatch_options["offer_safe_mode_confirmation"] = True
            result = self.dispatcher.dispatch(
                call.tool, call.args, actor="model", confidence=1.0, **dispatch_options
            )

            if isinstance(result, dict) and result.get("needs_confirmation"):
                # Suspend. The model does not get to answer its own question.
                token = secrets.token_urlsafe(8)
                self._suspended[token] = _Suspended(
                    messages=messages, iters_left=iters_left, deadline=deadline,
                    trace=trace, confirm_token=result["token"], capability=call.tool,
                    signature=signature, suspended_at=time.monotonic(),
                    has_action_evidence=has_action_evidence,
                    retried_completion=retried_completion, seen=seen,
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
                    "requires_safe_mode_off": result.get("requires_safe_mode_off", False),
                }

            if _successful_action(cap, result):
                has_action_evidence = True
            observation = _observation(call.tool, result)
            seen[signature] = result
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    def _finish(self, messages, trace, seen, deadline, why, on_token=None, *,
                has_action_evidence=False):
        """Reserve one answer-only turn; it has no path to tool dispatch."""
        if seen and time.monotonic() < deadline:
            final_messages = [*messages, {"role": "system", "content":
                "FINAL ANSWER REQUIRED. Tool use has ended for this request. "
                "Return one JSON object with a reply field and no tool field. "
                "Answer the user's original request using the OBSERVATION results already "
                "received. If a lookup failed, explain the actual failure; do not invent "
                "missing facts. Report any incomplete work honestly. Do not repeat a "
                "tool request or claim an action that was not observed."}]
            try:
                chat = getattr(self.llm, "chat_json", self.llm.chat)
                raw = chat(final_messages, on_token=on_token)
                call, _ = parse_tool_call(raw)
                if call is not None and call.is_reply and call.reply.strip():
                    if not has_action_evidence and _claims_action_completed(call.reply):
                        return self._give_up(trace, seen,
                            "could not verify the model's action-completion claim",
                            error="unverified_action")
                    self.mem.add("assistant", call.reply)
                    return {"plan": trace, "reply": call.reply, "thought": call.thought}
            except LLMError as exc:
                return self._give_up(trace, seen, str(exc), error="llm")
        return self._give_up(trace, seen, why)

    def _give_up(self, trace, seen, why: str, *, error=None) -> dict:
        """Preserve observed data even if the model cannot finish its answer."""
        reply = (why if error == "llm" else f"I couldn't finish the answer because I {why}.")
        if seen:
            summaries = [_result_summary(tool, result) for (tool, _), result in seen.items()]
            reply += "\n\nResults received:\n" + "\n\n".join(summaries[-3:])
        self.log(f"brain: {why}; {len(seen)} tool results retained")
        self.mem.add("assistant", reply)
        return {"plan": trace, "reply": reply, "error" if error else "exhausted": error or why}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _result_summary(tool, result):
    """A bounded literal result, not an inferred claim that the task succeeded."""
    source = ""
    if isinstance(result, dict):
        source = str(result.get("url") or result.get("path") or "")
        if result.get("cancelled"):
            text = result.get("reason") or "The action was cancelled."
        elif result.get("error") or result.get("denied") or result.get("ok") is False:
            text = "Error: " + str(result.get("error") or result.get("reason") or result)
        elif isinstance(result.get("text"), str):
            text = result["text"] or "The tool returned no readable text."
            if result.get("truncated"):
                text += "\n[Partial result: source was truncated.]"
        elif "stdout" in result:
            text = f"Exit code: {result.get('returncode', 'unknown')}\n{result['stdout']}"
            if result.get("stderr"):
                text += "\n" + str(result["stderr"])
        else:
            text = json.dumps(result, ensure_ascii=False, default=str)
    else:
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    text = str(text)
    if len(text) > 2000:
        text = text[:2000] + "\n[Result excerpt truncated.]"
    return f"{tool}{' — ' + source if source else ''}:\n{text}"


_COMPLETED_VERBS = (
    r"saved|created|written|wrote|opened|launched|deleted|removed|updated|changed|"
    r"copied|moved|installed|sent|scheduled|completed|executed|ran|closed"
)
_ACTION_CLAIM = re.compile(
    rf"^(?:(?:done|successfully)\b(?:\s*[,.:!—–-]\s*|\s+|$)|"
    rf"(?:{_COMPLETED_VERBS})\b|"
    rf"I\s*(?:have\s+|'ve\s+|’ve\s+)?(?:successfully\s+|just\s+)?(?:{_COMPLETED_VERBS})\b)|"
    rf"\b(?:has\s+been|have\s+been|is|are|was|were)\s+(?:now\s+|successfully\s+)?(?:{_COMPLETED_VERBS})\b",
    re.I,
)


def _claims_action_completed(text: str) -> bool:
    # Past-tense completion, not offers, instructions, hypotheticals or limits.
    # Code and literal quoted strings are data rather than execution claims.
    text = re.sub(r"```[\s\S]*?```|`[^`\n]*`|\"[^\"\n]*\"", "", text)
    for sentence in re.split(r"(?:[.!?]\s+|[;\n]|\bbut\b)", text, flags=re.I):
        sentence = sentence.strip()
        if re.match(r"^(?:according to|the (?:script|code|example)\b|previously\b|yesterday\b)", sentence, re.I):
            continue
        # Negation applies to its clause, not every claim in the sentence.
        # "It already exists, so nothing was created" is a refusal; "Nothing
        # was created, but Chrome was opened" still asserts an action.
        for clause in re.split(r",|\b(?:and|so|yet|however)\b", sentence, flags=re.I):
            clause = clause.strip()
            if re.match(r"^(?:no\b|none\b|nothing\b)", clause, re.I):
                continue
            if _ACTION_CLAIM.search(clause):
                return True
    return False


def _successful_action(cap, result) -> bool:
    if cap is None or cap.reversibility == "free" or result is None:
        return False
    # Fetching is non-free to prohibit speculative network requests, not
    # because reading a page proves a requested file/desktop change happened.
    if cap.name == "os_fetch_url":
        return False
    if not isinstance(result, dict):
        return True
    return not (
        result.get("error") or result.get("denied") or result.get("cancelled")
        or result.get("needs_confirmation") or result.get("ok") is False
        or result.get("returncode", 0) != 0
    )


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
