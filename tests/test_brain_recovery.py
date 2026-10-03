"""A bounded assistant still answers from the observations it already obtained."""

import json

import pytest

import core.brain as brain_mod
from core.actions import register_os_capabilities
from core.brain import Brain
from core.capabilities import capabilities
from core.llm import LLMError


WEATHER = "Boston: Sunny, +18 C"
WEATHER_URL = "https://wttr.in/Boston?format=3"


def propose(name="os_fetch_url", **args):
    if name == "os_fetch_url" and not args:
        args = {"url": WEATHER_URL}
    return json.dumps({"tool": name, "args": args})


def answer(text):
    return json.dumps({"reply": text})


class ScriptedLLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def chat(self, messages, on_token=None):
        self.calls.append([dict(message) for message in messages])
        assert self.turns, "the loop made an unexpected additional model call"
        result = self.turns.pop(0)
        if callable(result):
            result = result()
        if isinstance(result, Exception):
            raise result
        return result


class RecordingDispatcher:
    def __init__(self, result=None):
        self.result = result if result is not None else {
            "url": WEATHER_URL, "text": WEATHER, "status": 200,
        }
        self.calls = []
        self.confirmations = []

    def dispatch(self, name, args, **options):
        self.calls.append((name, dict(args), options))
        return self.result() if callable(self.result) else self.result

    def confirm(self, token, granted, **options):
        self.confirmations.append((token, granted, options))
        if granted:
            return {"returncode": 0, "stdout": "hello"}
        return {"ok": True, "cancelled": True, "capability": "run_local"}


class Clock:
    def __init__(self):
        self.now = 100.0

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def make_brain(memory):
    register_os_capabilities()

    def make(*turns, result=None, **options):
        llm = ScriptedLLM(turns)
        dispatcher = RecordingDispatcher(result)
        brain = Brain(
            llm, memory, "SYSTEM", {}, dispatcher=dispatcher,
            registry=capabilities, **options,
        )
        return brain, llm, dispatcher

    return make


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(brain_mod.time, "monotonic", lambda: clock.now)
    return clock


def test_last_planning_step_can_finish_from_fetched_weather(make_brain):
    brain, llm, dispatcher = make_brain(propose(), answer(WEATHER), max_iters=1)

    result = brain.step("Tell me the weather in Boston")

    assert result["reply"] == WEATHER
    assert "exhausted" not in result
    assert len(dispatcher.calls) == 1
    assert len(llm.calls) == 2
    assert any(WEATHER in message["content"] for message in llm.calls[-1])


def test_repeated_fetches_recover_early_without_refetching(make_brain):
    brain, llm, dispatcher = make_brain(
        propose(), propose(), propose(), answer(WEATHER), max_iters=8,
    )

    result = brain.step("Tell me the weather in Boston")

    assert result["reply"] == WEATHER
    assert "exhausted" not in result
    assert len(dispatcher.calls) == 1
    assert len(llm.calls) == 4
    assert any("already called" in message["content"] for message in llm.calls[2])


def test_repeated_fetch_finalization_never_executes_another_tool(make_brain):
    brain, llm, dispatcher = make_brain(
        propose(), propose(), propose(), propose("os_open_url", url=WEATHER_URL),
        max_iters=8,
    )

    result = brain.step("Tell me the weather in Boston")

    assert len(dispatcher.calls) == 1
    assert len(llm.calls) == 4
    assert WEATHER in result["reply"]
    assert result["exhausted"] == "repeated the same tool without progress"


def test_model_step_limit_finalization_cannot_dispatch_more_tools(make_brain):
    brain, llm, dispatcher = make_brain(
        propose(), propose("os_open_url", url=WEATHER_URL), max_iters=1,
    )

    result = brain.step("Tell me the weather in Boston")

    assert len(dispatcher.calls) == 1
    assert len(llm.calls) == 2
    assert WEATHER in result["reply"]
    assert result["exhausted"] == "reached the model-step limit"
    assert "tool-call limit" not in result["reply"]


def test_expired_deadline_preserves_tool_output_without_more_model_calls(make_brain, clock):
    def fetch():
        clock.advance(2)
        return {"url": WEATHER_URL, "text": WEATHER}

    brain, llm, dispatcher = make_brain(propose(), result=fetch, max_iters=1, budget_ms=1000)

    result = brain.step("Tell me the weather in Boston")

    assert WEATHER in result["reply"]
    assert result["exhausted"] == "ran out of time"
    assert len(llm.calls) == 1
    assert len(dispatcher.calls) == 1


@pytest.mark.parametrize("max_iters", [1, 5])
def test_model_failure_after_fetch_preserves_the_received_weather(make_brain, max_iters):
    brain, llm, dispatcher = make_brain(
        propose(), LLMError("Ollama disconnected during final response"),
        max_iters=max_iters,
    )

    result = brain.step("Tell me the weather in Boston")

    assert WEATHER in result["reply"]
    assert "Ollama disconnected" in result["reply"]
    assert result["error"] == "llm"
    assert len(llm.calls) == 2
    assert len(dispatcher.calls) == 1


def test_final_answer_cannot_turn_a_read_into_an_unsupported_action_claim(make_brain):
    brain, llm, dispatcher = make_brain(
        propose(), answer("Created a weather report on your desktop."), max_iters=1,
    )

    result = brain.step("Tell me the weather in Boston")

    assert result["error"] == "unverified_action"
    assert WEATHER in result["reply"]
    assert "Created a weather report" not in result["reply"]
    assert len(llm.calls) == 2
    assert len(dispatcher.calls) == 1


def test_failed_fetch_is_reported_instead_of_inventing_weather(make_brain):
    failure = "Weather provider did not respond within 10 seconds"
    brain, llm, dispatcher = make_brain(
        propose(), propose(), result={"error": failure}, max_iters=1,
    )

    result = brain.step("Tell me the weather in Boston")

    assert failure in result["reply"]
    assert WEATHER not in result["reply"]
    assert len(dispatcher.calls) == 1
    assert len(llm.calls) == 2


def test_zero_budget_never_starts_a_model_request(make_brain, clock):
    brain, llm, dispatcher = make_brain(budget_ms=0)

    result = brain.step("Tell me the weather in Boston")

    assert result["exhausted"] == "ran out of time"
    assert llm.calls == []
    assert dispatcher.calls == []


def test_late_model_proposal_cannot_start_a_tool_after_deadline(make_brain, clock):
    def slow_proposal():
        clock.advance(2)
        return propose()

    brain, llm, dispatcher = make_brain(slow_proposal, budget_ms=1000)

    result = brain.step("Tell me the weather in Boston")

    assert result["exhausted"] == "ran out of time"
    assert len(llm.calls) == 1
    assert dispatcher.calls == []


def test_finalization_tool_proposal_never_runs_even_if_it_arrives_late(make_brain, clock):
    def slow_proposal():
        clock.advance(2)
        return propose("os_open_url", url=WEATHER_URL)

    brain, llm, dispatcher = make_brain(propose(), slow_proposal, max_iters=1, budget_ms=1000)

    result = brain.step("Tell me the weather in Boston")

    assert WEATHER in result["reply"]
    assert len(llm.calls) == 2
    assert len(dispatcher.calls) == 1


def pending_command():
    return {
        "needs_confirmation": True,
        "token": "confirmation-token",
        "args": {"cmd": "echo hello"},
        "reason": "Command execution requires confirmation",
        "reversibility": "irreversible",
    }


def test_human_confirmation_wait_does_not_consume_execution_budget(make_brain, clock):
    brain, llm, dispatcher = make_brain(
        propose("run_local", cmd="echo hello"), answer("Executed the command."),
        result=pending_command(), budget_ms=1000,
    )
    parked = brain.step("run echo hello")
    assert parked["needs_confirmation"]

    clock.advance(60)
    result = brain.resume(parked["resume_token"], granted=True)

    assert result["reply"] == "Executed the command."
    assert "exhausted" not in result
    assert len(llm.calls) == 2
    assert len(dispatcher.confirmations) == 1


@pytest.mark.parametrize("granted", [True, False])
def test_confirmed_or_declined_actions_are_cached_against_repetition(make_brain, granted):
    command = propose("run_local", cmd="echo hello")
    response = "Executed the command." if granted else "Understood, I did not run it."
    brain, llm, dispatcher = make_brain(
        command, command, command, answer(response), result=pending_command(), max_iters=8,
    )
    parked = brain.step("run echo hello")

    result = brain.resume(parked["resume_token"], granted=granted)

    assert result["reply"] == response
    assert not result.get("needs_confirmation")
    assert len(dispatcher.calls) == 1
    assert len(dispatcher.confirmations) == 1
    assert len(llm.calls) == 4
    assert any("already called" in message["content"] for message in llm.calls[2])
    if not granted:
        assert any("declined" in message["content"] for message in llm.calls[-1])
