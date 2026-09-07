"""The tool loop, driven by a stubbed LLM.

Everything here runs without Ollama. The point of the stub is that the model's
output is the one thing we control precisely, so each failure mode a local model
actually exhibits — fences, preamble, malformed JSON, refusing to stop — gets a
test rather than a hope.
"""

import json

import pytest

from core import actions as actions_mod
from core.brain import Brain, extract_json_object, parse_tool_call, render_capabilities
from core.capabilities import capabilities, set_safe_mode
from core.llm import LLMError


class StubLLM:
    """Emits a scripted sequence of turns and records what it was asked."""

    def __init__(self, *turns):
        self.turns = list(turns)
        self.calls = []

    def chat(self, messages, on_token=None):
        self.calls.append(list(messages))
        if not self.turns:
            return json.dumps({"thought": "done", "reply": "out of script"})
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        if on_token:
            on_token(turn)
        return turn


def tool(name, **args):
    return json.dumps({"thought": f"calling {name}", "tool": name, "args": args})


def reply(text):
    return json.dumps({"thought": "finished", "reply": text})


@pytest.fixture
def brain_factory(wired, project):
    acts, journal, mem = wired

    def make(*turns, **kw):
        llm = StubLLM(*turns)
        b = Brain(llm, mem, "SYSTEM", {}, dispatcher=acts, registry=capabilities, **kw)
        return b, llm

    return make


# ── The headline: the model actually reads a file ───────────────────────────


def test_a_read_file_tool_call_really_reads_the_file(project, brain_factory):
    (project / "config.txt").write_text("timezone: Mars/Olympus", encoding="utf-8")

    brain, llm = brain_factory(
        tool("read_file", path="config.txt"),
        reply("The timezone is Mars/Olympus."),
    )
    out = brain.step("what timezone is configured?")

    assert out["reply"] == "The timezone is Mars/Olympus."
    assert out["plan"] == ["read_file(path=config.txt)"]

    # The contents must have actually reached the model, not been imagined by it.
    observations = [m["content"] for m in llm.calls[-1] if m["content"].startswith("OBSERVATION:")]
    assert any("Mars/Olympus" in o for o in observations)


def test_a_write_tool_call_really_writes_and_is_journalled(project, brain_factory, wired):
    _acts, journal, _mem = wired
    brain, _llm = brain_factory(
        tool("write_file", path="out.txt", text="hello"),
        reply("Written."),
    )
    brain.step("write hello to out.txt")

    assert (project / "out.txt").read_text(encoding="utf-8") == "hello"
    row = journal.recent()[0]
    assert row["capability"] == "write_file"
    assert row["actor"] == "model", "a model action must be journalled as the model's"


# ── Malformed output ────────────────────────────────────────────────────────


def test_markdown_fences_are_stripped(project, brain_factory):
    brain, _llm = brain_factory(
        '```json\n{"thought": "t", "tool": "list_dir", "args": {"path": "."}}\n```',
        reply("Listed."),
    )
    assert brain.step("ls")["reply"] == "Listed."


def test_preamble_before_the_json_is_tolerated(project, brain_factory):
    brain, _llm = brain_factory(
        'Sure! Here is what I will do:\n{"thought": "t", "reply": "hi"}',
        reply("unused"),
    )
    assert brain.step("hello")["reply"] == "hi"


def test_malformed_json_gets_one_correction_then_recovers(project, brain_factory):
    brain, llm = brain_factory(
        "{this is not json at all",
        reply("Recovered."),
    )
    out = brain.step("hello")

    assert out["reply"] == "Recovered."
    complaints = [m["content"] for m in llm.calls[-1] if m["content"].startswith("FORMAT ERROR:")]
    assert len(complaints) == 1, "the model must be told what was wrong, exactly once"


def test_persistent_malformed_json_does_not_crash_or_loop(project, brain_factory):
    """A model that simply cannot emit JSON still has to produce something."""
    brain, _llm = brain_factory("no json here", "still no json here", "and again")
    out = brain.step("hello")

    assert out.get("protocol_error")
    assert out["reply"], "the user must get the prose rather than an exception"


def test_trailing_commas_are_forgiven(project, brain_factory):
    brain, _llm = brain_factory('{"thought": "t", "reply": "ok",}')
    assert brain.step("hi")["reply"] == "ok"


def test_a_brace_inside_a_string_does_not_end_the_object(project, brain_factory):
    brain, _llm = brain_factory(json.dumps({"thought": "t", "reply": "use {} for a dict"}))
    assert brain.step("hi")["reply"] == "use {} for a dict"


def test_an_unknown_tool_name_is_reported_back_not_executed(project, brain_factory):
    brain, llm = brain_factory(tool("delete_everything"), reply("Understood."))
    out = brain.step("go")

    assert out["reply"] == "Understood."
    obs = [m["content"] for m in llm.calls[-1] if m["content"].startswith("OBSERVATION:")]
    assert any("no tool named" in o for o in obs)


# ── Bounds ──────────────────────────────────────────────────────────────────


def test_max_iters_is_honoured(project, brain_factory):
    """A stub that never stops must still terminate."""
    brain, llm = brain_factory(*[tool("list_dir", path=".")] * 50)
    out = brain.step("go", max_iters=3)

    assert out["exhausted"] == "reached the tool-call limit"
    assert len(out["plan"]) == 3
    assert len(llm.calls) == 3


def test_the_wall_clock_budget_is_honoured(project, brain_factory):
    brain, llm = brain_factory(*[tool("list_dir", path=".")] * 50)
    out = brain.step("go", max_iters=50, budget_ms=0)

    assert out["exhausted"] == "ran out of time"
    assert llm.calls == [], "an exhausted budget must not spend an LLM call"


def test_an_llm_failure_is_surfaced_not_swallowed(project, brain_factory):
    brain, _llm = brain_factory(LLMError("Cannot reach Ollama at http://127.0.0.1:11434"))
    out = brain.step("hello")

    assert out["error"] == "llm"
    assert "Cannot reach Ollama" in out["reply"]


# ── Confirmation ────────────────────────────────────────────────────────────


def test_run_local_halts_the_loop_and_surfaces_a_confirmation(project, brain_factory):
    set_safe_mode(False)
    brain, llm = brain_factory(tool("run_local", cmd="rm -rf ."), reply("never reached"))
    out = brain.step("clean the directory")

    assert out["needs_confirmation"]
    assert out["capability"] == "run_local"
    assert out["reversibility"] == "irreversible"
    assert out["resume_token"]
    assert len(llm.calls) == 1, "the loop must suspend, not carry on"


def test_the_model_cannot_approve_its_own_confirmation(project, brain_factory):
    """The suspended loop is resumable only through Brain.resume, which is
    reachable from the UI and not from anything the model emits."""
    set_safe_mode(False)
    brain, _llm = brain_factory(
        tool("run_local", cmd="echo hi"),
        # If the model's next turn could unpark it, this would run something.
        tool("run_local", cmd="echo hi"),
    )
    out = brain.step("run it")
    assert out["needs_confirmation"]
    assert len(brain._suspended) == 1


def test_declining_resumes_the_loop_with_the_refusal(project, brain_factory):
    set_safe_mode(False)
    brain, llm = brain_factory(
        tool("run_local", cmd="echo hi"),
        reply("Understood, I did not run it."),
    )
    parked = brain.step("run it")
    out = brain.resume(parked["resume_token"], granted=False)

    assert out["reply"] == "Understood, I did not run it."
    obs = [m["content"] for m in llm.calls[-1] if "declined" in m["content"]]
    assert obs, "the model must be told the human said no"


def test_granting_resumes_the_loop_and_runs_the_action(project, brain_factory, wired):
    _acts, journal, _mem = wired
    set_safe_mode(False)
    brain, _llm = brain_factory(
        tool("write_file", path="a.txt", text="x"),
        reply("done"),
    )
    # write_file is reversible and needs no confirmation, so drive the parked path
    # through delete_task, which is irreversible by manifest.
    brain2, _ = brain_factory(tool("delete_task", task_id=1), reply("Deleted."))
    parked = brain2.step("delete task 1")
    out = brain2.resume(parked["resume_token"], granted=True)

    assert out["reply"] == "Deleted."
    assert journal.recent()[0]["decision"] == "confirm_granted"


def test_a_stale_resume_token_is_refused(project, brain_factory):
    brain, _llm = brain_factory(reply("hi"))
    out = brain.resume("not-a-real-token", granted=True)
    assert out["error"] == "expired"


def test_a_denied_action_comes_back_as_an_observation(project, brain_factory):
    """The gate refuses, the model is told why, and the loop carries on."""
    brain, llm = brain_factory(
        tool("write_file", path=str(project.parent / "escape.txt"), text="x"),
        reply("I could not write outside the project."),
    )
    out = brain.step("write outside")

    assert out["reply"] == "I could not write outside the project."
    obs = [m["content"] for m in llm.calls[-1] if m["content"].startswith("OBSERVATION:")]
    assert any("outside" in o for o in obs)


# ── Prompt assembly ─────────────────────────────────────────────────────────


def test_the_tool_list_is_generated_from_the_manifest(project, brain_factory):
    brain, llm = brain_factory(reply("hi"))
    brain.step("hello")

    system = llm.calls[0][0]["content"]
    assert "AVAILABLE TOOLS" in system
    for name in ("read_file", "write_file", "run_local"):
        assert name in system
    # And the costs travel with the names, so the model is not told a tool is
    # available without being told what it costs.
    assert "irreversible" in system
    assert "needs confirmation" in system


def test_conversation_history_reaches_the_prompt(project, brain_factory, wired):
    _acts, _journal, mem = wired
    mem.add("user", "my name is Soheil")
    mem.add("assistant", "Noted.")

    brain, llm = brain_factory(reply("Soheil."))
    brain.step("what is my name?")

    contents = [m["content"] for m in llm.calls[0]]
    assert "my name is Soheil" in contents, "the prompt used to be [system, user] and nothing else"


def test_context_is_rendered_into_the_prompt(project, brain_factory):
    brain, llm = brain_factory(reply("ok"))
    brain.step("hi", context={"cwd": "/tmp/x", "git_branch": "main", "last_exit_code": None})

    system = llm.calls[0][0]["content"]
    assert "CURRENT SITUATION" in system
    assert "git_branch: main" in system
    assert "last_exit_code" not in system, "empty context fields should not pad the prompt"


def test_render_capabilities_marks_optional_arguments():
    text = render_capabilities(capabilities.manifest())
    assert "read_file(path: string)" in text
    assert "list_dir(path?: string)" in text


# ── Parser units ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('noise {"a": 1} trailing', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('{"a": "brace } inside"}', {"a": "brace } inside"}),
    ('{"a": {"b": 2}}', {"a": {"b": 2}}),
    ("not json", None),
    ("", None),
    ("{unbalanced", None),
])
def test_extract_json_object(raw, expected):
    assert extract_json_object(raw) == expected


def test_parse_tool_call_rejects_an_object_with_neither_tool_nor_reply():
    call, complaint = parse_tool_call('{"thought": "hmm"}')
    assert call is None
    assert "tool" in complaint and "reply" in complaint


def test_parse_tool_call_rejects_non_object_args():
    call, complaint = parse_tool_call('{"tool": "read_file", "args": "config.yaml"}')
    assert call is None
    assert "args" in complaint


def test_parse_tool_call_treats_missing_args_as_empty():
    call, complaint = parse_tool_call('{"tool": "hw_list"}')
    assert complaint is None
    assert call.tool == "hw_list"
    assert call.args == {}
