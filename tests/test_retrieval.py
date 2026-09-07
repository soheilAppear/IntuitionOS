"""Retrieval has to do two things the old code did not: rank, and arrive without
being asked. The prompt-budget tests are the ones that keep it safe at scale."""

import json
import sqlite3
import time

import pytest

from core.brain import Brain
from core.capabilities import capabilities
from core.context import Context
from core.memory import Memory
from core.retrieval import Retriever, estimate_tokens, render_notes


@pytest.fixture
def notes(memory):
    r = Retriever(memory)
    memory.add("note", "The Ollama model I use is llama3", tags="note")
    memory.add("note", "The HUD overlay is toggled with Alt+Space", tags="note")
    memory.add("note", "Deploy runs from the release branch, never main", tags="note")
    return r, memory


def ctx(cwd="/home/u/intuition", branch="main", prev=None, files=None):
    return Context(ts=time.time(), cwd=cwd, git_branch=branch, hour_of_day=10,
                   recent_commands=([{"text": prev, "exit": 0}] if prev else []),
                   recent_files=list(files or []))


class StubLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, on_token=None):
        self.calls.append(list(messages))
        return json.dumps({"thought": "t", "reply": "ok"})


# ── FTS5 ────────────────────────────────────────────────────────────────────


def test_the_index_is_available(notes):
    r, _mem = notes
    assert r.available, "FTS5 is not compiled into this SQLite build"


def test_search_finds_a_saved_note(notes):
    r, _mem = notes
    hits = r.search("llama3")
    assert hits
    assert "llama3" in hits[0].text


def test_search_ranks_rather_than_returning_everything_that_matches(notes):
    """LIKE returned rows in id order with no notion of which was a better match."""
    r, mem = notes
    mem.add("note", "deploy deploy deploy release release", tags="note")
    hits = r.search("deploy release")
    assert len(hits) >= 2
    assert "deploy deploy deploy" in hits[0].text


def test_stopwords_do_not_drown_the_query(notes):
    r, _mem = notes
    hits = r.search("what is the model that I use")
    assert hits
    assert "llama3" in hits[0].text


def test_a_query_of_only_stopwords_still_searches(notes):
    r, _mem = notes
    assert r.search("the a of") is not None


def test_an_empty_query_returns_nothing(notes):
    r, _mem = notes
    assert r.search("") == []
    assert r.search("   ") == []


def test_notes_added_after_the_index_are_searchable(notes):
    """The triggers, doing their job."""
    r, mem = notes
    mem.add("note", "a brand new thought about sqlite", tags="note")
    assert any("brand new" in h.text for h in r.search("sqlite"))


def test_a_deleted_note_leaves_the_index(notes):
    r, mem = notes
    hit = r.search("llama3")[0]
    mem.execute("DELETE FROM mem WHERE id=?", (hit.id,))
    assert r.search("llama3") == []


def test_recency_breaks_ties_between_equally_good_matches(memory):
    r = Retriever(memory)
    old = time.time() - 200 * 86400
    memory.execute("INSERT INTO mem (ts, role, text, tags) VALUES (?,?,?,?)",
                   (old, "note", "widget configuration notes", "note"))
    memory.add("note", "widget configuration notes", tags="note")

    hits = r.search("widget configuration")
    assert hits[0].ts > old, "a year-old note outranked this week's identical one"


# ── Migration ───────────────────────────────────────────────────────────────


def test_it_indexes_notes_that_predate_the_index(tmp_path):
    """The upgrade path: a user's existing notes must become searchable, and must
    not be lost in the process."""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE mem (id INTEGER PRIMARY KEY, ts REAL, role TEXT, text TEXT, tags TEXT)")
    conn.execute("INSERT INTO mem (ts, role, text, tags) VALUES (?,?,?,?)",
                 (time.time(), "note", "an important note from before the upgrade", "note"))
    conn.commit()
    conn.close()

    mem = Memory(str(db))
    r = Retriever(mem)
    hits = r.search("important upgrade")
    assert hits, "pre-existing notes were not backfilled into the index"
    assert "before the upgrade" in hits[0].text


def test_migration_is_idempotent(memory):
    Retriever(memory)
    Retriever(memory)
    r = Retriever(memory)
    memory.add("note", "still working", tags="note")
    assert r.search("working")


def test_it_falls_back_when_the_index_is_unavailable(notes, monkeypatch):
    """FTS5 is compiled into most SQLite builds, but not all."""
    r, _mem = notes
    monkeypatch.setattr(r, "available", False)
    hits = r.search("llama3")
    assert hits
    assert "llama3" in hits[0].text


def test_a_broken_index_falls_back_rather_than_raising(notes, monkeypatch):
    r, _mem = notes

    def explode(*a, **kw):
        raise sqlite3.OperationalError("no such module: fts5")

    monkeypatch.setattr(r, "_fts_search", explode)
    assert r.search("llama3")


# ── Cue-driven retrieval ────────────────────────────────────────────────────


def test_a_note_surfaces_from_the_situation_rather_than_the_words(notes):
    """The design point: notes arrive because the situation matches, not because
    the user typed /recall."""
    r, mem = notes
    mem.add("note", "On the release branch, always bump the version first", tags="note")

    found = r.retrieve("what now?", ctx(branch="release"))
    assert any("bump the version" in n.text for n in found)


def test_the_previous_command_acts_as_a_cue(notes):
    r, mem = notes
    mem.add("note", "pytest is slow here because of the fixtures", tags="note")
    found = r.retrieve("why", ctx(prev="pytest"))
    assert any("pytest is slow" in n.text for n in found)


def test_a_recently_touched_file_acts_as_a_cue(notes):
    r, mem = notes
    mem.add("note", "scheduler.py has the timezone bug in parse_when", tags="note")
    found = r.retrieve("remind me", ctx(files=["/home/u/intuition/core/scheduler.py"]))
    assert any("timezone bug" in n.text for n in found)


def test_retrieval_only_returns_notes_not_the_transcript(notes):
    """Appendix A #16: Brain wrote both sides of every conversation into mem, so
    an unfiltered search returns the transcript instead of the notes."""
    r, mem = notes
    for i in range(20):
        mem.add("user", "tell me about llama3 again")
        mem.add("assistant", "llama3 is a model")

    found = r.retrieve("llama3", ctx())
    assert found
    assert all(n.role == "note" for n in found)


# ── The token budget ────────────────────────────────────────────────────────


def test_a_huge_note_database_does_not_produce_an_oversized_prompt(memory):
    """The acceptance criterion. 10,000 notes, a hard ceiling."""
    memory.add_many(
        ("note", f"note {i} about deployment and the release process", "note")
        for i in range(10_000)
    )
    r = Retriever(memory, budget_tokens=200)

    found = r.retrieve("deployment release", ctx())
    rendered = render_notes(found)
    assert estimate_tokens(rendered) <= 260, f"rendered {estimate_tokens(rendered)} tokens"


def test_the_budget_caps_the_number_of_notes(notes):
    r, mem = notes
    for i in range(50):
        mem.add("note", "deployment " * 40, tags="note")
    assert len(r.retrieve("deployment", ctx(), k=4)) <= 4


def test_a_note_too_long_for_the_budget_is_skipped_not_truncated(memory):
    """A note cut off mid-sentence is worse than no note: it reads as a fact."""
    r = Retriever(memory, budget_tokens=40)
    memory.add("note", "deployment " * 500, tags="note")
    memory.add("note", "deployment is short", tags="note")

    found = r.retrieve("deployment", ctx())
    assert all(len(n.text) < 200 for n in found)


def test_history_is_trimmed_oldest_first(memory, project):
    """The turn the user is in the middle of must survive the trim."""
    brain = Brain(StubLLM(), memory, "SYSTEM", {}, registry=capabilities,
                  history_turns=50, prompt_budget_tokens=60)
    for i in range(50):
        memory.add("user", f"question number {i} " + "padding " * 20)
        memory.add("assistant", f"answer number {i} " + "padding " * 20)

    trimmed = brain._trim(brain._history())
    assert trimmed, "the trim discarded everything"
    assert "number 49" in trimmed[-1]["content"]
    assert not any("number 0 " in m["content"] for m in trimmed)


# ── Reaching the prompt ─────────────────────────────────────────────────────


def test_a_saved_note_is_demonstrably_present_in_the_prompt(memory, project):
    """The README's claim, asserted against a stubbed client. This is the thing
    that was simply not true before."""
    memory.add("note", "The deploy key lives in the ops vault", tags="note")
    llm = StubLLM()
    brain = Brain(llm, memory, "SYSTEM", {}, registry=capabilities,
                  retriever=Retriever(memory))

    brain.step("where is the deploy key?", context=ctx())

    system = llm.calls[0][0]["content"]
    assert "RELEVANT NOTES YOU SAVED EARLIER" in system
    assert "ops vault" in system


def test_no_note_block_appears_when_nothing_matches(memory, project):
    memory.add("note", "something entirely unrelated", tags="note")
    llm = StubLLM()
    brain = Brain(llm, memory, "SYSTEM", {}, registry=capabilities,
                  retriever=Retriever(memory))

    brain.step("zzzqqq", context=ctx())
    assert "RELEVANT NOTES" not in llm.calls[0][0]["content"]


def test_a_brain_with_no_retriever_still_works(memory, project):
    llm = StubLLM()
    brain = Brain(llm, memory, "SYSTEM", {}, registry=capabilities)
    assert brain.step("hello")["reply"] == "ok"


def test_a_failing_retriever_does_not_break_a_turn(memory, project):
    class Exploding:
        def retrieve(self, *a, **kw):
            raise RuntimeError("index corrupt")

    llm = StubLLM()
    brain = Brain(llm, memory, "SYSTEM", {}, registry=capabilities, retriever=Exploding())
    assert brain.step("hello")["reply"] == "ok"


def test_render_notes_is_empty_for_no_notes():
    assert render_notes([]) == ""
