"""The gate is the only thing standing between a probabilistic component and a
side effect, so each rule it enforces gets a test that would fail without it."""

import os
from pathlib import Path

import pytest

from core import actions as actions_mod
from core.capabilities import (
    Capability,
    GateDecision,
    capabilities,
    gate,
    jail_path,
    set_safe_mode,
)


def cap(name):
    c = capabilities.get(name)
    assert c is not None, f"{name} has no manifest entry"
    return c


# ── Rule 1: arguments are validated before dispatch ──────────────────────────


def test_unknown_argument_is_rejected():
    d = gate(cap("read_file"), {"path": "x", "encoding": "rot13"}, confidence=1.0, actor="user")
    assert d.verdict == "deny"
    assert "encoding" in d.reason


def test_missing_required_argument_is_rejected():
    d = gate(cap("read_file"), {}, confidence=1.0, actor="user")
    assert d.verdict == "deny"


def test_wrong_type_is_rejected():
    d = gate(cap("complete_task"), {"task_id": "seven"}, confidence=1.0, actor="user")
    assert d.verdict == "deny"


def test_hw_call_argument_not_in_driver_schema_is_denied_before_the_driver(monkeypatch):
    """Appendix A #3: hw_call used to splat unvalidated kwargs into d.call()."""
    reached = []

    class FakeDriver:
        name = "led_strip"

        def schema(self):
            return {"actions": [{"name": "set_brightness", "args": ["brightness"]}]}

        def call(self, action, **kwargs):
            reached.append((action, kwargs))
            return {"ok": True}

    monkeypatch.setitem(actions_mod._drivers, "led_strip", FakeDriver())

    d = gate(
        cap("hw_call"),
        {"device": "led_strip", "action": "set_brightness", "args": {"voltage": 240}},
        confidence=1.0,
        actor="user",
    )
    assert d.verdict == "deny"
    assert "voltage" in d.reason
    assert reached == [], "the driver must never be reached"


def test_hw_call_undeclared_action_is_denied(monkeypatch):
    class FakeDriver:
        name = "led_strip"

        def schema(self):
            return {"actions": [{"name": "status", "args": []}]}

        def call(self, action, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("driver reached")

    monkeypatch.setitem(actions_mod._drivers, "led_strip", FakeDriver())
    d = gate(cap("hw_call"), {"device": "led_strip", "action": "self_destruct"},
             confidence=1.0, actor="user")
    assert d.verdict == "deny"


# ── Rule 2: paths are resolved and jailed, never prefix-matched ──────────────


def test_write_file_outside_the_project_is_denied(project):
    outside = project.parent / "elsewhere.txt"
    d = gate(cap("write_file"), {"path": str(outside), "text": "x"}, confidence=1.0, actor="user")
    assert d.verdict == "deny"
    assert "outside" in d.reason


def test_write_file_inside_the_project_is_allowed(project):
    d = gate(cap("write_file"), {"path": "notes/a.txt", "text": "x"}, confidence=1.0, actor="user")
    assert d.verdict == "allow"
    assert Path(d.args["path"]) == (project / "notes" / "a.txt")


def test_traversal_out_of_the_project_is_denied(project):
    d = gate(cap("read_file"), {"path": "../../secrets.txt"}, confidence=1.0, actor="user")
    assert d.verdict == "deny"


def test_sibling_directory_prefix_escape_is_denied(tmp_path, monkeypatch):
    """Appendix A #1, the specific bug.

    The old check was target.startswith(base_dir). With a base of
    /home/u/proj the sibling /home/u/proj-evil shares that prefix as a *string*
    and passed. Comparing resolved paths component-wise closes it.
    """
    proj = tmp_path / "proj"
    evil = tmp_path / "proj-evil"
    proj.mkdir()
    evil.mkdir()
    monkeypatch.chdir(proj)

    # The precise shape of the old bug: the string test would have said yes.
    assert str(evil).startswith(str(proj)), "this test is meaningless without the prefix overlap"

    d = gate(cap("run_local"), {"cmd": "echo hi", "cwd": str(evil)}, confidence=1.0, actor="user")
    assert d.verdict == "deny"
    assert "outside" in d.reason


def test_run_local_relative_sibling_escape_is_denied(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    (tmp_path / "proj-evil").mkdir()
    proj.mkdir()
    monkeypatch.chdir(proj)
    d = gate(cap("run_local"), {"cmd": "echo hi", "cwd": "../proj-evil"}, confidence=1.0, actor="user")
    assert d.verdict == "deny"


def test_jail_path_accepts_the_root_itself(project):
    resolved, problem = jail_path(".", "project")
    assert problem is None
    assert Path(resolved) == project


# ── Rule 3: the anticipator is confined to free capabilities ─────────────────


@pytest.fixture
def stub_led(monkeypatch):
    class FakeDriver:
        name = "led_strip"

        def schema(self):
            return {"actions": [{"name": "status", "args": []},
                                {"name": "set_color", "args": ["hex"]}]}

        def call(self, action, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("driver reached")

    monkeypatch.setitem(actions_mod._drivers, "led_strip", FakeDriver())


@pytest.mark.parametrize("name,args", [
    ("write_file", {"path": "a.txt", "text": "x"}),
    ("create_task", {"text": "t", "when": "in 5m"}),
    ("run_local", {"cmd": "echo hi"}),
    ("delete_task", {"task_id": 1}),
    ("hw_call", {"device": "led_strip", "action": "status"}),
])
def test_anticipator_cannot_invoke_non_free_capabilities(name, args, project, stub_led):
    d = gate(cap(name), args, confidence=1.0, actor="anticipator")
    assert d.verdict == "deny"
    assert "anticipator" in d.reason


def test_anticipator_may_invoke_free_capabilities(project):
    d = gate(cap("list_dir"), {"path": "."}, confidence=1.0, actor="anticipator")
    assert d.verdict == "allow"


def test_anticipator_is_denied_even_at_full_confidence(project):
    d = gate(cap("write_file"), {"path": "a.txt", "text": "x"}, confidence=1.0, actor="anticipator")
    assert d.verdict == "deny", "no confidence may buy the anticipator a side effect"


# ── Rules 4 and 5: confirmation outranks confidence ─────────────────────────


def test_delete_task_always_requires_confirmation():
    set_safe_mode(False)
    d = gate(cap("delete_task"), {"task_id": 1}, confidence=1.0, actor="user")
    assert d.verdict == "confirm"


def test_irreversible_is_never_auto_executed_at_any_confidence(project):
    set_safe_mode(False)
    for conf in (0.5, 0.95, 0.999, 1.0):
        d = gate(cap("run_local"), {"cmd": "echo hi"}, confidence=conf, actor="model")
        assert d.verdict == "confirm", f"auto-executed an irreversible action at {conf}"


def test_irreversible_is_denied_outright_while_safe_mode_is_on(project):
    """Safe Mode is checked before the confirmation rule, so the user is never
    prompted to approve something that would then be refused."""
    set_safe_mode(True)
    for name, args in (("run_local", {"cmd": "echo hi"}), ("delete_task", {"task_id": 1})):
        d = gate(cap(name), args, confidence=1.0, actor="model")
        assert d.verdict == "deny"
        assert "Safe Mode" in d.reason


def test_scheduler_may_not_invoke_irreversible_capabilities():
    set_safe_mode(False)
    d = gate(cap("delete_task"), {"task_id": 1}, confidence=1.0, actor="scheduler")
    assert d.verdict == "deny"
    assert "scheduled" in d.reason


def test_scheduler_may_invoke_reversible_capabilities(project):
    d = gate(cap("write_file"), {"path": "a.txt", "text": "x"}, confidence=1.0, actor="scheduler")
    assert d.verdict == "allow"


# ── Rule 6: cost-gated thresholds ───────────────────────────────────────────


def test_model_below_the_auto_execute_threshold_must_confirm(project):
    th = {"free": 0.30, "auto_execute": 0.95}
    low = gate(cap("write_file"), {"path": "a.txt", "text": "x"},
               confidence=0.5, actor="model", thresholds=th)
    high = gate(cap("write_file"), {"path": "a.txt", "text": "x"},
                confidence=0.99, actor="model", thresholds=th)
    assert low.verdict == "confirm"
    assert high.verdict == "allow"


def test_thresholds_can_only_restrict_never_unlock(project):
    """A generous threshold must not talk the gate past a structural rule."""
    set_safe_mode(False)
    th = {"free": 0.0, "auto_execute": 0.0, "irreversible": 0.0}
    d = gate(cap("run_local"), {"cmd": "echo hi"}, confidence=1.0, actor="model", thresholds=th)
    assert d.verdict == "confirm"


# ── Manifest completeness ───────────────────────────────────────────────────


def test_every_registered_action_has_a_manifest_entry():
    missing = sorted(set(actions_mod.actions.names) - set(capabilities.names()))
    assert missing == [], f"registered without a declared cost: {missing}"


def test_manifest_is_serialisable():
    import json
    json.dumps(capabilities.manifest())


def test_capability_rejects_undo_without_a_capture_hook():
    with pytest.raises(ValueError):
        Capability(name="x", fn=lambda: None, arg_schema={}, reversibility="reversible",
                   est_cost_ms=1, requires_confirmation=False, undo=lambda p: None)


def test_capability_rejects_undo_on_an_irreversible_action():
    with pytest.raises(ValueError):
        Capability(name="x", fn=lambda: None, arg_schema={}, reversibility="irreversible",
                   est_cost_ms=1, requires_confirmation=False,
                   capture_undo=lambda a: {}, undo=lambda p: None)


# ── The OS surface ──────────────────────────────────────────────────────────


def test_os_capabilities_all_carry_a_manifest_entry():
    """register_os_capabilities is the only way these reach the registry, so a
    function added to os_sandbox and wired up without a declared cost fails here."""
    names = actions_mod.register_os_capabilities()
    if not names:
        pytest.skip("os sandbox unavailable on this platform")
    missing = sorted(set(actions_mod.actions.names) - set(capabilities.names()))
    assert missing == []


@pytest.mark.parametrize("name", [
    "os_shutdown_computer", "os_restart_computer", "os_kill_process",
    "os_type_text", "os_click", "os_move_mouse",
])
def test_destructive_os_actions_are_irreversible_and_confirmed(name):
    """These were reachable from a single regex match on speech with no gate at
    all. Nothing about them may be auto-executed."""
    actions_mod.register_os_capabilities()
    c = capabilities.get(name)
    if c is None:
        pytest.skip(f"{name} unavailable on this platform")
    assert c.reversibility == "irreversible"
    assert c.requires_confirmation


def test_shutdown_is_denied_in_safe_mode_and_confirmed_out_of_it():
    actions_mod.register_os_capabilities()
    c = capabilities.get("os_shutdown_computer")
    if c is None:
        pytest.skip("os sandbox unavailable on this platform")

    set_safe_mode(True)
    assert gate(c, {"delay_sec": 30}, confidence=1.0, actor="user").verdict == "deny"

    set_safe_mode(False)
    assert gate(c, {"delay_sec": 30}, confidence=1.0, actor="user").verdict == "confirm"


def test_os_reads_are_free_so_the_anticipator_may_prewarm_them():
    actions_mod.register_os_capabilities()
    c = capabilities.get("os_get_battery")
    if c is None:
        pytest.skip("os sandbox unavailable on this platform")
    assert gate(c, {}, confidence=1.0, actor="anticipator").verdict == "allow"
