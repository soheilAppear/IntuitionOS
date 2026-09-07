"""Fuzzy command correction (Appendix A #18).

It is one of the five headline features and had no test at all. What matters is
not just that it corrects typos, but that it knows when to leave things alone: a
correction that fires on something the user meant literally is worse than no
correction, because it silently runs a different command.
"""

import pytest

from interface.server import _KNOWN_CMDS, _fuzzy_cmd
from interface.terminal import fuzzy_slash


# Both interfaces implement this separately, so both are held to the same
# behaviour rather than one being assumed to follow the other.
@pytest.fixture(params=[_fuzzy_cmd, fuzzy_slash], ids=["hud", "terminal"])
def correct(request):
    return request.param


# ── Corrections that should happen ──────────────────────────────────────────


@pytest.mark.parametrize("typed,expected", [
    ("/hlp", "/help"),
    ("/taks", "/tasks"),
    ("/tsaks", "/tasks"),
    ("/memroy", "/memory"),
    ("/recal", "/recall"),
    ("/drema", "/dream"),
    ("/exti", "/exit"),
    ("/cofnig", "/config"),
    ("/capabilites", "/capabilities"),
    ("/calibraton", "/calibration"),
    ("/jounral", "/journal"),
])
def test_a_typo_is_corrected(correct, typed, expected):
    assert correct(typed) == expected


@pytest.mark.parametrize("command", _KNOWN_CMDS)
def test_a_known_command_is_returned_untouched(correct, command):
    assert correct(command) == command


# ── Corrections that should not happen ──────────────────────────────────────


@pytest.mark.parametrize("typed", [
    "/zzzzzzzz",
    "/qqq",
    "/",
    "/xyzzy",
])
def test_something_unlike_any_command_is_left_alone(correct, typed):
    """Returning the input unchanged produces an honest 'unknown command' rather
    than silently running something else."""
    assert correct(typed) == typed


def test_it_does_not_confuse_two_commands_that_differ_by_one_letter(correct):
    """/done and /dream are close enough to be worth checking explicitly."""
    assert correct("/done") == "/done"
    assert correct("/dream") == "/dream"


def test_it_does_not_turn_delete_into_dream(correct):
    assert correct("/delete") == "/delete"


def test_correction_is_stable(correct):
    """Correcting twice must not wander to a third command."""
    once = correct("/taks")
    assert correct(once) == once


# ── The two interfaces agree ────────────────────────────────────────────────


@pytest.mark.parametrize("typed", [
    "/hlp", "/taks", "/memroy", "/undoo", "/rulez", "/zzzz", "/help",
])
def test_the_hud_and_the_terminal_correct_identically(typed):
    """They are separate implementations over separate command lists, so they can
    drift. A command that exists in one and not the other is a real bug."""
    assert _fuzzy_cmd(typed) == fuzzy_slash(typed)


def test_both_interfaces_know_the_same_commands():
    from interface.terminal import fuzzy_slash as _f

    # fuzzy_slash keeps its list inline; compare via behaviour on each name.
    for command in _KNOWN_CMDS:
        assert _f(command) == command, f"the terminal does not know {command}"


def test_every_advertised_command_is_actually_handled():
    """A command in the fuzzy table that nothing implements would be corrected
    *to* and then rejected, which is worse than not being corrected at all."""
    import inspect

    from interface import server, terminal

    server_src = inspect.getsource(server)
    terminal_src = inspect.getsource(terminal)

    for command in _KNOWN_CMDS:
        assert f'"{command}' in server_src or f"'{command}" in server_src, \
            f"{command} is offered by the HUD's fuzzy table but never handled"
        assert f'"{command}' in terminal_src or f"'{command}" in terminal_src, \
            f"{command} is offered by the terminal's fuzzy table but never handled"
