"""Drive the terminal's actual prompt_toolkit renderer and key bindings.

The pipe sends ordinary keystrokes; no test dispatches a real shell command.
These tests catch the gap between a core resolver and what a user saw at Enter.
"""

import asyncio
import time
import threading
from io import StringIO
from dataclasses import replace
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from core.command_resolver import (
    CommandResolver, CorrectionFeedbackStore, IntuitionCommandProvider,
    StaticCommandProvider,
)
from core.context import Context
from core.calibration import CalibrationStore
from core.capabilities import capabilities, set_safe_mode
from core.consolidation import RuleStore
from core.episodes import EpisodeLog
from core.memory import Memory
from core.predictor import Predictor, PredictorStore
from interface import terminal


@pytest.fixture
def resolver(tmp_path):
    memory = Memory(str(tmp_path / 'terminal.db'))
    feedback = CorrectionFeedbackStore(memory)
    return CommandResolver(
        [IntuitionCommandProvider(), StaticCommandProvider(['git', 'python', 'cat', 'cut'])],
        shell='cmd', feedback=feedback,
    )


async def until(predicate):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if predicate():
            return
        # Windows' monotonic clock resolution can exceed 10 ms; sub-resolution
        # timers can fire immediately while the console input thread is ready.
        await asyncio.sleep(0.03)
    raise AssertionError('prompt did not reach the expected rendered state')


def drive(resolver, script):
    async def run():
        with create_pipe_input() as pipe:
            prompt = terminal.CorrectionPrompt(resolver, input=pipe, output=DummyOutput())
            task = asyncio.create_task(prompt.session.prompt_async('io> '))
            try:
                await until(lambda: prompt.session.app.is_running)
                await script(prompt, pipe, task)
                result = await asyncio.wait_for(task, timeout=3)
                return result, prompt
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
    return asyncio.run(run())


def test_enter_commits_rendered_correction_preserving_every_argument(resolver):
    raw = '  pyhton  "train file.py" --lr 0.001  --secret "A  B"  '
    expected = raw.replace('pyhton', 'python', 1)

    async def script(prompt, pipe, task):
        pipe.send_text(raw)
        await until(lambda: prompt._displayed_text == raw)
        toolbar = prompt._toolbar()
        assert expected in ''.join(text for _, text in toolbar)
        assert ('class:correction.changed', 'python') in toolbar
        pipe.send_text('\r')

    selected, prompt = drive(resolver, script)
    assert selected == expected
    assert prompt.submission_index == 0
    prompt.outcome('error')
    rows = resolver.feedback.mem.query(
        'SELECT original_token, selected_token, accepted, outcome, candidates_json FROM command_corrections'
    )
    assert rows[-1][:4] == ('pyhton', 'python', 1, 'error')
    assert 'secret' not in str(rows)
    assert 'train file' not in str(rows)


def test_exact_available_command_is_never_changed(resolver):
    raw = 'cut  --literal "cat"'

    async def script(prompt, pipe, task):
        pipe.send_text(raw)
        await until(lambda: prompt._displayed_text == raw)
        assert prompt._snapshot['status'] == 'exact'
        assert prompt._snapshot['candidates'] == []
        pipe.send_text('\r')

    selected, _ = drive(resolver, script)
    assert selected == raw


def test_cycle_selects_displayed_alternative(resolver):
    raw = 'cot  "unchanged argument"'

    async def script(prompt, pipe, task):
        pipe.send_text(raw)
        await until(lambda: prompt._displayed_text == raw)
        assert len(prompt._snapshot['candidates']) >= 2
        pipe.send_text('\x0e')  # Ctrl+N
        await until(lambda: prompt._displayed_index == 1)
        assert prompt._snapshot['candidates'][1]['text'] in ''.join(t for _, t in prompt._toolbar())
        pipe.send_text('\r')

    selected, prompt = drive(resolver, script)
    assert selected == prompt.submission_resolution.candidates[1].text
    assert selected.endswith('  "unchanged argument"')


def test_escape_keeps_original_without_silent_submit_correction(resolver):
    raw = '/hlp'

    async def script(prompt, pipe, task):
        pipe.send_text(raw)
        await until(lambda: prompt._displayed_text == raw)
        assert prompt._selected_index == 0
        pipe.send_text('\x1b')
        await until(lambda: prompt._displayed_index is None)
        pipe.send_text('\r')

    selected, prompt = drive(resolver, script)
    assert selected == raw
    row = resolver.feedback.mem.query('SELECT accepted, selected_token FROM command_corrections')[-1]
    assert row == (None, None)  # keeping original is not a rejected correction


def test_enter_before_first_render_waits_for_visible_suggestion(resolver):
    async def script(prompt, pipe, task):
        pipe.send_text('gti status\r')
        await until(lambda: prompt._displayed_text == 'gti status')
        assert not task.done(), 'unseen correction must not be committed'
        pipe.send_text('\r')

    selected, _ = drive(resolver, script)
    assert selected == 'git status'


def test_keep_original_before_first_render_is_not_overwritten(resolver):
    async def script(prompt, pipe, task):
        pipe.send_text('/hlp\x1b\r')
        await until(lambda: prompt._displayed_text == '/hlp')
        assert prompt._selected_index is None
        assert not task.done()
        pipe.send_text('\r')

    selected, _ = drive(resolver, script)
    assert selected == '/hlp'


def test_edit_and_enter_cannot_reuse_stale_arguments(resolver):
    async def script(prompt, pipe, task):
        pipe.send_text('gti status')
        await until(lambda: prompt._displayed_text == 'gti status')
        token = prompt._snapshot['token']
        pipe.send_text(' --short\r')
        await until(lambda: prompt._displayed_text == 'gti status --short')
        assert prompt._snapshot['token'] != token
        assert not task.done(), 'edited arguments need their own displayed commitment'
        pipe.send_text('\r')

    selected, _ = drive(resolver, script)
    assert selected == 'git status --short'


def test_cycle_and_enter_cannot_commit_unrendered_alternative(resolver):
    async def script(prompt, pipe, task):
        pipe.send_text('cot')
        await until(lambda: prompt._displayed_text == 'cot')
        pipe.send_text('\x0e\r')
        await until(lambda: prompt._displayed_index == 1)
        assert not task.done()
        pipe.send_text('\r')

    selected, prompt = drive(resolver, script)
    assert selected == prompt.submission_resolution.candidates[1].text


def test_unsupported_shell_syntax_remains_unchanged_and_explained(resolver):
    raw = 'gti status && echo secret'

    async def script(prompt, pipe, task):
        pipe.send_text(raw)
        await until(lambda: prompt._displayed_text == raw)
        assert prompt._snapshot['status'] == 'unsupported'
        assert prompt._snapshot['candidates'] == []
        assert 'unchanged' in ''.join(t for _, t in prompt._toolbar())
        pipe.send_text('\r')

    selected, _ = drive(resolver, script)
    assert selected == raw


def test_forget_invalidates_previously_visible_commitment(resolver):
    async def script(prompt, pipe, task):
        pipe.send_text('gti status')
        await until(lambda: prompt._displayed_text == 'gti status')
        old = prompt._snapshot['token']
        resolver.feedback.forget()
        pipe.send_text('\r')
        await until(lambda: prompt._snapshot is not None and prompt._snapshot['token'] != old)
        assert not task.done()
        pipe.send_text('\r')

    selected, _ = drive(resolver, script)
    assert selected == 'git status'


def test_disabled_logging_creates_no_feedback(resolver):
    resolver.feedback.enabled = False

    async def script(prompt, pipe, task):
        pipe.send_text('gti status')
        await until(lambda: prompt._displayed_text == 'gti status')
        pipe.send_text('\r')

    selected, prompt = drive(resolver, script)
    prompt.outcome('ok')
    assert selected == 'git status'
    assert resolver.feedback.mem.query('SELECT COUNT(*) FROM command_corrections')[0][0] == 0


@pytest.mark.parametrize('result,expected', [
    ({'denied': True, 'error': 'Safe Mode'}, 'denied'),
    ({'cancelled': True}, 'cancelled'),
    ({'returncode': 2}, 'error'),
    ({'code': 1}, 'error'),
    ({'error': 'missing executable'}, 'error'),
    ({'returncode': 0}, 'ok'),
])
def test_acceptance_and_execution_outcome_are_separate(result, expected):
    assert terminal._execution_outcome(result) == expected


def run_main(monkeypatch, memory, inputs, resolver):
    """Boot the real REPL with pipe input and isolated in-memory collaborators."""
    episodes = EpisodeLog(memory)
    predictor = Predictor(store=PredictorStore(memory))
    sensor = SimpleNamespace(
        snapshot=lambda: Context(ts=time.time(), cwd='.'),
        note_submission=lambda text: None, note_exit_code=lambda code: None,
    )
    monkeypatch.setattr(terminal, 'ContextSensor', lambda **kw: sensor)
    brain = SimpleNamespace(step=lambda *a, **kw: pytest.fail('recognized commands reached the LLM'))
    cfg = {'anticipation': {'enabled': False}}
    stopped = []
    scheduler = SimpleNamespace(stop=lambda: stopped.append('scheduler'))
    monkeypatch.setattr(terminal, 'bootstrap', lambda: (
        cfg, brain, memory, scheduler, episodes, sensor, predictor,
        RuleStore(memory), CalibrationStore(memory),
    ))
    monkeypatch.setattr(terminal, 'create_default_resolver', lambda *a, **kw: resolver)
    monkeypatch.setattr(terminal, 'make_anticipator', lambda *a, **kw: SimpleNamespace(
        update_buffer=lambda text: None, stop=lambda: stopped.append('anticipator'),
        try_serve=lambda text: None,
    ))
    monkeypatch.setattr(terminal, 'rprint', lambda *a, **kw: None)
    failures = []
    original_prompt = terminal.CorrectionPrompt
    script = iter(inputs)

    with create_pipe_input() as pipe:
        class ScriptedPrompt(original_prompt):
            def __init__(self, resolver, context_fn=None):
                super().__init__(resolver, context_fn, input=pipe, output=DummyOutput())

            def prompt(self, message='io> '):
                try:
                    raw = next(script)
                except StopIteration:
                    raise EOFError

                def feed():
                    deadline = time.monotonic() + 4
                    while time.monotonic() < deadline and not self.session.app.is_running:
                        time.sleep(0.03)
                    pipe.send_text('\x1b[200~' + raw + '\x1b[201~')
                    while time.monotonic() < deadline and self._displayed_text != raw:
                        time.sleep(0.03)
                    if self._displayed_text != raw:
                        failures.append('terminal did not display scripted input')
                        pipe.send_text('\x03')
                        return
                    pipe.send_text('\r')

                feeder = threading.Thread(target=feed, daemon=True)
                feeder.start()
                result = super().prompt(message)
                feeder.join(timeout=4)
                return result

        monkeypatch.setattr(terminal, 'CorrectionPrompt', ScriptedPrompt)
        terminal.main()
    assert failures == []
    assert 'scheduler' in stopped
    assert 'anticipator' in stopped
    return episodes, predictor


@pytest.mark.parametrize('safe_mode,answer,expected', [
    (True, 'yes', 'denied'),
    (False, 'no', 'cancelled'),
    (False, 'yes', 'error'),
])
def test_real_repl_correction_cannot_bypass_gate_or_confirmation(
    monkeypatch, project, wired, safe_mode, answer, expected,
):
    actions, journal, memory = wired
    feedback = CorrectionFeedbackStore(memory)
    resolver = CommandResolver(
        [IntuitionCommandProvider(), StaticCommandProvider(['python'])],
        shell='cmd', feedback=feedback,
    )
    called = []
    def fake_shell(cmd, cwd='.'):
        called.append((cmd, cwd))
        return {'returncode': 2, 'stdout': '', 'stderr': 'intentional test failure'}
    monkeypatch.setitem(capabilities._caps, 'run_command', replace(
        capabilities.get('run_command'), fn=fake_shell,
    ))
    monkeypatch.setattr('builtins.input', lambda _: answer)
    set_safe_mode(safe_mode)
    raw = '  pyhton  train.py --secret "DO NOT LEARN THIS"  '
    episodes, predictor = run_main(monkeypatch, memory, [raw, '/exit'], resolver)
    if expected == 'error':
        assert called == [(raw.replace('pyhton', 'python', 1), str(project))]
    else:
        assert called == []
    row = memory.query(
        "SELECT accepted,outcome FROM command_corrections WHERE original_token='pyhton'"
    )[-1]
    assert row == (1, expected)
    shell_episode = next(e for e in episodes.all() if e.capability == 'run_command')
    assert shell_episode.action == 'python'
    assert shell_episode.outcome == expected
    assert 'DO NOT LEARN' not in str(predictor.to_dict())
    assert journal.recent()[0]['capability'] == 'run_command'


def test_real_repl_preserves_existing_reminder_and_directory_shortcuts(monkeypatch, project, wired):
    actions, journal, memory = wired
    resolver = CommandResolver([IntuitionCommandProvider()], shell='cmd')
    calls = []
    monkeypatch.setattr(terminal, 'run_action', lambda name, **kw: calls.append((name, kw)) or {'ok': True})
    run_main(monkeypatch, memory, ['remind me check build in 10m', 'ls', 'tree', '/exit'], resolver)
    assert calls == [
        ('create_task', {'text': 'check build', 'when': 'in 10m'}),
        ('list_dir', {'path': '.'}),
        ('list_tree', {'path': '.'}),
    ]


@pytest.mark.parametrize('raw,expected', [
    ('  /exec\t  pyhton  "train file.py" --lr 0.001  ',
     ('run_command', {'cmd': '  python  "train file.py" --lr 0.001  ', 'cwd': '.'})),
    ('/exec "python  train.py --lr 0.001" .',
     ('run_local', {'cmd': 'python  train.py --lr 0.001', 'cwd': '.'})),
])
def test_exec_wrapper_keeps_complete_arguments_and_legacy_quoted_behavior(
    monkeypatch, project, wired, raw, expected,
):
    _, _, memory = wired
    resolver = CommandResolver(
        [IntuitionCommandProvider(), StaticCommandProvider(['python'])], shell='cmd',
    )
    calls = []
    monkeypatch.setattr(terminal, 'run_action', lambda name, **kw: calls.append((name, kw)) or {'ok': True})
    run_main(monkeypatch, memory, [raw, '/exit'], resolver)
    assert calls == [expected]


def test_repl_unsupported_shell_syntax_never_reaches_an_action_or_llm(monkeypatch, project, wired):
    _, _, memory = wired
    resolver = CommandResolver(
        [IntuitionCommandProvider(), StaticCommandProvider(['git'])], shell='cmd',
    )
    monkeypatch.setattr(terminal, 'run_action', lambda *a, **kw: pytest.fail('unsafe syntax dispatched'))
    episodes, _ = run_main(monkeypatch, memory, ['gti status | echo value', '/exit'], resolver)
    assert episodes.all()[0].outcome == 'unsupported'


def test_repl_forget_clears_persistent_learning_and_in_memory_predictor(monkeypatch, project, wired):
    _, _, memory = wired
    resolver = CommandResolver(
        [IntuitionCommandProvider(), StaticCommandProvider(['python'])], shell='cmd',
        feedback=CorrectionFeedbackStore(memory),
    )
    monkeypatch.setattr(terminal, 'run_action', lambda *a, **kw: {'returncode': 0})
    episodes, _ = run_main(monkeypatch, memory, ['pyhton secret.py', '/forget', '/exit'], resolver)
    assert [episode.action for episode in episodes.all()] == ['/exit']
    assert memory.query('SELECT original_token FROM command_corrections') == []
    state = PredictorStore(memory).load()
    assert state['seen'] == 1
    assert 'python' not in str(state)


def test_confirmation_shows_literal_command_arguments(monkeypatch):
    output = StringIO()
    console = Console(file=output, width=120, color_system=None)
    monkeypatch.setattr(terminal, 'rprint', console.print)
    monkeypatch.setattr(terminal.actions, 'call', lambda *a, **kw: {
        'needs_confirmation': True, 'token': 'one-use', 'capability': 'run_command',
        'args': {'cmd': 'echo "[red]literal[/red]"', 'cwd': '.'},
        'reason': 'Review command', 'reversibility': 'irreversible',
    })
    monkeypatch.setattr(terminal.actions, 'confirm', lambda token, granted: {'cancelled': not granted})
    monkeypatch.setattr('builtins.input', lambda _: 'no')
    assert terminal.run_action('run_command', cmd='echo "[red]literal[/red]"')['cancelled']
    assert 'echo "[red]literal[/red]"' in output.getvalue()
