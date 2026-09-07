"""Resolver behavior: availability, exact spans, chronology and user commitment."""
import json
import os
import shutil
import subprocess

import pytest

from core.command_resolver import (
    Command, CommandResolver, CorrectionFeedbackStore, CorrectionSession,
    EnvironmentCatalogProvider, GitSubcommandProvider, IntuitionCommandProvider,
    InstalledGitSubcommandProvider,
    PathExecutableProvider, ShellBuiltinProvider, StaticCommandProvider,
    learning_text, spelling_distance,
    create_default_resolver,
)


def resolver(names=('git', 'python', 'pytest'), **kwargs):
    return CommandResolver([IntuitionCommandProvider(), StaticCommandProvider(names)], **kwargs)


@pytest.mark.parametrize('typed,wanted', [('gti status', 'git status'), ('pyhton train.py --lr 0.001', 'python train.py --lr 0.001'), ('git statsu', 'git status'), ('/memroy', '/memory')])
def test_transpositions_work_without_history_or_llm(typed, wanted):
    result = resolver().resolve(typed)
    assert result.status == 'correction'
    assert result.candidates[0].text == wanted
    assert result.original == typed
    assert spelling_distance('statsu', 'status') == 1


@pytest.mark.parametrize('text', ['git status', 'python', 'pytest', '/tasks'])
def test_exact_valid_names_never_change(text, memory):
    feedback = CorrectionFeedbackStore(memory)
    model = resolver(feedback=feedback)
    offered = model.resolve('pytes')
    event = feedback.record_display(offered)
    feedback.record_selection(event, 0)
    result = model.resolve(text)
    assert result.status == 'exact'
    assert result.candidates == []


def test_prefix_is_incomplete_and_near_ties_are_ambiguous():
    assert resolver().resolve('pyth').status == 'incomplete'
    result = resolver(('cat', 'cut')).resolve('cot')
    assert result.status == 'ambiguous'
    assert [candidate.token for candidate in result.candidates] == ['cat', 'cut']


def test_unavailable_executable_is_not_suggested():
    assert not resolver(('git',)).resolve('pyhton train.py').candidates


@pytest.mark.parametrize('text', [
    '  pyhton\t"train data.py"  --lr 0.001 ',
    "pyhton 'file with spaces.py' --password 'keep these exact bytes'",
    '/exec  pyhton\ttrain.py --lr 0.001',
])
def test_only_command_span_changes(text):
    candidate = resolver().resolve(text).candidates[0]
    a, b = candidate.span
    assert text[a:b] == 'pyhton'
    assert candidate.text == text[:a] + 'python' + text[b:]


def test_cmd_preserves_windows_paths_and_quoted_arguments():
    text = 'pyhton "C:\\Program Files\\train.py" --lr 0.001'
    candidate = resolver(shell='cmd').resolve(text).candidates[0]
    assert candidate.text == text.replace('pyhton', 'python', 1)


@pytest.mark.parametrize('text', ['gti status | cat', 'gti && echo ok', 'gti > out', 'gti; echo hi', 'gti $(whoami)', 'gti `whoami`', 'gti "unterminated', 'gti\nstatus', 'VAR=value gti'])
def test_shell_syntax_is_never_guessed(text):
    result = resolver().resolve(text)
    assert result.status == 'unsupported'
    assert result.namespace == 'shell'
    assert not result.candidates
    assert result.original == text


def test_arguments_flags_and_non_git_subcommands_are_opaque():
    assert resolver().resolve('python statsu --lrr trainn.py').status == 'exact'
    assert resolver().resolve('git status --porclain').status == 'exact'
    assert resolver().resolve('git -C project statsu').status == 'exact'


def test_explicit_git_alias_is_valid():
    model = resolver(git_provider=GitSubcommandProvider(aliases=['statsu']))
    assert model.resolve('git statsu').status == 'exact'


def test_git_config_alias_is_not_corrected(tmp_path):
    gitdir = tmp_path / '.git'
    gitdir.mkdir()
    (gitdir / 'config').write_text('[alias]\n statsu = status\n')
    assert resolver().resolve('git statsu', {'cwd': str(tmp_path)}).status == 'exact'


def test_external_git_subcommand_is_valid():
    assert resolver(('git', 'git-statsu')).resolve('git statsu').status == 'exact'


def test_alias_function_catalog_is_bound_to_execution_shell():
    entries = [{'name': 'gst', 'kind': 'Alias'}, {'name': 'deploy', 'kind': 'Function'}]
    provider = EnvironmentCatalogProvider(entries, shell='powershell', supported_shell='powershell')
    model = CommandResolver([provider], shell='powershell')
    assert model.resolve('GST').status == 'exact'
    assert model.resolve('deplyo').candidates[0].text == 'deploy'
    wrong = EnvironmentCatalogProvider(entries, shell='cmd', supported_shell='powershell')
    assert list(wrong.commands()) == []


def test_powershell_qualified_names_preserve_module_namespace():
    entries = [{'name': 'Get-ChildItem', 'kind': 'Cmdlet', 'source': 'Microsoft.PowerShell.Management'},
               {'name': 'Get-ChildItms', 'kind': 'Function', 'source': 'OtherModule'}]
    provider = EnvironmentCatalogProvider(entries, shell='powershell', supported_shell='powershell')
    model = CommandResolver([provider], shell='powershell')
    assert model.resolve('Microsoft.PowerShell.Management\\Get-ChildItem .').status == 'exact'
    result = model.resolve('Microsoft.PowerShell.Management\\Get-ChlidItem .')
    assert result.candidates[0].text == 'Microsoft.PowerShell.Management\\Get-ChildItem .'
    assert all(c.token.startswith('Microsoft.PowerShell.Management\\') for c in result.candidates)
    assert not model.resolve('UnknownModule\\Get-ChlidItem .').candidates


@pytest.mark.parametrize('shell', ['powershell', 'pwsh'])
def test_actual_full_powershell_snapshot_metadata_without_candidate_execution(shell, tmp_path, monkeypatch):
    executable = shutil.which(shell)
    if not executable:
        pytest.skip(f'{shell} not installed')
    # Capture definitions in process memory and a temporary file only; never
    # print definitions, invoke captured functions, or run the UI launcher.
    script = r"""
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$entries = @(Microsoft.PowerShell.Core\Get-Command -CommandType Alias,Function -All -ListImported | Microsoft.PowerShell.Core\ForEach-Object {
 [pscustomobject]@{name=$_.Name; kind=[string]$_.CommandType; definition=$_.Definition}
})
[Console]::WriteLine((Microsoft.PowerShell.Utility\ConvertTo-Json -InputObject $entries -Depth 5 -Compress))
"""
    snapshot = subprocess.run([executable, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', script],
                              capture_output=True, text=True, encoding='utf-8', timeout=15,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert snapshot.returncode == 0, snapshot.stderr
    entries = json.loads(snapshot.stdout)
    catalog = tmp_path / 'full-shell-catalog.json'
    catalog.write_text(json.dumps({'shell': shell, 'commands': entries}), encoding='utf-8')
    monkeypatch.setenv('INTUITION_SHELL', shell)
    monkeypatch.setenv('INTUITION_SHELL_CATALOG', str(catalog))
    from core.shell_environment import discover_powershell_commands
    assert '%' in {item['name'] for item in discover_powershell_commands(shell)}
    model = create_default_resolver()
    for text in ['%', 'Get-ChildItem', 'Microsoft.PowerShell.Management\\Get-ChildItem .']:
        assert model.resolve(text).status == 'exact', model.resolve(text).to_dict()


def test_shell_case_rules():
    assert CommandResolver([ShellBuiltinProvider('cmd')], shell='cmd').resolve('DIR').status == 'exact'
    assert CommandResolver([ShellBuiltinProvider('sh')], shell='sh').resolve('CD').status != 'exact'


def test_existing_internal_bare_commands_are_preserved():
    model = CommandResolver([IntuitionCommandProvider()])
    for text in ['ls', 'tree', 'tree --depth 2']:
        result = model.resolve(text)
        assert result.status == 'exact' and result.namespace == 'intuitionos'


@pytest.mark.parametrize('text', ['set volume to 50', 'open chrome', 'remind me to call tomorrow', 'read file example.txt'])
def test_existing_app_intentions_precede_shell_corrections(text):
    result = resolver(('set', 'open', 'read')).resolve(text)
    assert result.status == 'exact' and result.namespace == 'intuitionos'


@pytest.mark.parametrize('text', ['git commit -m battery', 'echo battery', '/exec set volume to 50', '/exec open chrome'])
def test_app_intentions_do_not_claim_shell_arguments_or_explicit_exec(text):
    result = resolver(('git', 'echo', 'set', 'open')).resolve(text)
    assert result.status == 'exact' and result.namespace in {'shell', 'git'}


def test_cmd_single_quotes_do_not_hide_shell_operators():
    result = resolver(shell='cmd').resolve("gti 'x & echo y'")
    assert result.status == 'unsupported' and not result.candidates


def test_git_metadata_discovers_new_commands_and_aliases_without_invoking_them(monkeypatch):
    import subprocess
    calls = []
    monkeypatch.setattr('core.command_resolver.shutil.which', lambda name: '/known/git')
    def metadata(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='status backfill statsu\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', metadata)
    provider = InstalledGitSubcommandProvider()
    model = resolver(git_provider=provider)
    assert model.resolve('git backfill').status == 'exact'
    assert model.resolve('git statsu').status == 'exact'
    assert model.resolve('git sttaus').candidates[0].text == 'git status'
    assert calls == [['/known/git', '--list-cmds=main,others,alias']]


def test_git_metadata_failure_preserves_unknown_alias(monkeypatch):
    import subprocess
    monkeypatch.setattr('core.command_resolver.shutil.which', lambda name: '/known/git')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout='', stderr=''))
    model = resolver(git_provider=InstalledGitSubcommandProvider())
    result = model.resolve('git statsu')
    assert result.status == 'exact' and not result.candidates


def test_alias_named_git_does_not_inherit_git_grammar():
    model = CommandResolver([StaticCommandProvider([Command('git', kind='Alias')])])
    assert model.resolve('git statsu').status == 'exact'


def test_path_discovery_is_filesystem_only(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Discovery executed a command'))
    (tmp_path / 'available.exe').write_bytes(b'not run')
    (tmp_path / 'script.cmd').write_text('@echo no')
    (tmp_path / 'note.txt').write_text('not executable')
    if os.name == 'nt':
        provider = PathExecutableProvider(shell='cmd', environ={'PATH': str(tmp_path), 'PATHEXT': '.EXE;.CMD'})
        names = {c.name for c in provider.commands({'cwd': str(tmp_path)})}
        assert {'available', 'available.exe', 'script', 'script.cmd'} <= names
        assert 'note.txt' not in names
    else:
        (tmp_path / 'available.exe').chmod(0o755)
        provider = PathExecutableProvider(shell='sh', environ={'PATH': str(tmp_path)})
        assert {c.name for c in provider.commands()} == {'available.exe'}


def test_learning_is_chronological_and_success_independent(memory):
    feedback = CorrectionFeedbackStore(memory)
    model = resolver(('cat', 'cut'), feedback=feedback)
    context = {'cwd': '/project-a', 'ts': 100}
    before = model.resolve('cot --token secret', context)
    assert before.candidates[0].token == 'cat'
    event = feedback.record_display(before, context, ts=100)
    feedback.record_selection(event, 1, ts=110)
    feedback.record_outcome(event, 'error', ts=111)
    assert model.resolve('cot', {'cwd': '/project-a', 'ts': 105}).candidates[0].token == 'cat'
    assert model.resolve('cot', {'cwd': '/project-a', 'ts': 120}).candidates[0].token == 'cut'
    rows = memory.query('SELECT original_token,candidates_json,selected_token,outcome FROM command_corrections')
    assert rows[0][0] == 'cot' and rows[0][2:] == ('cut', 'error')
    assert 'secret' not in str(rows)
    assert '--token' not in str(rows)


def test_project_context_improves_only_relevant_preferences(memory):
    feedback = CorrectionFeedbackStore(memory)
    model = resolver(('cat', 'cut'), feedback=feedback)
    for project, desired in [('/project-a', 'cat'), ('/project-b', 'cut')]:
        result = model.resolve('cot', {'cwd': project, 'ts': 100})
        event = feedback.record_display(result, {'cwd': project}, ts=100)
        index = next(i for i, c in enumerate(result.candidates) if c.token == desired)
        feedback.record_selection(event, index, ts=110)
    assert model.resolve('cot', {'cwd': '/project-a', 'ts': 120}).candidates[0].token == 'cat'
    assert model.resolve('cot', {'cwd': '/project-b', 'ts': 120}).candidates[0].token == 'cut'


def test_ignored_and_manual_edits_are_ambiguous_evidence(memory):
    feedback = CorrectionFeedbackStore(memory)
    model = resolver(('cat', 'cut'), feedback=feedback)
    session = CorrectionSession(model)
    session.update('cot secret')
    session.invalidate()
    session.update('cut secret')
    assert memory.query('SELECT accepted,manual_token FROM command_corrections') == [(None, 'cut')]
    assert model.resolve('cot').candidates[0].token == 'cat'


@pytest.mark.parametrize('before,after,expected', [
    ('gti status --secret abc', 'git status --secret def', 'git'),
    ('git statsu --secret abc', 'git status --secret def', 'status'),
    ('/exec gti status --secret abc', '/exec git status --secret def', 'git'),
])
def test_manual_feedback_uses_original_correction_grammar(memory, before, after, expected):
    session = CorrectionSession(resolver(feedback=CorrectionFeedbackStore(memory)))
    session.update(before)
    session.update(after)
    assert memory.query('SELECT manual_token FROM command_corrections')[0][0] == expected


def test_sessions_bind_full_arguments_and_reject_stale_or_reused_selection():
    session = CorrectionSession(resolver())
    session.update('gti status')
    token, revision = session.token, session.revision
    with pytest.raises(ValueError):
        session.commit('gti reset --hard', token=token, revision=revision, candidate_index=0)
    assert session.commit('gti status', token=token, revision=revision, candidate_index=0) == 'git status'
    with pytest.raises(ValueError):
        session.commit('gti status', token=token, revision=revision, candidate_index=0)
    session.update('gti status')
    token, revision = session.token, session.revision
    session.update('gti log')
    with pytest.raises(ValueError):
        session.commit('gti status', token=token, revision=revision, candidate_index=0)


def test_keep_original_is_explicit_and_not_negative_feedback(memory):
    feedback = CorrectionFeedbackStore(memory)
    session = CorrectionSession(resolver(feedback=feedback))
    session.update('gti status')
    assert session.commit('gti status', token=session.token, revision=session.revision) == 'gti status'
    assert memory.query('SELECT accepted FROM command_corrections') == [(None,)]


def test_forget_invalidates_every_session_and_preference(memory):
    store_a, store_b = CorrectionFeedbackStore(memory), CorrectionFeedbackStore(memory)
    model = resolver(('cat', 'cut'), feedback=store_a)
    first, second = CorrectionSession(model), CorrectionSession(resolver(feedback=store_b))
    first.update('cot')
    first.commit('cot', token=first.token, revision=first.revision, candidate_index=1)
    assert model.resolve('cot').candidates[0].token == 'cut'
    second.update('gti status')
    token, revision = second.token, second.revision
    store_a.forget()
    assert model.resolve('cot').candidates[0].token == 'cat'
    with pytest.raises(ValueError):
        second.commit('gti status', token=token, revision=revision, candidate_index=0)


def test_disabled_logging_disables_learning_and_storage(memory):
    enabled = [False]
    store = CorrectionFeedbackStore(memory, enabled=lambda: enabled[0])
    session = CorrectionSession(resolver(feedback=store))
    session.update('gti status')
    session.commit('gti status', token=session.token, revision=session.revision, candidate_index=0)
    assert memory.query('SELECT * FROM command_corrections') == []


def test_forget_before_removes_only_older_feedback(memory):
    feedback = CorrectionFeedbackStore(memory)
    result = resolver().resolve('gti status')
    feedback.record_display(result, ts=100)
    feedback.record_display(result, ts=200)
    feedback.forget(before=150)
    assert memory.query('SELECT ts FROM command_corrections') == [(200.0,)]


def test_privacy_summary_has_no_argument_values():
    assert learning_text('python train.py --api-key SECRET') == 'python'
    assert learning_text('/exec python train.py --api-key SECRET') == '/exec python'
    assert learning_text('git status --token SECRET') == 'git status'
    assert learning_text('/save SECRET') == '/save'
