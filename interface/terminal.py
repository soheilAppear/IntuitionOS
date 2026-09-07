# Terminal UI with fuzzy commands, anticipator, and safe built-ins

import os, re, yaml, json, difflib, time, threading
from rich import print as rprint
from rich.panel import Panel
from prompt_toolkit import PromptSession

from core.llm import LLMClient
from core.memory import Memory
from core.brain import Brain
from core.logger import make_logger
from core.actions import (
    actions, set_scheduler, set_memory, set_logger, set_safe_mode_action,
    set_thresholds, undo_last, journal_recent, get_journal,
)
from core.calibration import CalibrationStore, load_thresholds, reliability
from core.capabilities import capabilities
from core.consolidation import RuleStore, consolidate, render_rules
from core.context import ContextSensor
from core.episodes import Episode, EpisodeLog, PredictionWindow
from core.predictor import Predictor, PredictorStore
from core.scheduler import Scheduler
from core.anticipator import Anticipator
from plugins.led_strip import LEDStrip
from plugins.gpu_nvml import GPUNVML

def load_config():
    # Load YAML config
    with open('config/config.yaml','r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def read_text(path:str)->str:
    # Read a whole text file
    with open(path,'r',encoding='utf-8') as f:
        return f.read()

def read_json(path:str)->dict:
    # Read a JSON file
    with open(path,'r',encoding='utf-8') as f:
        import json
        return json.load(f)

def show_help():
    # Print supported commands
    rprint(Panel.fit('\n'.join([
        '/help - show this help',
        '/exit - quit',
        '/memory - show recent memory',
        '/dream - consolidate the episode log into rules',
        '/rules [--all] - what the system believes about your habits',
        '/rules delete <id> - forget one belief',
        '/save "text" - save a note',
        '/recall "term" - search memory',
        '/actions - list actions',
        '/config - print current config',
        '/reload - reload config and prompts',
        '/tasks - list pending tasks',
        '/done <id> - mark a task done',
        '/delete <id> - delete a task',
        '/snooze <id> <15m|2h|1d> - snooze a task',
        '/hw - list hardware devices',
        '/hw schema <name> - show a device schema',
        '/safe on|off - toggle Safe Mode',
        '/undo - reverse the last reversible action',
        '/journal [n] - show recent gated actions',
        '/calibration - reliability of stated confidence',
        '/thresholds - show the cost-gated thresholds',
        '/episodes - show what the episode log has recorded',
        '/forget - erase the episode log',
        '/capabilities - list actions with their declared cost',
        '/exec "python your_script.py" [cwd] - run local venv python',
        '/write <path> "text" - write a file',
        '/read <path> - read a file',
        'ls - list directory',
        'tree - show a recursive directory view',
        '/task_payload \"{...json...}\" <when> - schedule a payload action at a time',
    ]), title='IntuitionOS Help'))

def fuzzy_slash(base:str)->str:
    # Known slash commands for fuzzy matching
    known=['/help','/exit','/memory','/dream','/save','/recall','/actions','/config','/reload','/tasks','/done','/delete','/snooze','/hw','/task_payload','/safe','/exec','/write','/read','/undo','/journal','/capabilities','/forget','/episodes','/calibration','/thresholds','/rules']
    # Return exact match if present
    if base in known:
        return base
    # Use difflib to find nearest command
    m=difflib.get_close_matches(base, known, n=1, cutoff=0.55)
    return m[0] if m else base

def run_action(name, **kwargs):
    """Dispatch through the gate, asking at the prompt if the gate says to.

    The REPL and the HUD answer a CONFIRM the same way — nothing has run when
    the parked result comes back, and the token is what unparks it — they just
    ask the question through different surfaces.
    """
    res = actions.call(name, **kwargs)
    if not (isinstance(res, dict) and res.get("needs_confirmation")):
        return res

    detail = ' '.join(f'{k}={v}' for k, v in (res.get('args') or {}).items())
    title = 'CANNOT BE UNDONE' if res.get('reversibility') == 'irreversible' else 'Confirm'
    rprint(Panel.fit(
        f"{res['capability']}  {detail}\n{res.get('reason','')}",
        title=title, border_style='red' if res.get('reversibility') == 'irreversible' else 'yellow',
    ))
    try:
        answer = input('proceed? [y/N] ').strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = 'n'
    return actions.confirm(res['token'], granted=answer in ('y', 'yes'))


def bootstrap():
    # Load config and collaborators
    cfg=load_config()
    sys_prompt=read_text(cfg.get('system_prompt_path','config/system_prompt.txt'))
    schema=read_json(cfg.get('planner_schema_path','config/planner_schema.json'))

    mem=Memory(cfg.get('memory_db_path','data/intuition.db'))
    logger=make_logger(cfg.get('log_path','data/log.txt'))
    set_logger(logger)
    set_memory(mem)

    thresholds = load_thresholds(cfg.get('thresholds'))
    set_thresholds(thresholds)

    llm=LLMClient(cfg.get('backend','ollama'), cfg.get('model','gpt-oss:20b'), cfg.get('temperature',0.2), cfg.get('max_tokens',600))
    bcfg=cfg.get('brain',{}) or {}
    brain=Brain(llm, mem, sys_prompt, schema, logger=logger,
                max_iters=int(bcfg.get('max_iters',5)),
                budget_ms=int(bcfg.get('budget_ms',20000)),
                history_turns=int(bcfg.get('history_turns',6)))

    def notify(task_id:int, title:str):
        rprint(Panel.fit(f'Remind #{task_id}: {title}', title='Reminder'))
        mem.add('reminder', f'#{task_id} {title}', tags='reminder')

    def execute(payload:dict):
        # whitelist payload execution
        allowed={'hw_call'}
        name=payload.get('action')
        kwargs=payload.get('kwargs',{})
        if name in allowed:
            res=actions.call(name, **kwargs)
            mem.add('tool', json.dumps({'scheduled':True,'action':name,'kwargs':kwargs,'result':res})[:2000])
            rprint(Panel.fit(f'Scheduled action {name} -> {res}', title='Run'))

    sched=Scheduler(db_path=cfg.get('memory_db_path','data/intuition.db'), tz=cfg.get('timezone','America/New_York'), tick_seconds=10, notify_cb=notify, execute_cb=execute)
    set_scheduler(sched)

    # Register hardware drivers
    drivers = cfg.get('hardware',{}).get('drivers',[])
    for d in drivers:
        if d.get('name')=='led_strip':
            from plugins.led_strip import LEDStrip
            from core.actions import register_driver
            register_driver(LEDStrip(simulate=d.get('simulate',True), port=d.get('port')))
        if d.get('name')=='gpu_nvml' and d.get('enabled',True):
            from plugins.gpu_nvml import GPUNVML
            from core.actions import register_driver
            register_driver(GPUNVML(enabled=True))
        if d.get('name')=='cpu_info' and d.get('enabled',True):
            from plugins.cpu_info import CPUInfo
            from core.actions import register_driver
            register_driver(CPUInfo())

    # Every submitted input is recorded whether or not the user asks. That is
    # the involuntary encoding /save lacks; it is disclosed in the README and can
    # be switched off in config.yaml.
    ecfg = cfg.get('episodes', {}) or {}
    episodes = EpisodeLog(mem, enabled=bool(ecfg.get('enabled', True)))
    sensor = ContextSensor(journal=get_journal())

    calibration_store = CalibrationStore(mem)
    rule_store = RuleStore(mem)
    pcfg = cfg.get('prediction', {}) or {}
    predictor = Predictor(store=PredictorStore(mem),
                          half_life_s=float(pcfg.get('half_life_s', 7*24*3600)),
                          min_episodes=int(pcfg.get('min_episodes', 50)),
                          calibrator=calibration_store.load(),
                          rules=rule_store)
    if predictor.seen == 0:
        # No saved state: relearn from the log rather than starting cold.
        predictor.fit(episodes.recent(limit=int(pcfg.get('replay_limit', 5000))))

    return cfg, brain, mem, sched, episodes, sensor, predictor, rule_store, calibration_store

def make_anticipator(brain, cfg, predictor=None, sensor=None):
    # What to warm now comes from the predictor rather than from four literals
    # that had never been compared to anything.
    def prewarm(prediction):
        # Speculative work runs as the anticipator, never as the user — that
        # actor is what confines it to free capabilities.
        text=prediction.action
        conf=prediction.confidence

        def warm(name, args):
            return str(actions.dispatch(name, args, actor='anticipator', confidence=conf))[:4000]

        t=text.strip()
        common={'confidence': conf, 'why': prediction.why, 'action': text}
        if t=='tree' or t.startswith('tree '):
            return (text, dict(common, reply=warm('list_tree', {'path': '.'})))
        if t=='ls':
            return (text, dict(common, reply=warm('list_dir', {'path': '.'})))
        if t.startswith('read file '):
            path=t[len('read file '):].strip()
            return (text, dict(common, reply=warm('read_file', {'path': path})))
        # Nothing cheap to precompute, but the prediction itself still has value.
        return (text, common)

    a=cfg.get('anticipation',{}) or {}
    ant = Anticipator(
        prewarm_fn=prewarm,
        predictor=predictor,
        context_fn=(lambda: sensor.snapshot()) if sensor else None,
        enabled=bool(a.get('enabled',True)),
        debounce_ms=int(a.get('debounce_ms',180)),
        match_threshold=float(a.get('match_threshold',0.6)),
        thresholds=cfg.get('thresholds'),
    )
    ant.start()
    return ant

def main():
    # Bootstrap everything
    cfg, brain, mem, sched, episodes, sensor, predictor, rules, calib_store = bootstrap()
    window = PredictionWindow()
    # Print banner
    rprint(Panel.fit('IntuitionOS v1.0 - sandboxed exec + anticipator + fuzzy commands - type /help', title='IntuitionOS'))

    # Create prompt session
    session=PromptSession('> ')
    # Make anticipator
    ant=make_anticipator(brain, cfg, predictor=predictor, sensor=sensor)
    # _ant_ref lets /reload swap the anticipator without losing the buffer hook
    _ant_ref = [ant]
    def _on_text(buf):
        _ant_ref[0].update_buffer(buf.text)
        window.note_keystroke(buf.text)

    session.default_buffer.on_text_changed += _on_text

    # Main REPL loop
    while True:
        try:
            user=session.prompt('io> ').strip()
        except (EOFError, KeyboardInterrupt):
            try:
                predictor.save()
            except Exception:
                pass
            rprint('\nbye.')
            break

        if not user:
            continue

        # Involuntary encoding: one row per submitted input, before anything is
        # dispatched, so it is recorded even if handling it raises. The REPL has
        # no ghost-hint surface, so accepted_prediction stays NULL here — it is
        # the HUD that can show a hint and therefore have one ignored.
        _signals = window.take(user)
        _ctx = sensor.snapshot()
        _episode_id = episodes.record(user, _ctx, **_signals)
        sensor.note_submission(user)
        # This submission is the ground truth for whatever was predicted a moment
        # ago, including the times the prediction was wrong.
        predictor.update(Episode(ts=time.time(), action=user, context=_ctx,
                                 keystroke_prefix=_signals.get('keystroke_prefix') or user))

        # Fuzzy only the first token for slash commands
        if user.startswith('/'):
            tokens=user.split()
            base=tokens[0]
            args=tokens[1:]
            fixed=fuzzy_slash(base)
            if fixed!=base:
                user=' '.join([fixed]+args).strip()

        # Built-ins
        if user=='/exit':
            try:
                predictor.save()
            except Exception:
                pass
            rprint('bye.')
            break
        if user=='/help':
            show_help(); continue
        if user=='/config':
            rprint(cfg); continue
        if user=='/reload':
            try:
                sched.stop()
                cfg, brain, mem, sched, episodes, sensor, predictor, rules, calib_store = bootstrap()
                _ant_ref[0].stop()
                ant = make_anticipator(brain, cfg, predictor=predictor, sensor=sensor)
                _ant_ref[0] = ant
                rprint({'result': 'reloaded'})
            except Exception as e:
                rprint(f'reload error: {e}')
            continue
        if user=='/actions':
            rprint(sorted(actions.names)); continue
        if user=='/memory':
            rows=mem.recent(limit=12)
            for _id, ts, role, text, tags in rows[::-1]:
                rprint(f'[{role}] {text}')
            continue
        if user=='/dream':
            ccfg=cfg.get('consolidation',{}) or {}
            report=consolidate(episodes.recent(limit=int(ccfg.get('window',2000))),
                               rules, llm=brain.llm,
                               min_support=int(ccfg.get('min_support',4)),
                               min_confidence=float(ccfg.get('min_confidence',0.5)),
                               calibrator=predictor.calibrator,
                               calibration_store=calib_store)
            rprint(Panel.fit(report.summary(), title='Consolidation'))
            continue
        if user.startswith('/rules'):
            parts=user.split()
            if len(parts)>=3 and parts[1]=='delete':
                try:
                    rid=int(parts[2])
                except ValueError:
                    rprint('usage: /rules delete <id>'); continue
                rprint(f'Deleted rule #{rid}.' if rules.delete(rid) else f'No rule #{rid}.')
                continue
            show_all = len(parts)>=2 and parts[1] in ('--all','all')
            rprint(render_rules(rules.all(active_only=not show_all)))
            continue
        if user.startswith('/save '):
            note=user[6:].strip().strip('"')
            mem.add('note', note, tags='note'); rprint('saved.'); continue
        if user.startswith('/recall '):
            term=user[8:].strip().strip('"')
            rows=mem.search(term, limit=12)
            for _id, ts, role, text, tags in rows[::-1]:
                rprint(f'[{role}] {text}')
            continue
        if user=='/tasks':
            rprint(actions.call('list_tasks', status='pending')); continue
        if user.startswith('/done '):
            try:
                tid=int(user.split(' ',1)[1]); rprint(actions.call('complete_task', task_id=tid))
            except Exception: rprint('usage: /done <id>')
            continue
        if user.startswith('/delete '):
            try:
                tid=int(user.split(' ',1)[1]); rprint(run_action('delete_task', task_id=tid))
            except Exception: rprint('usage: /delete <id>')
            continue
        if user.startswith('/snooze '):
            try:
                _, tid, d = user.split(' ')
                rprint(actions.call('snooze_task', task_id=int(tid), delta=d))
            except Exception: rprint('usage: /snooze <id> <15m|2h|1d>')
            continue
        if user=='/hw':
            rprint(actions.call('hw_list')); continue
        if user.startswith('/hw schema '):
            name=user.split(' ',2)[2]; rprint(actions.call('hw_schema', device=name)); continue
        if user.startswith('/task_payload '):
            try:
                _, payload_json, when = user.split(' ', 2)
                payload=json.loads(payload_json.strip('\"\''))
                rprint(actions.call('create_task', text=None, when=when, repeat='', payload=payload))
            except Exception as e:
                rprint(f"usage: /task_payload '{{...json...}}' <when>. error: {e}")
            continue
        if user=='/calibration':
            rprint(reliability(episodes.shown_predictions()).table()); continue
        if user=='/thresholds':
            for k, v in (cfg.get('thresholds') or {}).items():
                rprint(f"  {k:<14} {'never' if v is None else v}")
            continue
        if user=='/forget':
            rprint(f'Forgot {episodes.forget()} episode(s).'); continue
        if user=='/episodes':
            rows=episodes.recent(limit=15)
            if not rows:
                rprint('No episodes recorded yet.' if episodes.enabled
                       else 'Episode logging is disabled in config.yaml.')
            for e in rows:
                when=time.strftime('%H:%M:%S', time.localtime(e.ts))
                hint=''
                if e.accepted_prediction is not None:
                    hint='  hint taken' if e.accepted_prediction else f'  hint ignored ({e.predicted})'
                rprint(f'{when}  {e.action}{hint}')
            continue
        if user=='/undo':
            rprint(undo_last()); continue
        if user.startswith('/journal'):
            parts=user.split()
            limit=int(parts[1]) if len(parts)>1 and parts[1].isdigit() else 15
            rows=journal_recent(limit=limit).get('rows', [])
            if not rows:
                rprint('The journal is empty.')
            for r in rows:
                when=time.strftime('%H:%M:%S', time.localtime(r['ts']))
                mark=' (undone)' if r['undone_at'] else (' [undoable]' if r['undo'] else '')
                outcome=f"/{r['outcome']}" if r['outcome'] else ''
                rprint(f"#{r['id']} {when} {r['actor']}/{r['capability']} {r['decision']}{outcome}{mark}")
            continue
        if user=='/capabilities':
            for c in capabilities.manifest():
                flag='confirm' if c['requires_confirmation'] else ''
                rprint(f"{c['name']:<24} {c['reversibility']:<13} {flag:<8} {c['summary']}")
            continue
        if user.startswith('/safe'):
            parts=user.split()
            if len(parts)>=2:
                state=parts[1]; rprint(set_safe_mode_action(state=state))
            else:
                rprint('usage: /safe on|off')
            continue
        if user.startswith('/exec '):
            try:
                rest=user[6:].strip()
                cmd=None; cwd='.'
                if rest.startswith('"'):
                    idx=rest.find('"',1)
                    if idx!=-1:
                        cmd=rest[1:idx]; tail=rest[idx+1:].strip(); cwd=tail or '.'
                if not cmd:
                    parts=rest.split(' ',1); cmd=parts[0]; cwd=parts[1] if len(parts)==2 else '.'
                rprint(run_action('run_local', cmd=cmd, cwd=cwd))
            except Exception as e:
                rprint(f'usage: /exec "python your_script.py" [cwd]. error: {e}')
            continue
        if user.startswith('/write '):
            try:
                _, rest = user.split(' ', 1); rest=rest.strip()
                if '"' in rest:
                    path, txt = rest.split('"',1); path=path.strip()
                    if not txt.endswith('"'):
                        last=txt.rfind('"')
                        if last!=-1: txt=txt[:last+1]
                    text=txt.strip('"')
                else:
                    path, text = rest.split(' ',1)
                rprint(actions.call('write_file', path=path, text=text))
            except Exception as e:
                rprint(f'usage: /write <path> "text". error: {e}')
            continue
        if user.startswith('/read '):
            try:
                path=user.split(' ',1)[1].strip(); rprint(actions.call('read_file', path=path))
            except Exception as e:
                rprint(f'usage: /read <path>. error: {e}')
            continue

        # Shortcuts for common shell like inputs
        if user=='ls':
            rprint(actions.call('list_dir', path='.')); continue
        if user=='tree':
            rprint(actions.call('list_tree', path='.')); continue

        # Natural language remind: "remind me <title> in/at <when>"
        _m = re.match(r'^remind\s+me\s+(.+?)\s+((?:in|at)\s+\S.*)$', user, re.IGNORECASE)
        if _m:
            rprint(actions.call('create_task', text=_m.group(1).strip(), when=_m.group(2).strip()))
            continue

        # Try anticipator cache first
        pre = None
        try:
            pre = ant.try_serve(user)
        except Exception:
            pre = None
        if isinstance(pre, dict) and ('reply' in pre or 'plan' in pre):
            plan = pre.get('plan') or []
            if isinstance(plan,str): plan=[plan]
            if plan: rprint(Panel.fit('\n'.join(f'- {p}' for p in plan), title='Plan'))
            rprint(pre.get('reply',''))
            mem.add('assistant', pre.get('reply',''))
            continue

        # Default to LLM brain — a real propose/validate/execute/observe loop.
        out = brain.step(user)
        while out.get('needs_confirmation'):
            detail=' '.join(f'{k}={v}' for k,v in (out.get('args') or {}).items())
            title='CANNOT BE UNDONE' if out.get('reversibility')=='irreversible' else 'Confirm'
            rprint(Panel.fit(f"{out['capability']}  {detail}" + chr(10) + f"{out.get('reason','')}",
                             title=title,
                             border_style='red' if out.get('reversibility')=='irreversible' else 'yellow'))
            try:
                answer=input('proceed? [y/N] ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer='n'
            out = brain.resume(out['resume_token'], granted=answer in ('y','yes'))
        plan = out.get('plan') or []
        if isinstance(plan,str):
            plan_lines=[plan]
        elif isinstance(plan,list):
            plan_lines=plan
        else:
            plan_lines=[str(plan)]
        if plan_lines:
            rprint(Panel.fit('\n'.join(f'- {p}' for p in plan_lines), title='Plan'))
        rprint(out.get('reply',''))
