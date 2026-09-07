# Terminal UI with fuzzy commands, anticipator, and safe built-ins

import os, re, yaml, json, difflib, time, threading
from rich import print as rprint
from rich.panel import Panel
from prompt_toolkit import PromptSession

from core.llm import LLMClient
from core.memory import Memory
from core.brain import Brain
from core.logger import make_logger
from core.actions import actions, set_scheduler, set_memory, set_logger, set_safe_mode_action
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
        '/dream - run reflection',
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
        '/exec "python your_script.py" [cwd] - run local venv python',
        '/write <path> "text" - write a file',
        '/read <path> - read a file',
        'ls - list directory',
        'tree - show a recursive directory view',
        '/task_payload \"{...json...}\" <when> - schedule a payload action at a time',
    ]), title='IntuitionOS Help'))

def fuzzy_slash(base:str)->str:
    # Known slash commands for fuzzy matching
    known=['/help','/exit','/memory','/dream','/save','/recall','/actions','/config','/reload','/tasks','/done','/delete','/snooze','/hw','/task_payload','/safe','/exec','/write','/read']
    # Return exact match if present
    if base in known:
        return base
    # Use difflib to find nearest command
    m=difflib.get_close_matches(base, known, n=1, cutoff=0.55)
    return m[0] if m else base

def bootstrap():
    # Load config and collaborators
    cfg=load_config()
    sys_prompt=read_text(cfg.get('system_prompt_path','config/system_prompt.txt'))
    schema=read_json(cfg.get('planner_schema_path','config/planner_schema.json'))

    mem=Memory(cfg.get('memory_db_path','data/intuition.db'))
    logger=make_logger(cfg.get('log_path','data/log.txt'))
    set_logger(logger)
    set_memory(mem)

    llm=LLMClient(cfg.get('backend','ollama'), cfg.get('model','gpt-oss:20b'), cfg.get('temperature',0.2), cfg.get('max_tokens',600))
    brain=Brain(llm, mem, sys_prompt, schema, logger=logger)

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
            actions.register('led_strip', lambda: None)  # no op alias
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

    return cfg, brain, mem, sched

def make_anticipator(brain, cfg):
    # Predictor decides what to warm
    def predict(buf:str):
        t=buf.strip()
        if not t:
            return {'confidence':0.0}
        if t=='tree' or t.startswith('tree '):
            return {'intent':'tree','text':t,'confidence':0.9}
        if t=='ls':
            return {'intent':'ls','text':t,'confidence':0.9}
        if t.startswith('read file '):
            return {'intent':'read_file','text':t,'confidence':0.85}
        return {'intent':'plan','text':t,'confidence':0.65}

    # Prewarm runs the action quietly
    def prewarm(intent:dict):
        t=intent.get('text','')
        kind=intent.get('intent')
        if kind=='tree':
            val=actions.call('list_tree', path='.')
            return (t, {'reply': str(val)[:4000]})
        if kind=='ls':
            val=actions.call('list_dir', path='.')
            return (t, {'reply': str(val)[:4000]})
        if kind=='read_file':
            path=t[len('read file '):].strip()
            val=actions.call('read_file', path=path)
            return (t, {'reply': str(val)[:4000]})
        if kind=='plan':
            plan=brain.plan_dryrun(t)
            return (t, {'plan': plan.get('plan', [])})
        return None

    a=cfg.get('anticipation',{}) or {}
    ant = Anticipator(
        predict_fn=predict,
        prewarm_fn=prewarm,
        enabled=bool(a.get('enabled',True)),
        debounce_ms=int(a.get('debounce_ms',180)),
        match_threshold=float(a.get('match_threshold',0.6))
    )
    ant.start()
    return ant

def main():
    # Bootstrap everything
    cfg, brain, mem, sched = bootstrap()
    # Print banner
    rprint(Panel.fit('IntuitionOS v1.0 - sandboxed exec + anticipator + fuzzy commands - type /help', title='IntuitionOS'))

    # Create prompt session
    session=PromptSession('> ')
    # Make anticipator
    ant=make_anticipator(brain, cfg)
    # _ant_ref lets /reload swap the anticipator without losing the buffer hook
    _ant_ref = [ant]
    session.default_buffer.on_text_changed += lambda buf: _ant_ref[0].update_buffer(buf.text)

    # Main REPL loop
    while True:
        try:
            user=session.prompt('io> ').strip()
        except (EOFError, KeyboardInterrupt):
            rprint('\nbye.')
            break

        if not user:
            continue

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
            rprint('bye.')
            break
        if user=='/help':
            show_help(); continue
        if user=='/config':
            rprint(cfg); continue
        if user=='/reload':
            try:
                sched.stop()
                cfg, brain, mem, sched = bootstrap()
                _ant_ref[0].stop()
                ant = make_anticipator(brain, cfg)
                _ant_ref[0] = ant
                session.default_buffer.on_text_changed += lambda buf: _ant_ref[0].update_buffer(buf.text)
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
            rprint(brain.plan_dryrun("reflection")); continue
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
                tid=int(user.split(' ',1)[1]); rprint(actions.call('delete_task', task_id=tid))
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
                rprint(actions.call('run_local', cmd=cmd, cwd=cwd))
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

        # Default to LLM brain
        out = brain.step(user)
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
