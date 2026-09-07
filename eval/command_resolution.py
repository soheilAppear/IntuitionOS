"""Chronological correction benchmark and a reproducible, non-executing showcase.

The fixed vocabulary is a declared test environment, not a claim that its tools
are installed. Each interaction is scored before its explicit feedback arrives.
Warm latency also samples the real installed shell's command catalog.
"""
from __future__ import annotations

import argparse
import difflib
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

from core.command_resolver import (
    CommandResolver, CorrectionFeedbackStore, CorrectionSession,
    IntuitionCommandProvider, KNOWN_COMMANDS, StaticCommandProvider,
    create_default_resolver,
)
from core.memory import Memory

VOCABULARY = ['git', 'python', 'pytest', 'npm', 'node', 'docker', 'code',
              'gcc', 'gcp', 'stat', 'status', 'deploy', 'build', 'echo']


def fixture_resolver(feedback=None):
    return CommandResolver([IntuitionCommandProvider(),
                            StaticCommandProvider(VOCABULARY, case_sensitive=False)],
                           shell='cmd', feedback=feedback)


def correction_cases():
    pairs = [
        ('/hlp', '/help'), ('/taks', '/tasks'), ('/tsaks', '/tasks'),
        ('/memroy', '/memory'), ('/recal note', '/recall note'),
        ('/drema', '/dream'), ('/exti', '/exit'), ('/cofnig', '/config'),
        ('/capabilites', '/capabilities'), ('/calibraton', '/calibration'),
        ('/jounral', '/journal'), ('gti status', 'git status'),
        ('pyhton train.py --lr 0.001', 'python train.py --lr 0.001'),
        ('git statsu', 'git status'), ('git chekcout topic', 'git checkout topic'),
        ('git comimt -m "two words"', 'git commit -m "two words"'),
        ('nmp install', 'npm install'), ('pytesst -q', 'pytest -q'),
        ('  pyhton\t"train file.py"  --lr 0.001  ', '  python\t"train file.py"  --lr 0.001  '),
        ('/exec gti status', '/exec git status'),
    ]
    # Both gcc/gcp are plausible for gco. The project preference is supplied
    # by explicit human selections *after* each observation, including failures.
    pairs += [('gco file.txt', 'gcp file.txt')] * 6
    return pairs


VALID = ['/help', '/done 1', '/tasks', 'git status', 'python -V', 'GIT status',
         'gcc -v', 'gcp file.txt', 'stat file.txt', 'status', 'git -C "a b" status',
         'echo "gti pyhton statsu"', 'ls', 'tree']
UNSUPPORTED = ['gti status | more', 'gti status > out.txt', 'gti status && echo ok',
               'gti status; echo ok', 'pyhton "unfinished', 'pyhton $value',
               'pyhton %SECRET%', 'python train.py --lrr 0.001', 'no-such-command-xyz']


def legacy(text):
    """Faithful pre-milestone behavior: one slash candidate, split/join args."""
    if not text.startswith('/'):
        return []
    parts = text.split()
    if not parts or parts[0] in KNOWN_COMMANDS:
        return []
    match = difflib.get_close_matches(parts[0], KNOWN_COMMANDS, n=1, cutoff=0.55)
    return [' '.join([match[0], *parts[1:]])] if match else []


def _latencies(resolve, samples):
    for text in samples:
        resolve(text)
    elapsed = []
    for _ in range(12):
        for text in samples:
            begin = time.perf_counter_ns()
            resolve(text)
            elapsed.append((time.perf_counter_ns() - begin) / 1_000_000)
    ordered = sorted(elapsed)
    return {"median_ms": round(statistics.median(elapsed), 4),
            "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
            "samples": len(elapsed)}


def run():
    with tempfile.TemporaryDirectory(prefix='intuition-correction-eval-') as directory:
        mem = Memory(str(Path(directory) / 'evaluation.db'))
        try:
            store = CorrectionFeedbackStore(mem)
            deterministic = fixture_resolver()
            personal = fixture_resolver(store)
            counts = {name: [0, 0] for name in ('legacy', 'deterministic', 'personalized')}
            misses = {name: [] for name in counts}
            base_time = 1_700_000_000.0
            for i, (typed, expected) in enumerate(correction_cases()):
                ctx = {'cwd': '/fixture/project-a', 'ts': base_time + i * 10}
                baseline = deterministic.resolve(typed, ctx)
                learned = personal.resolve(typed, ctx)
                predictions = {'legacy': legacy(typed),
                               'deterministic': [c.text for c in baseline.candidates],
                               'personalized': [c.text for c in learned.candidates]}
                for name, options in predictions.items():
                    counts[name][0] += bool(options and options[0] == expected)
                    counts[name][1] += expected in options[:3]
                    if not options or options[0] != expected:
                        misses[name].append({'typed': typed, 'expected': expected, 'suggested': options[:3]})
                # Feedback is not visible to the resolver until the next step.
                shown = store.record_display(learned, ctx, ts=ctx['ts'])
                chosen = next((j for j, c in enumerate(learned.candidates) if c.text == expected), None)
                if chosen is not None:
                    store.record_selection(shown, chosen, ts=ctx['ts'] + 1)
                    store.record_outcome(shown, 'error' if i % 3 == 0 else 'ok', ts=ctx['ts'] + 2)
            summaries = []
            for name in counts:
                resolver = deterministic if name == 'deterministic' else personal
                resolve = legacy if name == 'legacy' else lambda text, r=resolver: [c.text for c in r.resolve(text).candidates]
                changed_valid = sum(bool(resolve(text)) for text in VALID)
                summaries.append({'name': name, 'cases': len(correction_cases()),
                                  'top1': counts[name][0] / len(correction_cases()),
                                  'top3': counts[name][1] / len(correction_cases()),
                                  'incorrect_changes_to_valid': changed_valid, 'valid_cases': len(VALID),
                                  'warm_latency': _latencies(resolve, [p[0] for p in correction_cases()[:20]]),
                                  'top1_misses': misses[name]})
            actual = create_default_resolver()
            installed_latency = _latencies(actual.resolve, ['gti status', 'pyhton train.py --lr 0.001', 'git statsu', '/taks'])
            return {'source': 'fixed synthetic command catalog; chronological score-before-feedback replay',
                    'environment': {'os': platform.platform(), 'python': platform.python_version(), 'shell': actual.shell},
                    'results': summaries, 'installed_catalog_warm_latency': installed_latency,
                    'unsupported_inputs_preserved': all(not deterministic.resolve(t).candidates for t in UNSUPPORTED),
                    'limitations': ['Synthetic preferences, not a user study.',
                                    'Legacy matcher exposed one candidate; its top-3 equals top-1.',
                                    'Warm timing excludes first filesystem/PowerShell catalog discovery and UI rendering.',
                                    'POSIX behavior is unit-tested where simulated; actual shells are reported separately.']}
        finally:
            mem.close()


def showcase():
    resolver = fixture_resolver()
    lines = ['IntuitionOS correction showcase (suggestions only; no commands executed)']
    for text in ['gti status', 'pyhton train.py --lr 0.001', 'git statsu',
                 'stat file.txt', 'gco file.txt', 'gti status | more']:
        result = resolver.resolve(text)
        lines.append(f'{text}\n  {result.status}: ' +
                     (' | '.join(c.text for c in result.candidates) if result.candidates else result.reason))
    session = CorrectionSession(resolver)
    session.update('gti status')
    old = session.snapshot()
    session.update('gti status --short')
    try:
        session.commit(old['original'], token=old['token'], revision=old['revision'], candidate_index=0)
    except ValueError as error:
        lines.append(f'Stale selection after editing:\n  BLOCKED: {error}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--showcase', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.showcase:
        print(showcase())
        return 0
    report = run()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(report['source'])
        print('matcher          top-1    top-3    valid changed    warm p50/p95 ms')
        for result in report['results']:
            latency = result['warm_latency']
            print(f"{result['name']:<16} {result['top1']:>6.1%}   {result['top3']:>6.1%}    "
                  f"{result['incorrect_changes_to_valid']}/{result['valid_cases']}             "
                  f"{latency['median_ms']:.3f}/{latency['p95_ms']:.3f}")
        print('Installed catalog:', report['installed_catalog_warm_latency'])
        print('Environment:', report['environment'])
    if args.check:
        old, baseline, learned = report['results']
        healthy = (baseline['top1'] > old['top1'] and learned['top1'] > baseline['top1']
                   and all(r['incorrect_changes_to_valid'] == 0 for r in report['results'])
                   and report['unsupported_inputs_preserved'])
        return 0 if healthy else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
