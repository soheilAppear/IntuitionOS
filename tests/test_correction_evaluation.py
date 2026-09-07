"""The new correction claim has a measured regression gate, like prediction."""
from eval.command_resolution import run, showcase


def test_chronological_correction_evaluation():
    report = run()
    legacy, baseline, learned = report['results']
    assert baseline['top1'] > legacy['top1']
    assert learned['top1'] > baseline['top1']
    assert learned['top3'] >= learned['top1']
    assert all(row['incorrect_changes_to_valid'] == 0 for row in report['results'])
    assert report['unsupported_inputs_preserved']


def test_showcase_includes_unchanged_and_stale_examples():
    output = showcase()
    assert 'git status' in output
    assert 'python train.py --lr 0.001' in output
    assert 'Available command' in output
    assert 'BLOCKED' in output
