"""Shared match list and dataset helpers for the 0.90 natural push."""
from pathlib import Path
import numpy as np

from motion_dead_experiment import dataset
from player_node_data import load_player_nodes
from eval_external import score_one, kwargs_for_version, run_labeller, smooth50_labels

MATCHES = [
    'J03WOH', 'J03WN1', 'J03WOY', 'J03WMX', 'J03WPY', 'J03WQQ', 'J03WR9',
    'Sample_Game_1', 'Sample_Game_2', 'Sample_Game_3',
]
METRICA = [m for m in MATCHES if m.startswith('Sample_Game_')]
SPORTEC = [m for m in MATCHES if m.startswith('J03')]
REPORTS = Path('evaluation/reports')


def load_match(game):
    f, gt, graded = dataset(game)
    nodes = load_player_nodes(game)
    return dict(game=game, features=f, gt=np.asarray(gt), graded=np.asarray(graded, bool),
                nodes=nodes)


def fold_train_games(holdout):
    return [g for g in MATCHES if g != holdout]


def natural_hide(features):
    return ~np.asarray(features['ball_present'], bool)


def score_natural(gt, pred, features):
    hide = natural_hide(features)
    metrics = score_one(gt, pred, hide)
    metrics['n_natural_missing_live'] = int(
        (hide & np.isin(gt, ['H', 'A'])).sum())
    return metrics


def mean_ignore_nan(values):
    vals = [v for v in values if v == v]
    return float(np.mean(vals)) if vals else float('nan')


def summarize_rows(rows, name_key='candidate'):
    """Equal-match means of natural spellF1 and hidden accuracy."""
    by = {}
    for row in rows:
        by.setdefault(row[name_key], []).append(row)
    out = {}
    for name, items in by.items():
        out[name] = dict(
            mean_spell_f1=float(np.mean([r['spell_f1_mean'] for r in items])),
            mean_hidden_team_acc=mean_ignore_nan(
                [r['hidden_team_acc'] for r in items]),
            n_matches=len(items),
            n_hidden_defined=sum(
                1 for r in items if r['hidden_team_acc'] == r['hidden_team_acc']),
        )
    return out


def predict_version(features, graded, version=(2, 3)):
    return run_labeller(features, graded, kwargs_for_version(*version))


def predict_smooth50(features):
    return smooth50_labels(features)
