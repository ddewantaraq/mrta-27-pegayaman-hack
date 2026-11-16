"""Crossfit sparse switch editor on frozen sm_v2_3 labels.

Edits only H↔A boundaries (retarget / delete / insert) in contested neighborhoods.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import (
    MATCHES, REPORTS, load_match, fold_train_games, score_natural,
    summarize_rows, predict_version)
from eval_external import write_csv
from state_machine import _merge_short_spells
from pipeline_090 import v23_team_logits
from contested_band_features import (
    FEATURE_VERSION, collect_candidate_events, ha_switches)
from switch_timing_audit import match_switches, MATCH_RADIUS


HARD_WEIGHT = 3.0
DEFAULT_R = 25
DEFAULT_MARGIN = 1.0
DEFAULT_MIN_ISLAND = 18
MERGE_MINIMUM = 18


def boundary_logits_for(features, graded, nodes, baseline):
    from numpy_set_temporal import load_parameters
    from numpy_boundary_ownership import predict_dual
    from boundary_ensemble_model import DEFAULT_BOUNDARY_MODEL_PATH
    _, boundary = predict_dual(
        features, graded, nodes, baseline,
        load_parameters(DEFAULT_BOUNDARY_MODEL_PATH))
    return boundary


def label_pred_events(pred_sw, gt_sw, R, match_radius=MATCH_RADIUS):
    """Per pred switch: ('delete',0) or ('keep_or_retarget', offset)."""
    matched = match_switches(pred_sw, gt_sw, radius=match_radius)
    out = []
    false_set = set(matched['false_pos'])
    # map pred -> offset
    offset_map = dict(zip(matched['matched_pred'], matched['offsets']))
    for p in pred_sw:
        p = int(p)
        if p in false_set:
            out.append(('delete', 0))
        else:
            off = int(offset_map.get(p, 0))
            off = int(np.clip(off, -R, R))
            out.append(('retarget', off))
    return out, matched


def label_insert_events(insert_idx, gt_sw, R):
    labels = []
    gt_sw = np.asarray(gt_sw)
    for p in insert_idx:
        if len(gt_sw) and np.min(np.abs(gt_sw - p)) <= R:
            labels.append(1)
        else:
            labels.append(0)
    return np.asarray(labels, np.int32)


def training_matrices(games, R=DEFAULT_R):
    xs_sw, ys_sw, ws_sw = [], [], []
    xs_ins, ys_ins, ws_ins = [], [], []
    for game in games:
        data = load_match(game)
        f, gt, graded, nodes = data['features'], data['gt'], data['graded'], data['nodes']
        baseline = predict_version(f, graded, (2, 3))
        team_logits = v23_team_logits(f, graded, nodes, baseline)
        blogits = boundary_logits_for(f, graded, nodes, baseline)
        ev = collect_candidate_events(
            f, graded, baseline, team_logits, blogits, retarget_radius=R)
        gt_sw = ha_switches(gt, graded)
        actions, matched = label_pred_events(ev['pred_switches'], gt_sw, R)
        # Pred-switch multiclass: 0=DELETE, 1=KEEP (offset0), 2=DELETE? 
        # Encode: class 0 = DELETE, class 1 = KEEP, classes 2..2R+1 = retarget nonzero
        # Simpler encoding: n_classes = 2 + 2R  -> 0=DELETE, 1=KEEP, 2..2R+1 map to offsets
        y = []
        w = []
        for kind, off in actions:
            if kind == 'delete':
                y.append(0)
                w.append(HARD_WEIGHT)
            elif off == 0:
                y.append(1)
                w.append(1.0)
            else:
                # map offset to class: 2 + (off + R) but skip 0 which would be R
                # offsets -R..-1,1..R -> indices 0..2R-1 then +2
                if off < 0:
                    y.append(2 + (-off - 1))          # 2..R+1
                else:
                    y.append(2 + R + (off - 1))       # 2+R .. 2R+1
                w.append(HARD_WEIGHT if abs(off) >= 8 else 1.5)
        if len(ev['x_pred']):
            xs_sw.append(ev['x_pred'])
            ys_sw.append(np.asarray(y, np.int32))
            ws_sw.append(np.asarray(w, np.float64))
        ins_y = label_insert_events(ev['insert_candidates'], gt_sw, R)
        if len(ev['x_insert']):
            xs_ins.append(ev['x_insert'])
            ys_ins.append(ins_y)
            hard = ins_y == 1
            ws_ins.append(np.where(hard, HARD_WEIGHT, 1.0).astype(np.float64))
        print('switch events', game, 'pred', len(ev['pred_switches']),
              'insert', len(ev['insert_candidates']),
              'false', len(matched['false_pos']),
              'miss', len(matched['missed']), flush=True)
    Xsw = np.concatenate(xs_sw) if xs_sw else np.zeros((0, 1))
    ysw = np.concatenate(ys_sw) if ys_sw else np.zeros(0, np.int32)
    wsw = np.concatenate(ws_sw) if ws_sw else np.zeros(0)
    Xin = np.concatenate(xs_ins) if xs_ins else np.zeros((0, 1))
    yin = np.concatenate(ys_ins) if ys_ins else np.zeros(0, np.int32)
    win = np.concatenate(ws_ins) if ws_ins else np.zeros(0)
    return Xsw, ysw, wsw, Xin, yin, win


def _n_switch_classes(R):
    # 0=DELETE, 1=KEEP, 2..(2R+1) = nonzero offsets (-R..-1, 1..R) = 2R classes
    return 2 + 2 * R


def _decode_switch_class(cls, R):
    cls = int(cls)
    if cls == 0:
        return 'delete', 0
    if cls == 1:
        return 'keep', 0
    # 2 .. R+1 -> -1 .. -R
    if 2 <= cls <= R + 1:
        return 'retarget', -(cls - 1)
    # R+2 .. 2R+1 -> +1 .. +R
    return 'retarget', cls - (R + 1)


def train(games, R=DEFAULT_R, seed=73):
    Xsw, ysw, wsw, Xin, yin, win = training_matrices(games, R=R)
    n_classes = _n_switch_classes(R)
    # Ensure all classes present by padding tiny synthetic if needed
    present = set(ysw.tolist()) if len(ysw) else set()
    if len(Xsw) and len(present) < n_classes:
        # add tiny zero rows for missing classes (won't matter much)
        miss = [c for c in range(n_classes) if c not in present]
        pad = np.zeros((len(miss), Xsw.shape[1]), np.float64)
        Xsw = np.concatenate([Xsw, pad])
        ysw = np.concatenate([ysw, np.asarray(miss, np.int32)])
        wsw = np.concatenate([wsw, np.full(len(miss), 1e-3)])
    switch_model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=200, max_leaf_nodes=31,
        min_samples_leaf=10, l2_regularization=1., max_bins=127,
        early_stopping=False, random_state=seed)
    insert_model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=150, max_leaf_nodes=31,
        min_samples_leaf=10, l2_regularization=1., max_bins=127,
        early_stopping=False, random_state=seed + 1)
    with threadpool_limits(limits=4):
        if len(Xsw):
            switch_model.fit(Xsw, ysw, sample_weight=wsw)
        if len(Xin) and len(np.unique(yin)) > 1:
            insert_model.fit(Xin, yin, sample_weight=win)
        elif len(Xin):
            insert_model.fit(Xin, yin)
    return dict(switch=switch_model, insert=insert_model, R=R,
                n_features_switch=int(Xsw.shape[1]) if len(Xsw) else 0,
                n_features_insert=int(Xin.shape[1]) if len(Xin) else 0)


def _export_trees(estimator):
    trees = []
    for iteration in estimator._predictors:
        # multiclass: one tree per class per iteration
        class_trees = []
        for tree in iteration:
            nodes = tree.nodes
            fields = dict(feature='feature_idx', threshold='num_threshold', value='value',
                          left='left', right='right', leaf='is_leaf',
                          missing_left='missing_go_to_left')
            class_trees.append({key: nodes[source].tolist() for key, source in fields.items()})
        trees.append(class_trees)
    baseline = estimator._baseline_prediction
    # baseline shape (1, n_classes) or (n_classes,)
    base = np.asarray(baseline, dtype=np.float64).reshape(-1).tolist()
    return trees, base


def export_models(bundle, path, training_matches, apply_cfg=None):
    sw_trees, sw_base = _export_trees(bundle['switch'])
    ins_trees, ins_base = _export_trees(bundle['insert'])
    payload = dict(
        feature_version=FEATURE_VERSION,
        R=int(bundle['R']),
        n_classes_switch=_n_switch_classes(bundle['R']),
        switch=dict(baseline=sw_base, trees=sw_trees,
                    n_features=int(bundle['switch'].n_features_in_)),
        insert=dict(baseline=ins_base, trees=ins_trees,
                    n_features=int(bundle['insert'].n_features_in_)),
        training_matches=list(training_matches),
        apply=apply_cfg or {},
    )
    Path(path).write_text(json.dumps(payload, separators=(',', ':')) + '\n')


def _boosted_multiclass_logits(x, packed):
    """Return (n, n_classes) logits from exported multiclass HGB trees."""
    n = len(x)
    n_classes = len(packed['baseline'])
    out = np.tile(np.asarray(packed['baseline'], float), (n, 1))
    for iteration in packed['trees']:
        for c, tree in enumerate(iteration):
            # single-tree boosted value contribution
            from boosted_dead_model import boosted_logits as _bl
            # boosted_logits expects binary-style baseline scalar; use per-tree value sum
            # Manual tree walk:
            nodes = tree
            active = np.zeros(n, np.int32)
            values = np.zeros(n, np.float64)
            pending = np.ones(n, bool)
            feat = np.asarray(nodes['feature'], np.int32)
            thr = np.asarray(nodes['threshold'], float)
            val = np.asarray(nodes['value'], float)
            left = np.asarray(nodes['left'], np.int32)
            right = np.asarray(nodes['right'], np.int32)
            leaf = np.asarray(nodes['leaf'], np.int32)
            missing_left = np.asarray(nodes['missing_left'], np.int32)
            while pending.any():
                idx = np.flatnonzero(pending)
                node = active[idx]
                is_leaf = leaf[node] != 0
                values[idx[is_leaf]] = val[node[is_leaf]]
                pending[idx[is_leaf]] = False
                go = idx[~is_leaf]
                if len(go) == 0:
                    break
                node = active[go]
                f = feat[node]
                xv = x[go, f]
                miss = ~np.isfinite(xv)
                to_left = np.where(miss, missing_left[node] != 0, xv <= thr[node])
                active[go] = np.where(to_left, left[node], right[node])
            out[:, c] += values
    return out


def load_packed(path):
    data = json.loads(Path(path).read_text())
    if data.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unsupported switch editor schema')
    def convert(block):
        trees = []
        for iteration in block['trees']:
            converted = []
            for tree in iteration:
                converted.append({
                    k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
                    for k, v in tree.items()})
            trees.append(converted)
        return dict(baseline=block['baseline'], trees=trees)
    return dict(
        R=int(data['R']),
        switch=convert(data['switch']),
        insert=convert(data['insert']),
        apply=data.get('apply') or {},
        n_classes_switch=int(data.get('n_classes_switch', _n_switch_classes(int(data['R'])))),
    )


def _spell_bounds(labels, graded, i):
    """Inclusive [start, end) of the H/A spell containing frame i."""
    n = len(labels)
    team = labels[i]
    a = i
    while a > 0 and graded[a - 1] and labels[a - 1] == team:
        a -= 1
    b = i + 1
    while b < n and graded[b] and labels[b] == team:
        b += 1
    return a, b


def apply_retarget(labels, graded, cut, offset):
    """Move H↔A cut at `cut` by offset frames; rewrite contiguous band only."""
    labels = labels.copy()
    n = len(labels)
    new_cut = int(np.clip(cut + offset, 1, n - 1))
    if new_cut == cut:
        return labels
    left_team = labels[cut - 1]
    right_team = labels[cut]
    if left_team not in ('H', 'A') or right_team not in ('H', 'A'):
        return labels
    lo, hi = (cut, new_cut) if new_cut > cut else (new_cut, cut)
    # When moving cut right: frames [cut, new_cut) become left_team
    # When moving cut left: frames [new_cut, cut) become right_team
    if new_cut > cut:
        labels[lo:hi] = left_team
    else:
        labels[lo:hi] = right_team
    return labels


def apply_delete(labels, graded, cut):
    """Remove a false cut by extending the longer neighboring spell across it."""
    labels = labels.copy()
    if cut <= 0 or cut >= len(labels):
        return labels
    left = labels[cut - 1]
    right = labels[cut]
    if left not in ('H', 'A') or right not in ('H', 'A') or left == right:
        return labels
    a_l, b_l = _spell_bounds(labels, graded, cut - 1)
    a_r, b_r = _spell_bounds(labels, graded, cut)
    # Prefer merging into the longer side
    if (b_l - a_l) >= (b_r - a_r):
        labels[a_r:b_r] = left
    else:
        labels[a_l:b_l] = right
    return labels


def apply_insert(labels, graded, cut, min_island):
    """Insert a single H↔A cut at `cut` if both sides stay >= min_island."""
    labels = labels.copy()
    n = len(labels)
    cut = int(np.clip(cut, 1, n - 1))
    if labels[cut] not in ('H', 'A') or labels[cut - 1] not in ('H', 'A'):
        return labels
    if labels[cut] != labels[cut - 1]:
        return labels  # already a boundary
    team = labels[cut]
    a, b = _spell_bounds(labels, graded, cut)
    left_len = cut - a
    right_len = b - cut
    if left_len < min_island or right_len < min_island:
        return labels
    other = 'A' if team == 'H' else 'H'
    # Insert means the right side becomes the other team only if ensemble wanted it;
    # Without a sign, flip the shorter side.
    if right_len <= left_len:
        labels[cut:b] = other
    else:
        labels[a:cut] = other
    return labels


def apply_switch_editor(features, graded, nodes, model_path,
                        margin=None, R=None, min_island=None, minimum=MERGE_MINIMUM,
                        baseline=None, joblib_bundle=None):
    if baseline is None:
        baseline = predict_version(features, graded, (2, 3))
    labels = np.asarray(baseline).copy()
    packed = None
    if joblib_bundle is not None:
        bundle = joblib_bundle
        R = int(bundle['R'] if R is None else R)
        margin = DEFAULT_MARGIN if margin is None else float(margin)
        min_island = DEFAULT_MIN_ISLAND if min_island is None else int(min_island)
        use_sklearn = True
    else:
        packed = load_packed(model_path)
        R = int(packed['R'] if R is None else R)
        apply_cfg = packed.get('apply') or {}
        if margin is None:
            margin = float(apply_cfg.get('margin', DEFAULT_MARGIN))
        if min_island is None:
            min_island = int(apply_cfg.get('min_island', DEFAULT_MIN_ISLAND))
        use_sklearn = False

    team_logits = v23_team_logits(features, graded, nodes, labels)
    blogits = boundary_logits_for(features, graded, nodes, labels)
    ev = collect_candidate_events(
        features, graded, labels, team_logits, blogits, retarget_radius=R)
    counts = dict(retarget=0, delete=0, insert=0, keep=0)

    # Pred-switch actions
    if len(ev['x_pred']):
        if use_sklearn:
            proba = bundle['switch'].predict_proba(ev['x_pred'])
            classes = bundle['switch'].classes_
            pred_cls = classes[np.argmax(proba, axis=1)]
            conf = np.max(proba, axis=1)
        else:
            logits = _boosted_multiclass_logits(ev['x_pred'], packed['switch'])
            # softmax confidence
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            proba = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
            pred_cls = np.argmax(proba, axis=1)
            conf = np.max(proba, axis=1)
        for i, cut in enumerate(ev['pred_switches']):
            if conf[i] < margin:
                counts['keep'] += 1
                continue
            action, offset = _decode_switch_class(int(pred_cls[i]), R)
            if action == 'delete':
                labels = apply_delete(labels, graded, int(cut))
                counts['delete'] += 1
            elif action == 'retarget' and offset != 0:
                labels = apply_retarget(labels, graded, int(cut), int(offset))
                counts['retarget'] += 1
            else:
                counts['keep'] += 1

    # Insert actions
    if len(ev['x_insert']):
        if use_sklearn:
            if hasattr(bundle['insert'], 'predict_proba'):
                proba = bundle['insert'].predict_proba(ev['x_insert'])
                # positive class
                classes = list(bundle['insert'].classes_)
                if 1 in classes:
                    pos = classes.index(1)
                    conf = proba[:, pos]
                    want = conf >= margin
                else:
                    want = np.zeros(len(ev['x_insert']), bool)
                    conf = np.zeros(len(ev['x_insert']))
            else:
                want = np.zeros(len(ev['x_insert']), bool)
        else:
            logits = _boosted_multiclass_logits(ev['x_insert'], packed['insert'])
            if logits.shape[1] == 1:
                # binary HGB still exports 1 tree column sometimes; treat as logit
                conf = 1 / (1 + np.exp(-logits[:, 0]))
                want = conf >= margin
            else:
                e = np.exp(logits - logits.max(axis=1, keepdims=True))
                proba = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
                # class 1 = insert
                conf = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                want = conf >= margin
        for i, cut in enumerate(ev['insert_candidates']):
            if not want[i]:
                continue
            before = labels.copy()
            labels = apply_insert(labels, graded, int(cut), int(min_island))
            if not np.array_equal(before, labels):
                counts['insert'] += 1

    return _merge_short_spells(labels, graded, minimum), counts


def nested_tune(train_games, model_path, priors, joblib_bundle=None):
    margins = sorted({float(priors.get('margin', DEFAULT_MARGIN)), 0.45, 0.6, 0.75})
    Rs = sorted({int(priors.get('retarget_radius', DEFAULT_R)), 18, 25})
    islands = sorted({int(priors.get('min_island', DEFAULT_MIN_ISLAND)), 18})
    nested_hold = train_games[-1]
    data = load_match(nested_hold)
    baseline = predict_version(data['features'], data['graded'], (2, 3))
    best = None
    for R in Rs:
        for margin in margins:
            for min_island in islands:
                pred, _ = apply_switch_editor(
                    data['features'], data['graded'], data['nodes'], model_path,
                    margin=margin, R=R, min_island=min_island, baseline=baseline,
                    joblib_bundle=joblib_bundle)
                metrics = score_natural(data['gt'], pred, data['features'])
                cand = dict(margin=margin, R=R, min_island=min_island,
                            spell=metrics['spell_f1_mean'])
                if best is None or cand['spell'] > best['spell']:
                    best = cand
    best['nested_hold'] = nested_hold
    return best


def load_priors(path=None):
    path = Path(path or (REPORTS / 'switch_timing_audit.json'))
    if path.exists():
        return json.loads(path.read_text()).get('priors', {})
    return dict(retarget_radius=DEFAULT_R, margin=DEFAULT_MARGIN, min_island=DEFAULT_MIN_ISLAND)


def crossfit(matches=None, priors=None):
    matches = matches or MATCHES
    priors = priors or load_priors()
    R0 = int(priors.get('retarget_radius', DEFAULT_R))
    fold_dir = REPORTS / 'crossfit_switch_editor'
    fold_dir.mkdir(parents=True, exist_ok=True)
    rows, tuned = [], []
    for holdout in matches:
        train_games = fold_train_games(holdout)
        joblib_path = fold_dir / ('switch_' + holdout + '.joblib')
        json_path = fold_dir / ('switch_' + holdout + '.json')
        if not joblib_path.exists():
            print('train fold', holdout, flush=True)
            bundle = train(train_games, R=R0)
            joblib.dump(bundle, joblib_path)
            export_models(bundle, json_path, train_games)
        bundle = joblib.load(joblib_path)
        params = nested_tune(train_games, json_path, priors, joblib_bundle=bundle)
        tuned.append(dict(holdout=holdout, **params))
        print('tune', holdout, params, flush=True)
        data = load_match(holdout)
        pred, counts = apply_switch_editor(
            data['features'], data['graded'], data['nodes'], json_path,
            margin=params['margin'], R=params['R'], min_island=params['min_island'],
            joblib_bundle=bundle)
        metrics = score_natural(data['gt'], pred, data['features'])
        base = predict_version(data['features'], data['graded'], (2, 3))
        m23 = score_natural(data['gt'], base, data['features'])
        present = data['features']['ball_present']
        live = present & np.isin(data['gt'], ['H', 'A']) & np.isin(pred, ['H', 'A'])
        acc = float(np.mean(pred[live] == data['gt'][live])) if live.any() else float('nan')
        live23 = present & np.isin(data['gt'], ['H', 'A']) & np.isin(base, ['H', 'A'])
        acc23 = float(np.mean(base[live23] == data['gt'][live23])) if live23.any() else float('nan')
        rows.append(dict(
            game=holdout, candidate='switch_editor',
            retarget=counts['retarget'], delete=counts['delete'], insert=counts['insert'],
            visible_team_acc=acc, visible_team_acc_v23=acc23,
            spell_f1_v23=m23['spell_f1_mean'],
            margin=params['margin'], R=params['R'], min_island=params['min_island'],
            **metrics))
        print(holdout, counts, 'spell', round(metrics['spell_f1_mean'], 4),
              'v23', round(m23['spell_f1_mean'], 4), flush=True)
    return rows, tuned


def gate_decision(rows):
    spell = float(np.mean([r['spell_f1_mean'] for r in rows]))
    spell23 = float(np.mean([r['spell_f1_v23'] for r in rows]))
    deltas = [r['spell_f1_mean'] - r['spell_f1_v23'] for r in rows]
    worst = float(np.min(deltas))
    ok = (spell >= spell23 - 1e-12) and (worst >= -0.02)
    return dict(
        pass_crossfit=bool(ok),
        mean_spell=spell, mean_spell_v23=spell23, worst_spell_delta=worst,
        per_match_spell_delta={r['game']: float(r['spell_f1_mean'] - r['spell_f1_v23'])
                               for r in rows},
    )


def majority_apply_params(tuned, priors):
    def mode(vals):
        vals = list(vals)
        return max(set(vals), key=vals.count)
    return dict(
        margin=float(mode(t['margin'] for t in tuned)),
        R=int(mode(t['R'] for t in tuned)),
        min_island=int(mode(t['min_island'] for t in tuned)),
        retarget_radius=int(priors.get('retarget_radius', DEFAULT_R)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', action='store_true')
    parser.add_argument('--train-all', action='store_true')
    parser.add_argument('--priors', default=str(REPORTS / 'switch_timing_audit.json'))
    args = parser.parse_args()
    priors = load_priors(args.priors)
    print('priors', priors, flush=True)
    R0 = int(priors.get('retarget_radius', DEFAULT_R))
    if args.train_all:
        bundle = train(MATCHES, R=R0)
        out = REPORTS / 'switch_editor_all10'
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out / 'model.joblib')
        # ship params filled later
        export_models(bundle, out / 'switch_editor_v1.json', MATCHES,
                      apply_cfg=dict(margin=priors.get('margin', DEFAULT_MARGIN),
                                     min_island=priors.get('min_island', DEFAULT_MIN_ISLAND),
                                     R=R0))
        print('wrote', out / 'switch_editor_v1.json', flush=True)
    if args.crossfit or not args.train_all:
        rows, tuned = crossfit(priors=priors)
        write_csv(REPORTS / 'crossfit_switch_editor.csv', rows)
        decision = gate_decision(rows)
        ship_params = majority_apply_params(tuned, priors)
        summary = dict(
            aggregate=summarize_rows(rows),
            gate=decision,
            ship_params=ship_params,
            tuned=tuned,
            rows=rows,
            priors=priors,
        )
        Path(REPORTS / 'crossfit_switch_editor.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(dict(gate=decision, ship_params=ship_params,
                              aggregate=summary['aggregate']), indent=2))


if __name__ == '__main__':
    main()
