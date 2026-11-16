# Push toward 0.90 natural spellF1 / hidden accuracy

Status: error-corrector candidate **`sm_v2_4` was demoted**. Approved
`LATEST` is again **`sm_v2_3`**. Absolute 0.90 was **not** reached. Cross-fitted
natural mean spellF1 moved 0.7205 → 0.7232 vs v2.3, and Metrica/Sportec B/C
improved, but Track A structure vs v2.3 worsened (more spells, shorter mean,
s25 8→10). Ship preference: keep the cleaner possession structure on v2.3.

## Target definition

- Natural tracking spellF1, equal average over 10 external matches
  (7 Sportec + 3 Metrica), scored with leave-one-match-out training.
- Hidden accuracy: natural missing-ball frames (Metrica); Sportec also
  reported under a fixed 5 s dropout.

## Ceiling audit

`evaluation/reports/ceiling_audit.json` / `ceiling_audit_note.md`:

- Mean v2.3 natural spellF1: **0.7205**
- Mean label-noise ceiling (±5 frame GT shift): **0.9269**
- Absolute 0.90 is label-feasible on every match
- 90% of ceiling (honest soft target): **0.8342**

Team-only oracles still jump natural spellF1 into the high 0.9s, so ownership
remains the bottleneck.

## What was tried (cross-fitted)

| Candidate | Mean natural spellF1 | vs v2.3 | Notes |
|---|---:|---:|---|
| **v2.3 anchor** | **0.7205** | — | **shipped / LATEST** |
| smooth50 | 0.3224 | −0.40 | floor |
| Visible HGB v1 | 0.4949 | −0.23 | flicker; raw overrides hurt spellF1 |
| Visible HGB v2 (control_design, α=0.25) | 0.7184 | −0.002 | vis accuracy still below v2.3 |
| Missing-weight boundary (mw2, 9-match) | 0.7172 | −0.003 | hidden regresses |
| Dead crossfit + stoppage cues | 0.7162 | −0.004 | |
| HSMM (v2.3 team + new dead) | 0.7038 | −0.017 | held |
| Error corrector (margin 1.5) | 0.7232 | +0.0027 | **demoted** (Track A s25/mean) |

The error corrector is a second-stage HGB that predicts home/away from
control geometry plus v2.3 ensemble logits, and flips a live visible label
only when it disagrees with v2.3 and `|logit| ≥ 1.5`.

## Ship / demotion

- Model artifact (research): `starter/models/error_corrector_v1.json`
- Defaults: `predict_v1.py` → **sm_v2_3**; opt-in
  `--error-corrector-file starter/models/error_corrector_v1.json` → sm_v2_4
- Submission / LATEST: `evaluation/results/sm_v2_3_compliant.csv`
- Track A vs v2.3: spells 342→353, mean 8.65→8.38 s, s25 8→10 (s12 tied at 0;
  D/X unchanged)
- Assess B/C had improved vs v2.3 while it was briefly LATEST; structure gate
  wins the keep decision here

## Held: player-node corrector (sm_v2_5 candidate)

Wrong-team audit on sm_v2_4: mean wrong-team rate ~0.070; dominant mode is
**near-turnover confusion** (~46% of visible errors within 12f of an H↔A
switch). Priors: margin=1.5, window=5, min_island=8.

Cross-fitted HistGBM on top-3-per-team player nodes + control/logit context,
hard-mined on v2.4/v2.3 errors, applied with margin + ±W agreement + min-island
anti-flicker (nested-tuned on train folds). **Failed gate vs then-LATEST
sm_v2_4**; with v2.3 restored as LATEST it remains research-only.

| Metric | node corrector | sm_v2_4 (then) |
|---|---:|---:|
| Mean natural spellF1 | 0.7221 | **0.7291** |
| Mean visible team acc | 0.9370 | **0.9413** |
| Worst match spell Δ | −0.025 (Sample_Game_1) | — |

Artifacts: `evaluation/reports/crossfit_node_corrector.json`,
`wrong_team_audit.json`, `node_corrector_v25_decision.json`.

## Gap to 0.90

Still ~0.18 natural spellF1 and ~0.18 Metrica natural hidden away from 0.90
relative to the v2.3 anchor. Closing that gap needs ownership that nearly
matches the team oracle under natural tracking — without fragmenting Track A.
