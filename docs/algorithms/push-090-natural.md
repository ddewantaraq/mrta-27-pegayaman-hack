# Push toward 0.90 natural spellF1 / hidden accuracy

Status: **`sm_v2_7` promoted** (Pareto vs sm_v2_6 on Tracks A/B/C/D). Absolute
0.90 was **not** reached. `sm_v2_5` is demoted: the leakage check found its
corrector was trained on evaluation matches. `sm_v2_6` replaced it with an
honest fit-only corrector, and `sm_v2_7` adds the event-first touch model.
Margin-1.5 `sm_v2_4` remains demoted (Track A).

## Ship: sm_v2_7 (event-first touch model)

Pipeline after the v2.6 corrector (`starter/event_first_model.py`):

1. **Touch candidates:** frames where the ball turns ≥ 25° or its speed jumps
   ≥ 2 m/s while the nearest player is within 1.5 m.
2. **Touch-team classifier:** 22 features (touch geometry, v2.3 ensemble team
   logits and the v2.6 reference sign at offsets 0 and ±10 frames). Trained
   with scikit-learn HGB and exported to NumPy trees; parity max |Δ| = 0.0.
3. **Decode:** two-state Viterbi over touches (switch cost 0.0).
4. **Conservative overwrite:** a touch segment's visible frames are rewritten
   only when the decode disagrees with v2.6 and |logit| ≥ 1.0. Short spells
   are merged afterwards (18 frames, unchanged).

Training: fit set (J03WOH, J03WN1, J03WOY, Sample_Game_3) plus the 16
SkillCorner training-split matches (sample weight 1.0). Selection: fit LOMO +
4 grouped folds over the SkillCorner training split; Track D is never seen.
Without the SkillCorner folds the selected model regressed Track D
(0.4328 < 0.4365), so that variant was rejected.

| Track | sm_v2_7 | sm_v2_6 | sm_v2_3 |
|---|---:|---:|---:|
| A spells / mean_s / s25 | **339 / 8.73 / 7** | 342 / 8.65 / 8 | 342 / 8.65 / 8 |
| B Metrica spellF1 / hidden | **0.6654** / 0.8014 | 0.6474 / 0.8014 | 0.6443 / 0.8014 |
| C Sportec spellF1 / hidden | **0.6218** / 0.8069 | 0.6187 / 0.8069 | 0.6161 / 0.8069 |
| D SkillCorner spellF1 / hidden | **0.4381 / 0.7272** | 0.4365 / 0.7271 | — |

Hidden ties are at 4 dp (Sportec net −2 frames for v2.7; Metrica Game 1 −1
frame for v2.6), the same tie convention as the v2.5 decision. Rollbacks:
`--no-event-first` → v2.6; `--error-corrector-file
starter/models/error_corrector_v1.json --error-corrector-margin 3
--no-event-first` → v2.5; `--no-error-corrector` → v2.3. All reproduce the
stored CSVs byte-for-byte. Decisions: `evaluation/reports/sm_v2_7_decision.json`,
`honest_anchor_decision.json`.

Not shipped: SkillCorner-augmented corrector (fit LOMO +0.0011; held because it
changes the event-first inputs), SkillCorner-augmented boundary head (−0.0033),
3-seed boundary weight soup (−0.0018), and a larger minimum spell (short spells
are not the bottleneck; `short_spell_ceiling.json`).

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
| v2.5 (demoted) | 0.7220 | +0.0015 | margin 3.0; corrector trained on eval matches |
| Boundary ladder best (rung_b) | 0.7208 | +0.0003 | held; loses vs v2.5 (−0.0023) |
| v2.3 anchor | 0.7205 | — | prior LATEST |
| smooth50 | 0.3224 | −0.40 | floor |
| Visible HGB v1 | 0.4949 | −0.23 | flicker; raw overrides hurt spellF1 |
| Visible HGB v2 (control_design, α=0.25) | 0.7184 | −0.002 | vis accuracy still below v2.3 |
| Missing-weight boundary (mw2, 9-match) | 0.7172 | −0.003 | hidden regresses |
| Dead crossfit + stoppage cues | 0.7162 | −0.004 | |
| HSMM (v2.3 team + new dead) | 0.7038 | −0.017 | held |
| Error corrector (margin 1.5) | 0.7232 | +0.0027 | demoted (Track A s25/mean) |

The error corrector is a second-stage HGB that predicts home/away from
control geometry plus v2.3 ensemble logits, and flips a live visible label
only when it disagrees with v2.3 and `|logit| ≥ 1.5`.

## Demoted: sm_v2_5 (margin 3.0, leaky corrector)

- Model artifact: `starter/models/error_corrector_v1.json`
- Was the default before v2.6 (margin **3.0**); now pinned as
  `V25_CORRECTOR_PATH` / `V25_CORRECTOR_MARGIN` for rollback only.
- Submission (no longer LATEST): `evaluation/results/sm_v2_5_compliant.csv`
- Track A vs v2.3: spells 342→**341**, mean 8.65→**8.68** s, s25 tied at 8
- Track B: spellF1 0.6443→**0.6509**, hidden ≥ (0.8014)
- Track C: spellF1 0.6161→**0.6191**, hidden ≥ (0.8069)
- Demoted sm_v2_4 remains margin **1.5** research (`V24_CORRECTOR_MARGIN`)

## Prior demotion (sm_v2_4)

Margin 1.5 briefly beat B/C natural but Track A worsened (spells 353, mean
8.38, s25 10). Not LATEST.

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

## Held: sparse switch editor (sm_v2_5 candidate)

Switch-timing audit on sm_v2_3: matched switches have median |offset| ~4f;
~32% of GT switches are missed (94% of those sit on contested peaks); ~19% of
pred switches are false. Contested-band HistGBM editor (retarget / delete /
insert) on frozen v2.3 labels, sparse ±R apply.

| Metric | switch editor | sm_v2_3 |
|---|---:|---:|
| Mean natural spellF1 | 0.6688 | **0.7205** |
| Worst match spell Δ | −0.116 | — |

Track A cand over-merged (183 spells vs 342). **Keep sm_v2_3.** Artifacts:
`evaluation/reports/crossfit_switch_editor.json`, `switch_timing_audit.json`,
`switch_editor_v25_decision.json`.

## Held: insert-only contested switches

Safer follow-up to the switch editor: insert-only at contested peaks where
left/right ensemble polarity disagrees; flip only the disagreeing side;
top-K + min-island/sep. Nested tune on mean Δ over 3 train matches.

| Metric | insert-only | sm_v2_3 |
|---|---:|---:|
| Mean natural spellF1 | 0.7191 | **0.7205** |
| Worst match spell Δ | −0.0077 | — |

Boundary-logit gated probe also failed to clear the gate. **Keep sm_v2_3.**
Artifacts: `evaluation/reports/crossfit_insert_only.json`,
`insert_only_v25_decision.json`.

## Held: boundary transition retrain ladder

v2.5 still misses ~32% of GT H↔A switches (~94% contested). Deterministic
10-fold boundary crossfit control reproduced ~0.7204. Training ladder
(boundary loss strength, sharper σ/truncation, contested loss boost,
transition `ball_context` extras) all failed vs sm_v2_5 (~0.723); best was
rung_b at 0.7208 / Δv2.5 −0.0023 with hidden regression. No ship model.
**Keep sm_v2_5.** See `evaluation/reports/boundary_v2_decision.json`.

## Gap to 0.90

Still ~0.18 natural spellF1 away from 0.90 relative to the v2.5 anchor
(~0.722). Closing that gap needs ownership that nearly matches the team
oracle under natural tracking — without fragmenting Track A. Post-hoc
patches and light boundary-loss/feature retrains appear saturated; larger
gains likely need a different transition architecture.
