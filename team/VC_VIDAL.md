# Vidal-Codina possession vs sm_v1_5

**Date:** 2026-11-09  
**Decision:** not kept. Default stays compliant `sm_v1_6`.

The candidate replaces “nearest player, sustained for 1 s” with the possession step in Vidal-Codina, Evans, El Fakir, and Billingham, *Automatic event detection in football using tracking data*, Sports Engineering, 2022 ([arXiv:2202.00804](https://arxiv.org/abs/2202.00804), §2.3 only). Dead ball, missing-ball hold, backdate, gap-bridge mid, short-spell merge, and the half-time kickoff rule are unchanged.

## What was implemented

On each visible frame:

- **Possession zone.** Closest player within `R_pz` of the ball is a candidate touch.
- **Duel zone.** If both teams have a player within `R_dz`, the frame is a duel. The paper lets the duel win, so nobody is awarded the ball. Our rule then keeps the last team.
- **Gain.** A same-player run becomes a real possession only if the ball changes direction from the start of the run to the end, or changes speed on any frame. A straight fly-by is not a turnover. Direction and speed thresholds are the published databallpy defaults of this method: 10 degrees and 5 m/s. The paper’s own epsilons are in the supplement and were not used. Ball coordinates are Savitzky-Golay smoothed (window 7, order 2) on each visible stretch, as the paper specifies.
- From a validated gain, that team keeps the ball through loose frames until the other team gains. The existing 25-frame confirm and backdate still apply.

Radii were not searched on BAR-ATH. The three published settings were scored on Metrica only:

| Setting | Source |
|---------|--------|
| pz 0.5 m, dz 1.0 m | paper, provider A |
| pz 1.0 m, dz 1.0 m | paper, providers B and C |
| pz 1.5 m, dz 1.5 m | databallpy default possession zone; duel zone at least as large |

## External result (the keep test)

Metrica Sample Game 1 and 2, event-derived labels, natural missing plus 2/5/10/30 s dropouts. Mean over 2 games × 5 dropouts. `sm_v1_5` reproduced the logged 0.3701 / 0.5102, so the harness matches the 7 Nov run.

| Candidate | spellF1 | hidden-frame team acc | macroF1 | pred spells | mean dur s |
|-----------|---------|------------------------|---------|-------------|------------|
| **vc pz 1.0 / dz 1.0** | **0.4247** | 0.4703 | 0.6912 | 377 | 10.94 |
| vc pz 1.5 / dz 1.5 | 0.4119 | 0.4662 | 0.6898 | 363 | 11.35 |
| vc pz 0.5 / dz 1.0 | 0.4060 | 0.4597 | 0.6830 | 369 | 11.16 |
| **sm_v1_5** | 0.3701 | **0.5102** | **0.7018** | 365 | 11.29 |
| smooth50-equivalent | 0.3210 | 0.1528 | 0.5920 | 491 | 7.46 |
| sm_v1_3 | 0.3141 | 0.5052 | 0.6856 | 376 | 10.95 |

Leave-one-game-out on spellF1, pz 1.0 m: Sample Game 1 0.431 vs 0.379, Sample Game 2 0.419 vs 0.361. Both games beat `sm_v1_5`. Hidden-frame accuracy does not: 0.545 vs 0.610 and 0.395 vs 0.410.

The hidden-frame gap is small on natural missing and grows with the dropout:

| Dropout | sm_v1_5 spell / hidden | vc pz 1.0 spell / hidden |
|---------|------------------------|--------------------------|
| natural | 0.401 / 0.479 | 0.490 / 0.463 |
| 2 s | 0.361 / 0.556 | 0.391 / 0.539 |
| 5 s | 0.362 / 0.529 | 0.401 / 0.475 |
| 10 s | 0.368 / 0.514 | 0.417 / 0.448 |
| 30 s | 0.359 / 0.473 | 0.425 / 0.427 |

A gain needs a visible change in the ball’s path. On a long hidden stretch the rule holds the last team and often never sees the turnover, so gap-bridge has no new team to place in the dark. `sm_v1_5` still switches from whoever is simply near the ball when it reappears. That is why spellF1 rises (fewer false touches while the ball is visible) and hidden-frame team accuracy falls.

Zone coverage on Metrica, ball-present frames: about 53% inside the 1.0 m possession zone, about 3–4% duels. On BAR-ATH tactical ATD the same radii hit only **23.6%** of ball-present frames (duels 0.7%). The paper’s 1 m zone is tight for this tracking.

## Local structure (not a keep driver)

Graded frames only. Dead-ball counts do not move, because the dead-ball rules were not changed. Compared with `baseline_smooth50` (594 spells, mean 7.24 s, 124 spells under 25 frames) every row below is still the clearer possession signal.

| Model | spells | mean s | short &lt;12f | short &lt;25f | D | X | X after D |
|-------|--------|--------|---------------|---------------|---|---|-----------|
| sm_v1_5 | 421 | 11.92 | 6 | 16 | 33212 | 177 | 0 |
| sm_v1_6 (default) | 416 | 11.83 | 6 | 16 | 35660 | 177 | 0 |
| vc pz 1.0 on v1.5 | 385 | 13.02 | 4 | 11 | 33212 | 281 | 0 |
| vc pz 1.0 on v1.6 | 380 | 12.94 | 4 | 11 | 35660 | 281 | 0 |
| smooth50 | 594 | 7.24 | 62 | 124 | 51368 | 0 | 0 |

Local structure is better than `sm_v1_5`: fewer spells, longer mean, fewer short islands, same dead-ball count. Graded `X` rises by about 100 frames. That is not enough to offset the external hidden-frame miss.

The public 5-minute scorer was not run. It is not the keep test.

## Why it is not the new default

Team rule for an external-data change: beat the reference on spellF1 **and** hidden-frame team accuracy, on both games. Vidal-Codina at the best published radius wins spellF1 on both games and loses hidden-frame accuracy on both, especially when the ball is missing for 10–30 s. About 41% of graded BAR-ATH frames have no ball, so that miss is the one that matters here.

`sm_v1_6` stays the default. The possession-zone gain is available locally with `--vc-possession` (pz 1.0 m, dz 1.0 m) and is not the submission path.
