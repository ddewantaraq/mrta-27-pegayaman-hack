# Gap-edge rules on the 1 m possession zone

**Date:** 2026-11-09  
**Decision:** not kept. Default stays compliant `sm_v1_6`.

Two additions to the Vidal-Codina gain, both at the published possession zone of 1.0 m and duel zone of 1.0 m. The zone was not widened. While the ball stays visible, a touch still becomes a turnover only if the ball's path changes. The new rules apply only in the first control window (25 frames, about 1 s) after a live missing gap:

- **reappear.** If the nearby team (nearest player, majority when the duel is contested) is the other team, that team may open a switch without a path change. The existing gap-bridge then places the switch in the middle of the gap.
- **edge.** Same window, and only when the ball's direction on the last visible frame and the first visible frame differ by more than 10°. The angle is measured on the visible edges. The missing coordinates are not used.

## External result

Metrica Sample Game 1 and 2, same dropouts as the `sm_v1_5` harness. Mean over 2 games × 5 dropouts.

| Candidate | spellF1 | hidden-frame team acc |
|-----------|---------|------------------------|
| vc reappear | **0.4273** | 0.4785 |
| vc edge | 0.4254 | 0.4730 |
| vc plain (path change only) | 0.4247 | 0.4703 |
| **sm_v1_5** | 0.3701 | **0.5102** |

SpellF1 is higher than `sm_v1_5` on both games for every variant. Hidden-frame team accuracy is lower on both games for every variant.

| Game | sm_v1_5 spell / hidden | reappear | edge |
|------|------------------------|----------|------|
| Sample Game 1 | 0.379 / 0.610 | 0.426 / 0.549 | 0.428 / 0.548 |
| Sample Game 2 | 0.361 / 0.410 | 0.428 / 0.408 | 0.423 / 0.398 |

By dropout, mean of the two games. Reappear lifts the hidden-frame number a little above plain Vidal-Codina and never reaches `sm_v1_5`. The direction requirement removes some of that lift.

| Dropout | sm_v1_5 spell / hidden | plain | reappear | edge |
|---------|------------------------|-------|----------|------|
| natural | 0.401 / 0.479 | 0.490 / 0.463 | 0.487 / 0.467 | 0.485 / 0.461 |
| 2 s | 0.361 / 0.556 | 0.391 / 0.539 | 0.396 / 0.543 | 0.396 / 0.541 |
| 5 s | 0.362 / 0.529 | 0.401 / 0.475 | 0.408 / 0.480 | 0.404 / 0.474 |
| 10 s | 0.368 / 0.514 | 0.417 / 0.448 | 0.419 / 0.457 | 0.416 / 0.449 |
| 30 s | 0.359 / 0.473 | 0.425 / 0.427 | 0.427 / 0.446 | 0.426 / 0.441 |

One second of "whoever is near the ball" at the moment the ball returns is not enough to recover the team that took over during a long gap. The path-change test still decides most of the timeline, and many dark turnovers never show up as a sustained nearby team in that first second. Requiring a 10° turn across the gap rejects some of the turnovers the nearby-team window would have kept.

## Local structure

Graded frames, possession zone 1.0 m. Dead-ball counts match the model they build on. On this match, reappear and edge produced the same graded counts: the direction gate did not move a label.

| Model | spells | mean s | short &lt;25f | D | X |
|-------|--------|--------|---------------|---|---|
| sm_v1_5 | 421 | 11.92 | 16 | 33212 | 177 |
| sm_v1_6 (default) | 416 | 11.83 | 16 | 35660 | 177 |
| vc plain | 385 | 13.02 | 11 | 33212 | 281 |
| vc reappear and vc edge | 388 | 12.92 | 11 | 33212 | 281 |
| same two, plus half-time kickoff | 383 | 12.84 | 11 | 35660 | 281 |

## Why it is not the default

The keep rule is both metrics, both games. SpellF1 passes. Hidden-frame team accuracy does not, on either game, for either gap rule. `sm_v1_6` stays the submission path.

Local flags, off by default: `--vc-possession --vc-gap reappear` or `--vc-possession --vc-gap edge`, with the 1.0 m zone.
