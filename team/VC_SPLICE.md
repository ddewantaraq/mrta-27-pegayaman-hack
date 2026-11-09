# Visible Vidal-Codina, missing frames stay sm_v1_5

**Date:** 2026-11-09  
**Decision:** kept. This is the default, `sm_v1_7` (`sm_v1_6` plus this splice).

Two full passes, then one join:

- Ball coordinate present: Vidal-Codina gain. Possession zone 1.0 m, duel zone 1.0 m. A touch counts only if the ball's direction changes by more than 10° or its speed changes by more than 5 m/s. A duel does not take the ball.
- Ball missing: the `sm_v1_5` label, including gap-bridge mid. On the submission path the same pass also includes the half-time kickoff rule (`sm_v1_6`).
- One short-spell merge on the joined row.

The zone was not widened. The gap-edge variants from [VC_GAP.md](VC_GAP.md) are not part of this default.

## External result

Metrica Sample Game 1 and 2, natural missing plus 2/5/10/30 s dropouts. Mean over 2 games × 5 dropouts. The `sm_v1_5` row matches the logged 0.3701 / 0.5102.

| Candidate | spellF1 | hidden-frame team accuracy |
|-----------|---------|----------------------------|
| **vc splice** | **0.4286** | **0.5117** |
| sm_v1_5 | 0.3701 | 0.5102 |

Both games:

| Game | sm_v1_5 spell / hidden | splice |
|------|------------------------|--------|
| Sample Game 1 | 0.379 / 0.610 | 0.437 / 0.611 |
| Sample Game 2 | 0.361 / 0.410 | 0.420 / 0.412 |

Hidden-frame accuracy is the `sm_v1_5` number, plus a small gain from the short-spell merge on the seam. It is not lower on either game.

## Local structure

Graded frames. Dead-ball counts match the base model. Compared with `baseline_smooth50` (594 spells, mean 7.24 s, 124 spells under 25 frames) the splice is still the longer possession signal.

| Model | spells | mean s | short &lt;25f | D | X |
|-------|--------|--------|---------------|---|---|
| sm_v1_5 | 421 | 11.92 | 16 | 33212 | 177 |
| sm_v1_6 | 416 | 11.83 | 16 | 35660 | 177 |
| splice on v1.5 | 411 | 12.20 | 11 | 33212 | 281 |
| splice on v1.6 (default) | 406 | 12.11 | 11 | 35660 | 281 |

## Rollback

`--no-vc-splice` returns to `sm_v1_6`.
