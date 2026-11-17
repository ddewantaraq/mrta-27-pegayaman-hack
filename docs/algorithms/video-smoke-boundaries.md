# BAR-ATH video smoke — rule boundaries (frozen before any run)

Written **before** the smoke runs. Nothing here may be edited after
seeing smoke output; any change is a new, separately logged experiment.

## What is allowed (organiser rules)

- Own detector on provided video: RULES.md "Run your own detector…
  as long as they are produced by an algorithm"; pre-trained public
  models inspectable by organisers (FootAndBall, MIT, weights sha256
  `42fbc1f457fc34d804b403e26138524fc40facde303ccd7b4be73121e8da4cd5`).
- Video as algorithm input + automatic checks on provided inputs:
  team clarification (TEAM.md), which takes precedence.

## Boundaries (hard)

1. **No calibration on the provided game.** Detector, thresholds
   (ball/player 0.7, repo defaults), segment and kill criterion are
   frozen here from external evidence (ISSIA a-ext) only. Smoke output
   must never be used to adjust them on BAR-ATH.
2. **No hand labels, no visual checking.** Nobody watches output
   against video; nobody corrects a detection. Algorithm output only.
3. **Local only.** No frame, crop or extract leaves this machine
   (no cloud AI services per TEAM.md).
4. **Agent sees aggregates only.** The smoke script processes video
   and ATD in code; its stdout/report contain counts and quantiles
   only. Per-frame detector positions are held in memory, never
   written to disk, never committed or published (derived frame-level
   tracking exports are banned from publication by RULES.md).
5. **Smoke is a pipeline check, not evidence of accuracy.** ATD is
   itself a CV output from the same video; agreement cannot validate
   accuracy and must not select anything.

## Pre-registered smoke

- Segment: tactical video frames **11185–13184** (2000 frames; start
  of the published first-half graded block, DATA.md — chosen without
  any ground truth).
- Detector: FootAndBall `fb1`, weights above, thresholds 0.7/0.7.
- Comparison: detector centre → `unproject(H[frame])` → metres
  (105.3×68.0) vs ATD ball where ATD present.
- Kill criterion: median disagreement **> 2 m** ⇒ pipeline/homography
  broken, stop the direction. Otherwise report aggregates only:
  ATD present rate, detector present rate, median/p90 disagreement,
  % > 1.5 m.
