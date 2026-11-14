# sm_v1_10: calibrated mixed-provider boosted live/dead model

## What changed

The live/dead head now combines ball geometry with ball-independent player
movement. Numeric gradient-boosted trees capture interactions that the previous
linear model missed. The team-control core, published halftime/kickoff handling
and submission timeline remain the same. Existing dead-bridge and short-spell
rules run **after** the learned D overlay so it cannot introduce untreated tiny
H/A fragments. No challenge labels or video were used.

Training uses only Sportec **J03WOH** and Metrica **Sample_Game_1**, the same
external fitting matches as sm_v1_9. The feature matrix has 173 columns:
41 ball/context features plus 132 movement/context features. Permutation-invariant
movement summaries do not require persistent player identities. Methods and
rejected candidates are in [the research log](research-round-2026-11-14.md).

The classifier uses 150 boosting iterations, learning rate 0.1, 15 leaves,
minimum 100 samples/leaf, L2=10, 63 bins and seed 73. No random frame-wise
early-stopping split is used. After balanced-class fitting, a training-only
prior correction of **-0.579630983906959 log-odds** restores the observed
training class prior (35.9017508683% D). This correction is not fitted to any
validation match or target-match statistic. The existing two-state temporal
decoder runs at 5 Hz and returns labels on the 25 Hz submission timeline.

The prior protected hidden-H/A override is not used in this model. Removing it
is supported by the external validation below; it is not a guarantee against
per-frame regressions. Rollbacks remain available.

## External validation

All five matches below were excluded from parameter fitting. They have been
used to choose between candidate variants, so these are validation results,
not an untouched final test. Scores average natural observations and 41%
2/5/10/30-second block dropouts across the complete timeline, including dead
ball. Training uses a different seed and 5/15-second blocks.

| Match | sm_v1_9 spellF1 | sm_v1_10 spellF1 | Old hidden H/A accuracy | New hidden H/A accuracy |
|---|---:|---:|---:|---:|
| Sportec J03WMX | 0.402103 | 0.459776 | 0.580085 | 0.678237 |
| Sportec J03WPY | 0.461009 | 0.527212 | 0.566622 | 0.639340 |
| Sportec J03WQQ | 0.416153 | 0.490340 | 0.649513 | 0.678726 |
| Sportec J03WR9 | 0.424553 | 0.501658 | 0.614164 | 0.665169 |
| Metrica Sample_Game_2 | 0.393622 | 0.426366 | 0.642444 | 0.664460 |
| Mean across five matches | 0.419488 | 0.481070 | 0.610566 | 0.665186 |

For continuity, the **original Metrica live-window dropout suite** also
improves: mean spellF1 0.496131 -> 0.571573 and hidden accuracy 0.678216 ->
0.749873. This mean includes training Game 1. Excluded Game 2 alone improves
spellF1 0.489793 -> 0.549717 and hidden accuracy 0.616913 -> 0.687964.
Keep these absolute scores separate from the full-timeline suite above.

The Sportec references use directly supplied frame-level BallStatus and
BallPossession. Metrica's historical reference derives labels from events and
omits some foul stoppages. Its historical proxy is retained unchanged here;
a separate reference-repair experiment does not overwrite old results.

These numbers are not the hidden challenge score. Clean-track scores above
0.70 in individual cases do not imply 0.70 under realistic missing-ball gaps,
nor establish a winning cutoff. The challenge has no live leaderboard or
submission feedback. External references and tracking quality differ from
the challenge, and all known validation matches have now informed development.

## Challenge structure and material risk

No challenge GT was used to select these counts:

| Model | Possessions | Mean duration, s | <12f | <25f | Graded D | Graded X |
|---|---:|---:|---:|---:|---:|---:|
| sm_v1_9 | 331 | 16.8413 | 3 | 8 | 19214 | 281 |
| sm_v1_10 | 294 | 10.0673 | 0 | 2 | 84581 | 281 |
| smooth50 | 594 | 7.2383 | 62 | 124 | 51368 | 0 |

The much larger D count is a material change. External gains across both
providers support this candidate, but target-domain transfer is unverified.
Low short-spell counts alone are not evidence of correctness.

## Reproduction

Inference uses only NumPy and pandas; scikit-learn is **not** required. Numeric
trees and attribution are packaged in `starter/models/boosted_dead_v1.json`.
Portable NumPy logits matched scikit-learn to 1e-12 on 16,758 training-feature
probes, including missing-value routes. Full ATD regeneration matched the
experimental candidate CSV exactly. No external data downloads are needed
for inference.

```sh
python -m pip install -r requirements-inference.txt
python starter/predict_v1.py --out evaluation/results/sm_v1_10_compliant.csv
python evaluation/validate_submission.py evaluation/results/sm_v1_10_compliant.csv
python starter/predict_v1.py --dead-model logistic --out evaluation/results/sm_v1_9_rollback.csv
python starter/predict_v1.py --no-learned-dead --out evaluation/results/sm_v1_8_rollback.csv
```

Training and validation (optional research dependencies and external assets):

```sh
python -m pip install -r requirements-research.txt
python starter/external/fetch_metrica.py
python starter/external/fetch_sportec.py --matches all
python starter/external/boosted_motion_experiment.py --train --calibrate-prior --match J03WMX
python starter/external/export_boosted_model.py --mixed-training --calibrate-prior
python starter/external/evaluate_approved.py --candidate 1.10 --reference 1.9
python -m unittest discover -s starter/external -p 'test_*.py'
```

The training exporter produces
`evaluation/reports/boosted_motion_mixed_v1_prior.json`; the packaged model
contains the identical parameters. Rebuilding all feature caches requires
several gigabytes of downloaded XML; these and all frame-level outputs remain
gitignored. Broadcast fusion and the learned team-control experiments are
not enabled in the approved pipeline.

## Attribution

Sportec: Bassek, Rein, Weber & Memmert (2025),
[An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y),
[figshare v1](https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0.
Metrica: sample tracking/event data provided by
[Metrica Sports sample-data](https://github.com/metrica-sports/sample-data).

Player movement and temporal context were motivated by
[Lang et al. (2022)](https://doi.org/10.1038/s41598-022-19948-1), and the
factorized state/team approach by
[Borghesi et al. (2023)](https://doi.org/10.1016/j.eswa.2023.119780).
This implementation is not a reproduction of either paper's model or metric.
