# Research beyond sm_v2_2

Status: boundary ensemble `sm_v2_3` is promoted and named in `LATEST`.
No challenge possession
reference or public score is used for fitting, selection or these experiments.

## Why investigate boundaries

The external-only diagnostic in `diagnose_v22.py` replaces selected decisions
with external reference labels. These are counterfactual diagnostics, not
achieved scores or deployable models.

| Natural tracking | Actual spellF1 | Reference-assisted team-only | Reference-assisted live/dead with latent team restoration |
|---|---:|---:|---:|
| Sportec WMX | .663303 | .969421 | .662727 |
| Metrica 2 | .652055 | .906539 | .705506 |
| Metrica 3 | .679070 | .904598 | .738264 |

The team-only diagnostic corrects reference H/A where the model already
predicts H/A; it leaves D/X errors. The live/dead diagnostic overlays reference
D and restores false D using the model's latent team estimate. That estimate
can still be wrong, so this is not a pure upper bound. The larger team-only
gain motivates ownership and boundary work. On natural Metrica 3, missing live
frames contain 6527 wrong-team predictions versus 2785 false-D predictions.

## Research basis

[Ishikawa et al., ASRF, WACV 2021](https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html)
separate frame classification from boundary estimation to improve action
segmentation. Our joint head adapts that idea to football ownership; it is not
a reproduction of ASRF or its benchmark scores. The temporal backbone and
smoothing build on the earlier MS-TCN-inspired ownership experiments.
[Lang et al.](https://www.nature.com/articles/s41598-022-19948-1) motivate the
separate player-motion live/dead experiment. Paper accuracies and action F1
scores are not this challenge's spellF1.

## Fit and evaluation separation

Fit only: Sportec WOH/WN1/WOY and Metrica Game 1. Development: WMX and Metrica
Game 2. Additional checks: WPY/WQQ/WR9 and Metrica Game 3. These matches have
been examined in earlier rounds; they are excluded from parameter fitting,
but are not untouched final tests. Game 3 also informed error diagnosis.

Training dropout: natural and 2/5/10/30-second blocks, seed 73, natural streams
weighted three times each artificial condition. Full-timeline evaluation:
natural plus 2/5/10/30-second blocks, seed 41. Historical Metrica live-window
dropout is reported separately. Conditions within a match are correlated;
scores below average conditions within each match and then matches equally.

## Joint ownership/boundary head

Initialize from the v2.2 residual network and add a boundary output to its
shared temporal representation. Team logits include the same fixed baseline
prior. Boundary targets are Gaussians with sigma 5 original frames, truncated
at +/-15 frames, centered on external H/A switches within the same period.
D/X transitions are not turnover targets. The boundary BCE positive weight
is 10 and its loss multiplier .2; the original team BCE and .15 temporal
smoothing remain. Fit 12 fixed epochs, 80 batches of four 512-sample windows,
supervised center 256, AdamW .0002, seed 373. No validation-based early stopping.

The selected policy equally averages original v2.2 and new team logits.
On observed-ball frames it uses their raw sign. Missing-frame decoding uses
boundary-dependent transition costs: subtract log(10) from the class-weighted
boundary logit, apply sigmoid, clip to [.001,.3], and decode at 5 Hz within
predicted H/A spans and periods. D/X remain the v1.10 decisions. The existing
18-frame merge follows; its threshold was not raised to force a low short
spell count. Team-swap antisymmetry and boundary symmetry are enforced.

| Match | v2.2 spellF1 | Ensemble spellF1 | v2.2 hidden accuracy | Ensemble hidden accuracy |
|---|---:|---:|---:|---:|
| WMX | .528764 | .549033 | .784941 | .806170 |
| WPY | .616393 | .624089 | .773989 | .795028 |
| WQQ | .617058 | .642677 | .794547 | .813398 |
| WR9 | .620848 | .648701 | .796561 | .813056 |
| Metrica 2 | .517666 | .531386 | .724188 | .730387 |

Equal-match means: spellF1 .580146 -> .599177, hidden accuracy
.774845 -> .791608. Additional Metrica 3 suite: spellF1 .555991 -> .584742,
hidden accuracy .688976 -> .711571. Natural Metrica 3: spellF1
.679070 -> .702290; hidden accuracy .530622 -> .560815. The latter still
does not regain v1.10's .575231.

Natural spellF1 slightly regresses on WMX (.663303 -> .659798), Metrica 2
(.652055 -> .646512), and WR9 (.762556 -> .754848), while improving WPY/WQQ.
This tradeoff is not concealed by the suite averages.

The original Metrica protocol also passes: spellF1 .634473 -> .644263,
hidden accuracy .787772 -> .801381. Both metrics improve in both games.
Game 1 is training; excluded Game 2 alone improves spellF1
.611713 -> .616270 and hidden accuracy .728064 -> .747213. These absolute
scores use a different dropout protocol and must not be merged with the
full-timeline results above.

Promotion prioritizes consistent primary-score gains across the checked
matches and the original two-game gate, while accepting the small natural
regressions listed above. Official challenge feedback is unavailable; no
private winning score or guaranteed ranking is claimed.

The NumPy export matches the two PyTorch heads on sequence lengths
1/17/257/1400, maximum absolute errors 1.44e-6 team and 6.26e-7 boundary.
Packaged ensemble labels on natural Metrica 2 match the native research
pipeline exactly. Challenge preview: 342 possessions, mean 8.65439 seconds,
zero <12f and eight <25f (v2.2: 319, 9.27837 s, zero and thirteen).
Graded D/X remain 84581/281. No private correctness conclusion follows from
these structural counts alone. Format validation passes, 199319 rows. Three
generations (preview, explicit artifact, and default base-Python inference
without PyTorch) are byte-identical, SHA256
`C6D6ECB8B04D8DBA1AF26D944C857A94BD0C0A930E24D5F554D4103AA250D9C6`.
`--no-boundary-model` exactly reproduces v2.2; `--no-team-model` restores v1.10.

## Rejected or held alternatives

- A temporal player-set residual to the boosted D emissions was trained on
  the same four matches (12 epochs, seed 473). WMX smooth-D spellF1 changes
  .528764 -> .529040, hidden .784941 -> .781743; Metrica 2 changes
  .517666 -> .516470, hidden .724188 -> .700522. No promotion.
- Simply merging <25f spells changes WMX suite spellF1 .528764 -> .521733
  and Metrica 2 .517666 -> .523504. A 50f threshold hurts both. Removing
  short D-H/A-D islands also fails to improve both. Lower short-spell counts
  are therefore not a substitute for correctness.
- Decoding the old logits at 25/12.5 Hz with evidence scaled to preserve
  5 Hz density yields no consistent gain. Unscaled dense evidence improves
  WMX but hurts Metrica 2, including natural tracking.
- A 12.5 Hz residual model uses freshly extracted player geometry, seven
  temporal layers and v2.2 logits as its fixed prior. It is fitted for 12
  epochs, seed 573, 1024-sample windows with center 512. WMX fine-grid suite
  spellF1 .528764 -> .540518, hidden .784941 -> .778143; Metrica 2
  .517666 -> .521780, hidden .724188 -> .701761. Natural tracking also
  regresses. It is held, not the deployed model.

New training scripts save atomic epoch checkpoints including optimizer and
both CPU RNG states. Resume behavior is unit-tested. Thirty-nine tests pass.

## Reproduction

```sh
python starter/external/diagnose_v22.py --match Sample_Game_2
python starter/external/temporal_dead_experiment.py --train --match J03WMX
python starter/external/boundary_ownership_experiment.py --train --match J03WMX
python starter/external/boundary_ownership_experiment.py --match Sample_Game_2
python starter/external/export_boundary_ownership.py
python starter/external/evaluate_approved.py --candidate 2.3 --reference 2.2
python starter/predict_v1.py --out evaluation/results/reproduced_sm_v2_3.csv
python starter/external/v22_short_ablation.py --match J03WMX
python starter/external/highres_ownership_experiment.py --train --match J03WMX
python -m unittest discover -s starter/external -p 'test_*.py'
```

Use `--resume` on the new training scripts after an interruption. Research
requires `requirements-neural.txt`; deployment remains NumPy-only. Predictions,
external frame data, caches and research checkpoints remain ignored.
