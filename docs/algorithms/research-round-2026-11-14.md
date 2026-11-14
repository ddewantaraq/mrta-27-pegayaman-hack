# Further research round (2026-11-14)

**Outcome:** promoted sm_v1_10, the calibrated mixed-provider boosted model
with existing short-spell postprocessing after the D overlay. It improves
both metrics on excluded Metrica Game 2 and four Sportec matches under the
whole-timeline dropout protocol. The original Metrica protocol also improves
both metrics in both games (Game 1 is fitting/development).
See [final numbers and reproduction](boosted-dead-ball.md).

## Confidence target

No winning spellF1 cutoff can be estimated from the available evidence. The
organisers explicitly provide
[no live leaderboard or submission feedback](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md).
Use 0.70 as an aspirational external-development target and 0.80 as a stretch
target, not as a probability of winning. A score must state its match,
reference construction and dropout conditions. The public challenge window
must not select parameters or models.

## Sources that informed the experiments

- Lang et al. (2022),
  [Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data](https://doi.org/10.1038/s41598-022-19948-1):
  player movement and temporal context can help detect stoppages without a ball
  track. Its accuracy/stoppage F1 numbers are not challenge spellF1.
- Borghesi et al. (2023),
  [Using Temporal Convolutional Networks to estimate ball possession in soccer games](https://doi.org/10.1016/j.eswa.2023.119780):
  temporal context, permutation-invariant team features, and separate live/dead
  and team predictions motivate the factorized candidates. No claim is made
  to reproduce its neural network or its reported accuracy.
- [PathCRF (2026)](https://arxiv.org/abs/2602.12080) motivates consistent
  possession sequences from player trajectories. We did not deploy its network.
- External geometry/status data:
  [Bassek et al. (2025), Sportec Open DFL](https://doi.org/10.1038/s41597-025-04505-y),
  [figshare v1](https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0.
- [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data)
  supplies a cross-provider stress check. The historical event-derived D proxy
  misses some foul stoppages; its historical scores remain unchanged.

## Data separation

All seven Sportec matches were downloaded and publisher-size/MD5 verified.
No challenge possession labels or video were loaded.

- Linear motion and mixed-provider boosted candidates fit J03WOH + Metrica 1.
- Direct-status boosted and team-control candidates fit J03WOH/J03WN1/J03WOY.
- J03WMX is development validation for this round. J03WPY/J03WQQ/J03WR9 were
  excluded from fitting. The boosted hyperparameters were fixed before their
  first evaluation; these matches are now validation data, not pristine tests
  for further tuning.

Every dropout suite includes natural observations plus deterministic 41%
2/5/10/30-second blocks across the entire timeline, including stoppages.
Training uses natural observations and 5/15-second blocks with another seed.
Scores average five conditions; no-hidden-frame accuracy is excluded from
the hidden-accuracy mean. A clean-track score and a dropout-suite score are
different measurements and must not be presented interchangeably.

## Models and diagnostics

Twelve permutation-invariant movement summaries use player coordinates only:
speed mean/quantiles, moving/fast/still fractions, positional and motion
coverage, and spatial spread. Velocity uses a 0.4-second span, excludes
missing tracks, period crossings and >12 m/s jumps. Temporal offsets up to
four seconds and context averages produce 132 motion features; combined with
the prior 41 ball features there are 173 inputs.

Linear motion models did not consistently beat sm_v1_9. Histogram gradient
boosting uses 150 iterations, learning rate 0.1, 15 leaves, minimum leaf size
100, L2=10, 63 bins, balanced classes, seed 73, and no random frame-wise
early-stopping split. This avoids validating on near-duplicate temporal
neighbours. It uses the existing temporal decoder and team-control pipeline.

Direct-status boosted model, before final short-spell postprocessing:

| Match excluded from fitting | sm_v1_9 spellF1 | Boosted spellF1 | Old hidden accuracy | Boosted hidden accuracy |
|---|---:|---:|---:|---:|
| J03WMX | 0.402103 | 0.466087 | 0.580085 | 0.679493 |
| J03WPY | 0.461009 | 0.531815 | 0.566622 | 0.646146 |
| J03WQQ | 0.416153 | 0.484037 | 0.649513 | 0.680335 |
| J03WR9 | 0.424553 | 0.506979 | 0.614164 | 0.672009 |

For example, J03WPY clean-track spellF1 is 0.731642, while its suite average is
0.531815. A clean score above 0.70 does not establish robust performance when
the ball is missing.

The direct-status model also loses hidden accuracy on the Metrica 2 proxy:
0.642444 -> 0.335346. A separate conservative reference repair adds D only
when FAULT RECEIVED, CARD or explicit OFFSIDE is followed by a SET PIECE as
the next on-ball event, preserving the historical proxy. The loss does not
disappear under this repair: this is a real transfer concern, not something
that can be dismissed as reference error. Mixed-provider boosted validation
was evaluated separately. A training-prevalence correction (-0.579631
log-odds, derived without target/validation statistics) resolved the small
hidden-accuracy loss of the uncalibrated mixed model. This corrected variant
was selected. The prior correction does not force a target D-frame count.

An external-only oracle diagnostic on J03WN1 with 5-second missing blocks gives
actual spellF1 0.336188, perfect-D-only 0.444178, and perfect-team-only 0.754225.
The oracle variants deliberately use external reference labels to isolate
bottlenecks. They are not deployable models, submission results, or claims of
achieved accuracy. The team-emission candidate improved hidden accuracy but
lost spellF1; possession-balanced external fitting is tested separately.
That balanced variant also lost spellF1 (J03WMX 0.319540 versus sm_v1_9
0.402103) and was rejected.

## Broadcast attempt

The supplied broadcast ATD is present; the MP4 files are absent. We aligned
moving-ball trajectory templates with KD-tree retrieval, geometric validation
and offset-consistent anchor pairs, testing all four coordinate orientations.
The accepted orientation yielded 546 anchors (median geometric error 0.489 m)
and 93,385 mapped tactical frames. Strict local agreement recovered only 617
missing ball observations. Fusion is experimental and not enabled in the
approved pipeline; cuts/replays remain a risk. No manual time alignment or
possession labels were used. Mapping/coordinate artifacts stay gitignored.

## Reproduction

Research dependencies are isolated in `.venv`, with versions pinned in
`requirements-research.txt`. The approved model remains dependency-light.

```sh
python -m pip install -r requirements-research.txt
python starter/external/fetch_sportec.py --matches all
python starter/external/motion_dead_experiment.py --train --match J03WMX
python starter/external/boosted_motion_experiment.py --train --match J03WMX
python starter/external/boosted_motion_experiment.py --sportec-training --train --match J03WMX
python starter/external/boosted_motion_experiment.py --sportec-training --match J03WPY
python starter/external/team_control_experiment.py --spell-balanced --train --match J03WMX
python starter/external/export_boosted_model.py
python starter/external/camera_alignment_experiment.py
python -m unittest discover -s starter/external -p 'test_*.py'
```

Numeric boosted trees can be exported to JSON and evaluated with NumPy only.
Export parity was verified against scikit-learn on 25,279 feature samples,
including missing-feature routing, at 1e-12 tolerance. The JSON contains model
parameters and external-data provenance, not target-match coordinates.
The selected mixed-provider export was additionally verified on 16,758 probes.
