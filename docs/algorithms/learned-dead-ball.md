# Protected learned dead-ball model: sm_v1_9 (2026-11-14)

## Method

The sm_v1_8 geometric dead-ball rules conflate near-edge observations with
stoppages. A physical-boundary candidate improved Metrica but slightly lost
spellF1 on an independent Sportec match; it was not promoted.

sm_v1_9 instead learns live/dead emissions from licensed external matches and
decodes a temporally consistent state sequence. It retains the existing
possession-zone/nearest-team splice for H/A control.

- 41 input features: visible ball edge distance, outside/near-edge indicators,
  ball speed, nearest-player distance, nearby player count, stillness;
  previous/next observed values; visibility and time since/until observation;
  centered context windows of 0.48, 2 and 5 seconds.
- Class-balanced, ridge-regularized logistic regression (ridge 10, at most 15
  Newton iterations), trained at 5 Hz on natural observations and fixed 41%
  block-dropout patterns of 5/15 seconds, seed 73. Total 167,574 samples.
- Two-state Viterbi decoding at 5 Hz, transition probability 0.01 per step,
  specified before held-out-match evaluation. Repeated states restore 25 Hz.
  Published half blocks and tracking period boundaries reset decoding.
- Generate team labels with geometric dead-ball detection disabled, then
  overlay decoded D. Published kickoff handling remains in effect.
- **Protection:** on missing-ball frames where sm_v1_8 says H or A, preserve
  that exact answer. Elsewhere use the new prediction. This cannot lose a
  previously correct hidden H/A answer; it does not guarantee improved spellF1.

This is a logistic/Viterbi model implemented here, not an implementation of
PathCRF. No target-match human labels, public GT, private GT or video were used.
The additional optional state-machine controls leave sm_v1_8 unchanged.

## External data and separation

Training sources:

1. Sportec Open DFL **J03WOH**, geometry and frame-level BallStatus only.
   BallPossession is used for external evaluation, not feature construction.
2. Metrica **Sample_Game_1**, geometry and event-derived live/dead labels.

Both **J03WN1** and **Sample_Game_2** are excluded from parameter fitting.
Their first tests informed the protected-vs-unprotected design comparison;
they are now development validation, not untouched tests for future changes.
Further progress should reserve additional external matches before tuning.

Sportec: Bassek, Rein, Weber & Memmert (2025),
[An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y).
Data [figshare v1](https://doi.org/10.6084/m9.figshare.28196177.v1), **CC BY 4.0**.
The XML explicitly supplies frame-level active/inactive and owning-team fields.
The loader tests verify changing those annotations does not change any model
input features. Downloaded files are verified against publisher size and MD5.

Metrica: [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data).
Acknowledgement: sample tracking and event data provided by Metrica Sports.
Event-derived labels are an imperfect reference, particularly for stoppages.

## Results versus approved sm_v1_8

| External evaluation | Old spellF1 | Protected spellF1 | Old hidden H/A accuracy | Protected hidden H/A accuracy |
|---|---:|---:|---:|---:|
| Metrica 1, original dropout suite (training match) | 0.494674 | 0.502469 | 0.611399 | 0.739519 |
| Metrica 2, original dropout suite | 0.486448 | 0.489793 | 0.412309 | 0.616913 |
| Metrica 2, natural + 41% block dropout | 0.387076 | 0.393622 | 0.503314 | 0.642444 |
| Sportec J03WN1, natural + 41% block dropout | 0.378973 | 0.414292 | 0.483740 | 0.548852 |

The original suite uses natural missing plus 2/5/10/30-second gaps on live
windows. The 41% suite uses natural observations plus random 2/5/10/30-second
blocks over the whole timeline, including stoppages (seed 41). Values average
the five configurations; natural Sportec has no hidden frames, so its hidden
accuracy is undefined and excluded from that mean. These are different
protocols: do not compare their absolute scores across rows.

Local structure, no challenge GT:

| Model | Possessions | Mean seconds | <12f | <25f | Graded D | Graded X |
|---|---:|---:|---:|---:|---:|---:|
| sm_v1_8 | 359 | 13.7626 | 3 | 9 | 35057 | 281 |
| sm_v1_9 protected | 331 | 16.8413 | 3 | 8 | 19214 | 281 |
| smooth50 | 594 | 7.2383 | 62 | 124 | 51368 | 0 |

The substantially lower D count is a material domain-transfer uncertainty.
Two validation matches cannot establish hidden-challenge performance or a win.
The unprotected model had larger spellF1 gains (Sportec 0.463905; Metrica 2
original 0.533037), but reduced Sportec hidden accuracy to 0.469773. It remains
experimental. Same/midpoint gap reconciliation also lost hidden accuracy and
was not promoted. The visibility-only island candidate remains separate.

## Reproduce

Inference needs only numpy, pandas, the challenge ATD and the packaged
`starter/models/dead_ball_mixed_v1.json`; downloading external data is unnecessary.

```sh
python starter/predict_v1.py --out evaluation/results/sm_v1_9_compliant.csv
python evaluation/validate_submission.py evaluation/results/sm_v1_9_compliant.csv
python starter/predict_v1.py --no-learned-dead --out evaluation/results/sm_v1_8_rollback.csv
python -m unittest discover -s starter/external -p 'test_*.py'
```

Rebuild the fitted parameters (requires the named external data):

```sh
python starter/external/fetch_sportec.py --matches J03WOH J03WN1
python starter/external/fetch_metrica.py
python starter/external/learned_dead_experiment.py --mixed-training --train --match J03WN1
python starter/external/learned_dead_experiment.py --mixed-training --metrica --match Sample_Game_2 --legacy-dropouts
```

Fitting creates `evaluation/reports/dead_model_sportec_J03WOH_mixed.npz`.
The packaged JSON stores the identical mean, scale and coefficients as decimal
float64 values, plus training provenance. Reports, feature caches, external
data and submission CSVs stay gitignored. There are no target tracking arrays
or labels in the packaged model.
