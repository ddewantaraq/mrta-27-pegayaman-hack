# Temporal ownership experiments

Status: residual hybrid `sm_v2_2` is promoted; earlier candidates remain research only.
The current approved successor is `sm_v2_3`; see
[boundary ownership research](boundary-ownership-research.md).
No public challenge possession labels are used for training or selection.

## Design and evidence

The remaining ownership problem motivates a temporal model rather than another
distance threshold. [Borghesi et al.](https://iris.polito.it/retrieve/d9aa6bfd-b26e-44cd-bf9e-6004e676211a/1-s2.0-S0957417423002816-main.pdf)
use temporal convolutions and team features for possession recognition.
[MS-TCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.pdf)
motivates dilated convolutions and log-probability smoothing to reduce temporal
fragmentation. Our networks are adaptations, not reproductions of either paper.
[PathCRF](https://arxiv.org/html/2602.12080v2) also motivates examining individual
player trajectories; its event metrics and reported cross-provider results
must not be equated with this challenge's spellF1.

Fit matches are Sportec J03WOH/J03WN1/J03WOY and Metrica Sample Game 1.
Development matches are J03WMX and Metrica Sample Game 2. Sample Game 3 was newly
downloaded and initially reserved: only geometry was prepared before the first
candidate was frozen. Its later evaluations are described below. Sportec WPY/WQQ/WR9 have been used
in previous validation rounds and are therefore additional robustness checks,
not untouched final tests.

Both models operate at 5 Hz with 25 Hz interpolated output. Training uses
natural tracking plus 2/5/15/30-second full-timeline dropout, seed 73. Evaluation
uses natural tracking plus 2/5/10/30-second dropout, seed 41. These five
conditions are correlated stress tests, not five independent matches.

The flat network takes 45 signed ownership features and 12 player-motion
summaries. Six residual blocks have paired convolutions at dilations
1/2/4/8/16/32. Loss is possession-duration-weighted binary cross entropy plus
same-label temporal smoothing (lambda .15, truncation 4). Fixed 12 epochs,
80 batches per epoch, batch size 4, AdamW .0005, seed 73. Inference averages
the normal and team-swapped predictions to enforce team symmetry.

| Development match | Reference spellF1 | Flat, missing only | Flat, missing + logit gate | Reference hidden accuracy | Gated hidden accuracy |
|---|---:|---:|---:|---:|---:|
| J03WMX | .459776 | .488677 | .466953 | .678237 | .696550 |
| Metrica 2 | .426366 | .433213 | .433001 | .664460 | .696571 |

The logit gate is `abs(logit) >= log(3)`; its sigmoid value is .75, but this is
not a validated probability calibration or a 75% chance of winning. Replacements
apply only to missing-ball reference H/A frames. The reference D/X decisions
remain, followed by the existing 18-frame short-spell merge. Ungated variants
improve spellF1 but lose hidden accuracy on Metrica 2. No variant is promoted
based on these development results alone.

The second model encodes anonymous player sets with a shared MLP and masked
mean/max pooling for each team before temporal convolutions. Geometry-only
nodes contain position, velocity, speed, validity and attack orientation.
Ball context uses only observed endpoints around gaps. Player identities and
event annotations are excluded. Team-swap and lateral-reflection augmentations
are used; player ordering is irrelevant by construction and tested explicitly.
Fixed 16 epochs, 80 batches, seed 173.

| Match | Reference spellF1 | Player-set full overlay | Reference hidden accuracy | Full overlay hidden accuracy |
|---|---:|---:|---:|---:|
| J03WMX | .459776 | .548874 | .678237 | .808169 |
| J03WPY | .527212 | .587810 | .639340 | .787271 |
| J03WQQ | .490340 | .612168 | .678726 | .799744 |
| J03WR9 | .501658 | .635441 | .665169 | .810210 |
| Metrica 2 | .426366 | .482277 | .664460 | .673663 |

Equal-match mean spellF1 improves .481070 -> .573314, and hidden accuracy
.665186 -> .775811. Hidden accuracy is averaged within each match over its
nonempty conditions, then equally over matches; pooling rows instead gives
different weights because natural Sportec has no missing ball.

The candidate and promotion rule were frozen before reading Game 3 events.
The same frozen model improved the reserved suite mean spellF1
.446470 -> .525379 and hidden accuracy .630178 -> .678087. **However, the
natural-only breakdown regressed:** Metrica 2 spellF1 .629772 -> .565258 and
hidden accuracy .700056 -> .590569; Metrica 3 spellF1 .653978 -> .619425 and
hidden .575231 -> .509602. The historical live-window dropout protocol also
failed its hidden-accuracy gate (.7499 -> .6838, with the
training game included). The model was therefore **held, not promoted**,
despite passing its initial aggregate-only criterion. Challenge structure also
became more fragmented: 294 -> 420 possessions, short <12f 0 -> 1 and <25f
2 -> 21. Public challenge labels were not evaluated.

NumPy-only inference was exported and checked against PyTorch on sequences
of lengths 1/17/257/1400 (maximum absolute logit difference 9.54e-7).
Two independent challenge generations were byte-identical, SHA256
`A61723E89F6375B1B9B7AF539B079926F8CE7E1EEAB5D175E98BB293BEF79D17`.
The research artifact `sm_v2_compliant.csv` is **not LATEST**.

## Selective replacement and further correction training

Frame confidence, observed possession-zone endpoints, Viterbi smoothing and
coherent change-run gates were examined on WMX/Metrica 2. Frame confidence
alone does not establish calibrated error probability. The selected conservative
gate requires a missing-ball disagreement run of at least 25 frames with mean
absolute logit >= log(9), then retains the existing 18-frame merge.

This gate improves both suite means on all five previously checked matches,
including natural Metrica 2 (.629772 -> .640845 spellF1,
.700056 -> .731083 hidden accuracy). Challenge structure matches v1.10:
294 possessions, zero <12f and two <25f. But an additional frozen Game 3
validation **fails** the hidden-accuracy criterion: suite spellF1
.446470 -> .451602, hidden accuracy .630178 -> .624692. No promotion.
Game 3 is no longer an untouched test after the first frozen evaluation; its
gate result is explicitly additional validation, not a new independent test.

A new research model conditions on the automatic v1.10 baseline prediction
and learns a correction with a fixed baseline log-odds prior. It uses the same
four external training matches, gives natural tracking three times the sampling
weight of each synthetic condition, and excludes reference D/X frames from
the ownership loss because this head cannot change them at inference. It does
not use Game 2/3 or excluded Sportec labels for fitting. The correction is
initialized from the player-set model, with its output weights scaled by .25,
then fitted for 12 fixed epochs, 80 batches of four sequences, AdamW .0002,
seed 273. Training dropouts are natural plus 2/5/10/30 seconds, seed 73.
The baseline log-odds prior is log(9); it is a design constant, not a claim
of calibrated 90% correctness.

The chosen hybrid retains raw network decisions on observed-ball frames and
uses the existing .01-transition Viterbi on missing frames within predicted
live spans. It was selected on WMX/Metrica 2 before the additional checks.

| Match | Reference spellF1 | Residual hybrid spellF1 | Reference hidden accuracy | Residual hidden accuracy |
|---|---:|---:|---:|---:|
| J03WMX | .459776 | .528764 | .678237 | .784941 |
| J03WPY | .527212 | .616393 | .639340 | .773989 |
| J03WQQ | .490340 | .617058 | .678726 | .794547 |
| J03WR9 | .501658 | .620848 | .665169 | .796561 |
| Metrica 2 | .426366 | .517666 | .664460 | .724188 |

Equal-match mean spellF1 .481070 -> .580146; hidden accuracy
.665186 -> .774845. Natural spellF1 also improves on all five matches. For
Metrica 2 specifically, natural spellF1 .629772 -> .652055 and hidden
accuracy .700056 -> .717215.

Additional Metrica 3 validation: suite spellF1 .446470 -> .555991 and hidden
accuracy .630178 -> .688976; natural spellF1 .653978 -> .679070 and macroF1
.798459 -> .803933. **Natural missing-ball accuracy falls .575231 -> .530622.**
Therefore the frozen strict auxiliary gate reports `passed: false`; it must
not be described as having passed all criteria. The primary spell metric
improves on every checked match, including natural conditions.

The original protocol subsequently passes the repository's two-game gate:
mean spellF1 .571573 -> .634473 and hidden accuracy .749873 -> .787772. Game 1
is training; excluded Game 2 alone improves .549717 -> .611713 spellF1 and
.687964 -> .728064 hidden accuracy. This is a different dropout protocol and
must not be combined numerically with the full-timeline suite.

**Promotion decision:** select `sm_v2_2` for the official primary spell metric,
which improves on all six checked external matches and all six natural
conditions. This explicitly accepts the natural Game 3 auxiliary accuracy
regression and the increased challenge <25f count. The earlier frozen strict
auxiliary gate remains marked failed; its definition/results are not edited
to manufacture a pass. There is no claim of a known private or winning score.
`--no-team-model` restores v1.10; the older logistic/legacy rollback flags
also disable the new head by default.

Challenge preview: 319 possessions, mean 9.27837 seconds, zero <12f and
13 <25f; graded D/X remain 84581/281. This is more fragmented than v1.10,
but substantially less than the original unrestricted player-set candidate.
Its format passes. Preview SHA256:
`27EA6EF8745D3D0641C0D50EB127F48D8FC40CD214AEB9DB4066CD3E35E297BA`.
Numeric export maximum absolute difference from PyTorch is 1.35e-6 on the
random sequence probes and 1.20e-6 on a complete external match, with zero
frame sign differences. The packaged Sportec registry also reproduces the
native research labels exactly. Three challenge generations, including default
inference without importing PyTorch, have the same SHA256. Inference remains
NumPy-only; 31 tests pass.

## Official pretrained PathCRF experiment

The official repository and public trial 140 weights were downloaded with
commit and SHA256 manifests under ignored `data_external/pathcrf`. The source
license is MPL 2.0 and remains intact. Its arguments specify Sportec-only
leave-one-match-out fold 6; matches used for that pretrained model must not be
reported as independent tests. Initial testing is on Metrica 2, which is
outside the checkpoint's stated training source.

The initial adapter tests macro and source-marginal **emissions only**, not a
reproduction of the paper's full event CRF evaluation. Inputs are player
geometry, goalkeeper/team indicators and physical derivatives following the
official 25 Hz Savitzky-Golay filtering. Geometry alone defines roster phases;
there are no event/possession boundaries or labels in model inputs. Observed
ball positions are excluded. Model windows run at 5 Hz, 10 seconds with
50% overlap, before output interpolation. Published event F1 is not the
challenge's spellF1. Macro/source-marginal emission results on Metrica 2 fail
to beat the baseline: suite spellF1 .426366 -> .411826/.417620; natural
spellF1 .629772 -> .426269/.427099. These emission-only adapters are rejected;
they do not establish that the full published CRF fails.

The subsequent sequence experiment uses the official dynamic CRF on fixed
geometry windows. A second variant adds soft source-team log-odds log(9) at
observed possession-zone contacts; these are geometric observations, not human
labels. Only majority-consistent votes (absolute vote >= .5) replace reference
H/A. No reference episode boundaries or final possession nodes are supplied.
Both sequence variants are rejected as replacements. On Metrica 2, ball-free
CRF suite spellF1 is .391679 (reference .426366), hidden accuracy .648857
(reference .664460), and natural spellF1 .410286 (reference .629772).
Observed-team hints raise suite spellF1 to .452605, but hidden accuracy is
.648988 and natural spellF1/hidden accuracy .501198/.518945. These lose to
the residual hybrid (.517666 suite spellF1, .652055/.717215 natural metrics).
This is evidence about our transfer adapter, not a reproduction or refutation
of the paper's original event-detection benchmark. No PathCRF parameters or
code are required for approved inference.

## Reproduction

Install `requirements-neural.txt` in the research environment (CPU PyTorch is
optional and is not needed for the current approved NumPy predictor).

```sh
python starter/external/fetch_metrica3.py
python starter/external/temporal_team_experiment.py --train --match J03WMX
python starter/external/temporal_team_experiment.py --match Sample_Game_2
python starter/external/set_temporal_experiment.py --train --match J03WMX
python starter/external/set_temporal_experiment.py --match Sample_Game_2
python starter/external/residual_set_experiment.py --train --match J03WMX
python starter/external/export_residual_set.py
python starter/external/evaluate_approved.py --candidate 2.2 --reference 1.10
python starter/predict_v1.py --out evaluation/results/reproduced_sm_v2_2.csv
python -m unittest discover -s starter/external -p 'test_*.py'
```

Research weights, external frame-level caches and predictions remain ignored.
Training has no GPU requirement. A winning score cannot be inferred without
comparable competitor results and the private evaluation; .70 is not a known
winning threshold.
