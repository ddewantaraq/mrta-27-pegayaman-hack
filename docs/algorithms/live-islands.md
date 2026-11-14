# Visible live-island experiment (2026-11-14)

Baseline: approved sm_v1_8, regenerated from ATD. No public or private challenge
ground truth was used for this experiment.

Hypothesis: a sub-second H/A run enclosed immediately by D on both sides can
be an erroneous apparent restart caused by edge tracking flicker. Suppress
only D-H-D / D-A-D runs shorter than 25 frames, and only when the ball is
visible throughout the island. Never cross an ungraded block or X boundary,
and never absorb a live team turnover. This is a new heuristic, not a rule
established by a paper. Real brief restarts can be lost.

The 12-frame and 25-frame thresholds were specified before external evaluation.
External validation uses Metrica Games 1 and 2, each with natural missing ball
and the existing deterministic 2/5/10/30-second dropout configurations.

## Reproduce

```sh
python starter/external/short_spell_experiment.py
python starter/external/short_spell_experiment.py --generate evaluation/results/visible_island25_candidate.csv
python evaluation/validate_submission.py evaluation/results/visible_island25_candidate.csv
python -m unittest discover -s starter/external -p test_short_spell_experiment.py
```

The generator executes the approved default model then the candidate postprocess.
It never loads a submission as an input. The evaluation report is local and
gitignored: `evaluation/reports/short_spell_experiment.csv`.

## Local structure (not correctness)

| Model | Possessions | Mean seconds | <12f | <25f | Graded D | Graded X |
|---|---:|---:|---:|---:|---:|---:|
| sm_v1_8 | 359 | 13.7626 | 3 | 9 | 35057 | 281 |
| visible island25 | 353 | 13.9870 | 0 | 3 | 35141 | 281 |
| smooth50 | 594 | 7.2383 | 62 | 124 | 51368 | 0 |

An initial unguarded variant improved external mean spellF1 from 0.490561
to 0.493174, but reduced hidden accuracy from 0.511854 to 0.511812.
The visibility guard was introduced to preserve all missing-frame predictions.
These experiments reuse the same two external games as previous development;
they are not independent evidence of generalization to unseen matches.

Guarded final result (mean over five dropout configurations per game):

| External game | sm_v1_8 spellF1 | Candidate spellF1 | Hidden accuracy (both) |
|---|---:|---:|---:|
| Game 1 | 0.494674 | 0.497271 | 0.611399 |
| Game 2 | 0.486448 | 0.487256 | 0.412309 |
| Mean | 0.490561 | 0.492264 | 0.511854 |

Both games improve slightly in spellF1 and preserve hidden accuracy exactly.
The assessment code accepts nondecreasing hidden accuracy (the prose says
"beat both"). This candidate remains experimental because the gain is tiny;
it was not rejected by a strict-hidden-improvement requirement in the code.

## Research directions

- [Vidal-Codina et al. (2022)](https://arxiv.org/abs/2202.00804) underlies the
  existing possession-zone method; it does not establish this island heuristic.
- [Mills et al. (2026)](https://link.springer.com/article/10.1007/s12283-026-00549-4)
  reports limitations from visibility and ball errors in broadcast auto-eventing.
- [PathCRF (2026)](https://arxiv.org/abs/2602.12080) infers possession paths from
  player trajectories with structured decoding. Its
  [official implementation](https://github.com/hyunsungkim-ds/pathcrf) supplies
  trained models and uses seven Sportec Open DFL matches. This is a potential
  larger missing-ball improvement, not an implemented component here. Adoption
  needs an audit of training matches, licenses, track identity requirements and
  independent external validation; fragmented challenge player tracks complicate
  direct application.

This island candidate was not promoted. Later on the same day, the separate
protected learned model sm_v1_9 was promoted; see learned-dead-ball.md. A smaller
number of short spells alone does not establish a winning submission.
