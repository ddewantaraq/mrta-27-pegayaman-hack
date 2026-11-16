# How we assess (three tracks)

We do **not** keep/drop from the public 5-minute board. Run all tracks when judging a candidate.

**Approved best:** `evaluation/results/LATEST` → **`sm_v2_3_compliant.csv`**. Tracks compare **cand vs ref** (default LATEST; pass `--ref` to override). Track A also includes smooth50. Promote by editing `LATEST` only when RESULTS keeps a newer model.

**sm_v2_3:** equal-logit ownership ensemble with a learned turnover-boundary
head. Five excluded-fit-match suite spellF1 .580146 -> .599177 and hidden
accuracy .774845 -> .791608; both improve on each match. Additional Metrica 3
suite and natural metrics also improve. The original Metrica protocol improves
both metrics in both games (Game 1 is training). Some natural Sportec/Metrica 2
scores regress slightly; these tradeoffs are recorded explicitly. Challenge
<25f spells decrease 13 -> 8, with zero <12f. Rollback `--no-boundary-model`.
See [evidence and reproduction](algorithms/boundary-ownership-research.md).

**sm_v2_4 (demoted):** sm_v2_3 + live-team error corrector. Crossfit natural
and Metrica/Sportec B/C improved vs v2.3, but Track A structure worsened
(spells 342→353, mean 8.65→8.38 s, s25 8→10). Opt-in via
`--error-corrector-file`. See [push-090-natural](algorithms/push-090-natural.md).

**sm_v2_2:** baseline-aware player-set ownership with raw visible estimates and
temporal decoding during missing-ball stretches. The five excluded-fit-match
full-timeline suite improves mean spellF1 .481070 -> .580146 and hidden team
accuracy .665186 -> .774845. The original protocol improves both metrics in
both Metrica games (Game 1 is training). Additional Game 3 validation improves
primary spellF1, but natural missing-ball accuracy declines; the stricter
auxiliary gate is recorded as failed. Promotion prioritizes the official spell
metric, which improves on every checked match, including natural conditions.
Challenge structure has more <25f spells (2 -> 13), with zero <12f spells.
See [the full evidence and decision](algorithms/temporal-ownership-research.md).
Rollback: `--no-team-model`. Broader reproduction:
`PYTHONPATH=starter:starter/external:research/training:research/experiments python research/experiments/evaluate_approved.py --candidate 2.2 --reference 1.10`.

**sm_v1_10:** calibrated mixed-provider boosted live/dead model. The original
Metrica suite improves vs sm_v1_9 on both metrics in both games, with Game 1
remaining a training/development check. Additional whole-timeline 41% dropout
validation improves both metrics on excluded Metrica Game 2 and four Sportec
matches (J03WMX/J03WPY/J03WQQ/J03WR9). The two dropout protocols have different
absolute scores; report them separately. Reproduce the broader comparison with
`PYTHONPATH=starter:starter/external:research/training:research/experiments python research/experiments/evaluate_approved.py --candidate 1.10 --reference 1.9`.
See [method, tables and transfer risk](algorithms/boosted-dead-ball.md).

**sm_v1_9:** the protected learned dead-ball model uses external Sportec J03WOH
and Metrica Sample_Game_1 for training. Metrica Game 1 scores are development
checks, not independent validation. Additional validation uses excluded
Metrica Game 2 and Sportec J03WN1; both improve spellF1 and hidden accuracy.
See [method and limitations](algorithms/learned-dead-ball.md).

## Track A — local structure (BAR-ATH CSV, no GT)

**Question:** do possessions look football-realistic (not flicker)?

Compare the candidate CSV to:

1. **LATEST** (default from `evaluation/results/LATEST`; pass `--ref` to override)
2. **`baseline_smooth50.csv`**

Metrics: `n_spells`, mean spell duration (s), short H/A **&lt;12f** / **&lt;25f**, graded D / X.

**Gate:** fewer shorts and longer mean than LATEST when possible; clearly better than smooth50.

## Track B — H/A correctness (Metrica only)

**Question:** is the team letter right, especially when the ball is missing?

Same three columns as Track A: **cand vs LATEST vs smooth50**.

- Games: Metrica Sample Game 1 & 2 (event-derived labels + dropouts)
- **External spellF1:** same `possession_metrics` formula as the challenge; GT is Metrica, not BAR-ATH
- **Hidden-frame team accuracy:** on hidden frames where GT is H or A, did we predict that team?
- Cand = current code defaults; LATEST = kwargs for the version named in `LATEST` (BAR-ATH CSV is not replayed on Metrica)

**Gate (team / missing-ball changes):** beat **ref** on external spellF1 **and** hidden accuracy, both games.

## Track C — Sportec H/A (full-timeline suite)

**Question:** does the model transfer to other-provider tracking used in the v2.x line?

- Matches (default, excluded from v2.3 fit): Sportec **J03WMX / J03WPY / J03WQQ / J03WR9**
- Protocol: natural missing + 41% 2/5/10/30 s block dropouts across the whole timeline (same suite as `evaluate_approved.py`)
- Metrics: external spellF1 + hidden-frame team accuracy (cand vs ref)
- Data: `data_external/sportec/` via `python starter/external/fetch_sportec.py --matches …` or `--fetch-sportec`

**Gate:** beat **ref** on spellF1 and hidden across the suite matches. Report Game-1-style in-sample scores separately if ever included.

## One command

```bash
source .venv/bin/activate
bash assess.sh evaluation/results/sm_v2_3_compliant.csv --ref evaluation/results/sm_v2_2_compliant.csv
bash assess.sh evaluation/results/sm_v2_3_compliant.csv --skip-sportec
bash assess.sh evaluation/results/sm_v2_3_compliant.csv --fetch-sportec
bash assess.sh evaluation/results/sm_v2_3_compliant.csv --public-log   # optional log only
```

After a new prediction:

```bash
bash predict_v1.sh                 # predict → validate → assess
PUBLIC_LOG=1 bash predict_v1.sh    # also public log
```

Auto predict writes the **next version after LATEST** (prefer bare `sm_v*.csv`; add `_compliant` only if that bare name already exists). It does **not** update `LATEST`.

## Keep checklist

1. Rule / physics / external-data citation  
2. Track A vs ref + smooth50  
3. Track B if the change touches who has the ball / missing frames  
4. Track C (Sportec) for ownership / dead-ball model changes on the v2.x line  
5. Log model version in [../team/RESULTS.md](../team/RESULTS.md); update `LATEST` only on promote  

`sm_v2_3` (boundary ensemble, **LATEST**): [algorithms/boundary-ownership-research.md](algorithms/boundary-ownership-research.md).  
`sm_v2_4` (error corrector, demoted): [algorithms/push-090-natural.md](algorithms/push-090-natural.md).  
`sm_v2_2` / `sm_v1_10` / `sm_v1_8` notes: [algorithms/](algorithms/).  
Research archive (experiments not needed to regenerate CSV): [../research/](../research/).  
T27 closed: [algorithms/controlled-possession.md](algorithms/controlled-possession.md).
