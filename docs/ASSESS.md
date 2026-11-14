# How we assess (two tracks)

We do **not** keep/drop from the public 5-minute board. Run both tracks when judging a candidate.

**Approved best:** `evaluation/results/LATEST`. Both tracks compare **cand vs LATEST vs smooth50**. Promote by editing `LATEST` only when RESULTS keeps a newer model.

**sm_v2_3:** equal-logit ownership ensemble with a learned turnover-boundary
head. Five excluded-fit-match suite spellF1 .580146 -> .599177 and hidden
accuracy .774845 -> .791608; both improve on each match. Additional Metrica 3
suite and natural metrics also improve. The original Metrica protocol improves
both metrics in both games (Game 1 is training). Some natural Sportec/Metrica 2
scores regress slightly; these tradeoffs are recorded explicitly. Challenge
<25f spells decrease 13 -> 8, with zero <12f. Rollback `--no-boundary-model`.
See [evidence and reproduction](algorithms/boundary-ownership-research.md).

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
`python starter/external/evaluate_approved.py --candidate 2.2 --reference 1.10`.

**sm_v1_10:** calibrated mixed-provider boosted live/dead model. The original
Metrica suite improves vs sm_v1_9 on both metrics in both games, with Game 1
remaining a training/development check. Additional whole-timeline 41% dropout
validation improves both metrics on excluded Metrica Game 2 and four Sportec
matches (J03WMX/J03WPY/J03WQQ/J03WR9). The two dropout protocols have different
absolute scores; report them separately. Reproduce the broader comparison with
`python starter/external/evaluate_approved.py --candidate 1.10 --reference 1.9`.
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

**Gate (team / missing-ball changes):** beat **LATEST** on external spellF1 **and** hidden accuracy, both games.

## One command

```bash
source .venv/bin/activate
bash assess.sh evaluation/results/sm_v1_7_compliant.csv
bash assess.sh evaluation/results/sm_v1_7_compliant.csv --skip-metrica
bash assess.sh evaluation/results/sm_v1_7_compliant.csv --public-log   # optional log only
```

After a new prediction:

```bash
bash predict_v1.sh                 # predict → validate → assess
PUBLIC_LOG=1 bash predict_v1.sh    # also public log
```

Auto predict writes the **next version after LATEST** (prefer bare `sm_v*.csv`; add `_compliant` only if that bare name already exists). It does **not** update `LATEST`.

## Keep checklist

1. Rule / physics / external-data citation  
2. Track A vs LATEST + smooth50  
3. Track B if the change touches who has the ball / missing frames  
4. Log model version in [../team/RESULTS.md](../team/RESULTS.md); update `LATEST` only on promote  

`sm_v1_8` (D-bridge): [algorithms/dead-bridge.md](algorithms/dead-bridge.md).  
`sm_v1_7` (VC): [algorithms/vidal-codina.md](algorithms/vidal-codina.md).  
T27 closed: [algorithms/controlled-possession.md](algorithms/controlled-possession.md).
