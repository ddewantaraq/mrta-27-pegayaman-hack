# How we assess (two tracks)

We do **not** keep/drop from the public 5-minute board. Run both tracks when judging a candidate.

## Track A — local structure (BAR-ATH CSV, no GT)

**Question:** do possessions look football-realistic (not flicker)?

Compare the candidate CSV to:

1. **Latest reference** (default `evaluation/results/sm_v1_7_compliant.csv`; pass `--ref` to compare a newer cand against it)
2. **`baseline_smooth50.csv`**

Metrics: `n_spells`, mean spell duration (s), short H/A **&lt;12f** / **&lt;25f**, graded D / X.

**Gate:** fewer shorts and longer mean than ref when possible; clearly better than smooth50.

## Track B — H/A correctness (Metrica only)

**Question:** is the team letter right, especially when the ball is missing?

- Games: Metrica Sample Game 1 & 2 (event-derived labels + dropouts)
- **External spellF1:** same `possession_metrics` formula as the challenge; GT is Metrica, not BAR-ATH
- **Hidden-frame team accuracy:** on hidden frames where GT is H or A, did we predict that team?

Reported model versions: **`sm_v1_5`** vs **`sm_v1_7`** vs **`smooth50`**.

**Gate (team / missing-ball changes):** beat `sm_v1_5` on external spellF1 **and** hidden accuracy, both games.

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

## Keep checklist

1. Rule / physics / external-data citation  
2. Track A vs ref + smooth50  
3. Track B if the change touches who has the ball / missing frames  
4. Log model version in [../team/RESULTS.md](../team/RESULTS.md)  

`sm_v1_7` background: [algorithms/vidal-codina.md](algorithms/vidal-codina.md).
