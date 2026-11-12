# How we assess (two tracks)

We do **not** keep/drop from the public 5-minute board. Run both tracks when judging a candidate.

**Approved best:** `evaluation/results/LATEST` (one line → currently `sm_v1_8_compliant.csv`). Both tracks compare **cand vs LATEST vs smooth50**. Promote by editing `LATEST` only when RESULTS keeps a newer model.

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
