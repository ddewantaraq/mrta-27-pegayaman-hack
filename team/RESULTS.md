# Results log

**Metrics only** — no match coordinates, no GT label dumps, no video.

Public 5-minute GT ballpark (sanity, not leaderboard): ~**17** possessions, mean ~**14.2** s on the public window. Ranking uses private full-match GT.

How to fill: run `python evaluation/run_evaluation.py`, then add a row. See [BEGINNER.md](BEGINNER.md).

---

## Score table

| Date | File | Owner | macroF1 | spellF1 | poss | mean_dur_s | Notes |
|------|------|-------|---------|---------|------|------------|-------|
| _YYYY-MM-DD_ | baseline_raw | shipped | | | | | Fill after T10 |
| _YYYY-MM-DD_ | baseline_smooth25 | shipped | | | | | Fill after T10 |
| _YYYY-MM-DD_ | baseline_smooth50 | shipped | | | | | Floor to beat on **full** match: spellF1 **0.2433** |
| _YYYY-MM-DD_ | baseline_smooth125 | shipped | | | | | Fill after T10 |
| | | | | | | | |
| | | | | | | | |

Add a new blank row for every model CSV (T30).

---

## Setup / ops notes

| Date | Note |
|------|------|
| | T01 — Python version: |
| | T02 — Data completeness: |
| | T03 — Google account ready: yes/no |
| | |

---

## Experiment notes (E1 / E2)

Short bullets only — why a change was kept or dropped. Prefer poss/dur and failure modes over chasing public spellF1 alone.

| Date | Change | Keep? | Why |
|------|--------|-------|-----|
| | | | |
