# Results log

**Metrics only** — no match coordinates, no GT label dumps, no video.

Public 5-minute GT ballpark (sanity, not leaderboard): ~**17** possessions, mean ~**14.2** s on the public window. Ranking uses private full-match GT.

How to fill: run `python evaluation/run_evaluation.py`, then add a row. See [BEGINNER.md](BEGINNER.md).

---

## Score table

Scored **2026-10-24** against public GT via `.venv/bin/python evaluation/run_evaluation.py` (T10). Baselines regenerated the same day with `starter/baseline.py` (T11).

| Date | File | Owner | macroF1 | spellF1 | poss | mean_dur_s | Notes |
|------|------|-------|---------|---------|------|------------|-------|
| 2026-10-24 | baseline_raw | shipped | 0.5972 | 0.0378 | 131 | 2.0 | Public window; far too many possessions vs GT 17 |
| 2026-10-24 | baseline_smooth25 | shipped | 0.6200 | 0.2868 | 36 | 7.6 | |
| 2026-10-24 | baseline_smooth50 | shipped | 0.6207 | 0.3256 | 26 | 10.8 | Full-match floor spellF1 **0.2433** (not this public number) |
| 2026-10-24 | baseline_smooth125 | shipped | 0.5812 | 0.4242 | 16 | 18.5 | Best on **public** spellF1; loses to smooth50 on full match |
| 2026-10-26 | sm_v1 | E1 | 0.5131 | 0.3000 | 27 | 8.8 | T21 state machine; poss << raw 131; validate OK |
| 2026-10-26 | sm_v1_1 | E1 | 0.6051 | 0.3111 | 28 | 9.3 | Tweak v1.1; validate OK; graded X down (1239→237) |
| | | | | | | | |

Add a new blank row for every model CSV (T30).

---

## Setup / ops notes

| Date | Note |
|------|------|
| 2026-10-24 | T01 — Python 3.14.0 via `.venv/`; numpy 2.5.4, pandas 3.1.0; `import numpy,pandas` OK |
| 2026-10-24 | T02 — Data completeness: OK — tactical atd.txt/xml/homo/smart_tagging + broadcast atd.txt/xml/homo present (mp4 optional, skipped) |
| 2026-10-24 | T03 — Google account ready: yes |
| 2026-10-24 | T04 — `git check-ignore -v data/tactical/` → `.gitignore:32:data/*`; nothing under `data/` staged |
| 2026-10-24 | T11 — `starter/baseline.py` regenerated four `baseline_*.csv` (ball on 113488/199319 frames, 56.9%) |
| 2026-10-24 | T13r signed off 2026-10-24 |
| 2026-10-26 | T20/T21 — `starter/features.py` + `state_machine.py` + `predict_v1.py` → `evaluation/results/sm_v1.csv` |

---

## Experiment notes (E1 / E2)

Short bullets only — why a change was kept or dropped. Prefer poss/dur and failure modes over chasing public spellF1 alone.

| Date | Change | Keep? | Why |
|------|--------|-------|-----|
| 2026-10-24 | Public vs full-match baseline order | note | smooth125 wins public spellF1 (0.4242); docs say smooth50 wins full match (0.2433). Do not pick windows on public 5 min alone. |
| 2026-10-26 | sm_v1 defaults: control_frames=15, control_dist_m=3.0, out_margin=0.02, min_spell_frames=8 | yes (v1) | Hold on missing ball; hysteresis vs flicker; out-of-pitch ≠ missing. Public poss 27 vs raw 131; spellF1 0.30. D recall still weak (margin-only dead-ball). Next: T22 docs / T23 tags. |
| 2026-10-26 | Tweak v1.1 → sm_v1_1: control_frames=25, min_spell=12, missing_near_edge=50, still_near_edge=25 @0.5m/s; keep last_live on D | yes (v1.1) | vs sm_v1: spellF1 0.30→0.31, macroF1 0.51→0.61, poss 27→28, graded X 1239→237. Slight spell gain; D/macro better; poss not closer to 17 yet. |
