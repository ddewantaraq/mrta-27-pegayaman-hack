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
| 2026-10-28 | sm_v1_2 | E1 | 0.6051 | 0.3111 | 28 | 9.3 | Auto-version smoke (same model as v1.1) |
| 2026-10-28 | sm_v1_3 | E1 | 0.6162 | 0.2381 | 25 | 9.2 | T23 Smart Tagging priors; validate OK |
| 2026-10-28 | sm_v1_4 | E1 | 0.6271 | 0.3130 | 29 | 8.9 | Soft set-piece cap 50f + narrow phases; v1.1 rules |
| 2026-10-28 | sm_v1_5 | E1 | 0.6312 | 0.2800 | 23 | 11.2 | Same priors + control=37 / min_spell=25; spell down |
| 2026-10-28 | sm_v1_6 | E1 | 0.6271 | 0.3130 | 29 | 8.9 | T24 sandwich merge only; tie with sm_v1_4 |
| 2026-10-28 | sm_v1_7 | E1 | 0.6288 | 0.3083 | 31 | 8.3 | T24 merge + restart 12/50; spell down |
| 2026-10-28 | sm_v1_8 | E1 | 0.6178 | 0.3130 | 29 | 8.9 | T24 merge + no phase votes; macro down |
| 2026-10-29 | sm_v1_9 | E1 | 0.6341 | 0.4205 | 22 | 11.9 | Fill short D <50f; beats public smooth50 |
| 2026-10-29 | sm_v1_10 | E1 | 0.6406 | 0.3905 | 25 | 10.4 | + restart reset after D≥50f; spell down vs v1.9 |
| 2026-10-29 | sm_v2 | E1 | 0.6397 | 0.4649 | 20 | 13.1 | min_spell=50 sandwich + short-X fill; beats v1.9 |
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
| 2026-10-28 | T22 — `predict_v1.bat` and `predict_v1.sh` write `evaluation/results/sm_v1_1.csv`, validate, then score. Validate OK (199319 rows). Public score matches the 2026-10-26 sm_v1_1 row (spellF1 0.3111, poss 28, dur 9.3 s) |
| 2026-10-28 | T23 — `smart_priors.py` + state_machine SET PIECES→D / weak phase votes; `sm_v1_3.csv` |
| 2026-10-28 | Soft priors: set-piece dead capped at 50f; drop final-third/box phase votes; CLI `--full-set-piece-dead` / `--no-phase-votes`. `sm_v1_4` / `sm_v1_5`. |
| 2026-10-28 | T24 diagnose `sm_v1_4` @IoU0.5: ~10 missed GT spells; ~7 near-miss (lagged/shifted edges, lags often +19…+59f); ~3 zero same-team overlap; long GT spells often fragmented by predicted D. Spurious: short islands after D / between opposing runs. |
| 2026-10-28 | T24 code: sandwich-only `_merge_short_spells`; `--restart-control-frames` / `--restart-window-frames` (default window=0). `sm_v1_6`–`sm_v1_8`. |
| 2026-10-29 | Short-D fill: `_fill_short_dead` (`min_dead_frames=50`); CLI `--min-dead-frames`. Post-stoppage reset CLI `--restart-reset-dead-frames` / `--restart-establish-frames` (default reset=0). `sm_v1_9` / `sm_v1_10`. |
| 2026-10-29 | `DEFAULT_MIN_SPELL_FRAMES=50` + `_fill_short_abstain`; `sm_v2.csv` (VERSION 2.0). |

---

## Experiment notes (E1 / E2)

Short bullets only — why a change was kept or dropped. Prefer poss/dur and failure modes over chasing public spellF1 alone.

| Date | Change | Keep? | Why |
|------|--------|-------|-----|
| 2026-10-24 | Public vs full-match baseline order | note | smooth125 wins public spellF1 (0.4242); docs say smooth50 wins full match (0.2433). Do not pick windows on public 5 min alone. |
| 2026-10-26 | sm_v1 defaults: control_frames=15, control_dist_m=3.0, out_margin=0.02, min_spell_frames=8 | yes (v1) | Hold on missing ball; hysteresis vs flicker; out-of-pitch ≠ missing. Public poss 27 vs raw 131; spellF1 0.30. D recall still weak (margin-only dead-ball). Next: T22 docs / T23 tags. |
| 2026-10-26 | Tweak v1.1 → sm_v1_1: control_frames=25, min_spell=12, missing_near_edge=50, still_near_edge=25 @0.5m/s; keep last_live on D | yes (v1.1) | vs sm_v1: spellF1 0.30→0.31, macroF1 0.51→0.61, poss 27→28, graded X 1239→237. Slight spell gain; D/macro better; poss not closer to 17 yet. |
| 2026-10-28 | T23 Smart Tagging → sm_v1_3: SET PIECES force D; weak phase team votes; `--no-smart-tagging` to A/B | keep (tune later) | vs sm_v1_2: spellF1 0.31→0.24 (down), macroF1 0.61→0.62, poss 28→25, graded X→0. Set-piece windows over-long as D (~35s tags). |
| 2026-10-28 | Soft set-piece + narrow phases → sm_v1_4: cap=50f; drop ball-in-final-third / box votes; rules still v1.1 | yes (prior default) | vs sm_v1_1: spellF1 0.311→0.313, macroF1 0.605→0.627, D recall 0.31→0.34, abstain 237→19. vs sm_v1_3: spell recovered. Set-piece D frames 43987→2950. |
| 2026-10-28 | Rule bump → sm_v1_5: control=37, min_spell=25 + soft priors | **no** (defaults stay v1.1) | vs sm_v1_4: spellF1 0.313→0.280 (down); poss 29→23 / dur 8.9→11.2 closer to ~17/14 but spell lost. Soft priors kept; hysteresis knobs left at 25/12. |
| 2026-10-28 | T24 sandwich merge → sm_v1_6 | **yes** (correctness; public tie) | Same spell/macro/matched@0.5 as sm_v1_4 on public window. Safer merge (no prev-alone / post-D wipe). Keep as default merge. |
| 2026-10-28 | T24 asymmetric restart → sm_v1_7 (restart_control=12, window=50) | **no** | spellF1 0.313→0.308; matched@0.5 still 8 but spurious 21→23; poss 29→31. Default `restart_window_frames=0`. |
| 2026-10-28 | T24 `--no-phase-votes` → sm_v1_8 | **no** | spellF1 tied 0.313; macro 0.627→0.618. Keep phase votes on. |
| 2026-10-29 | Fill short D <50f → sm_v1_9 | **yes (default)** | vs sm_v1_4: spellF1 0.313→0.421, macro 0.627→0.634, matched@0.5 8→9, poss 29→22 / dur 8.9→11.9. Beats public smooth50 (0.326). FULL-match sanity poss 449→366 / mean dur 11.1→13.8 s (plausible). Physical reason: GT stoppages ≥138f; SM short D blips are tracking noise splitting spells. Cutoff 25≈50 in prior probe (stable). |
| 2026-10-29 | Restart reset after D≥50f → sm_v1_10 | **no** (reset default stays 0) | vs sm_v1_9: spellF1 0.421→0.391; matched@0.5 tied 9 but spur 13→16; abstain 15→144; FULL poss 366→418 / dur 13.8→11.5. Macro up slightly; spell/structure worse. Keep `--restart-reset-dead-frames` behind flag. |
| 2026-10-29 | Sandwich min_spell 12→50 + short graded-X fill → sm_v2 | **yes (default)** | vs sm_v1_9: spellF1 0.421→0.465, macro 0.634→0.640, poss 22→20 / dur 11.9→13.1 (closer to GT 17/14.2), spur 13→11, abstain 15→0, matched@0.5 held at 9. Gates passed. Stable with merge 50–75 in prior probe; merge≥100 banned. |
