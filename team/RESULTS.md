# Results log

### 2026-11-14: sm_v2_3 boundary-aware ownership ensemble — promoted

An ASRF-inspired joint boundary/ownership head is fitted on WOH/WN1/WOY and
Metrica Game 1 only, initialized from v2.2. Its team logits are equally
averaged with v2.2; its boundary head controls missing-frame transition costs.
Observed-ball predictions use raw ensemble signs. Baseline D/X remain.

- Five excluded-fit-match full-timeline suite: spellF1 .580146 -> .599177;
  hidden accuracy .774845 -> .791608, both improve in every match.
- Original Metrica protocol: spellF1 .634473 -> .644263, hidden
  .787772 -> .801381, both improve in each game. Game 1 is training;
  excluded Game 2 alone .611713 -> .616270 spellF1, .728064 -> .747213 hidden.
- Reused additional Game 3 validation: suite spellF1 .555991 -> .584742,
  hidden .688976 -> .711571; natural spellF1 .679070 -> .702290 and hidden
  .530622 -> .560815. No untouched-test claim. Natural spellF1 regresses
  slightly on WMX, Metrica 2 and WR9; complete values are in the method note.
- Challenge: 342 possessions, mean 8.65439 s, <12f 0, <25f 8 (v2.2: 13).
  Graded D/X remain 84581/281. The merge threshold stays 18 frames.
- Thirty-nine tests pass. NumPy export max errors 1.44e-6 team and 6.26e-7
  boundary; natural Metrica 2 packaged labels equal native research labels.
  Format passes, 199319 rows. Three generations are byte-identical, including
  default inference without PyTorch. SHA256:
  `C6D6ECB8B04D8DBA1AF26D944C857A94BD0C0A930E24D5F554D4103AA250D9C6`.
  `--no-boundary-model` exactly restores v2.2. `LATEST` points to v2.3.

Temporal D correction, blunt short-spell suppression, dense decoding and a
12.5 Hz correction model were also tested and held/rejected. Details and
sources: [boundary ownership research](../docs/algorithms/boundary-ownership-research.md).
Official challenge feedback is unavailable; this is the best tested model,
not a claim of a known private score or guaranteed win.

### 2026-11-14: sm_v2_2 residual ownership hybrid — promoted on primary metric

The network sees anonymous player geometry/motion, ball context and the
automatic v1.10 baseline, with a fixed baseline prior. Fit: Sportec
WOH/WN1/WOY + Metrica Game 1 only. It uses raw visible predictions and
Viterbi during missing-ball frames, retaining baseline D/X.

- Five excluded-fit-match full-timeline suite: equal-match spellF1
  .481070 -> .580146; hidden accuracy .665186 -> .774845. Both improve on
  each match; natural spellF1 also improves on each.
- Original Metrica protocol: spellF1 .571573 -> .634473, hidden
  .749873 -> .787772. Both improve in each game. Game 1 is training;
  excluded Game 2 alone: spellF1 .549717 -> .611713, hidden .687964 -> .728064.
- Additional Game 3 validation: suite spellF1 .446470 -> .555991, hidden
  .630178 -> .688976. Natural spellF1 .653978 -> .679070, macroF1
  .798459 -> .803933, but hidden accuracy .575231 -> .530622.
  The strict auxiliary natural-hidden criterion **failed**. Promotion is a
  deliberate tradeoff for the official primary spell metric, not a claim
  that every diagnostic passed. Game 3 was not untouched after prior trials.
- Challenge: 319 possessions, mean 9.27837 s, <12f = 0, <25f = 13;
  graded D/X 84581/281. More <25f spells than v1.10 (2), but fewer than
  the unrestricted set model (21). No challenge GT/public score was used.
- 31 tests pass. NumPy/PyTorch maximum logit difference 1.35e-6 on random
  probes and 1.20e-6 on a full external match; no frame sign differences.
  Packaged Sportec registry labels equal native research labels exactly.
  Three challenge generations are byte-identical, including default inference
  in the base Python environment without importing PyTorch:
  SHA256 `27EA6EF8745D3D0641C0D50EB127F48D8FC40CD214AEB9DB4066CD3E35E297BA`.
  Format validation passes, 199319 rows. Rollback `--no-team-model` exactly
  reproduces v1.10. `LATEST` now points to `sm_v2_2_compliant.csv`.

Details: [temporal ownership research](../docs/algorithms/temporal-ownership-research.md).

The additional official pretrained PathCRF transfer experiment is complete.
Macro/micro emissions, full dynamic CRF, and observed-team constrained CRF
were tested without reference episode boundaries. None outperformed the
residual hybrid on Metrica 2; observed-team CRF suite spellF1 .452605 versus
the hybrid .517666, with worse natural spellF1 and hidden accuracy. These
adapters remain research only. All experiment processes have finished.

### 2026-11-14: temporal ownership research — candidates held

At this stage the approved pointer remained sm_v1_10. A geometry-only player-set temporal
network improves the five-match full-timeline dropout mean spellF1
.481070 -> .573314, but natural Metrica tracks and the historical protocol
regress. It is not promoted. A selective segment gate improves development
results and preserves challenge structure, but fails the additional Metrica 3
hidden-accuracy check. The research CSV `sm_v2_compliant.csv` is not the best
approved submission. The later baseline-aware result is recorded above.
See [temporal ownership research](../docs/algorithms/temporal-ownership-research.md)
for split provenance, the reserved-test limitations, failed checks and exact
reproduction hashes. No challenge public possession labels were scored.

### 2026-11-14: sm_v1_10 calibrated mixed-provider boosted model — promoted

Training is still Sportec J03WOH + Metrica Game 1, with no challenge GT/video.
Numeric tree JSON permits NumPy/pandas-only inference. Training-prior
correction is derived from fitting labels, not target or validation statistics.

- Track A: sm_v1_9 -> sm_v1_10: 331 -> 294 spells, mean 16.8413 -> 10.0673 s;
  <12f 3 -> 0, <25f 8 -> 2; graded D 19214 -> 84581, X remains 281.
  The large D-count change is a material transfer risk, not a tuning target.
- Original Metrica suite: spellF1 0.496131 -> 0.571573; hidden accuracy
  0.678216 -> 0.749873. Both metrics improve in both games. Game 1 is training;
  excluded Game 2 alone: spellF1 0.489793 -> 0.549717, hidden 0.616913 -> 0.687964.
- Whole-timeline natural + 41% block dropout, five excluded fitting matches:
  mean spellF1 0.419488 -> 0.481070, hidden 0.610566 -> 0.665186. Both improve
  separately in Metrica Game 2 and Sportec WMX/WPY/WQQ/WR9. Different protocol:
  these absolute values must not be mixed with the preceding original suite.
- Twenty-three tests pass; portable logits match scikit-learn to 1e-12 on
  16758 probes, including missing routes. Packaged inference reproduces the
  experimental CSV exactly (SHA256 6CB8F5B98618B890FF2F262304A779430313B014294B8791BBFE84E6D7053AA9).
  Validation passes, 199319 rows. Rollback `--dead-model logistic` -> sm_v1_9.

Kept for consistent cross-match and cross-provider gains, with prior correction
and established short-spell rules after the D overlay. Validation matches have
informed variant selection; none is a pristine test for future development.
The hidden challenge score/rank is unknown. Research attempts and rejected
variants are recorded in the [research log](../docs/algorithms/research-round-2026-11-14.md).
[Final method and reproduction](../docs/algorithms/boosted-dead-ball.md).

### 2026-11-14: sm_v1_9 protected learned dead-ball model — promoted

External-only training: Sportec J03WOH and Metrica Game 1. Excluded from fitting:
Sportec J03WN1 and Metrica Game 2. No challenge labels/video were used.
Packaged JSON parameters require no external downloads for inference.

- Track A: spells 359 -> 331, mean 13.7626 -> 16.8413 s,
  <12f 3 -> 3, <25f 9 -> 8; graded D 35057 -> 19214, X stays 281.
- Original Metrica suite: mean spellF1 0.490561 -> 0.496131;
  hidden accuracy 0.511854 -> 0.678216; both games improve both metrics.
  **Game 1 is training/development, not held out.**
- Excluded Sportec J03WN1, natural + 41% block dropout:
  spellF1 0.378973 -> 0.414292; hidden 0.483740 -> 0.548852.
- Excluded Metrica Game 2, natural + 41% block dropout:
  spellF1 0.387076 -> 0.393622; hidden 0.503314 -> 0.642444.
- Full assess command passed against prior LATEST + smooth50. Fifteen tests
  pass. Submission validates (199319 rows). Packaged/default inference is
  SHA256-identical to the experimental protected candidate. Legacy core
  output is unchanged. Rollback: `--no-learned-dead`.

Kept for external evidence and the challenge's live/dead distinction, with
explicit preservation of previous hidden H/A answers. Lower challenge D count
is a material transfer risk. No hidden-challenge performance or winning rank
is established. [Full method/results](../docs/algorithms/learned-dead-ball.md).

Rejected/deferred: same/midpoint gap reconciliation lost hidden accuracy;
physical-boundary-only change lost Sportec spellF1; unprotected learned model
had larger spell gains but lost Sportec hidden accuracy. Do not promote them.

### 2026-11-14: visible live-island candidate (not promoted)

Regenerated sm_v1_8: 359 spells, mean 13.7626 s, <12f 3, <25f 9.
Experimental visible-only D-H/A-D island suppression (<25f): 353 spells,
mean 13.9870 s, <12f 0, <25f 3. Metrica external mean spellF1
0.490561 -> 0.492264, improves both games; hidden accuracy unchanged
0.511854. Kept experimental for the tiny gain; the assessment code actually
accepts nondecreasing hidden accuracy. This candidate did not change LATEST.
No challenge GT used. Four boundary tests pass; candidate submission validates.
See [experiment and reproduction](../docs/algorithms/live-islands.md).

**Metrics only** — no match coordinates, no GT label dumps, no video.

### Active baseline (Nov 2026 clarification)

- **Forward path:** … → **`sm_v1_7`** → **`sm_v1_8`** → **`sm_v1_9`** → **`sm_v1_10`** (calibrated boosted live/dead). **Approved best:** `evaluation/results/LATEST` → `sm_v1_10_compliant.csv`. Historical public-GT-era bare files are not this compliant line.
- **Archive:** rows from historical `sm_v1_1`…`sm_v2` (Oct 2026) used public GT as a keep/drop signal — we **do not continue** that process. Auto predict floors on **LATEST**, not max bare archive.
- **How we assess (not Oct):** two tracks via `assess.sh` — **A** and **B** both **cand vs LATEST vs smooth50** (A = structure CSV; B = Metrica H/A). Public scorer: **optional log only**. Keep? cites **challenge rule / ATD physics / licensed external data**. Ritual: [../docs/ASSESS.md](../docs/ASSESS.md) · [TEAM.md](TEAM.md).

How to fill: `bash assess.sh <cand.csv>` (or predict → assess); add a RESULTS row. See [BEGINNER.md](BEGINNER.md).

---

## Score table

Baselines scored **2026-10-24** (T10). Team models: archive through `sm_v2`, then restore / compliant line.

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
| 2026-10-28 | sm_v1_7 | E1 | 0.6288 | 0.3083 | 31 | 8.3 | Oct archive T24 restart; **not from this branch** (`newrules#1`); do not use |
| 2026-10-28 | sm_v1_8 | E1 | 0.6178 | 0.3130 | 29 | 8.9 | T24 merge + no phase votes; macro down |
| 2026-10-29 | sm_v1_9 | E1 | 0.6341 | 0.4205 | 22 | 11.9 | Fill short D <50f; beats public smooth50 |
| 2026-10-29 | sm_v1_10 | E1 | 0.6406 | 0.3905 | 25 | 10.4 | + restart reset after D≥50f; spell down vs v1.9 |
| 2026-10-29 | sm_v2 | E1 | 0.6397 | 0.4649 | 20 | 13.1 | ARCHIVE: min_spell=50 + short-X; public-GT era |
| 2026-11-03 | sm_v1_restore | E1 | 0.5131 | 0.3000 | 27 | 8.8 | Bit-identical to archived sm_v1; clarification restore |
| 2026-11-03 | sm_v1_1_compliant | E1 | 0.6051 | 0.3111 | 28 | 9.3 | Compliant sm_v1_1 defaults; pipeline log only (not a keep driver) |
| 2026-11-03 | sm_v1_2_compliant | E1 | 0.6065 | 0.3111 | 28 | 9.3 | Compliant sm_v1_2; pipeline log only (not a keep driver) |
| 2026-11-03 | sm_v1_3_compliant | E1 | 0.6063 | 0.3022 | 28 | 9.3 | Compliant sm_v1_3 cluster majority; ≠ Oct archive sm_v1_3; log only |
| 2026-11-07 | sm_v1_4_compliant | E1 | 0.6189 | 0.3043 | 29 | 8.9 | Compliant soft ST; ≠ Oct archive sm_v1_4; log only; later archived vs sm_v1_3 structure |
| 2026-11-07 | sm_v1_5_compliant | E1 | 0.6113 | 0.3111 | 28 | 9.3 | Compliant missing-ball H/A (Metrica external); ≠ Oct archive sm_v1_5; log only |
| 2026-11-08 | sm_v1_6_compliant | E1 | 0.6113 | 0.3111 | 28 | 9.3 | Half-kickoff reset; ≠ Oct archive; log only |
| 2026-11-09 | sm_v1_7_compliant | E1 | 0.6090 | 0.3238 | 25 | 10.4 | Prior LATEST; public numbers log only |
| 2026-11-12 | sm_v1_8_compliant | E1 | — | — | — | — | Prior LATEST; T32 D-bridge<25f; local spells 359 / mean 13.8; Metrica spellF1 0.491; public log TBD |
| 2026-11-14 | sm_v1_9_compliant | Codex | — | — | — | — | Prior LATEST; protected learned D; local 331 spells / mean 16.84 / s12 3 / s25 8; external Metrica spellF1 0.4961; Sportec excluded-match spellF1 0.4143 |
| 2026-11-14 | sm_v1_10_compliant | Codex | — | — | — | — | **LATEST**; calibrated boosted D; local 294 spells / mean 10.07 / s12 0 / s25 2; original Metrica suite 0.5716 (includes training Game 1); five excluded-match whole-timeline suite 0.4811 |
| | | | | | | | |

Add a new blank row for every model CSV (T30). Pipeline metrics in Notes are logs, not keep/drop drivers.

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
| 2026-11-03 | Organiser clarification adopted: no GT/video parameter tuning. Code restored to sm_v1 (`sm_v1_restore.csv` == `sm_v1.csv`). |
| 2026-11-03 | Merged upstream `c073f29` + `f8f2629`: public 5 min not training/tuning; no hand labels on this game; external data = other games only. |
| 2026-11-03 | Compliant sm_v1_1 defaults shipped; `sm_v1_1_compliant.csv`. Local checks: graded X 29754→237; X-after-D 151→0; graded D↑ (edge heuristics). |
| 2026-11-03 | Compliant sm_v1_2: establish-without-X + min_spell 18; `sm_v1_2_compliant.csv`. Local vs v1_1: graded X 237→177; short&lt;12f 4→6; mean dur/n_spells flat; X-after-D 0. |
| 2026-11-03 | Compliant sm_v1_3 (T25): contested majority-near-ball; `sm_v1_3_compliant.csv`. Local vs v1_2: structure flat (X 177, n_spells 428, short&lt;12f 6); ~1.7% frames differ (H/A rebalance). ≠ Oct archive sm_v1_3. |
| 2026-11-07 | Compliant sm_v1_4 (T26): missing-aware soft SET PIECES→D; `sm_v1_4_compliant.csv`. Explained missing (D∩soft ST prior) 10.7%→21.7% of graded missing; graded D↑; n_spells 428→456; mean dur 11.73→10.33; short&lt;12f 6→10. ≠ Oct archive sm_v1_4. |
| 2026-11-07 | Compliant sm_v1_5 (T28): backdate-switch + gap-bridge mid on sm_v1_3 base; Smart Tagging off. External Metrica: mean spellF1 0.314→0.370, hidden-frame team acc 0.505→0.510; LOGO both games. Local vs sm_v1_3: n_spells 428→421; mean dur 11.73→11.92; short&lt;25f 24→16; X/D flat. ≠ Oct archive sm_v1_5. |
| 2026-11-08 | Compliant sm_v1_6 (T29): clear last_live at HT; D on half-2 graded before published 2nd KO (113742); seed opposite of 1st KO team. Local vs sm_v1_5: D 33212→35660; n_sp 421→416; mean 11.92→11.83; s12/s25/X flat; ≪ smooth50. Sanity: 1st live A → 2nd KO H. |
| 2026-11-09 | `sm_v1_7_compliant.csv` on `newrules#1` (VC on visible + sm_v1_6 on missing + half-kickoff-reset). Oct `sm_v1_7.csv` archive-only, **not from this branch**. Local vs sm_v1_6: n_sp 416→406; mean 11.83→12.11; s12 6→4; s25 16→11; D flat 35660; X 177→281; ≪ smooth50. Metrica: spellF1 0.429 / hidden 0.512 vs sm_v1_5. **Promoted** to forward path. Assess: [../docs/ASSESS.md](../docs/ASSESS.md). |
| 2026-11-12 | T27 pass-travel fill (`_apply_pass_travel`) abandoned: force-fill worse; X-only no-op. |
| 2026-11-12 | T27 redesign `_confirm_control`. Track A ↑; Metrica spellF1 ↓. **Closed not promoted.** |
| 2026-11-12 | T32 `_bridge_short_dead` (max 25f, same-team D sandwich). Track A: spells 406→359, mean 12.11→13.76, s25 11→9. Track B: spellF1 0.429→0.491, hidden flat, both games. **Promoted** `sm_v1_8_compliant.csv` → LATEST. [../docs/algorithms/dead-bridge.md](../docs/algorithms/dead-bridge.md). |

---

## Experiment notes (E1 / E2)

Short bullets only — why a change was kept or dropped. **Keep? must cite challenge rule / physics / external data.** Do not keep because public spellF1 moved.

| Date | Change | Keep? | Why |
|------|--------|-------|-----|
| 2026-10-24 | Public vs full-match baseline order | note | smooth125 wins public spellF1 (0.4242); docs say smooth50 wins full match (0.2433). Do not pick windows on public 5 min alone. |
| 2026-10-26 | sm_v1 defaults: control_frames=15, control_dist_m=3.0, out_margin=0.02, min_spell_frames=8 | yes (v1) | Hold on missing ball; hysteresis vs flicker; out-of-pitch ≠ missing. Public poss 27 vs raw 131; spellF1 0.30. D recall still weak (margin-only dead-ball). Next: T22 docs / T23 tags. |
| 2026-10-26 | Tweak v1.1 → sm_v1_1: control_frames=25, min_spell=12, missing_near_edge=50, still_near_edge=25 @0.5m/s; keep last_live on D | yes (v1.1) | vs sm_v1: spellF1 0.30→0.31, macroF1 0.51→0.61, poss 27→28, graded X 1239→237. Slight spell gain; D/macro better; poss not closer to 17 yet. |
| 2026-10-28 | T23 Smart Tagging → sm_v1_3: SET PIECES force D; weak phase team votes; `--no-smart-tagging` to A/B | keep (tune later) | vs sm_v1_2: spellF1 0.31→0.24 (down), macroF1 0.61→0.62, poss 28→25, graded X→0. Set-piece windows over-long as D (~35s tags). |
| 2026-10-28 | Soft set-piece + narrow phases → sm_v1_4: cap=50f; drop ball-in-final-third / box votes; rules still v1.1 | yes (prior default) | vs sm_v1_1: spellF1 0.311→0.313, macroF1 0.605→0.627, D recall 0.31→0.34, abstain 237→19. vs sm_v1_3: spell recovered. Set-piece D frames 43987→2950. |
| 2026-10-28 | Rule bump → sm_v1_5: control=37, min_spell=25 + soft priors | **no** (defaults stay v1.1) | vs sm_v1_4: spellF1 0.313→0.280 (down); poss 29→23 / dur 8.9→11.2 closer to ~17/14 but spell lost. Soft priors kept; hysteresis knobs left at 25/12. |
| 2026-10-28 | T24 sandwich merge → sm_v1_6 | **yes** (correctness; public tie) | Same spell/macro/matched@0.5 as sm_v1_4 on public window. Safer merge (no prev-alone / post-D wipe). Keep as default merge. |
| 2026-10-28 | T24 asymmetric restart → sm_v1_7 (restart_control=12, window=50) | **no** | spellF1 0.313→0.308; matched@0.5 still 8 but spurious 21→23; poss 29→31. Default `restart_window_frames=0`. Oct `sm_v1_7.csv` **not from this branch**. |
| 2026-10-28 | T24 `--no-phase-votes` → sm_v1_8 | **no** | spellF1 tied 0.313; macro 0.627→0.618. Keep phase votes on. |
| 2026-10-29 | Fill short D <50f → sm_v1_9 | **yes (default)** | vs sm_v1_4: spellF1 0.313→0.421, macro 0.627→0.634, matched@0.5 8→9, poss 29→22 / dur 8.9→11.9. Beats public smooth50 (0.326). FULL-match sanity poss 449→366 / mean dur 11.1→13.8 s (plausible). Physical reason: GT stoppages ≥138f; SM short D blips are tracking noise splitting spells. Cutoff 25≈50 in prior probe (stable). |
| 2026-10-29 | Restart reset after D≥50f → sm_v1_10 | **no** (reset default stays 0) | vs sm_v1_9: spellF1 0.421→0.391; matched@0.5 tied 9 but spur 13→16; abstain 15→144; FULL poss 366→418 / dur 13.8→11.5. Macro up slightly; spell/structure worse. Keep `--restart-reset-dead-frames` behind flag. |
| 2026-10-29 | Sandwich min_spell 12→50 + short graded-X fill → sm_v2 | ARCHIVE (not forward path) | Public-GT era keep. Superseded by Nov clarification restore to sm_v1. |
| 2026-11-03 | Restore defaults to sm_v1 (15/3.0/0.02/8; no ST priors; no edge D knobs) | **yes (active baseline)** | Rule-motivated T21; bit-identical to archived sm_v1.csv. Clarification: stop GT keep/drop. |
| 2026-11-03 | Compliant sm_v1_1 (control=25, min_spell=12, edge D, freeze last_live on D) | yes (then superseded by sm_v1_2) | Challenge: controlled switch ~1s; touch≠turnover; dead near edge; keep last through stoppage. Local: graded X 29754→237; X-after-D 151→0. Pipeline numbers logged once — not used to keep/drop. |
| 2026-11-03 | Compliant sm_v1_2: establish-without-X + min_spell 12→18 | yes (then superseded by sm_v1_3) | CHALLENGE: live ball → H/A (not X while establishing); touch ≠ turnover (milder merge ~0.72 s). Local vs sm_v1_1: graded X 237→177; X-after-D 0; short&lt;12f 4→6 (near); mean dur 11.72→11.73; n_spells 428 flat. Pipeline log only — not used to keep/drop. |
| 2026-11-03 | Compliant sm_v1_3: contested majority near ball (radius 3 m, contest ≤1 m) | yes (structure reference; then sm_v1_5) | Physics: team presence in control circle, not only nearest track; override nearest only when duel contested. Local vs sm_v1_2: graded X/n_spells/short islands flat; short&lt;25f 25→24; ~1.7% frames H↔A. Pipeline log only — not used to keep/drop. Rollback: `--no-majority-near`. |
| 2026-11-07 | Compliant sm_v1_4: missing-aware soft SET PIECES → D (cap 50f live; full tag while ball missing; no phase votes) | archived (not default) | Challenge: set piece = dead. Local vs sm_v1_3: explained missing ↑ but mean dur 11.73→10.33 and short islands ↑ — worse structure than sm_v1_3. Kept reachable via `--smart-tagging`. |
| 2026-11-07 | Compliant sm_v1_5: backdate-switch + gap-bridge mid (long-gap D dropped) | yes (then sm_v1_6) | Link & Hoernig 2017: switch at first opposing control; TBP continues through void phases then places turnover in the dark when teams disagree across a gap. Chosen on Metrica Sample_Game_1/2 (event-derived GT + synthetic dropout) vs sm_v1_3; LOGO both games; long-gap→D hurt hidden-frame acc. Local vs sm_v1_3: structure flat/better (n_sp 421, mean 11.92, s25 16); ≪ smooth50. Rollback: `--no-backdate-switch --gap-bridge off`. |
| 2026-11-08 | Compliant sm_v1_6: half-kickoff-reset (HT clear; D pre-2nd KO; seed opposite) | **yes (default)** | Laws of the Game: other team kicks off 2nd half; play dead until restart. Uses published `KICKOFF_VIDEO_FRAMES` (FAQ/DATA), not GT. Local vs sm_v1_5: pre-KO gap all D; short islands flat; mean dur 11.83; ≪ smooth50. Rollback: `--no-half-kickoff-reset`. |
| 2026-11-09 | Vidal-Codina 2022 possession-zone gain as cand, on the sm_v1_5 rules | **no** | Paper §2.3; published radii only. Metrica spellF1↑ but hidden↓. Full table: [../docs/algorithms/vidal-codina.md](../docs/algorithms/vidal-codina.md) §1. |
| 2026-11-09 | VC gap edges at pz=1.0 m: reappear / edge | **no** | SpellF1↑ hidden↓ both games. [../docs/algorithms/vidal-codina.md](../docs/algorithms/vidal-codina.md) §2. |
| 2026-11-09 | **sm_v1_7**: Vidal-Codina on visible frames; missing frames stay sm_v1_5 / sm_v1_6 | **yes (default)** | Metrica vs sm_v1_5: spellF1 0.370→0.429, hidden 0.510→0.512, both games. Local vs sm_v1_6: spells 416→406, mean 11.83→12.11, short&lt;25f 16→11; ≪ smooth50. ≠ Oct archive `sm_v1_7.csv`. Rollback → `sm_v1_6`. [../docs/algorithms/vidal-codina.md](../docs/algorithms/vidal-codina.md) §3. |
| 2026-11-12 | T27 pass-travel post-fill (`--pass-travel`) | **no** | Hop fill abandoned. |
| 2026-11-12 | T27 control-confirm sandwich gate | **no** | Track B lost. Closed. |
| 2026-11-12 | **sm_v1_8** T32 short D-sandwich bridge (≤25f, same H/A) | **yes (default)** | Tracking flicker ≠ stoppage. Track A + B beat sm_v1_7. Rollback `--no-dead-bridge`. [../docs/algorithms/dead-bridge.md](../docs/algorithms/dead-bridge.md). |
