# Task board

Update **Owner**, **Status**, and check **Acceptance** when done.
Statuses: `todo` | `doing` | `blocked` | `done`

**Closed 2026-11-17:** `sm_v2_7_compliant.csv` accepted as the final model
(SHA256 `017b9546…`, validated 199,319 rows). All tasks below are `done`;
the video direction (T90) was explored and killed under frozen smoke
boundaries — see [RESULTS.md](RESULTS.md).

Roles: **E1** lead model · **E2** infra/eval · **B** ops/docs — see [TEAM.md](TEAM.md).

---

## Setup / ops

| ID | Task | Owner | Week | Status | Due | Acceptance | Depends on |
|----|------|-------|------|--------|-----|------------|------------|
| T01 | Confirm Python env + `pip install -r requirements.txt` | B | 1 | done | Oct 22 | `python -c "import numpy,pandas"` works; note Python version in RESULTS | — |
| T02 | Confirm non-mp4 data complete vs `data/README.md` | B | 1 | done | Oct 22 | Checklist logged in RESULTS (tactical ATD/xml/homo/tags + broadcast ATD/xml/homo present; mp4 optional) | — |
| T03 | Google account ready for submission form | B | 1 | done | Oct 26 | Team member who will upload can sign in to Google Forms | — |
| T04 | Ensure `data/` never committed (gitignore check) | E2 | 1 | done | Oct 22 | `git check-ignore -v data/tactical/` (or equivalent) shows ignore; no data paths staged | — |

---

## Baseline / understanding

| ID | Task | Owner | Week | Status | Due | Acceptance | Depends on |
|----|------|-------|------|--------|-----|------------|------------|
| T10 | Run `python evaluation/run_evaluation.py` on shipped baselines; paste board into RESULTS | B | 1 | done | Oct 23 | Four baseline rows in [RESULTS.md](RESULTS.md) with macroF1 / spellF1 | T01 |
| T11 | Run `python starter/baseline.py`; confirm four CSVs regenerate | E2 | 1 | done | Oct 24 | Command finishes; `baseline_*.csv` present under `evaluation/results/` | T02, T04 |
| T12 | Plain-language possession rules summary → TECH_DOC_DRAFT | B | 1 | done | Oct 26 | Section 3 in [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) filled from `CHALLENGE.md` (no jargon dump) | — |
| T13 | Document spellF1 vs macroF1 for the team | B | 1 | done | Oct 26 | Short section in TECH_DOC_DRAFT (or TEAM metrics already OK + E1 reviewed) | T10 |
| T13r | E1 reviews T13 accuracy | E1 | 1 | done | Oct 26 | E1 signs off in RESULTS notes or task Status done | T13 |

---

## Modeling (tactical ATD only)

| ID | Task | Owner | Week | Status | Due | Acceptance | Depends on |
|----|------|-------|------|--------|-----|------------|------------|
| T20 | Feature extract from tactical ATD (ball present, nearest team, distances) | E1 | 2 | done | Oct 30 | Module/script produces per-frame features for all 199319 frames; no match data pasted to cloud LLMs | T11 |
| T21 | State machine v1: hold last team; control hysteresis; basic dead-ball | E1 | 2 | done | Nov 2 | Labels H/A/D/X; fewer possessions than `baseline_raw` on public window | T20 |
| T22 | One-command predict script → valid 199319-row CSV | E2 | 2 | done | Nov 2 | Documented command writes CSV; `validate_submission.py` passes | T20 |
| T23 | Smart Tagging soft priors (set pieces → D / restart; phases as weak votes) | E1 | 3 | done | Nov 7 | Priors wired; do **not** copy phase intervals as labels; logged in RESULTS | T21, T22 |
| T24 | Rule/physics upgrades + local structure checks (no GT-driven keep/drop) | E1 | 3 | done | Nov 9 | Compliant sm_v1_1 shipped; RESULTS cites challenge rules; local non-GT checks; no public-board selection | T21, T22 |
| T25 | Cluster / majority near ball (contested override of nearest) | E1 | 3–4 | done | Nov 3 | Features `n_near_*` + `majority_team`; SM contested cand; `sm_v1_3_compliant`; local vs sm_v1_2 flat; Keep? cites team presence | T24 |
| T26 | Soft Smart Tagging redesign (capped set-piece D; missing-aware) | E1 | 4 | done | Nov 7 | Missing-aware SET PIECES→D; `sm_v1_4_compliant`; explained missing ↑; phase votes off; local vs sm_v1_3; later archived vs sm_v1_3 structure | T25 |
| T28 | Missing-ball H/A via Metrica external validation | E1 | 4 | done | Nov 7 | External harness; backdate+gap-mid wins vs sm_v1_3; `sm_v1_5_compliant`; local structure ≥ sm_v1_3 ≪ smooth50; ST off | T25 |
| T29 | Half-time reset + opposite 2nd-half kickoff | E1 | 4 | done | Nov 8 | Clear last_live at HT; D before published 2nd KO; seed opposite; `sm_v1_6_compliant`; local vs sm_v1_5 ≪ smooth50 | T28 |
| T27 | Pass-like travel / control-confirm | E1 | 4 | done | Nov 12 | Closed not promoted (Track B lost); `--control-confirm` archive only | T29 |
| T32 | Short same-team D-sandwich bridge | E1 | 4 | done | Nov 12 | `_bridge_short_dead` max 25f; A+B beat sm_v1_7; promoted sm_v1_8 (later superseded by sm_v2_3) | T27 |

---

## Eval loop

| ID | Task | Owner | Week | Status | Due | Acceptance | Depends on |
|----|------|-------|------|--------|-----|------------|------------|
| T30 | After every model CSV: validate + one pipeline score + log RESULTS | B | 2–4 | done | ongoing | Closed: every model CSV through sm_v2_7 + all research attempts has a RESULTS row; scores logged as gates per ASSESS, not public-board keep/drop | T10, T22 |
| T31 | Log own poss / mean dur (structure sanity) | B | 2–3 | done | Nov 2 | Closed: structure logged per RESULTS entry (final: 339 spells / 8.73 s / s25 7); GT 17/14.2 never used as a tuning target | T30 |

---

## Ship

| ID | Task | Owner | Week | Status | Due | Acceptance | Depends on |
|----|------|-------|------|--------|-----|------------|------------|
| T40 | Public repo skeleton (no data); pinned deps; README run steps | E2 | 3 | done | Nov 9 | Done: README reproduce steps + `requirements-inference.txt` (NumPy/pandas) + packaged weights; `data/` gitignored, never committed | T22 |
| T41 | Tech doc complete from TECH_DOC_DRAFT | B + E1 + E2 | 4 | done | Nov 14 | Done: all 6 sections filled for final sm_v2_7 (approach, assets, rules, missing/noise, limitations, reproduce) | T12, T40 |
| T42 | Code freeze: regenerate bit-identical CSV | E2 | 4 | done | Nov 13 | Done: frozen at sm_v2_7; default predict reproduces `sm_v2_7_compliant.csv` byte-identically (SHA256 `017b9546…`), validate passes | T40, T24 |
| T43 | Soft submit Nov 15 (CSV + repo URL + doc) | B (submit) / E2 (check) | 4 | done | Nov 15 23:59 GMT+7 | Done (team decision 2026-11-17): sm_v2_7 accepted as final; validated CSV + repo + tech doc are the submission package; name files `<team_name>.*` at upload | T41, T42 |

---

## Deferred (stretch — do not start before freeze)

| ID | Task | Owner | Week | Status | Due | Acceptance | Depends on |
|----|------|-------|------|--------|-----|------------|------------|
| T90 | Broadcast sync / video CNN | — | — | done (closed) | after freeze | Explored after freeze: FootAndBall + ROI-gate smokes killed on domain shift under frozen boundaries; YOLO fine-tune (v3) is a separate future commission, not part of the final ship | T42 |

---

## Quick filter by owner

- **B:** T01, T02, T03, T10, T12, T13, T30, T31, T41 (draft), T43 (submit)
- **E1:** T13r, T20, T21, T23, T24, T41 (accuracy)
- **E2:** T04, T11, T22, T40, T41 (reproduce), T42, T43 (check)
