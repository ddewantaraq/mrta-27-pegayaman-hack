# Task board

Update **Owner**, **Status**, and check **Acceptance** when done.
Statuses: `todo` | `doing` | `blocked` | `done`

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
| T22 | One-command predict script → valid 199319-row CSV | E2 | 2 | todo | Nov 2 | Documented command writes CSV; `validate_submission.py` passes | T20 |
| T23 | Smart Tagging soft priors (set pieces → D / restart; phases as weak votes) | E1 | 3 | todo | Nov 7 | Priors wired; do **not** copy phase intervals as labels; logged in RESULTS | T21, T22 |
| T24 | Tune via failure modes + possession-count sanity (not public spellF1 alone) | E1 | 3 | todo | Nov 9 | Tuning notes in RESULTS; avoid ranking models only on public 5 min | T23 |

---

## Eval loop

| ID | Task | Owner | Week | Status | Due | Acceptance | Depends on |
|----|------|-------|------|--------|-----|------------|------------|
| T30 | After every model CSV: validate + score + log RESULTS | B | 2–4 | todo | ongoing | Every new CSV has a RESULTS row the same day | T10, T22 |
| T31 | Compare poss / mean dur to public GT (~17 poss, ~14.2 s) | B | 2–3 | todo | Nov 2 | RESULTS notes include poss/dur vs GT ballpark; E1 interprets | T30 |

---

## Ship

| ID | Task | Owner | Week | Status | Due | Acceptance | Depends on |
|----|------|-------|------|--------|-----|------------|------------|
| T40 | Public repo skeleton (no data); pinned deps; README run steps | E2 | 3 | todo | Nov 9 | Clone + data bundle + one command regenerates submission; `data/` not in repo | T22 |
| T41 | Tech doc complete from TECH_DOC_DRAFT | B + E1 + E2 | 4 | todo | Nov 14 | All 6 sections filled; B draft, E1 accuracy, E2 reproduce | T12, T40 |
| T42 | Code freeze: regenerate bit-identical CSV | E2 | 4 | todo | Nov 13 | Two runs produce identical file hash/bytes | T40, T24 |
| T43 | Soft submit Nov 15 (CSV + repo URL + doc) | B (submit) / E2 (check) | 4 | todo | Nov 15 23:59 GMT+7 | Form fields filled; files named `<team_name>.*`; E2 confirmed validate on exact upload | T41, T42 |

---

## Deferred (stretch — do not start before freeze)

| ID | Task | Owner | Week | Status | Due | Acceptance | Depends on |
|----|------|-------|------|--------|-----|------------|------------|
| T90 | Broadcast sync / video CNN | — | — | deferred | after freeze | Only if tactical-only system is frozen and strong | T42 |

---

## Quick filter by owner

- **B:** T01, T02, T03, T10, T12, T13, T30, T31, T41 (draft), T43 (submit)
- **E1:** T13r, T20, T21, T23, T24, T41 (accuracy)
- **E2:** T04, T11, T22, T40, T41 (reproduce), T42, T43 (check)
