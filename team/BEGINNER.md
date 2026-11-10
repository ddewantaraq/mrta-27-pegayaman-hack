# Beginner guide (role **B**)

You do **not** need to write the possession model. You keep the team honest: run the official tools, log scores, write plain-language docs, and handle submission logistics.

Roles overview: [TEAM.md](TEAM.md) · Your tasks: [TASKS.md](TASKS.md) (filter **B**) · Assess: [../docs/ASSESS.md](../docs/ASSESS.md)

---

## Setup (once)

From the repo root (`more-than-a-hack-2027/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import numpy, pandas; print('ok')"
```

Mark **T01** done when that works. Note your Python version in [RESULTS.md](RESULTS.md).

**Clarification:** do not ask E1 to keep a change because the public scoreboard improved. Your job is log + validate, not GT tuning. See [TEAM.md](TEAM.md).

---

## Commands you will use often

### Dual-track assess (keep ritual — preferred)

```bash
source .venv/bin/activate
bash assess.sh evaluation/results/<file>.csv
```

Track A = structure vs latest + smooth50; Track B = Metrica H/A. See [../docs/ASSESS.md](../docs/ASSESS.md).

### Optional public board log (not a keep gate)

```bash
python evaluation/run_evaluation.py
# or: bash assess.sh evaluation/results/<file>.csv --public-log
```

### Check a submission before anyone “trusts” it

```bash
python evaluation/validate_submission.py evaluation/results/<file>.csv
```

### Regenerate, validate, and assess the current model

```bash
source .venv/bin/activate
bash predict_v1.sh
```

Windows CMD: `predict_v1.bat`. Writes next `sm_v*.csv`, then runs assess. Add a RESULTS row (**T30**). Do not commit the CSV.

---

## Your tasks (IDs)

| ID | What you do |
|----|-------------|
| **T01** | Install requirements |
| **T02** | Tick data files present (names only — see checklist below) |
| **T03** | Confirm Google account for the submission form |
| **T10** | Run scorer on baselines → RESULTS |
| **T12** | Write possession rules in plain language in TECH_DOC_DRAFT |
| **T13** | Write spellF1 vs macroF1 for the team (E1 reviews) |
| **T30** | After every new model CSV: validate + one pipeline score + RESULTS row (log only) |
| **T31** | Note our own poss / mean duration (structure); not a GT target |
| **T41** | Draft tech doc; E1/E2 fill technical holes |
| **T43** | Submit the form on soft-deadline day (E2 checks files first) |

Week checklists: [WEEK1](WEEK1.md) · [WEEK2](WEEK2.md) · [WEEK3](WEEK3.md) · [WEEK4](WEEK4.md)

---

## Data completeness checklist (T02)

Confirm these exist (ignore `.mp4` unless someone asks):

- `data/tactical/BAR-ATH_tactical_atd.xml`
- `data/tactical/BAR-ATH_tactical_atd.txt`
- `data/tactical/BAR-ATH_tactical_homography.csv`
- `data/tactical/BAR-ATH_tactical_smart_tagging.xml`
- `data/broadcast/BAR-ATH_broadcast_atd.xml`
- `data/broadcast/BAR-ATH_broadcast_atd.txt`
- `data/broadcast/BAR-ATH_broadcast_homography.csv`

Write “T02 complete” + date in RESULTS notes. Do **not** paste file contents into Cursor.

---

## What to put in RESULTS.md

One row per scored file:

`Date | File | Owner | macroF1 | spellF1 | poss | mean_dur_s | Notes`

Example note: `fewer flickers than raw; graded X down vs sm_v1 restore`.

---

## Cursor / cloud LLM safety

**Never paste or @-attach:**

- Anything under `data/`
- Video / screenshots of frames
- ATD lines or coordinates
- Ground-truth label dumps

**OK to paste:**

- macroF1, spellF1, poss, mean_dur_s
- Validation error messages (row count, bad header, etc.)
- Short symptoms: “too many tiny possessions”, “D recall looks weak”

---

## How to claim work

1. Open [TASKS.md](TASKS.md)
2. Set your task **Status** to `doing`
3. When finished → `done` and meet **Acceptance**
