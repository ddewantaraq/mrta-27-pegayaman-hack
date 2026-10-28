# Beginner guide (role **B**)

You do **not** need to write the possession model. You keep the team honest: run the official tools, log scores, write plain-language docs, and handle submission logistics.

Roles overview: [TEAM.md](TEAM.md) · Your tasks: [TASKS.md](TASKS.md) (filter **B**)

---

## Setup (once)

From the repo root (`more-than-a-hack-2027/`):

```bash
pip install -r requirements.txt
python -c "import numpy, pandas; print('ok')"
```

Mark **T01** done when that works. Note your Python version in [RESULTS.md](RESULTS.md).

---

## Commands you will use often

### Score everything in `evaluation/results/`

```bash
python evaluation/run_evaluation.py
```

Copy the printed board into [RESULTS.md](RESULTS.md) (see template rows).

### Check a submission before anyone “trusts” it

```bash
python evaluation/validate_submission.py evaluation/results/<file>.csv
```

Replace `<file>.csv` with the real name. If validation fails, the official scorer will skip it — tell E2 immediately.

### Regenerate official baselines (usually E2 owns this; you may watch)

```bash
python starter/baseline.py
```

### Regenerate, validate, and score the current model (T22)

From the repo root, with the match bundle in `data\` (or `MTAH_DATA_DIR` set). One command does all three.

Command Prompt:

```bat
predict_v1.bat
```

Mac Terminal, or Git Bash on Windows:

```bash
bash predict_v1.sh
```

It writes the **next** auto-incremented `evaluation/results/sm_v*.csv` (after `sm_v1` / `sm_v1_1` the next run is `sm_v1_2`, … up to `sm_v1_10` then `sm_v2`, …). The script prints `SUBMISSION …` and `VERSION major.minor`; the wrapper validates that path, then scores. Pass `--out path.csv` only if you must force a name. If predict or validate fails, they stop and do not score. Then add a RESULTS row (T30) using the printed filename. Do not commit the CSV.

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
| **T30** | After every new model CSV: validate + score + RESULTS row |
| **T31** | Note poss / mean duration vs public GT (~17 poss, ~14.2 s) |
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

Example note: `fewer flickers than raw; still more poss than GT 17`.

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
