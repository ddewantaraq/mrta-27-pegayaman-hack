# Team model — how to run and score

This folder is our team workspace for **More than a Hack 2027** (ball possession).  
The **runnable model** lives under [`starter/`](../starter/) (ATD features + state machine). This README is for anyone who wants to **regenerate our submission**.

Challenge overview: [../README.md](../README.md) · Rules we encode: [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) · Score log: [RESULTS.md](RESULTS.md) · Process: [TEAM.md](TEAM.md)

---

## What you get

| Piece | Path |
|-------|------|
| Predict | `starter/predict_v1.py` |
| One-shot (predict → validate → score) | `predict_v1.sh` (Mac/Linux/Git Bash) or `predict_v1.bat` (Windows) |
| Features / rules | `starter/features.py`, `state_machine.py` |
| Public scorer | `evaluation/run_evaluation.py` (pipeline **log** only — not a tuning loop) |

**Current default (compliant path):** **`sm_v1_3`** (contested majority-near-ball on sm_v1_2). Restored `sm_v1` kept as `--control-frames 15 …` rollback; `--no-majority-near` for sm_v1_2 cand. Oct 2026 `sm_v2` / archive `sm_v1_3` are **not** the forward default.

**Organiser clarification:** do not train/tune/calibrate parameters on possession GT or the provided video. See [TEAM.md](TEAM.md).

---

## Prerequisites

1. **Python 3.8+** with `numpy` and `pandas` in a venv:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Match data** (not in git). Unpack into `./data` or set `MTAH_DATA_DIR`. See [DATA.md](../DATA.md). Need tactical ATD (Smart Tagging optional; not default).

3. From the **repository root**.

---

## Quick start (recommended)

**Mac / Linux / Git Bash:**

```bash
source .venv/bin/activate
bash predict_v1.sh
```

`predict_v1.sh` uses `python`/`python3` on `PATH` — activate `.venv` first or you get `No module named 'numpy'`.

**Windows (Command Prompt):**

```bat
.venv\Scripts\activate
predict_v1.bat
```

That will predict → validate → run the public scorer once for the **log**. Do **not** keep/drop model changes because the board moved.

`sm_v*.csv` files are **gitignored**. Do not commit them.

---

## Predict only / force a path

```bash
source .venv/bin/activate
python starter/predict_v1.py
python starter/predict_v1.py --out evaluation/results/mine.csv
python evaluation/validate_submission.py evaluation/results/mine.csv
# optional one-off log:
python evaluation/run_evaluation.py
```

---

## What a good run looks like

**Validate:** `OK` and `199319 rows`.

**Structure (no GT):** compare graded-half metrics (short H/A islands, mean spell dur, n_spells, graded X, X-after-D) to the **previous compliant CSV** and to **`evaluation/results/baseline_smooth50.csv`** (structure reference — prefer fewer short islands / longer mean dur than smooth50). Public scoreboard numbers are for the RESULTS log only — not a success gate. See [TEAM.md](TEAM.md) § How we assess.

Full table: [RESULTS.md](RESULTS.md).

---

## Useful flags (defaults = compliant `sm_v1_3`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--control-frames` / `--min-spell-frames` | `25` / `12` | Control hysteresis (~1 s) / short H–A merge |
| `--out-margin` | `0.02` | Out-of-pitch → `D` |
| `--missing-near-edge-frames` / `--still-near-edge-frames` | `50` / `25` | Near-edge missing / still → `D` |
| `--keep-last-live-on-dead` | on | Freeze last team through `D` |
| `--clear-last-live-on-dead` | — | sm_v1 rollback behaviour |

---

## Layout (team)

| File | Purpose |
|------|---------|
| [TASKS.md](TASKS.md) | Board |
| [RESULTS.md](RESULTS.md) | Every scored CSV + experiment notes |
| [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) | Approach + possession rules as implemented |
| [WEEK3.md](WEEK3.md) / [WEEK4.md](WEEK4.md) | Week plans |

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Predict fails opening ATD | `data/` missing or wrong; check `MTAH_DATA_DIR` |
| `No module named 'numpy'` | `.venv` not activated |
| Validate fails row count | Wrong output / truncated write |
| Scoreboard only shows baselines | No `sm_v*.csv` yet — run `predict_v1` first |

Questions about the official challenge: see [FAQ.md](../FAQ.md). Team notes: edit [RESULTS.md](RESULTS.md) when you add a new CSV.
