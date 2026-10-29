# Team model — how to run and score

This folder is our team workspace for **More than a Hack 2027** (ball possession).  
The **runnable model** lives under [`starter/`](../starter/) (state machine + Smart Tagging soft priors). This README is for anyone who wants to **regenerate our submission and see public scores**.

Challenge overview: [../README.md](../README.md) · Rules we encode: [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) · Score log: [RESULTS.md](RESULTS.md)

---

## What you get

| Piece | Path |
|-------|------|
| Predict | `starter/predict_v1.py` |
| One-shot (predict → validate → score) | `predict_v1.sh` (Mac/Linux/Git Bash) or `predict_v1.bat` (Windows) |
| Features / rules / priors | `starter/features.py`, `state_machine.py`, `smart_priors.py` |
| Public scorer | `evaluation/run_evaluation.py` |

**Current default (best kept run):** behaviour of **`sm_v2`** — ATD state machine + capped set-piece priors + fill short same-team dead-ball blips (`min_dead_frames=50`) + sandwich merge (`min_spell_frames=50`) + fill short graded `X`.

---

## Prerequisites

1. **Python 3.8+** with `numpy` and `pandas`:

   ```bash
   pip install -r requirements.txt
   ```

2. **Match data** (not in git). Enrol, download the bundle, then either:
   - unpack into `./data` at the repo root, or  
   - set `MTAH_DATA_DIR` to the bundle root  

   See [DATA.md](../DATA.md). You need at least tactical ATD + Smart Tagging XML (same as our predict path).

3. From the **repository root** (directory that contains `starter/` and `evaluation/`).

---

## Quick start (recommended)

**Mac / Linux / Git Bash:**

```bash
pip install -r requirements.txt
bash predict_v1.sh
```

**Windows (Command Prompt):**

```bat
pip install -r requirements.txt
predict_v1.bat
```

That will:

1. Extract features and run the state machine  
2. Write the **next** auto-versioned file under `evaluation/results/`  
   (`sm_v1.csv` → `sm_v1_1.csv` … → after `sm_v1_10.csv` comes `sm_v2.csv`)  
3. Validate the CSV (199,319 rows)  
4. Score **all** CSVs in `evaluation/results/` against the **public 5-minute** ground truth  

`sm_v*.csv` files are **gitignored** (only `baseline_*.csv` ship in a fresh clone). You must run predict locally to produce our labels.

---

## Predict only / force a path

```bash
# next auto sm_v* name
python starter/predict_v1.py

# fixed path (skips auto-version bump)
python starter/predict_v1.py --out evaluation/results/mine.csv

python evaluation/validate_submission.py evaluation/results/mine.csv
python evaluation/run_evaluation.py
```

---

## What a good run looks like

**Validate:** `OK` and `199319 rows`.

**Scoreboard** (public GT only — not the private full-match board): look for your new `sm_v*` row. With current defaults you should land near our logged **`sm_v2`** public numbers:

| Metric | Our `sm_v2` (logged) | Public `baseline_smooth50` | Public GT ballpark |
|--------|----------------------|----------------------------|--------------------|
| spellF1 | ~0.46 | 0.326 | — |
| macroF1 | ~0.64 | 0.621 | — |
| possessions | ~20 | 26 | ~17 |
| mean dur (s) | ~13.1 | 10.8 | ~14.2 |

Full table and keep/drop notes: [RESULTS.md](RESULTS.md).  
**Ranking in the competition** uses private full-match spellF1; local scores are a proxy. Do not tune only to beat public `baseline_smooth125` (see [EVALUATION.md](../EVALUATION.md)).

---

## Useful flags (defaults = `sm_v2` path)

| Flag | Default | Meaning |
|------|---------|---------|
| `--min-dead-frames` | `50` | Fill short `D` islands when same team on both sides (`0` = off) |
| `--set-piece-cap-frames` | `50` | Only first N frames of each SET PIECES tag → force `D` |
| `--no-smart-tagging` | off | Disable Smart Tagging priors |
| `--no-phase-votes` | off | Keep set-piece prior; disable weak phase-team votes |
| `--full-set-piece-dead` | off | Legacy: full tagged set-piece span as `D` (hurts spellF1) |
| `--restart-reset-dead-frames` | `0` | After long `D`, clear last team and re-establish (`50` tried as `sm_v1_10`, dropped) |
| `--control-frames` / `--min-spell-frames` | `25` / `50` | Mid-play switch hysteresis / sandwich H–A flicker merge |

Example A/B without Smart Tagging:

```bash
python starter/predict_v1.py --no-smart-tagging --out evaluation/results/ab_no_tags.csv
python evaluation/validate_submission.py evaluation/results/ab_no_tags.csv
python evaluation/run_evaluation.py
```

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
| Predict fails opening ATD / tags | `data/` missing or wrong; check `MTAH_DATA_DIR` |
| Validate fails row count | Wrong output / truncated write |
| Your spellF1 ≪ 0.46 with defaults | Stale code, or `--min-dead-frames 0` / `--min-spell-frames 12` / `--full-set-piece-dead` |
| Scoreboard only shows baselines | No `sm_v*.csv` yet — run `predict_v1` first |

Questions about the official challenge: see [FAQ.md](../FAQ.md). Team notes: edit [RESULTS.md](RESULTS.md) when you add a new CSV.