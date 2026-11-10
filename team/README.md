# Team model — how to run and score

This folder is our team workspace for **More than a Hack 2027** (ball possession).  
The **runnable model** lives under [`starter/`](../starter/) (ATD features + state machine). This README is for anyone who wants to **regenerate our submission**.

Challenge overview: [../README.md](../README.md) · Assess: [../docs/ASSESS.md](../docs/ASSESS.md) · Rules: [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) · Scores: [RESULTS.md](RESULTS.md) · Process: [TEAM.md](TEAM.md)

---

## What you get

| Piece | Path |
|-------|------|
| Predict | `starter/predict_v1.py` |
| One-shot (predict → validate → **assess**) | `predict_v1.sh` / `predict_v1.bat` |
| Dual-track assess only | `assess.sh` / `assess.bat` → `starter/assess.py` |
| Features / rules | `starter/features.py`, `state_machine.py` |
| Public scorer | optional `--public-log` / `PUBLIC_LOG=1` (not a keep gate) |

**Current code default:** **`sm_v1_7`**. Rollback to **`sm_v1_6`**: `--no-vc-splice` (implementation flag) / `--no-half-kickoff-reset` / `--no-backdate-switch --gap-bridge off`. Oct archive rows are **not** the forward path. Algorithm note: [../docs/algorithms/vidal-codina.md](../docs/algorithms/vidal-codina.md).

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

That runs predict → validate → **Track A + Track B assess**. Optional public log: `PUBLIC_LOG=1 bash predict_v1.sh`. Do **not** keep/drop from the public board.

`sm_v*.csv` files are **gitignored**. Do not commit them.

---

## Assess only / predict only

```bash
source .venv/bin/activate
bash assess.sh evaluation/results/sm_v1_7_compliant.csv
bash assess.sh evaluation/results/sm_v1_7_compliant.csv --skip-metrica

python starter/predict_v1.py --out evaluation/results/mine.csv
python evaluation/validate_submission.py evaluation/results/mine.csv
bash assess.sh evaluation/results/mine.csv
```

---

## What a good run looks like

**Validate:** `OK` and `199319 rows`.

**Assess:** see [../docs/ASSESS.md](../docs/ASSESS.md) — Track A structure vs ref + smooth50; Track B Metrica H/A. Full table: [RESULTS.md](RESULTS.md).

---

## Useful flags (defaults = compliant `sm_v1_7`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--control-frames` / `--min-spell-frames` | `25` / `18` | Control hysteresis (~1 s) / short H–A merge |
| `--out-margin` | `0.02` | Out-of-pitch → `D` |
| `--missing-near-edge-frames` / `--still-near-edge-frames` | `50` / `25` | Near-edge missing / still → `D` |
| `--keep-last-live-on-dead` | on | Freeze last team through `D` |
| `--half-kickoff-reset` / `--no-half-kickoff-reset` | on | HT clear; D before 2nd KO; seed opposite team |
| `--no-vc-splice` | off (sm_v1_7 on) | Roll back to **sm_v1_6** (no Vidal-Codina on visible frames) |
| `--backdate-switch` / `--no-backdate-switch` | on | Relabel pending frames once switch confirms |
| `--gap-bridge` | `mid` | `off\|start\|mid\|end` — place switch inside live missing gap |
| `--long-gap-dead-frames` | `0` | Optional missing streak → D (off; hurt externally) |
| `--no-majority-near` | — | Disable contested majority (sm_v1_2 cand) |
| `--smart-tagging` | off | Archived sm_v1_4 soft SET PIECES → D |
| `--set-piece-dead-cap-frames` | `50` | First N frames of set-piece tag → D when ball visible |
| `--clear-last-live-on-dead` | — | sm_v1 rollback behaviour |

---

## Layout (team)

| File | Purpose |
|------|---------|
| [TASKS.md](TASKS.md) | Board |
| [RESULTS.md](RESULTS.md) | Every scored CSV + experiment notes |
| [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) | Approach + possession rules as implemented |
| [../docs/ASSESS.md](../docs/ASSESS.md) | Two-track assess command |
| [../docs/algorithms/](../docs/algorithms/) | Algorithm decision notes |
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
