# Team model — how to run and score

This folder is our team workspace for **More than a Hack 2027** (ball possession).  
The **runnable model** lives under [`starter/`](../starter/). This README is for anyone who wants to **regenerate our submission and run assess** with no extra model choices.

Challenge overview: [../README.md](../README.md) · Assess: [../docs/ASSESS.md](../docs/ASSESS.md) · Rules: [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) · Scores: [RESULTS.md](RESULTS.md) · Process: [TEAM.md](TEAM.md)

---

## What you get

| Piece | Path |
|-------|------|
| Predict (ship) | `starter/predict_v1.py --out evaluation/results/sm_v2_7_compliant.csv` |
| Assess (A + B + C + D) | `assess.sh` / `assess.bat` → `starter/assess.py` |
| Features / rules | `starter/features.py`, `state_machine.py` |
| Packaged weights | `starter/models/` (NumPy JSON/NPZ; no training step) |
| Fetch external assess data | `starter/external/fetch_metrica.py`, `fetch_sportec.py`, `fetch_skillcorner.py` |

**Approved best (`LATEST`):** **`sm_v2_7`** → `evaluation/results/sm_v2_7_compliant.csv`  
(boosted live/dead + residual ownership + boundary ensemble + fit-only error corrector + event-first touch model).  

Inference: `requirements-inference.txt` (NumPy/pandas only).  
Rollbacks: `--no-event-first` → sm_v2_6; `--no-error-corrector` → sm_v2_3; `--no-boundary-model` → sm_v2_2; `--no-team-model` → sm_v1_10; `--dead-model logistic` → sm_v1_9; `--no-learned-dead` → sm_v1_8.  

Method notes: [push-090-natural](../docs/algorithms/push-090-natural.md) · [boundary](../docs/algorithms/boundary-ownership-research.md) · [temporal ownership](../docs/algorithms/temporal-ownership-research.md) · [boosted dead-ball](../docs/algorithms/boosted-dead-ball.md).

**Organiser clarification:** do not train/tune/calibrate on possession GT or the provided video. See [TEAM.md](TEAM.md).

---

## Prerequisites

1. **Python 3.10+** (3.8+ often works for inference) and a venv:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements-inference.txt
   ```

2. **Challenge match data** (not in git). Unpack into `./data` or set `MTAH_DATA_DIR`. See [DATA.md](../DATA.md). Need tactical ATD.

3. Run all commands from the **repository root**.

---

## Quick start — regenerate submission (no manual effort)

```bash
source .venv/bin/activate
python starter/predict_v1.py --out evaluation/results/sm_v2_7_compliant.csv
python evaluation/validate_submission.py evaluation/results/sm_v2_7_compliant.csv
```

Defaults already are **sm_v2_7**. Always use a **fixed `--out`** for the approved file.

Avoid `bash predict_v1.sh` when you only want the ship artifact — it **auto-bumps** the filename after `LATEST` (e.g. `sm_v2_8_…`) for experiments, and does not update `LATEST`.

`sm_v*.csv` files are **gitignored**. Do not commit them.

---

## External data for Tracks B + C + D

Predict does **not** need these. Full assess does.

### Preferred: fetch scripts

```bash
python starter/external/fetch_metrica.py
python starter/external/fetch_sportec.py --matches J03WMX J03WPY J03WQQ J03WR9
# optional full dump (~2.6 GB): --matches all
python starter/external/fetch_skillcorner.py
```

| Script | Destination |
|--------|-------------|
| `fetch_metrica.py` | `data_external/metrica/` |
| `fetch_sportec.py` | `data_external/sportec/` |
| `fetch_skillcorner.py` | `data_external/skillcorner/` |

All three trees are **gitignored**.

### If fetch returns 403 / fails

Download yourself, then place files so the layout matches a successful fetch:

| Dataset | Manual fallback |
|---------|-----------------|
| **Metrica** sample games 1–2 | From [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) into `data_external/metrica/` (same layout `fetch_metrica.py` creates: per-game folders / CSVs). |
| **Sportec** Open DFL | From [figshare 28196177 v1](https://doi.org/10.6084/m9.figshare.28196177.v1) (CC BY 4.0). Extract so you have `data_external/sportec/*.xml` (e.g. unpack a `sportec.rar` into `data_external/` if the archive already contains a `sportec/` folder). |
| **SkillCorner** Open Data | From [SkillCorner/opendata](https://github.com/SkillCorner/opendata) (MIT). Prefer `fetch_skillcorner.py` (pinned commit + sha256). |

Then re-run assess. Use `--skip-metrica` or `--skip-sportec` only if that track’s data is still missing; omit `--skillcorner` if Track D data is missing.

---

## Assess (Tracks A + B + C + D)

```bash
source .venv/bin/activate
bash assess.sh evaluation/results/sm_v2_7_compliant.csv \
  --ref evaluation/results/sm_v2_6_compliant.csv --skillcorner
# bash assess.sh evaluation/results/sm_v2_7_compliant.csv --skip-sportec
```

| Track | What |
|-------|------|
| **A** | BAR-ATH structure vs ref + `baseline_smooth50` |
| **B** | Metrica H/A (spellF1 + hidden) |
| **C** | Sportec full-timeline suite (J03WMX / WPY / WQQ / WR9) |
| **D** | SkillCorner hold-out (`--skillcorner`) |

Optional: `--public-log` (pipeline log only — not a keep gate).  
Windows: `assess.bat` with the same arguments.

Detail: [../docs/ASSESS.md](../docs/ASSESS.md) · numbers: [RESULTS.md](RESULTS.md).

---

## Useful flags (defaults = compliant `sm_v2_7`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--dead-model` | boosted | live/dead base: boosted, logistic (sm_v1_9), or legacy (sm_v1_8) |
| `--no-team-model` | off | Disable ownership correction → sm_v1_10 |
| `--no-boundary-model` | off | Restore sm_v2_2 ownership head |
| `--no-error-corrector` | off | Disable fit-only corrector → sm_v2_3 |
| `--no-event-first` | off | Disable event-first touch model → sm_v2_6 |
| `--error-corrector-file` | packaged v2 | Override corrector weights (v1 + margin 3 = demoted sm_v2_5) |
| `--boundary-team-model-file` | packaged NPZ | Override boundary ensemble weights |
| `--residual-team-model-file` | packaged NPZ | Override residual ownership weights |
| `--boosted-model-file` | packaged JSON | Override boosted live/dead weights |
| `--no-learned-dead` | off | Roll back to geometric sm_v1_8 dead-ball |
| `--control-frames` / `--min-spell-frames` | `25` / `18` | Control hysteresis / short H–A merge |
| `--half-kickoff-reset` / `--no-half-kickoff-reset` | on | HT clear; D before 2nd KO; seed opposite |
| `--no-vc-splice` | off | No Vidal-Codina on visible frames |
| `--gap-bridge` | `mid` | Place switch inside live missing gap |
| `--smart-tagging` | off | Archived soft SET PIECES → D |

---

## Layout (team)

| File | Purpose |
|------|---------|
| [TASKS.md](TASKS.md) | Board |
| [RESULTS.md](RESULTS.md) | Every scored CSV + experiment notes |
| [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) | Approach + possession rules |
| [../docs/ASSESS.md](../docs/ASSESS.md) | Four-track assess + leakage check |
| [../research/README.md](../research/README.md) | Archived experiments (not needed for CSV) |
| [../docs/algorithms/](../docs/algorithms/) | Method decision notes |

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Predict fails opening ATD | `data/` missing or wrong; check `MTAH_DATA_DIR` |
| `No module named 'numpy'` | `.venv` not activated |
| Validate fails row count | Wrong / truncated output |
| Track B fails / no Metrica | Run `fetch_metrica.py` or fill `data_external/metrica/` |
| Track C fails / no Sportec | Run `fetch_sportec.py` or extract into `data_external/sportec/` |
| Track D fails / no SkillCorner | Run `fetch_skillcorner.py` or omit `--skillcorner` |
| Fetch 403 | Manual download into `data_external/` (table above) |
| Assess fails on auto-bump kwargs | Use fixed `--out …/sm_v2_7_compliant.csv` |

Questions about the official challenge: [FAQ.md](../FAQ.md). Team notes: [RESULTS.md](RESULTS.md).
