# Research archive (not required to regenerate the submission)

Training scripts, rejected candidates, PathCRF / camera / high-res probes, and
unit tests from the Nov 2026 external-data rounds.

**Ship / reproduce path:** use the repo root README and `starter/predict_v1.py`
(defaults = **sm_v2_3**). You do **not** need this folder to regenerate
`evaluation/results/sm_v2_3_compliant.csv`.

| Path | Contents |
|------|----------|
| `experiments/` | Fit / ablation / export scripts formerly under `starter/external/` |
| `training/` | PyTorch training modules (NumPy inference stays in `starter/`) |
| `tests/` | Research unit tests |
| `docs/` | Long research logs |
| `requirements/` | Optional heavy deps (`neural`, `pathcrf`, `research`) |

Inference deps: `requirements-inference.txt` at repo root.

To re-run an experiment:

```bash
source .venv/bin/activate
python -m pip install -r research/requirements/requirements-research.txt
PYTHONPATH=starter:starter/external:research/training:research/experiments \
  python research/experiments/evaluate_approved.py --candidate 2.3 --reference 2.2
```
