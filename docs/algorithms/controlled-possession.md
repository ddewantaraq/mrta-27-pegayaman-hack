# Controlled-possession confirmation (T27 redesign)

## Rule

After sm_v1_7 labels (VC splice + gap-bridge), find **sandwich** H/A spells: live team before and after (skipping X; stopping at D / ungraded) is the same and differs from the middle spell.

Keep the middle spell only if either:

1. **Duration** ≥ `confirm_frames` (default 75 ≈ 3 s), or  
2. **Same-team hops** inside the spell ≥ `confirm_passes` (default 3), counted by [`starter/pass_travel_t27.py`](../../starter/pass_travel_t27.py)

Otherwise relabel the middle to the surrounding team.

CLI: `--control-confirm` / `--no-control-confirm`, `--confirm-frames`, `--confirm-passes`.  
Code: `state_machine._confirm_control`.

## CHALLENGE citation

From [CHALLENGE.md](../../CHALLENGE.md):

- Controlled possession proxies: keep the ball for a few seconds; complete more than two consecutive passes.
- Not a change of possession: simple touch without control; deflection; aerial duel without follow-up control.
- Contested / loose: possession stays with the last team that had it.

## Research context (post-2022 and earlier)

| Work | Year | Relevance |
|------|------|-----------|
| Vidal-Codina et al., *Automatic event detection…* ([arXiv:2202.00804](https://arxiv.org/abs/2202.00804)) | 2022 | PZ/DZ + ball path change → already used as sm_v1_7 VC gain |
| Link & Hoernig, *Individual ball possession in soccer* (PLOS One) | 2017 | Player-level possession; inspired backdate-on-confirm |
| databallpy `add_individual_player_possession` | ongoing | Open implementation of Vidal-Codina |
| Kim et al., **ELASTIC** ([arXiv:2508.09238](https://arxiv.org/abs/2508.09238), [github](https://github.com/hyunsungkim-ds/elastic)) | 2025 | Pass-like kick/receive via ball acceleration + kick distance; optional next hop-recall boost |
| Peral et al., VISAPP 2025 *Temporally Accurate Events…* | 2025 | Min possession duration filter; false positives when ball flies past a player |
| PathCRF (Kim et al.) | 2025 | Learned per-frame possession from trajectories; needs labelled training — not used before code freeze |

## Experiment (2026-11-12)

**Track A (BAR-ATH, confirm 75/3):** spells 406→350, mean 12.11→14.05, s12/s25 flat vs LATEST — structure better.

**Track B (Metrica grid):** all cells `{50,75,125} × {2,3}` lost mean spellF1 vs sm_v1_7 (~0.429 → 0.34–0.40); hidden roughly flat. Gate failed.

**Decision:** **closed, not promoted.** Default OFF. Forward path moved to T32 / sm_v1_8 ([dead-bridge.md](dead-bridge.md)).

```bash
python starter/predict_v1.py --control-confirm --out evaluation/results/sm_v1_8_compliant.csv
bash assess.sh evaluation/results/sm_v1_8_compliant.csv
```
