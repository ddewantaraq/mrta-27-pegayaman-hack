# Vidal-Codina possession (§2.3) → sm_v1_7

Paper: Ferran Vidal-Codina et al., *Automatic event detection in football using tracking data*, Sports Engineering, 2022 ([arXiv:2202.00804](https://arxiv.org/abs/2202.00804), §2.3 only).

**Model version:** `sm_v1_7` · Artifact: `evaluation/results/sm_v1_7_compliant.csv` · **LATEST** pointer: `evaluation/results/LATEST`  
(Oct archive `sm_v1_7.csv` is unrelated — not from this branch.)

---

## 1. Plain VC on all frames (not kept)

**Date:** 2026-11-09 · **Decision:** not kept.

Replaces “nearest player, sustained for 1 s” with the paper’s possession step on every frame. Dead ball / missing-ball / backdate / gap-bridge / half-kickoff unchanged around it.

- Possession zone / duel zone; gain needs direction or speed change (databallpy 10° / 5 m/s); SG smooth 7 / order 2.
- Radii tried on Metrica only: 0.5/1.0, **1.0/1.0**, 1.5/1.5.

| Candidate | spellF1 | hidden-frame team acc |
|-----------|---------|------------------------|
| vc pz 1.0 / dz 1.0 | **0.4247** | 0.4703 |
| **sm_v1_5** | 0.3701 | **0.5102** |
| smooth50-equivalent | 0.3210 | 0.1528 |

SpellF1↑ but **hidden↓** both games → dropped. About 41% of graded BAR-ATH frames have no ball, so hidden matters.

---

## 2. Gap-edge variants (not kept)

**Date:** 2026-11-09 · **Decision:** not kept.

After a live missing gap, briefly allow nearby-team switch (**reappear**) or nearby + &gt;10° across the gap (**edge**). Path-change still required on continuous visible play.

| Candidate | spellF1 | hidden |
|-----------|---------|--------|
| reappear | 0.4273 | 0.4785 |
| edge | 0.4254 | 0.4730 |
| plain VC | 0.4247 | 0.4703 |
| **sm_v1_5** | 0.3701 | **0.5102** |

Hidden still loses → not kept.

---

## 3. sm_v1_7 (kept)

**Date:** 2026-11-09 · **Decision:** kept as **`sm_v1_7`**.

- Ball **visible:** Vidal-Codina gain (pz/dz 1.0 m).
- Ball **missing:** same missing-ball rules as `sm_v1_5` / `sm_v1_6` (backdate + gap-bridge mid; half-kickoff on submission path).
- One short-spell merge after join.

### External (Metrica) — model versions

| Model | spellF1 | hidden-frame team accuracy |
|-------|---------|----------------------------|
| **sm_v1_7** | **0.4286** | **0.5117** |
| sm_v1_5 | 0.3701 | 0.5102 |

| Game | sm_v1_5 spell / hidden | sm_v1_7 |
|------|------------------------|---------|
| Sample Game 1 | 0.379 / 0.610 | 0.437 / 0.611 |
| Sample Game 2 | 0.361 / 0.410 | 0.420 / 0.412 |

### Local structure (BAR-ATH graded)

| Model | spells | mean s | short &lt;25f | D | X |
|-------|--------|--------|---------------|---|---|
| sm_v1_5 | 421 | 11.92 | 16 | 33212 | 177 |
| sm_v1_6 | 416 | 11.83 | 16 | 35660 | 177 |
| **sm_v1_7** | **406** | **12.11** | **11** | 35660 | 281 |
| smooth50 | 594 | 7.24 | 124 | 51368 | 0 |

Higher X is mostly 1st-kickoff until first validated gain (paper “no possession” until path-change). Most leftover shorts are D-sandwiched beeps.

**Rollback to sm_v1_6:** disable the visible-VC join in predict defaults (see `predict_v1.py` / team README flags).

Assess: [../ASSESS.md](../ASSESS.md).
