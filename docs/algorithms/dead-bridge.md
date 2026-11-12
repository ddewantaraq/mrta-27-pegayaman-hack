# Short D-sandwich bridge (T32) → sm_v1_8

## Rule

After sm_v1_7 labels, find graded **D** runs whose live neighbors (skipping X; stopping at another D / ungraded) are the **same** H or A team. If `len(D) < max_dead_bridge_frames` (default **25** = 1 s), fill those D frames to that team.

CLI: `--dead-bridge` / `--no-dead-bridge`, `--max-dead-bridge-frames`.  
Code: `state_machine._bridge_short_dead`.

Far from T27 (no pass/travel/control proxies).

## Citation

- Challenge: dead ball = referee stoppage, and `H…D…H` is two possessions when the stoppage is real.
- Heuristic: a **sub-second** D between the **same** live team is treated as tracking / edge-flicker noise, not a stoppage. Longer D and D between different teams stay D.

## Keep (2026-11-12)

| Track | vs LATEST (sm_v1_7) |
|-------|---------------------|
| A | spells 406→359, mean 12.11→13.76, s12 4→3, s25 11→9, D 35660→35057 |
| B | mean spellF1 0.429→0.491, hidden ~flat; both Metrica games beat LATEST |

**Promoted:** `evaluation/results/LATEST` → `sm_v1_8_compliant.csv`. Rollback: `--no-dead-bridge`.
