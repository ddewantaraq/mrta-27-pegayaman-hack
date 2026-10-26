# Week 2 — First real system

**Dates:** 27 October – 2 November 2026  
**Theme:** Features + state machine v1 + one-command predict  
**Weekend lead:** E1 (model) + E2 (predict script); B on validate/score

Parent board: [TASKS.md](TASKS.md) · Roles: [TEAM.md](TEAM.md)

---

## Checklist (task IDs)

- [x] **T20** — E1: Feature extract (ball present, nearest team, distances)
- [x] **T21** — E1: State machine v1 (hold, control hysteresis, basic dead-ball)
- [ ] **T22** — E2: One-command predict → valid 199319-row CSV
- [ ] **T30** — B: Validate + score + log every new CSV (start ongoing habit)
- [ ] **T31** — B (+ E1 interpret): poss / mean dur vs public GT (~17 / ~14.2 s)

Do **not** start Smart Tagging fusion (**T23**) until T21+T22 exit criteria are met.

---

## Day focus

| Day | Focus |
|-----|--------|
| Mon 27–Tue 28 | T20 features; E2 scaffolds predict CLI |
| Wed 29–Thu 30 | T21 state machine; first CSV |
| Fri 31 | B: T30 on first model CSV; E1/E2 fix validate failures |
| Sat 1 | Pair: reduce flicker / dead-ball basics; T31 notes |
| Sun 2 | Stabilize v1; document run command for B |

---

## Exit criteria

Week 2 is done when **all** are true:

1. A **non-baseline** CSV validates (`validate_submission.py` OK).
2. Public-window scores are logged in [RESULTS.md](RESULTS.md).
3. Possession count is **clearly less insane** than `baseline_raw` (direction toward GT ballpark, not perfect).
4. B has a copy-paste command from E2 to regenerate / score the latest file.

When met, mark tasks `done` and start [WEEK3.md](WEEK3.md).

---

## Notes

- **2026-10-26 — Tweak v1.1:** ran `predict_v1.py` → `evaluation/results/sm_v1_1.csv` (defaults: control_frames=25, min_spell_frames=12, missing/still near-edge dead-ball). See [RESULTS.md](RESULTS.md).
