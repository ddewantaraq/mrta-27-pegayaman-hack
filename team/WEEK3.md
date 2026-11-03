# Week 3 — Compliant rules path + harden

**Dates:** 3–9 November 2026  
**Theme:** Organiser clarification compliance, restore `sm_v1`, rule-justified `sm_v1_1`, reproducible public repo  
**Weekend lead:** E1 (rules path) + E2 (repo pack); B on RESULTS + tech doc

Parent board: [TASKS.md](TASKS.md) · Roles: [TEAM.md](TEAM.md)

---

## Checklist (task IDs)

- [x] **T23** — Smart Tagging module exists (archived era); **not** default under compliant path until rule-only redesign
- [x] **T24** — E1: Compliant upgrades from restored `sm_v1` → `sm_v1_1` (rule citations + local non-GT checks)
- [ ] **T30** — B: Log every CSV (validate + one pipeline score; no keep/drop from board)
- [ ] **T31** — B: Log own poss/dur only (not “closer to GT 17/14.2”)
- [ ] **T40** — E2: Public repo skeleton, pinned deps, README run steps (no `data/`)
- [ ] Tech doc → ~70%: fill more of [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) (B), E1 reviews technical claims

**Deferred:** T90 broadcast / video — leave `deferred`.

---

## Day focus

| Day | Focus |
|-----|--------|
| Mon 3–Tue 4 | Clarification: freeze GT tuning; restore `sm_v1` as active baseline |
| Wed 5–Thu 6 | T24: compliant `sm_v1_1` rule upgrades; RESULTS citations |
| Fri 7 | E2: T40 repo skeleton |
| Sat 8 | Repro dry-run: clean clone path documented |
| Sun 9 | Doc push to ~70%; freeze feature wishlist for Week 4 |

---

## Exit criteria

Week 3 is done when **all** are true:

1. Active defaults are the **compliant** line (`sm_v1` restore and/or approved `sm_v1_1`), not archived `sm_v2`.
2. Someone else can follow README and regenerate the submission from provided data (**T40** path ready).
3. [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) is ~70% filled and states no GT parameter tuning.
4. Every Keep? note cites **rule / physics / external data**, never public spellF1.

When met, start [WEEK4.md](WEEK4.md).

---

## Notes

- **2026-11-03 — Clarification:** No train/tune/calibrate on possession GT or provided video. Public scorer = pipeline log only.
- **2026-11-03 — Restore:** Code defaults match archived `sm_v1.csv` bit-for-bit (`sm_v1_restore.csv`).
- **Archive:** Oct 2026 `sm_v1_1`…`sm_v2` rows kept in RESULTS for history; not the forward path.
