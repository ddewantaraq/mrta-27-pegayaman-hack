# Week 3 — Fusion + harden

**Dates:** 3–9 November 2026  
**Theme:** Smart Tagging priors, tuning discipline, reproducible public repo  
**Weekend lead:** E1 (priors/tune) + E2 (repo pack); B on RESULTS + tech doc

Parent board: [TASKS.md](TASKS.md) · Roles: [TEAM.md](TEAM.md)

---

## Checklist (task IDs)

- [x] **T23** — E1: Smart Tagging soft priors (set pieces → D / restart; weak phase votes)
- [x] **T24** — E1: Tune on failure modes + poss-count sanity (**not** public spellF1 alone)
- [ ] **T30** — B: Keep logging every CSV
- [ ] **T31** — B: Keep poss/dur notes when metrics move
- [ ] **T40** — E2: Public repo skeleton, pinned deps, README run steps (no `data/`)
- [ ] Tech doc → ~70%: fill more of [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) (B), E1 reviews technical claims

**Deferred:** T90 broadcast / video — leave `deferred`.

---

## Day focus

| Day | Focus |
|-----|--------|
| Mon 3–Tue 4 | T23 tagging priors |
| Wed 5–Thu 6 | T24 tune; log before/after in RESULTS |
| Fri 7 | E2: T40 repo skeleton |
| Sat 8 | Repro dry-run: clean clone path documented |
| Sun 9 | Doc push to ~70%; freeze feature wishlist for Week 4 |

---

## Exit criteria

Week 3 is done when **all** are true:

1. Smart Tagging priors are in the pipeline (**T23**) without copying phase intervals as labels.
2. Someone else can follow README and regenerate the submission from provided data (**T40** path ready).
3. [TECH_DOC_DRAFT.md](TECH_DOC_DRAFT.md) is ~70% filled (approach, assets, rules, handling noise started).
4. Tuning notes exist in RESULTS explaining *why* a change was kept (not “public spellF1 went up”).

When met, start [WEEK4.md](WEEK4.md).

---

## Notes

- **2026-10-28 — T23:** Smart Tagging priors in `starter/smart_priors.py`; default on in `predict_v1.py` (`--no-smart-tagging` to disable). Output `sm_v1_3.csv`. See [RESULTS.md](RESULTS.md).
- **2026-10-28 — T24:** Boundary diagnose on `sm_v1_4`; sandwich merge kept; asymmetric restart + no-phase dropped. Best public SM still `sm_v1_4`/`sm_v1_6` (spell 0.313).
