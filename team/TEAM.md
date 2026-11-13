# Team workspace — More than a Hack 2027

Single source of truth for **who owns what**, deadlines, and how we work.
Task board: [TASKS.md](TASKS.md) · Weeks: [WEEK1](WEEK1.md) · [WEEK2](WEEK2.md) · [WEEK3](WEEK3.md) · [WEEK4](WEEK4.md) · Beginner guide: [BEGINNER.md](BEGINNER.md)

---

## Goal

Produce a validated `frame,label` CSV (199,319 rows: `H` / `A` / `D` / `X`) that beats the published floor **spellF1 0.2433** (`baseline_smooth50`), plus a **public code repo** and a **short technical document**.

Technical direction: tactical ATD → possession state machine → CSV. Optimize for **possession structure** (what private spellF1 rewards), not per-frame accuracy.

**Active model line:** compliant **`sm_v1_8`** (`evaluation/results/LATEST` → `sm_v1_8_compliant.csv` = sm_v1_7 + short D-sandwich bridge). T27 closed not promoted. `sm_v1_4` archived. Oct `sm_v1_1`…`sm_v2` archive only.

**Next compliant levers** (rule/physics citations + local non-GT structure; no public-floor keep/drop):

1. ~~Cluster / majority near ball~~ — **done** (`sm_v1_3`).
2. ~~Soft Smart Tagging redesign~~ — **done / archived** (`sm_v1_4`).
3. ~~Missing-ball H/A (external Metrica)~~ — **done** (`sm_v1_5`).
4. ~~Half-time / opposite 2nd kickoff~~ — **done** (`sm_v1_6`).
5. ~~Visible-ball possession-zone gain~~ — **done** (`sm_v1_7`).
6. ~~Pass-like / control-confirm~~ — **closed not promoted** (T27).
7. ~~Short same-team D-sandwich bridge~~ — **done / promoted** (`sm_v1_8`). [../docs/algorithms/dead-bridge.md](../docs/algorithms/dead-bridge.md).
8. **Homography / broadcast** — deferred (T90).

Old A/B with public floors + GT duration bands is **archived** — do not revive.

---

## Deadlines

| | When |
|--|------|
| **Team soft deadline** | **15 November 2026, 23:59 GMT+7** |
| Official deadline | 17 November 2026, 23:59 CET |
| Code freeze | 13 November 2026 |

Soft deadline is intentional buffer for form issues / last fixes. Do not plan work into Nov 16–17 except emergencies.

---

## Success bar

| Tier | Meaning |
|------|---------|
| **Must** | Valid CSV + reproducible public repo + tech doc by Nov 15 |
| **Target** | Plausibly above floor spellF1 0.2433 (better possession structure vs baselines locally) |
| **Stretch** | Smart Tagging fusion polished; only then consider broadcast |

---

## Capacity

| Day type | Hours / person |
|----------|----------------|
| Weekday | 1–2 h |
| Weekend | 3–4 h |

---

## Roles — fill names

| ID | Role | Name (fill in) | Owns |
|----|------|----------------|------|
| **E1** | Lead model (expert) | _Name_ | Features, state machine, possession/control rules, spellF1 |
| **E2** | Infra + eval (expert) | _Name_ | Scripts, validate/score loop, reproducibility, public repo pack |
| **B** | Ops + docs (beginner) | _Name_ | Run commands, [RESULTS.md](RESULTS.md) log, checklists, tech-doc prose |

Use **E1 / E2 / B** as Owner in [TASKS.md](TASKS.md) until names are filled, then replace with real names if you prefer.

---

## Metrics in one minute

| Metric | What it asks | Ranking? |
|--------|--------------|----------|
| **spellF1** | Did we recover whole possessions (intervals)? | **Yes — board ranks on this** |
| **macroF1** | Right H/A/D letter per frame? | Tie-break only |

### Organiser clarification (Nov 2026) — binding

Takes precedence over earlier docs/emails. **Not allowed:** checking output against the video; hand labels / manual data on this game; uploading provided data to AI services; **training, fitting, or calibrating (including parameter tuning) on the ground-truth possession data or on the provided video.** **Allowed:** automatic checks on provided inputs; inspecting data/video to *understand* possession; using ATD / Smart Tagging / video as *algorithm inputs*; intermediate files; licensed external datasets. Tune parameters only on **external** data (or set them from challenge/physics rules without a GT search).

Public 5-minute GT: **one-off pipeline check** (validate + scorer runs) only. **Forbidden:** iterative keep/drop because public spellF1 / macro / matched / “closer to GT 17/14.2” moved. Official wording (upstream Oct 2026): the 5 minutes are **not training data and not a tuning set** — [RULES.md](../RULES.md), [EVALUATION.md](../EVALUATION.md), [FAQ.md](../FAQ.md). See also [RESULTS.md](RESULTS.md).

### How we assess (different from the Oct public-GT era)

**One command:** `bash assess.sh <cand.csv>` (or `predict_v1.sh` after predict). Full write-up: [../docs/ASSESS.md](../docs/ASSESS.md).

Keep/drop using **rule/physics citation** — not public spellF1.

1. **Track A — structure:** vs prior compliant CSV + `baseline_smooth50.csv`: short H/A &lt;12f/&lt;25f, n_spells, mean duration, graded X/D.
2. **Track B — H/A:** Metrica Sample_Game_1/2 + dropouts; external spellF1 (same `possession_metrics` as the challenge) + hidden-frame team accuracy. Team/missing-ball changes must beat the reference on both, both games.
3. **Public scorer** → optional `--public-log` only. Never the keep driver.

Algorithm notes: [../docs/algorithms/](../docs/algorithms/). Oct public-GT keep/drop is archived. See [RESULTS.md](RESULTS.md).

---

## Data and LLM rules

- Match assets live under `data/` — **do not commit** them; do not redistribute.
- **Do not** upload / paste into cloud Cursor (or other cloud LLMs): video, ATD dumps, frame crops, GT label sequences, or any extracts.
- **Do** share in chat: own poss count / mean duration / short structure notes; metrics from a one-off pipeline log if needed — never as a tuning target.
- Beginner workflow: [BEGINNER.md](BEGINNER.md).

---

## How to claim a task

1. Open [TASKS.md](TASKS.md).
2. Set **Owner** to E1 / E2 / B (or your name).
3. Set **Status** to `doing`.
4. When finished, set Status to `done` and meet **Acceptance**.
5. If stuck, set `blocked` and note why in the daily async.

Statuses: `todo` | `doing` | `blocked` | `done`

---

## Daily async ritual (~15 min)

| Who | Posts |
|-----|--------|
| **B** | Latest line in [RESULTS.md](RESULTS.md) (or “no new CSV today”) — log only, do not push keep/drop from the board |
| **E1** | Next change + **rule / physics / external-data citation** (not “raise spellF1”) |
| **E2** | Blockers / infra status |

Weekend: E1+E2 pair on modeling; B joins for validate + one pipeline log + doc writing.
