# Team workspace — More than a Hack 2027

Single source of truth for **who owns what**, deadlines, and how we work.
Task board: [TASKS.md](TASKS.md) · Weeks: [WEEK1](WEEK1.md) · [WEEK2](WEEK2.md) · [WEEK3](WEEK3.md) · [WEEK4](WEEK4.md) · Beginner guide: [BEGINNER.md](BEGINNER.md)

---

## Goal

Produce a validated `frame,label` CSV (199,319 rows: `H` / `A` / `D` / `X`) that beats the published floor **spellF1 0.2433** (`baseline_smooth50`), plus a **public code repo** and a **short technical document**.

Technical direction: tactical ATD + Smart Tagging → possession state machine → CSV. Optimize for **spellF1**, not per-frame accuracy. Broadcast / video CNN are out of scope until after code freeze.

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

Public 5-minute GT (~17 possessions, mean ~14.2 s) is for **pipeline sanity**, not for picking the final model. See repo `EVALUATION.md`.

---

## Data and LLM rules

- Match assets live under `data/` — **do not commit** them; do not redistribute.
- **Do not** upload / paste into cloud Cursor (or other cloud LLMs): video, ATD dumps, frame crops, GT label sequences, or any extracts.
- **Do** share in chat: macroF1, spellF1, poss count, mean duration, short notes (“too many flickers”).
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
| **B** | Latest line in [RESULTS.md](RESULTS.md) (or “no new CSV today”) |
| **E1** | Next experiment / what they will try |
| **E2** | Blockers / infra status |

Weekend: E1+E2 pair on modeling; B joins for validate/score + doc writing.
