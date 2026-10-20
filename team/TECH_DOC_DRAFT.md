# Technical document draft

This becomes the submission write-up (PDF or Markdown) described in repo `SUBMISSION.md`.
Keep it short — a few pages is enough.

**Section owners:** draft prose **B** · technical accuracy **E1** · reproduce steps **E2**

Team soft deadline for final doc: **14 November 2026** (task **T41**).

---

## 0. Metrics primer (B draft · E1 review — T13)

_Owner: B · Reviewer: E1_

- **spellF1** (ranking): …
- **macroF1** (tie-break / per-frame): …
- Why we do not tune only on the public 5-minute window: …

---

## 1. Approach and why

_Owner: B (structure) · E1 (method)_

What we built, and why that rather than something else.

- Pipeline overview (tactical ATD → features → state machine → CSV): …
- Why optimize for possession spells (spellF1), not frame accuracy: …
- What we deliberately did **not** build (e.g. broadcast sync, video CNN): …

---

## 2. Assets used and how

_Owner: E1 · B lists “dropped”_

| Asset | Used? | How |
|-------|-------|-----|
| Tactical ATD | yes / no | |
| Smart Tagging | yes / no | |
| Tactical homography | yes / no | |
| Tactical video | yes / no | |
| Broadcast (all) | yes / no | |

Saying what you tried and **dropped** is useful.

---

## 3. Possession rules implemented

_Owner: B from CHALLENGE.md (T12) · E1 maps to code_

Plain language first; then how the code approximates it.

- Live ball → one of H / A: …
- Dead ball → D (not “who owns the restart”): …
- Change of possession = controlled takeover: …
- Contested / loose → keep last team: …
- What does **not** count as a change (deflection, touch without control, aerial without control): …
- How our state machine / thresholds implement the above: …

---

## 4. Missing and noisy tracking

_Owner: E1_

- Ball missing (~41% graded): what we do (hold / X / …): …
- Track identity noise / no player IDs: …
- Dead-ball detection vs “ball missing ≠ dead”: …
- Smart Tagging as prior (not ground truth): …

---

## 5. Limitations

_Owner: B + E1_

- Where it fails: …
- What we would fix with more time: …
- Known biases (e.g. over-smoothing transitions): …

---

## 6. How to reproduce

_Owner: E2_

Someone with the data bundle and our public repo must regenerate the **same** submission file.

```bash
# install
…

# run (exact command that writes the submission CSV)
…

# validate
python evaluation/validate_submission.py path/to/<team_name>.csv
```

- Pinned dependency file: …
- Expected output path / filename: …
- Note: match `data/` is **not** in the public repo; point `MTAH_DATA_DIR` or unpack into `./data` per `DATA.md`.

---

## Revision log

| Date | Who | Change |
|------|-----|--------|
| | | Initial stub created |
