# Technical document draft

This becomes the submission write-up (PDF or Markdown) described in repo `SUBMISSION.md`.
Keep it short — a few pages is enough.

**Section owners:** draft prose **B** · technical accuracy **E1** · reproduce steps **E2**

Team soft deadline for final doc: **14 November 2026** (task **T41**).

---

## 0. Metrics primer (B draft · E1 review — T13)

_Owner: B · Reviewer: E1_ — **T13r signed off 2026-10-24**

- **spellF1** (ranking): Treats each possession as one object — a maximal run of `H` or of `A` (dead ball is a stoppage, not a possession). Predicted spells are matched one-to-one to ground-truth spells by temporal overlap (IoU), then F1 is averaged over IoU thresholds 0.3–0.7. The leaderboard ranks on this. A constant guess scores 0. Flickering labels that invent many tiny possessions hurt spellF1 badly even if many frames look “half right.”
- **macroF1** (tie-break / per-frame): Mean of the per-class F1 scores for `H`, `A`, and `D` on every graded frame. Measures “right letter, right frame.” Useful diagnostic; reported beside spellF1 and breaks ties only.
- Why we do not tune only on the public 5-minute window: That window has only ~17 possessions. On it (2026-10-24 scores), `baseline_smooth125` leads spellF1 (0.4242) while `baseline_smooth50` is lower (0.3256) — but on the **full** match the published floor is `baseline_smooth50` at spellF1 **0.2433**, and smooth125 is worse. Picking hyperparameters on the public board can choose the wrong system. Use the public window to check the pipeline (alignment, file shape, poss/dur sanity), not as the final ranking signal.

Illustration from our T10 run (public GT): raw invents 131 possessions (mean 2.0 s) vs GT 17 / 14.2 s → spellF1 0.0378 despite macroF1 still ~0.60. Structure matters more than frame accuracy for the board.

---

## 1. Approach and why

_Owner: B (structure) · E1 (method)_

What we built, and why that rather than something else.

- Pipeline overview (tactical ATD → features → state machine → CSV): … _(Week 2+)_
- Why optimize for possession spells (spellF1), not frame accuracy: see section 0.
- What we deliberately did **not** build (e.g. broadcast sync, video CNN): … _(Week 2+)_

---

## 2. Assets used and how

_Owner: E1 · B lists “dropped”_

| Asset | Used? | How |
|-------|-------|-----|
| Tactical ATD | planned yes | Core input for possession (Week 2+) |
| Smart Tagging | planned yes | Soft priors for dead ball / restarts (Week 3) |
| Tactical homography | planned optional | Debug / video gap-fill only |
| Tactical video | planned optional | Debug only; not required for ATD path |
| Broadcast (all) | planned no (v1) | Separate timeline; deferred until after freeze |

Saying what you tried and **dropped** is useful.

---

## 3. Possession rules implemented

_Owner: B from CHALLENGE.md (T12) · E1 maps to code_

Plain language first; then how the code approximates it.

- **Live ball → one of H / A:** While the ball is in play, exactly one team has possession: home Barcelona (`H`) or away Athletic Club (`A`). There is no “contested” or “unknown” label in the ground truth on graded frames.
- **Dead ball → D (not “who owns the restart”):** When play is stopped, the answer is simply `D`. We are not asked who will take the throw-in or free kick.
- **Change of possession = controlled takeover:** Possession switches only when the other team gains **controlled** possession. Proxies suggested by the challenge include keeping the ball for a few seconds, or a team completing more than two consecutive passes. A restart after a stoppage has a clear possessor; the hard part is detecting later hand-overs.
- **Contested / loose → keep last team:** If the ball is loose or fought over, possession stays with the last team that had it — do not flip on a single messy frame.
- **What does not count as a change:** A simple touch without control; a deflection that does not intercept; winning an aerial duel without that team then controlling the ball.
- **Possession as an interval:** A possession is a maximal run of one team with the ball alive. `H H H A A` is two possessions. `H H D D H H` is two possessions with a stoppage between them, not one. Dead ball is never a possession.
- **Reference:** If our reading of these rules disagrees with the public 5-minute ground truth, the ground truth is right.
- **How our state machine / thresholds implement the above (T20–T22, `sm_v1_1`):** Features from tactical ATD (`starter/features.py`): ball present, nearest team, min distances (m). `starter/state_machine.py` then: (1) ungraded halves → `X`; (2) ball missing → **hold** last `H`/`A`, except a long gap after the ball was last seen near the edge; (3) switch team only after 25 frames (~1 s) of the other team being nearest and within 3 m; (4) contested/far → keep last team; (5) dead ball = ball outside the pitch, a still ball near the edge for 25 frames, or missing for 50 frames after a near-edge sighting; (6) merge H/A islands shorter than 12 frames (~0.5 s). One command: `predict_v1.bat` on Windows, or `bash predict_v1.sh` on Mac (writes the next auto `sm_v*.csv`, then validates and scores). Smart Tagging not used yet (T23).

---

## 4. Missing and noisy tracking

_Owner: E1_

- Ball missing (~41% graded): what we do (hold / X / …): **Hold** last live team; `X` only if no live team established yet on graded frames.
- Track identity noise / no player IDs: distances are team-collective (min home vs min away); no per-player identity.
- Dead-ball detection vs “ball missing ≠ dead”: a missing ball still holds the last live team, unless it stays missing for 50 frames after a near-edge sighting. `D` also when the ball is outside the pitch, or still (≤0.5 m/s) near the edge for 25 frames (v1.1). Stoppages away from the edge are still under-called vs GT.
- Smart Tagging as prior (not ground truth): not wired yet (T23).

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

From the repo root. No virtual environment required. Python 3.8+ with `numpy` and `pandas`.

```bat
pip install -r requirements.txt
predict_v1.bat
```

Mac, from the repo root:

```bash
pip install -r requirements.txt
bash predict_v1.sh
```

On Windows, Git Bash can run the same `.sh`: `"C:\Program Files\Git\bin\bash.exe" predict_v1.sh`.

Both scripts run, in order: `starter/predict_v1.py`, then `validate_submission.py` on the path that script prints as `SUBMISSION ...`, then `run_evaluation.py`. With no `--out`, predict picks the **next** `sm_v*` under `evaluation/results/` (scan latest → bump). Naming: `sm_v1.csv` = 1.0, `sm_v1_1.csv` … `sm_v1_10.csv` = 1.10, then `sm_v2.csv` = 2.0, etc. Minor max is **10**. Pass `--out path.csv` to skip auto-versioning. They stop if predict or validate fails. The `.sh` uses `python` when it is on `PATH`, otherwise `python3`.

- Dependency file: `requirements.txt` (`numpy`, `pandas`). Exact version pins are T40.
- Output: next `evaluation/results/sm_v*.csv` — 199,319 data rows, header `frame,label`, labels `H`/`A`/`D`/`X`. Those filenames are gitignored.
- Defaults (v1.1, no flags): `control_frames=25`, `control_dist_m=3.0`, `out_margin=0.02`, `min_spell_frames=12`, `missing_near_edge_frames=50`, `still_near_edge_frames=25`, `still_speed_m=0.5`.
- Match `data/` is **not** in the public repo. Unpack the bundle into `./data`, or set `MTAH_DATA_DIR` first (`set MTAH_DATA_DIR=D:\mtah2027` in Command Prompt, `export MTAH_DATA_DIR=/path/to/mtah2027` on Mac). See `DATA.md`.

---

## Revision log

| Date | Who | Change |
|------|-----|--------|
| 2026-10-20 | team | Initial stub created |
| 2026-10-24 | B / E1 | T12 section 3 possession rules; T13/T13r section 0 metrics primer |
| 2026-10-26 | E1 | T20/T21 sm_v1 implementation notes in §3–§4 |
| 2026-10-26 | E1 | Tweak v1.1: sm_v1_1 (longer control, edge dead-ball heuristics, keep last_live on D) |
| 2026-10-28 | E2 | T22: `predict_v1.bat` (Windows) and `predict_v1.sh` (Mac / Git Bash) predict, validate, and score. |
| 2026-10-28 | E2 | Auto-increment `sm_v*` CSV versions on predict (1.0…1.10 → 2.0). |
