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
| Smart Tagging | yes | Soft priors: capped SET PIECES→D; weak phase votes (`sm_v1_4`) |
| Tactical homography | planned optional | Debug / video gap-fill only |
| Tactical video | planned optional | Debug only; not required for ATD path |
| Broadcast (all) | planned no (v1) | Separate timeline; deferred until after freeze |

Saying what you tried and **dropped** is useful.

---

## 3. Possession rules implemented

_Owner: B from CHALLENGE.md (T12) · E1 maps to code (`sm_v1_9` defaults)_

### Challenge rules (what we aim for)

- **Live ball → one of H / A:** While the ball is in play, exactly one team has possession: home Barcelona (`H`) or away Athletic Club (`A`). There is no “contested” or “unknown” label in the ground truth on graded frames.
- **Dead ball → D (not “who owns the restart”):** When play is stopped, the answer is simply `D`. We are not asked who will take the throw-in or free kick.
- **Change of possession = controlled takeover:** Possession switches only when the other team gains **controlled** possession. Proxies suggested by the challenge include keeping the ball for a few seconds, or a team completing more than two consecutive passes. A restart after a stoppage has a clear possessor; the hard part is detecting later hand-overs.
- **Contested / loose → keep last team:** If the ball is loose or fought over, possession stays with the last team that had it — do not flip on a single messy frame.
- **What does not count as a change:** A simple touch without control; a deflection that does not intercept; winning an aerial duel without that team then controlling the ball.
- **Possession as an interval:** A possession is a maximal run of one team with the ball alive. `H H H A A` is two possessions. `H H D D H H` is two possessions with a stoppage between them, not one. Dead ball is never a possession.
- **Reference:** If our reading of these rules disagrees with the public 5-minute ground truth, the ground truth is right.

### How code approximates this (`starter/features.py` + `state_machine.py` + `smart_priors.py`)

Features (tactical ATD): ball present, ball speed/position, nearest team and distance (m).

1. **Ungraded halves → `X`.**
2. **Hold on missing ball:** keep last live `H`/`A` (ball missing ≠ dead). Exception: missing for 50 frames after last sighting near the edge → `D`.
3. **Controlled switch:** opposing team must be nearest and within 3 m for 25 consecutive frames (~1 s) before possession flips; otherwise keep last team.
4. **Loose / far / no nearest:** keep last live team (or soft prior / nearest while first establishing).
5. **Dead ball (`D`):** out of pitch; still (≤0.5 m/s) near edge for 25 frames; missing-near-edge as above; plus Smart Tagging SET PIECES for only the **first 50 frames** of each tagged event (full ~35 s tags as `D` hurt spellF1 in `sm_v1_3`). Through short dead stretches, `last_live` stays frozen.
6. **Post-process — sandwich merge:** H/A islands shorter than 12 frames absorbed only when both live neighbours are the **same** opposing team (flicker), not via prev-alone or post-`D` wipe.
7. **Post-process — fill short dead blips (`sm_v1_9`):** `D` islands shorter than 50 frames (~2 s) with the **same** graded live team on both sides → rewrite as that team (tracking noise; public GT stoppages are all ≥ ~5.5 s).
8. **Smart Tagging soft priors (not labels):** weak phase-team votes (build up / progression / attacking transition / creating chances) only when establishing — never overwrite a clear opposing controlled possession; ignore `Team=N/A`. Never copy phase spans wholesale to `H`/`A`.
9. **Establish / cold-start:** while `last_live` is unset, prefer nearest team or soft phase vote over `X` when the ball is visible and close enough.

**Tried and not default:** longer global hysteresis (`sm_v1_5`); asymmetric mid-play restart window (`sm_v1_7`); clear `last_live` after every long `D` (`sm_v1_10`). CLI still exposes `--restart-reset-dead-frames` / `--min-dead-frames` / `--no-smart-tagging` for A/B.

One command: `predict_v1.bat` / `predict_v1.sh` (writes next `sm_v*`, validates, scores).

---

## 4. Missing and noisy tracking

_Owner: E1_

- Ball missing (~41% graded): **Hold** last live team. If no live team yet and the ball is visible/close, prefer nearest team or soft phase vote over `X`; `X` mainly when still nothing to latch onto (or ungraded).
- Track identity noise / no player IDs: distances are team-collective (min home vs min away); no per-player identity.
- Dead-ball detection vs “ball missing ≠ dead”: a missing ball still holds the last live team, unless it stays missing for 50 frames after a near-edge sighting. `D` also when the ball is outside the pitch, or still (≤0.5 m/s) near the edge for 25 frames (v1.1). Short same-team `D` blips (&lt;50 frames) are filled back to the live team post-pass (`sm_v1_9`). Stoppages away from the edge can still be under-called vs GT.
- Smart Tagging as prior (not ground truth): capped set-piece→D + narrow phase votes (`starter/smart_priors.py`). `--full-set-piece-dead` / `--no-phase-votes` / `--no-smart-tagging` for A/B.

---

## 5. Limitations

_Owner: B + E1_

- Where it fails: …
- What we would fix with more time: …
- Known biases (e.g. over-smoothing transitions): …

---

## 6. How to reproduce

_Owner: E2_

Someone with the data bundle and our public repo must regenerate a submission that matches our
kept defaults (`sm_v1_9` path). Step-by-step for external developers: **[README.md](README.md)**.

From the repo root. Python 3.8+ with `numpy` and `pandas`.

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
- Defaults (`sm_v1_9` path, no flags): `control_frames=25`, `control_dist_m=3.0`, `out_margin=0.02`, `min_spell_frames=12`, `missing_near_edge_frames=50`, `still_near_edge_frames=25`, `still_speed_m=0.5`, `set_piece_dead_cap_frames=50`, phase votes on, `restart_window_frames=0`, `min_dead_frames=50`, `restart_reset_dead_frames=0`. Short H/A merge is sandwich-only. Tried `control=37`/`min_spell=25` (`sm_v1_5`), restart 12/50 (`sm_v1_7`), and post-stoppage reset (`sm_v1_10`) — spellF1 down; not default.
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
| 2026-10-28 | E1 | T23: Smart Tagging soft priors (SET PIECES→D; weak phase votes); `sm_v1_3`. |
| 2026-10-28 | E1 | Soft set-piece cap 50f + narrow phases (`sm_v1_4` keep); hysteresis 37/25 (`sm_v1_5`) dropped. |
| 2026-10-28 | E1 | T24: sandwich-only short-spell merge; restart hysteresis tried (`sm_v1_7`) and off by default. |
| 2026-10-29 | E1 | Fill short D <50f (`sm_v1_9` keep; beats public smooth50); post-stoppage reset (`sm_v1_10`) dropped. |
| 2026-10-29 | E1 | §3 rewrite: challenge rules vs code mapping for `sm_v1_9`; §4 cold-start / short-D notes aligned. |
| 2026-10-29 | E1 | Developer runbook [README.md](README.md); root README link; §6 points at it. |
