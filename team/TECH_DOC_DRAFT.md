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
- **Organiser clarification (Nov 2026):** parameters must **not** be trained, fitted, or calibrated on the possession ground truth or the provided video. The public 5-minute GT is for **pipeline verification** only (file shape, scorer runs). We set knobs from challenge rules / ATD physics (or licensed external data), not from public spellF1 keep/drop.
- Why public spellF1 alone was always a weak private proxy: smooth125 can lead the public window while losing on the full match (published floor smooth50 spellF1 **0.2433**). That is a second reason not to chase the public board — the first is the clarification itself.
- **How we assess (vs Oct public-GT keep/drop):** local CSV structure vs **`sm_v1_3_compliant`** (structure reference) **and** vs shipped `baseline_smooth50.csv`. Missing-ball H/A correctness is measured only on **external** Metrica sample games (event-derived labels + synthetic ball dropout). Public board = log only. Detail: team [TEAM.md](TEAM.md) § How we assess.

Illustration (pipeline sanity, T10): raw invents far more possessions than a coach would count → structure matters more than frame accuracy.

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
| Tactical ATD | yes | Core input for possession (nearest + cluster majority near ball) |
| Smart Tagging | archived | sm_v1_4 path via `--smart-tagging`; **off** in sm_v1_5 default |
| Tactical homography | no | Lever deferred; optional debug only |
| Tactical video | no | May inspect to understand possession; not an algorithm input; no output-vs-video check |
| Broadcast (all) | no | Deferred / T90; separate timeline |
| Metrica sample-data (other games) | yes (validation) | External label source for missing-ball H/A rules; not an algorithm input on BAR-ATH |

**Compliant levers (order):** (1) cluster/majority ~~done~~ → (2) Soft ST ~~done/archived (`sm_v1_4`)~~ → (3) missing-ball H/A ~~done (`sm_v1_5`)~~ → (4) pass-like same-team travel → (5) homography/broadcast deferred. Keep gates = rule citation + local structure vs sm_v1_3 + smooth50 (+ external Metrica for missing-ball). Old public-floors + GT duration-band A/B is archived.

Oct 2026 public-GT-tuned variants (archive `sm_v1_3`…`sm_v2`) are **not** the submitted method narrative. Compliant `sm_v1_*_compliant` files are distinct from those Oct rows.

**External data acknowledgement:** Sample tracking and event data from [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data). Used only for compliant validation of missing-ball rules on *other* games.

---

## 3. Possession rules implemented

_Owner: B from CHALLENGE.md (T12) · E1 maps to code (`sm_v1` restore → compliant `sm_v1_5`)_

### Challenge rules (what we aim for)

- **Live ball → one of H / A:** While the ball is in play, exactly one team has possession: home Barcelona (`H`) or away Athletic Club (`A`).
- **Dead ball → D:** When play is stopped, the answer is simply `D`.
- **Change of possession = controlled takeover:** Proxies include keeping the ball for a few seconds, or more than two consecutive passes.
- **Contested / loose → keep last team.**
- **What does not count as a change:** A simple touch without control; deflection without intercept; aerial duel without then controlling.
- **Possession as an interval:** Maximal run of one live team; `D` breaks possessions.

### How code approximates this (`starter/features.py` + `state_machine.py`)

Features (tactical ATD): ball present, position, nearest team and distance (m), raw ATD ball speed field.

**Restored `sm_v1`:**
1. Ungraded halves → `X`.
2. Hold on missing ball (missing ≠ dead).
3. Controlled switch: opposing nearest within 3 m for **15** frames; else keep last.
4. Loose / far / no nearest → keep last (or `X` if never set).
5. Dead ball: visible ball **out of pitch** (`out_margin=0.02`); clears `last_live` on that `D`.
6. Merge H/A islands shorter than **8** frames into neighbours (flicker).

**Compliant `sm_v1_1` (rule upgrades, no GT selection):**
- Control **25** frames (~1 s) — “few seconds” / sustained control.
- Merge threshold **12** frames — touch without control ≠ turnover.
- Edge dead-ball: missing ≥50f after near-edge sighting; still ≤0.5 m/s near edge ≥25f → `D`.
- **Freeze** `last_live` through `D` — restart keeps last clear possessor / contested hold.

**Compliant `sm_v1_2` (builds on sm_v1_1):**
- **Establish-without-X:** while `last_live` is unset and a close nearest team is present, emit that team's letter during the pending streak (CHALLENGE: live ball → H/A). Still require `control_frames` before locking `last_live`. `X` only for ungraded, never-established + ball missing / no close nearest.
- Merge threshold **18** frames (~0.72 s) — touch without control ≠ turnover; still well below smooth125 / merge-50.

**Compliant `sm_v1_3` (builds on sm_v1_2):**
- Features: `n_near_home` / `n_near_away` within **3 m** cluster radius; `majority_team` when counts differ.
- **Contested override:** if majority disagrees with nearest and `|min_dist_home − min_dist_away| ≤ 1.0` m, use majority as cand. Clear 1v1 nearest stays nearest. Rollback: `--no-majority-near`.

**Compliant `sm_v1_4` (archived; builds on sm_v1_3):**
- Soft Smart Tagging SET PIECES → D (not phase intervals as H/A).
- **Missing-aware:** first **50** frames of each set-piece tag → D; additionally any **ball-missing** frame inside the full tag span → D.
- Phase votes **off**. Reachable via `--smart-tagging`. Not the structure reference (mean dur / short islands worse than sm_v1_3).

**Compliant `sm_v1_5` (default; builds on sm_v1_3; ST off):**
- **Backdate switch** (`--backdate-switch`): once opposing cand confirms for `control_frames`, relabel the pending window to the new team (Link & Hoernig 2017: switch at first opposing control).
- **Gap bridge mid** (`--gap-bridge mid`): on a live missing gap, if the confirmed team after the gap differs from the team before it, place the switch at the gap midpoint (turnover happened in the dark; TBP continues through void phases).
- Long-gap → D was A/B'd externally and **dropped** (hurt hidden-frame team accuracy).
- Chosen on Metrica Sample_Game_1/2 only; BAR-ATH used for non-GT structure vs sm_v1_3 + smooth50. Rollback: `--no-backdate-switch --gap-bridge off`.

One command: `predict_v1.bat` / `predict_v1.sh` (writes next `sm_v*`, validates, optional score log).

### External validation (missing-ball H/A)

Harness: `starter/external/` — fetch Metrica Sample_Game_1/2, build features, derive H/A/D/X from events (PASS/RECOVERY/SHOT/SET PIECE/BALL LOST set team; BALL OUT→D until SET PIECE; X pre-kickoff/halftime), hide the ball on natural missing + 2/5/10/30 s live gaps, score with `possession_metrics.evaluate` plus **hidden-frame team accuracy**.

Aggregate (mean over 2 games × 5 dropouts):

| Candidate | spellF1 | hidden-frame team acc |
|-----------|---------|------------------------|
| sm_v1_3 | 0.3141 | 0.5052 |
| **sm_v1_5** (mid + backdate) | **0.3701** | **0.5102** |
| smooth50-equivalent | 0.3210 | 0.1528 |

Leave-one-game-out: sm_v1_5 beat sm_v1_3 on both Sample_Game_1 and Sample_Game_2. Literature: Link & Hoernig, *Individual ball possession in soccer*, PLOS ONE 2017.

---

## 4. Missing and noisy tracking

_Owner: E1_

- Ball missing (~41% graded): **Hold** last live team; sm_v1_5 may **retime** the H/A switch inside a gap when the team after reappear differs from the team before (gap-bridge mid) and backdates the pending window once control confirms. `X` mainly when never established + no ball/nearest, or ungraded. During establish with a close nearest, sm_v1_2+ labels H/A instead of X.
- No player IDs: team-collective nearest + cluster majority in the control circle (sm_v1_3).
- Dead-ball: out-of-pitch always; sm_v1_1+ near-edge missing/still. Soft SET PIECES archived (`--smart-tagging`). Mid-pitch stoppages remain hard.
- Smart Tagging: **off** in sm_v1_5 default; phase votes off.

---

## 5. Limitations

_Owner: B + E1_

- Nearest-player distance is a weak proxy for control; wrong-team spans remain.
- Without GT tuning we cannot chase public spellF1; private ranking still uses full-match spellF1.
- Earlier Oct 2026 line (`sm_v2`) was selected with public GT keep/drop — archived after clarification.

---

## 6. How to reproduce

_Owner: E2_

Someone with the data bundle and our public repo must regenerate a submission that matches our
**compliant defaults** (`sm_v1_5`). Step-by-step: **[README.md](README.md)**.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash predict_v1.sh
```

Windows: activate `.venv`, then `predict_v1.bat`.

- Output: next `evaluation/results/sm_v*.csv` — 199,319 rows, `frame,label`. Gitignored.
- Defaults (compliant `sm_v1_5`): sm_v1_3 knobs plus `backdate_switch=True`, `gap_bridge=mid`, `use_smart_tagging=False`. Rollback: `--no-backdate-switch --gap-bridge off` (sm_v1_3) or `--smart-tagging` (archived sm_v1_4) / restored sm_v1 CLI.
- External validation (optional): `python starter/external/fetch_metrica.py` then `python starter/external/eval_external.py --sweep`.
- Match `data/` and `data_external/` are **not** in the public repo. See `DATA.md`.

---

## Revision log

| Date | Who | Change |
|------|-----|--------|
| 2026-10-20 | team | Initial stub created |
| 2026-10-24 | B / E1 | T12 section 3 possession rules; T13/T13r section 0 metrics primer |
| 2026-10-26 | E1 | T20/T21 sm_v1 implementation notes in §3–§4 |
| 2026-10-26 | E1 | Tweak v1.1: sm_v1_1 (longer control, edge dead-ball heuristics, keep last_live on D) |
| 2026-10-28 | E2 | T22: `predict_v1.bat` / `predict_v1.sh`; auto-increment `sm_v*`. |
| 2026-10-28–29 | E1 | ARCHIVE: Smart Tagging / short-D / sm_v2 public-GT era |
| 2026-11-03 | E1 | Clarification compliance: restore sm_v1; §0/§3/§6 rewritten; sm_v1_1 rule path |
| 2026-11-03 | E1 | sm_v1_2 default: establish-without-X + min_spell 18; §3/§4/§6 |
| 2026-11-03 | E1 | sm_v1_3 default: contested majority-near-ball; levers 1–4 in §2; §3/§6 |
| 2026-11-07 | E1 | sm_v1_4: missing-aware soft SET PIECES → D; later archived vs sm_v1_3 structure |
| 2026-11-07 | E1 | sm_v1_5 default: backdate-switch + gap-bridge mid (Metrica external); §2/§3/§4/§6 |
