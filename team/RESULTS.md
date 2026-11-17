# Results log

### 2026-11-17: YOLO fine-tune v3 — cancelled by team decision

Scoped and boundaries frozen, but cancelled by team decision before
any result. No weights were produced or evaluated; nothing ships
from this line. `sm_v2_7_compliant.csv` remains the final submission.

### 2026-11-17: smoke v2 (ISSIA-selected detector + ROI gate) — STOP

Boundaries v2 frozen first: detector selection on ISSIA only,
5m ROI gate pre-registered, identical segment/kill. SoccerNet-
fine-tuned YOLO (Roboflow sports, sha `678fbad0…`) evaluated on
ISSIA cams 5+6: detection 0.63/0.80, cam-5 far-FP rate 34% (p90
1223px) vs FootAndBall 0.83/0.97 with 100% ≤10px — FootAndBall
wins the external selection. Smoke v2 (FootAndBall + gate):
ungated present 15.25%, **gated compared = 0** — no detection
falls within 5m of the ATD ball in 2000 frames. Firings are
systematically elsewhere, not near-misses; gating cannot rescue.
STOP per boundaries; no detector shopping on BAR-ATH output.
Artifacts: `evaluation/reports/issia_yolo.json`,
`barath_smoke_v2.json`, `research/experiments/issia_yolo_eval.py`,
`barath_smoke_v2.py`, `docs/algorithms/video-smoke-v2-boundaries.md`.

### 2026-11-17: BAR-ATH video smoke — STOP (detector domain shift)

Tactical mp4 placed by team (renamed `-002` → expected name; 5.22GB,
199,319 frames verified by code). Frozen smoke re-run unchanged:
detector present 15.25% (ATD 92.25%), compared 302 frames, median
disagreement **39.6 m**, p90 74.3 m, 100% > 1.5 m → pre-registered
kill (> 2 m) triggers, direction STOPPED. Diagnostics (aggregates
only): seek lands exactly on frame 11185, preprocessing/labels match
the FootAndBall repo, pixel-space distance median ~905 px — the
detector fires far from the ATD ball on this camera (domain shift:
moving tactical feed vs static ISSIA). No threshold/segment tuning
was done after seeing output, per frozen boundaries. Detector
weights remain external-only assets; v2.7 unchanged and final.

### 2026-11-17: BAR-ATH video smoke — inconclusive (no tactical mp4 in bundle)

Boundaries frozen before the run
([video-smoke-boundaries](../docs/algorithms/video-smoke-boundaries.md)):
FootAndBall thresholds 0.7, segment 11185–13184, kill = median
disagreement > 2m, aggregates only, no calibration on BAR-ATH.
The code-only smoke (`research/experiments/barath_smoke.py`) read
the ATD stream fine (ball present 92.3% in segment) but the tactical
mp4 is absent from this bundle (setup note T02: mp4 optional,
skipped), so zero frames decoded — verdict INCONCLUSIVE, not PASS.
Unblocking is a team action: place the enrolment tactical mp4 in the
data bundle, then re-run the identical frozen script. No `data/`
file was listed or inspected by the agent; only aggregate counts
were emitted.

### 2026-11-17: new direction a-ext — FootAndBall falsification passes on ISSIA

Downloaded FootAndBall weights (sha256 `42fbc1f4…`, 199,649 params,
MIT) + ISSIA-CNR (Sequences 292MB + Annotations, Dropbox links from
the dataset page) into gitignored `data_external/`. Pretrained
detector at native 1080p, threshold 0.7, held-out cameras 5+6:
241 GT ball frames, detection rate 0.83 / 0.97, median pixel error
2.1 / 2.5 px, p90 < 5 px, 100% within 10 px (~0.06–0.15 m at the
tactical Jacobian). Caveats: ISSIA is the detector's home domain
(paper protocol trains cams 1–4), small sample, static cameras vs
BAR-ATH moving tactical feed. Next step is the BAR-ATH smoke
(detector-vs-ATD agreement, non-tuning), which needs the tactical
video as pipeline input — still banned for research reads here, so
it waits on a team decision. Artifacts:
`evaluation/reports/issia_falsification.json`,
`research/experiments/issia_falsification.py`.

### 2026-11-17: new direction a-sim/b — trajectory refinement passes RMSE, weak nearest

GT-arbitrated simulation (FIT-4): degrade GT ball to ATD proxy (41%
block dropout + contested jitter σ∈{0.15,0.3,0.6}), refine = ≤25f gap
fill + Savitzky-Golay(7,2). Contested-window RMSE improves
25–29% across the sweep (kill threshold 20% — pass). But nearest-team
agreement at contested windows moves only 0.5484 → 0.5505 (+0.002):
xy error is not the binding constraint at <1m duels. This tests
trajectory refinement, not a video detector (which could recover true
positions where interp cannot). Next step requires FootAndBall +
ISSIA downloads and a team decision on BAR-ATH video-as-input (still
banned for research reads here). Artifacts:
`evaluation/reports/ball_refine_sim.json`, `ball_refine_nearest.json`,
`research/experiments/ball_refine_sim.py`, `ball_refine_nearest.py`.

### 2026-11-17: touch seed ensemble — held (identical decode)

Averaging touch HGB logits over 5 seeds (29/73/101/202/307) reproduces
the single-seed decode exactly on FIT LOMO (0.7702, same per-match
bests). No variance to reduce at the margin threshold. Held. Artifact:
`evaluation/reports/touch_seed_ensemble_crossfit.json`,
`research/experiments/touch_seed_ensemble_experiment.py`.

### 2026-11-17: duel-conditional margin decode — held (below noise)

Decoder-only: overwrite margin 0.5 at duel touches (|min_dist_home −
min_dist_away| ≤ 1m), 1.0 elsewhere, shipped touch weights unchanged.
FIT LOMO 0.7727 vs 0.7702 (+0.0025, worst +0.0059); clean external
eval (G2 + Sportec suite + SC hold-out, shipped model): mean spell
0.6028 vs 0.6017 (+0.0011), hidden identical, WR9 −0.0043. Positive
trend on two protocols but an order below the ~0.04 spell noise floor
— insufficient evidence to promote or to change starter decode.
Artifacts: `evaluation/reports/touch_duel_margin_crossfit.json`,
`touch_duel_margin_eval.json`,
`research/experiments/touch_duel_margin_experiment.py`,
`touch_duel_margin_eval.py`.

### 2026-11-17: final lever contested-weighted touch head — held (tie)

Touch HGB with extra sample weight on touches within 25f of a GT
switch (boundary head had this, touch head never did): weight 2 →
0.7690, weight 4 → 0.7704 vs unweighted 0.7702 FIT LOMO, hidden
−0.0001 — a tie inside the ~0.04 spell noise floor, not a beat.
SoccerTrack (Rank 7) skipped by team decision (retrieval steps);
PFF (Rank 4) skipped earlier. All ranked levers are now dispositioned
and none beats sm_v2_7. `LATEST` stays `sm_v2_7_compliant.csv`.
Artifact: `evaluation/reports/touch_contested_weight_crossfit.json`,
`research/experiments/touch_contested_weight_experiment.py`.

### 2026-11-17: mad-scientist Ranks 5–8 — held / blocked (v2.7 unbeaten)

Rank 5 GNN probe (2-round distance-weighted message passing, 5Hz,
contested visible frames, FIT LOMO): GNN 0.659 beats pooled control
0.642 but not v2.3 logit 0.772 — kill gate, no NumPy export. Rank 6
synthetic duel snippets (control-outcome labels, physics simulator):
mixed touch training 0.7687 vs real-only 0.7702 — held. Rank 7
SoccerTrack v2: schema verified via search (`ball/` per-frame track
exists, BAS team/actor), but dataset is HF-gated (agreement required)
and HF is proxy-denied in this environment — blocked on access, not
merit; needs a manual download into `data_external/` to proceed.
Rank 8 SSL probe (masked-ball forecasting pretrain + frozen linear
probe): 0.502 vs v2.3 0.772 — chance, kill gate at probe. Rank 5
strengthened retest (GNN residual on v2.3 logit + nearest features,
60 epochs): 0.7659 vs v2.3 0.7720, pooled control 0.7644 — message
passing adds ~0.001 over pooled residual, still below v2.3 alone.
Rank 5 closed definitively (`gnn_probe_v23res.json`). `LATEST`
stays `sm_v2_7_compliant.csv`. Artifacts:
`evaluation/reports/gnn_probe.json`, `touch_synthetic_crossfit.json`,
`ssl_probe.json`, `research/experiments/gnn_probe.py`,
`synthetic_duels.py`, `touch_synthetic_experiment.py`, `ssl_probe.py`.

### 2026-11-17: mad-scientist Ranks 1–3 — held (PFF skipped by team decision)

PFF WC2022 (Rank 4) skipped: gated access needs organiser approval.
Rank 1 physics intercept features (Spearman race, 8 cols from 5Hz
nodes) into touch HGB: FIT LOMO 0.7700 vs shipped-protocol baseline
0.7702. Rank 2 dwell-outcome label (25f majority): 0.7649. Combined:
FIT-only 0.7728 but under ship selection with SkillCorner folds 0.7509
vs shipped 0.7530 (SC spell +0.0004, hidden −0.0002) — FIT-only gain
does not survive the ship gate. Rank 3 duel-winner probe (Sportec
TacklingGame windows, 586 duels): duel head 0.576 vs touch logit 0.579
— kill gate at probe, no downstream wiring. All held; `LATEST` stays
`sm_v2_7_compliant.csv`. Artifacts:
`evaluation/reports/touch_physics_*_crossfit.json`,
`duel_winner_probe.json`, `research/experiments/touch_physics_experiment.py`,
`duel_winner_probe.py`, `research/tests/test_touch_physics.py`.
Queued (public, unverified): Rank 5 GNN-distill, Rank 6 synthetic
duels, Rank 7 SoccerTrack folds, Rank 8 SSL probe.

### 2026-11-17: P4 missing-span audit — held (no drift on clean GT)

Probe: are hidden-span team errors onset handoff or mid-span drift?
v2.7 labeller, 5s-block dropout spans + natural Metrica spans. On Sportec,
hidden-team accuracy does not decay with distance into the span (d0–4 →
d25+: 0.75→0.82 WMX, 0.76→0.84 WPY, 0.80→0.86 WQQ, 0.80→0.85 WR9) —
the Viterbi settles, it does not drift. Only Sample_Game_3 natural spans
(87 spans, noisy event GT) decay (0.78→0.70→0.55). Mid-span possessor
heads / imputation have no headroom on clean GT; span errors are onset
handoff + in-the-dark turnovers. Not pursued. Artifacts:
`evaluation/reports/missing_span_audit.json`,
`research/experiments/missing_span_audit.py`.

### 2026-11-17: P5 dead-ball hedge — held (no headroom)

Scored v2.7 D against frame-exact BallStatus on Sportec
(`diagnose_v27_baseline.json`): dead precision/recall ≈ 0.996 on all
four Track C matches (~200–260 D errors vs ~5–6k wrong-team per match).
Metrica D errors concentrate where the event-derived GT itself is noisy
(visible dead recall 0.09–0.12). A dead-alive head has nothing to fix on
clean GT and cannot fix GT noise. Not pursued. P3 (contested boundary
head) stays held on the pre-existing ladder (best rung 0.7208 < 0.723);
P7 (broadcast/video) stays team-deferred.

### 2026-11-17: P6 SkillCorner phase dead-ball proxy — held

Probe: do `phases_of_play` gaps mark dead ball on SC_1874553 (train)?
P(gap|phase) = 0.04, P(gap|no phase) = 0.95 — phases mirror group
coverage almost exactly, so they add no information beyond
`possession.group`. Phase gaps mix dead frames with live loose-ball
frames (challenge GT keeps last team there, never D), so they are not a
dead-ball label. Not pursued as supervision. Artifact:
`evaluation/reports/sc_phase_proxy.json`,
`research/experiments/sc_phase_proxy.py`.

### 2026-11-17: event-team labeled touch head — held

P1 follow-up: train the event-first touch classifier on Sportec play-event
team labels (GT lead+3 backoff beyond 12f) instead of pure GT lead+3, rule
candidates, LOMO over the three Sportec FIT matches. 206/8675 touch labels
differ (2.4%). Event labels lose on every decoder config (c0.0_m1.0:
0.7779 → 0.7701, worst +0.0059 → +0.0043); the disagreeing frames are
annotation noise, not signal. Neither variant clears the hidden gate on
the 3-match subset. `LATEST` stays `sm_v2_7_compliant.csv`. Artifact:
`evaluation/reports/touch_eventlabel_crossfit.json`,
`research/experiments/touch_eventlabel_experiment.py`.

### 2026-11-17: P2 premise check — full CRF decode was already tested, pivot

Deep-research P2 proposed "full dynamic CRF path decoding (untested)" for
missing-ball spans. Repo history shows it was tested:
`research/experiments/pathcrf_sequence_experiment.py` runs official
`net.crf.decode` per geometry window, free and possession-zone-hinted
variants, on Metrica 2 — ball-free suite spellF1 0.3917 (ref 0.4264),
natural 0.4103 (ref 0.6298); hinted suite 0.4526 but natural 0.5012 /
hidden 0.5189, losing to the residual hybrid. Rejected as a replacement
(see [temporal ownership](../docs/algorithms/temporal-ownership-research.md)
§ PathCRF). The only untested slice is missing-spans-only application of
CRF votes, but no vote cache survives (`prediction_cache` has none) so it
needs full torch re-inference against strongly negative priors — skipped
as low expected value. Pivoting to the P1 follow-up (event-team labeled
touch head) and P6 (SkillCorner dead-ball proxy).

### 2026-11-17: union touch candidates (rule + ELASTIC-style accel) — held

P1 from the deep-research review: parse the previously unused Sportec
`events_raw` XML (touch time → frame N via the ball clock) and SkillCorner
`dynamic_events.csv` (player_possession + on_ball_engagement; duel
`team_id` is the pressing side, possessor team resolved via `match.json`)
into touch supervision. Event-team labels agree with GT lead+3 at 0.93
(Sportec play), 0.83 (SkillCorner), 0.65–0.68 (Sportec duels).

An acceleration-based kick/receive detector (accel ≥ 10 m/s² or slowdown ≥
3 m/s near a player; physics constants, untuned) lifts event recall @tol12
from 0.56 → 0.69 (J03WOH fit), 0.71 → 0.80 (J03WMX readout) and 0.37 →
0.79 (SC train). But a retrained event-first head on union candidates loses
FIT LOMO natural spellF1 0.7702 → 0.7635 (hidden 0.5617 → 0.5612), and a
dual-margin sweep (accel-only margin 1–4) peaks at 1.0. Denser Viterbi
nodes reshape the decode; recall does not convert. No B/C/D track was
touched for selection. `LATEST` stays `sm_v2_7_compliant.csv`. Artifacts:
`evaluation/reports/touch_supervision_align.json`,
`touch_union_crossfit.json`, `touch_union_margin.json`,
`research/experiments/touch_supervision.py`, `touch_align.py`,
`touch_union_experiment.py`, `research/tests/test_touch_supervision.py`.
Method: [touch supervision](../docs/algorithms/touch-supervision.md).

### 2026-11-16: sm_v2_7 event-first touch model (+ SkillCorner) — promoted

sm_v2_6 plus a touch-team model: ball touch candidates, a HGB touch classifier
exported to NumPy trees, a two-state Viterbi decode and a conservative
overwrite (visible frames only, only where the decode disagrees with v2.6 and
|logit| ≥ 1.0). Trained on the fit set plus the 16 SkillCorner training-split
matches; selected by fit LOMO and SkillCorner training folds (never Track D).

| Track | sm_v2_7 | sm_v2_6 |
|---|---:|---:|
| A spells / mean_s / s25 | **339 / 8.73 / 7** | 342 / 8.65 / 8 |
| B Metrica spellF1 / hidden | **0.6654** / 0.8014 | 0.6474 / 0.8014 |
| C Sportec spellF1 / hidden | **0.6218** / 0.8069 | 0.6187 / 0.8069 |
| D SkillCorner spellF1 / hidden | **0.4381 / 0.7272** | 0.4365 / 0.7271 |

Hidden accuracy: Track B +22 frames, Track D +14, Track C net −2 frames (tied
at 4 dp, same tie convention as v2.5/v2.6). All four Sportec matches improve
spellF1. The fit-only event-first variant was rejected (Track D 0.4328).
SkillCorner retrains of the corrector (+0.0011 fit LOMO, held) and boundary
head (−0.0033, rejected) were not shipped. NumPy parity 0.0; default predict
is byte-identical to `sm_v2_7_compliant.csv`. `LATEST` → `sm_v2_7_compliant.csv`.
Rollback `--no-event-first` → sm_v2_6. Artifacts:
`evaluation/reports/sm_v2_7_decision.json`, `assess_sm_v2_7.log`,
`event_first_crossfit*.json`, `research/experiments/event_first.py`.

### 2026-11-16: sm_v2_6 honest corrector — promoted (sm_v2_5 demoted as leaky)

New leakage guard (`starter/external/leakage_guard.py`, run by `assess.py`)
showed `error_corrector_v1` was trained on matches in Tracks B/C. Retrained it
on the fit set only (J03WOH, J03WN1, J03WOY, Sample_Game_3) with the margin
picked by fit LOMO (2.5) → `error_corrector_v2.json`.

| Track | sm_v2_6 | sm_v2_3 |
|---|---:|---:|
| A spells / mean_s / s25 | 342 / 8.65 / 8 | 342 / 8.65 / 8 |
| B Metrica spellF1 / hidden | **0.6474** / 0.8014 | 0.6443 / 0.8014 |
| C Sportec spellF1 / hidden | **0.6187** / 0.8069 | 0.6161 / 0.8069 |

Also tried and rejected: a 3-seed boundary weight soup (−0.0018 crossfit) and
raising the minimum spell (short spells are not the bottleneck; stays 18 f,
`short_spell_ceiling.json`). Artifacts: `honest_anchor_decision.json`,
`honest_corrector.json`, `assess_sm_v2_6.log`.

### 2026-11-16: boundary transition retrain — held (keep sm_v2_5)

v2.5 audit still shows ~32% missed GT switches (~94% contested). Built
deterministic `boundary_crossfit.py` (control ≈ 0.7204 vs v2.3 0.7205). Ladder
of training-side boundary upgrades all failed vs sm_v2_5 natural crossfit
(~0.723):

| Tag | mean spell | Δv2.5 | worst Δv2.5 |
|---|---:|---:|---:|
| control_v1 | 0.7204 | −0.0027 | −0.018 |
| rung_a (loss 0.5 / pos 20) | 0.7198 | −0.0033 | −0.016 |
| rung_b (σ=3 / trunc=10) | **0.7208** | −0.0023 | −0.009 |
| rung_c contested×2/×3 | 0.7197 / 0.7191 | −0.0034 / −0.0040 | −0.015 / −0.016 |
| rung_d transition ctx | 0.7196 | −0.0035 | −0.015 |
| stacked B+D / A+B+D | 0.7202 / 0.7188 | −0.0029 / −0.0043 | −0.014 / −0.017 |

No corrector retrain or ship model. `LATEST` remains `sm_v2_5_compliant.csv`.
Artifacts: `evaluation/reports/boundary_v2_decision.json`,
`evaluation/reports/boundary_cf/`, `v25_audit_*.json`,
`research/experiments/boundary_crossfit.py`.

### 2026-11-16: sm_v2_5 structure-preserving corrector — promoted

High-margin retune of the live-team error corrector (`margin=3.0`, window=0,
min-island=1, no structure_guard) on frozen sm_v2_3. Only apply regime that
cleared Track A without a challenge-only guard while still beating B/C.

| Track | sm_v2_5 | sm_v2_3 |
|---|---:|---:|
| A spells / mean_s / s25 | **341 / 8.68 / 8** | 342 / 8.65 / 8 |
| B Metrica spellF1 / hidden | **0.6509 / 0.8014** | 0.6443 / 0.8014 |
| C Sportec spellF1 / hidden | **0.6191 / 0.8069** | 0.6161 / 0.8069 |
| Crossfit natural spellF1 | **0.7220** (+0.0015) | 0.7205 |

`LATEST` → `sm_v2_5_compliant.csv`. Default predict enables the corrector;
rollback `--no-error-corrector`. Artifacts:
`evaluation/reports/sm_v2_5_decision.json`,
`structure_preserving_corrector.json`, `assess_sm_v2_5.log`,
`research/experiments/structure_preserving_corrector.py`.

### 2026-11-16: insert-only contested switches — held (keep sm_v2_3)

Follow-up to the failed sparse switch editor: **insert-only** proposals at
contested peaks with ensemble polarity disagreement across the peak, single-side
flip, min-island/sep caps, top-K. No delete/retarget. Nested tune on mean Δ over
3 train matches.

**Crossfit gate vs sm_v2_3: FAIL** (mean natural spellF1 0.7191 < 0.7205;
worst Δ −0.0077). Boundary-gated probe also flat/down. Not promoted.
`LATEST` remains `sm_v2_3_compliant.csv`. Artifacts:
`evaluation/reports/crossfit_insert_only.json`, `insert_only_v25_decision.json`,
`research/experiments/insert_only_switches.py`.

### 2026-11-16: sm_v2_5 sparse switch editor — held (keep sm_v2_3)

Switch-timing audit on sm_v2_3: median |offset| ~4f on matched switches; miss
rate ~0.32 with ~95% of missed GT switches sitting on contested peaks; false
rate ~0.19. Built contested-band features + crossfit HistGBM editor
(retarget/delete/insert) applied only in ±R neighborhoods on frozen v2.3.

**Crossfit gate vs sm_v2_3 (10 matches): FAIL**
- mean natural spellF1 0.6688 < 0.7205
- worst spell Δ −0.116 (Sample_Game_3)
- Track A cand looked “smoother” (183 spells / s25=4) but that was over-merging,
  not better ownership

Not promoted. `LATEST` remains `sm_v2_3_compliant.csv`. Artifacts:
`evaluation/reports/crossfit_switch_editor.json`, `switch_timing_audit.json`,
`switch_editor_v25_decision.json`, `research/experiments/switch_editor.py`.

### 2026-11-16: sm_v2_4 demoted — restore sm_v2_3 as LATEST

Track A structure vs sm_v2_3: spells 353 vs 342, mean 8.38 vs 8.65 s, s25 10 vs 8
(s12 tied at 0; D/X unchanged). Prefer cleaner possession structure on v2.3
despite earlier crossfit natural / Metrica–Sportec B/C gains from the error
corrector. `DEFAULT_USE_ERROR_CORRECTOR=False`; opt-in via
`--error-corrector-file`. `LATEST` → `sm_v2_3_compliant.csv`.

Method note: [push-090-natural](../docs/algorithms/push-090-natural.md).

### 2026-11-16: sm_v2_5 player-node corrector — held (keep then-LATEST sm_v2_4)

Attempted live H↔A fix: top-3-per-team player-node features + hard-mined HGB
corrector replacing the v2.4 control-design head, with anti-flicker apply
(margin / ±W window / min_island; nested-tuned per fold). Audit priors from
`wrong_team_audit.py`: margin=1.5, window=5, min_island=8 (near-turnover
dominant, ~46% of visible wrong frames within 12f of GT switch).

**Crossfit gate vs sm_v2_4 (10 matches): FAIL**
- mean natural spellF1 0.7221 < 0.7291
- mean visible team acc 0.9370 < 0.9413
- worst spell Δ −0.025 on Sample_Game_1 (limit −0.02)

Track A cand CSV more fragmented than v2.4 (395 spells / mean 7.49s / s25=21
vs 353 / 8.38 / 10). Not promoted. (Then-LATEST was sm_v2_4; later demoted —
see above.) Artifacts: `evaluation/reports/crossfit_node_corrector.json`,
`wrong_team_audit.json`, `node_corrector_v25_decision.json`,
`research/experiments/node_corrector.py`.

### 2026-11-16: sm_v2_4 live-team error corrector — briefly promoted, then demoted

Cross-fitted push toward 0.90 natural spellF1 / hidden across 10 external
matches. Ceiling audit: mean noise floor 0.927 (absolute 0.90 label-feasible);
v2.3 natural mean 0.7205. Visible HGB, missing-weight boundary retrain, dead
crossfit and HSMM all failed to beat v2.3. A second-stage error corrector
(control geometry + v2.3 ensemble logits; flip only when disagreeing with
`|logit| ≥ 1.5`) wins the leave-one-out natural mean **0.7205 → 0.7232**,
with no match dropping more than 0.02.

- Metrica protocol assess: spellF1 0.6443 → **0.6650**, hidden 0.8014 → 0.8015
- Sportec Track C: spellF1 0.6161 → **0.6226**, hidden flat 0.8069
- Track A: 353 spells / mean 8.38 s / s25=10 (v2.3: 342 / 8.65 / 8); D/X unchanged
- Absolute 0.90 **not** reached (gap ~0.18 natural spellF1). Soft target
  (90% of ceiling) is 0.834.
- SHA256 `cd684645899f25b09eac7d54416aa6ed05b8923c86969cdb8a271b0e1b49fa70`.
  Format OK, 199319 rows. Two regenerations byte-identical.
  Initially set `LATEST` → `sm_v2_4_compliant.csv`; later demoted on Track A
  structure (see demotion entry above). Opt-in:
  `--error-corrector-file starter/models/error_corrector_v1.json`.

Method: [push-090-natural](../docs/algorithms/push-090-natural.md).

### 2026-11-16: sm_v2_4 improvement round — held (keep sm_v2_3)

Attempted three fixes on top of sm_v2_3 under a leave-one-match-out gate
(suite spellF1/hidden ≥ v2.3 on five eval matches; non-lose on ≥5/6 including
Metrica 3; natural spellF1 on WMX/Metrica2/WR9 ≥ v2.3; Track B ≫ smooth50).
No challenge GT used. Merge threshold stayed 18.

1. **Boundary gate / blend ablation** (inference only): blend ∈ {0.3,0.5,0.7},
   scale ∈ {0.5,1.0}, gate ∈ {none, missing_only, confidence}. Only the
   v2.3-identical setting (`b0.5_s1.0_none`) passed. Blend 0.7 raised suite
   spellF1 (~0.601) but failed natural on Sample_Game_2.
2. **Missing-ball retrain**: new heads with `--missing-weight` 2 and 3
   (seeds 473/573, long-dropout boost), init from packaged residual npz,
   exported with NumPy parity ≤1.6e-6. Both failed the gate (suite spell
   0.5968 / 0.5931 vs v2.3 0.5992; natural regressions).
3. **Cautious D**: skipped — live/dead errors dominate wrong-team on missing
   frames in only 2/6 natural matches (Metrica 2/3), not a majority.

**Decision:** do not promote. `LATEST` remains `sm_v2_3_compliant.csv`.
Assess confirmation vs sm_v2_2 + smooth50: Track B spellF1 0.6443 / hidden
0.8014 (smooth50 0.3210 / 0.1528); Track C spellF1 0.6161 / hidden 0.8069.
Artifacts: `evaluation/reports/boundary_gate_ablation.json`,
`mw2_vs_v23_gate.json`, `mw3_vs_v23_gate.json`, `v24_round_decision.json`,
`diagnose_v23_baseline.json`. Harness:
`research/experiments/evaluate_approved.py`, `diagnose_errors.py`.
CLI knobs for future trials remain on `predict_v1.py`
(`--boundary-blend/scale/gate`, `--dead-confidence`) with defaults = v2.3.

### 2026-11-14: sm_v2_3 boundary-aware ownership ensemble — promoted

An ASRF-inspired joint boundary/ownership head is fitted on WOH/WN1/WOY and
Metrica Game 1 only, initialized from v2.2. Its team logits are equally
averaged with v2.2; its boundary head controls missing-frame transition costs.
Observed-ball predictions use raw ensemble signs. Baseline D/X remain.

- Five excluded-fit-match full-timeline suite: spellF1 .580146 -> .599177;
  hidden accuracy .774845 -> .791608, both improve in every match.
- Original Metrica protocol: spellF1 .634473 -> .644263, hidden
  .787772 -> .801381, both improve in each game. Game 1 is training;
  excluded Game 2 alone .611713 -> .616270 spellF1, .728064 -> .747213 hidden.
- Reused additional Game 3 validation: suite spellF1 .555991 -> .584742,
  hidden .688976 -> .711571; natural spellF1 .679070 -> .702290 and hidden
  .530622 -> .560815. No untouched-test claim. Natural spellF1 regresses
  slightly on WMX, Metrica 2 and WR9; complete values are in the method note.
- Challenge: 342 possessions, mean 8.65439 s, <12f 0, <25f 8 (v2.2: 13).
  Graded D/X remain 84581/281. The merge threshold stays 18 frames.
- Thirty-nine tests pass. NumPy export max errors 1.44e-6 team and 6.26e-7
  boundary; natural Metrica 2 packaged labels equal native research labels.
  Format passes, 199319 rows. Three generations are byte-identical, including
  default inference without PyTorch. SHA256:
  `C6D6ECB8B04D8DBA1AF26D944C857A94BD0C0A930E24D5F554D4103AA250D9C6`.
  `--no-boundary-model` exactly restores v2.2. `LATEST` points to v2.3.

Temporal D correction, blunt short-spell suppression, dense decoding and a
12.5 Hz correction model were also tested and held/rejected. Details and
sources: [boundary ownership research](../docs/algorithms/boundary-ownership-research.md).
Official challenge feedback is unavailable; this is the best tested model,
not a claim of a known private score or guaranteed win.

### 2026-11-14: sm_v2_2 residual ownership hybrid — promoted on primary metric

The network sees anonymous player geometry/motion, ball context and the
automatic v1.10 baseline, with a fixed baseline prior. Fit: Sportec
WOH/WN1/WOY + Metrica Game 1 only. It uses raw visible predictions and
Viterbi during missing-ball frames, retaining baseline D/X.

- Five excluded-fit-match full-timeline suite: equal-match spellF1
  .481070 -> .580146; hidden accuracy .665186 -> .774845. Both improve on
  each match; natural spellF1 also improves on each.
- Original Metrica protocol: spellF1 .571573 -> .634473, hidden
  .749873 -> .787772. Both improve in each game. Game 1 is training;
  excluded Game 2 alone: spellF1 .549717 -> .611713, hidden .687964 -> .728064.
- Additional Game 3 validation: suite spellF1 .446470 -> .555991, hidden
  .630178 -> .688976. Natural spellF1 .653978 -> .679070, macroF1
  .798459 -> .803933, but hidden accuracy .575231 -> .530622.
  The strict auxiliary natural-hidden criterion **failed**. Promotion is a
  deliberate tradeoff for the official primary spell metric, not a claim
  that every diagnostic passed. Game 3 was not untouched after prior trials.
- Challenge: 319 possessions, mean 9.27837 s, <12f = 0, <25f = 13;
  graded D/X 84581/281. More <25f spells than v1.10 (2), but fewer than
  the unrestricted set model (21). No challenge GT/public score was used.
- 31 tests pass. NumPy/PyTorch maximum logit difference 1.35e-6 on random
  probes and 1.20e-6 on a full external match; no frame sign differences.
  Packaged Sportec registry labels equal native research labels exactly.
  Three challenge generations are byte-identical, including default inference
  in the base Python environment without importing PyTorch:
  SHA256 `27EA6EF8745D3D0641C0D50EB127F48D8FC40CD214AEB9DB4066CD3E35E297BA`.
  Format validation passes, 199319 rows. Rollback `--no-team-model` exactly
  reproduces v1.10. `LATEST` now points to `sm_v2_2_compliant.csv`.

Details: [temporal ownership research](../docs/algorithms/temporal-ownership-research.md).

The additional official pretrained PathCRF transfer experiment is complete.
Macro/micro emissions, full dynamic CRF, and observed-team constrained CRF
were tested without reference episode boundaries. None outperformed the
residual hybrid on Metrica 2; observed-team CRF suite spellF1 .452605 versus
the hybrid .517666, with worse natural spellF1 and hidden accuracy. These
adapters remain research only. All experiment processes have finished.

### 2026-11-14: temporal ownership research — candidates held

At this stage the approved pointer remained sm_v1_10. A geometry-only player-set temporal
network improves the five-match full-timeline dropout mean spellF1
.481070 -> .573314, but natural Metrica tracks and the historical protocol
regress. It is not promoted. A selective segment gate improves development
results and preserves challenge structure, but fails the additional Metrica 3
hidden-accuracy check. The research CSV `sm_v2_compliant.csv` is not the best
approved submission. The later baseline-aware result is recorded above.
See [temporal ownership research](../docs/algorithms/temporal-ownership-research.md)
for split provenance, the reserved-test limitations, failed checks and exact
reproduction hashes. No challenge public possession labels were scored.

### 2026-11-14: sm_v1_10 calibrated mixed-provider boosted model — promoted

Training is still Sportec J03WOH + Metrica Game 1, with no challenge GT/video.
Numeric tree JSON permits NumPy/pandas-only inference. Training-prior
correction is derived from fitting labels, not target or validation statistics.

- Track A: sm_v1_9 -> sm_v1_10: 331 -> 294 spells, mean 16.8413 -> 10.0673 s;
  <12f 3 -> 0, <25f 8 -> 2; graded D 19214 -> 84581, X remains 281.
  The large D-count change is a material transfer risk, not a tuning target.
- Original Metrica suite: spellF1 0.496131 -> 0.571573; hidden accuracy
  0.678216 -> 0.749873. Both metrics improve in both games. Game 1 is training;
  excluded Game 2 alone: spellF1 0.489793 -> 0.549717, hidden 0.616913 -> 0.687964.
- Whole-timeline natural + 41% block dropout, five excluded fitting matches:
  mean spellF1 0.419488 -> 0.481070, hidden 0.610566 -> 0.665186. Both improve
  separately in Metrica Game 2 and Sportec WMX/WPY/WQQ/WR9. Different protocol:
  these absolute values must not be mixed with the preceding original suite.
- Twenty-three tests pass; portable logits match scikit-learn to 1e-12 on
  16758 probes, including missing routes. Packaged inference reproduces the
  experimental CSV exactly (SHA256 6CB8F5B98618B890FF2F262304A779430313B014294B8791BBFE84E6D7053AA9).
  Validation passes, 199319 rows. Rollback `--dead-model logistic` -> sm_v1_9.

Kept for consistent cross-match and cross-provider gains, with prior correction
and established short-spell rules after the D overlay. Validation matches have
informed variant selection; none is a pristine test for future development.
The hidden challenge score/rank is unknown. Research attempts and rejected
variants are recorded in the [research log](../research/docs/research-round-2026-11-14.md).
[Final method and reproduction](../docs/algorithms/boosted-dead-ball.md).

### 2026-11-14: sm_v1_9 protected learned dead-ball model — promoted

External-only training: Sportec J03WOH and Metrica Game 1. Excluded from fitting:
Sportec J03WN1 and Metrica Game 2. No challenge labels/video were used.
Packaged JSON parameters require no external downloads for inference.

- Track A: spells 359 -> 331, mean 13.7626 -> 16.8413 s,
  <12f 3 -> 3, <25f 9 -> 8; graded D 35057 -> 19214, X stays 281.
- Original Metrica suite: mean spellF1 0.490561 -> 0.496131;
  hidden accuracy 0.511854 -> 0.678216; both games improve both metrics.
  **Game 1 is training/development, not held out.**
- Excluded Sportec J03WN1, natural + 41% block dropout:
  spellF1 0.378973 -> 0.414292; hidden 0.483740 -> 0.548852.
- Excluded Metrica Game 2, natural + 41% block dropout:
  spellF1 0.387076 -> 0.393622; hidden 0.503314 -> 0.642444.
- Full assess command passed against prior LATEST + smooth50. Fifteen tests
  pass. Submission validates (199319 rows). Packaged/default inference is
  SHA256-identical to the experimental protected candidate. Legacy core
  output is unchanged. Rollback: `--no-learned-dead`.

Kept for external evidence and the challenge's live/dead distinction, with
explicit preservation of previous hidden H/A answers. Lower challenge D count
is a material transfer risk. No hidden-challenge performance or winning rank
is established. [Full method/results](../docs/algorithms/learned-dead-ball.md).

Rejected/deferred: same/midpoint gap reconciliation lost hidden accuracy;
physical-boundary-only change lost Sportec spellF1; unprotected learned model
had larger spell gains but lost Sportec hidden accuracy. Do not promote them.

### 2026-11-14: visible live-island candidate (not promoted)

Regenerated sm_v1_8: 359 spells, mean 13.7626 s, <12f 3, <25f 9.
Experimental visible-only D-H/A-D island suppression (<25f): 353 spells,
mean 13.9870 s, <12f 0, <25f 3. Metrica external mean spellF1
0.490561 -> 0.492264, improves both games; hidden accuracy unchanged
0.511854. Kept experimental for the tiny gain; the assessment code actually
accepts nondecreasing hidden accuracy. This candidate did not change LATEST.
No challenge GT used. Four boundary tests pass; candidate submission validates.
See [experiment and reproduction](../research/docs/live-islands.md).

**Metrics only** — no match coordinates, no GT label dumps, no video.

### Active baseline (Nov 2026 clarification)

- **Forward path:** … → **`sm_v1_8`** → **`sm_v1_9`** → **`sm_v1_10`** → **`sm_v2_2`** → **`sm_v2_3`** → **`sm_v2_6`** (honest fit-only corrector) → **`sm_v2_7`** (event-first touch model). **Approved best:** `evaluation/results/LATEST` → `sm_v2_7_compliant.csv`. (`sm_v2_5` demoted: corrector trained on evaluation matches; `sm_v2_4` margin-1.5 demoted on Track A.) Historical public-GT-era bare files are not this compliant line. Research scripts live under [../research/](../research/).
- **Archive:** rows from historical `sm_v1_1`…`sm_v2` (Oct 2026) used public GT as a keep/drop signal — we **do not continue** that process. Auto predict floors on **LATEST**, not max bare archive.
- **How we assess (not Oct):** four tracks via `assess.sh` — **A** structure (cand vs ref vs smooth50); **B** Metrica H/A; **C** Sportec full-timeline suite (optional `--skip-sportec`); **D** SkillCorner hold-out (`--skillcorner`), plus a leakage check. Public scorer: **optional log only**. Keep? cites **challenge rule / ATD physics / licensed external data**. Ritual: [../docs/ASSESS.md](../docs/ASSESS.md) · [TEAM.md](TEAM.md).

How to fill: `bash assess.sh <cand.csv>` (or predict → assess); add a RESULTS row. See [BEGINNER.md](BEGINNER.md).

---

## Score table

Baselines scored **2026-10-24** (T10). Team models: archive through `sm_v2`, then restore / compliant line.

| Date | File | Owner | macroF1 | spellF1 | poss | mean_dur_s | Notes |
|------|------|-------|---------|---------|------|------------|-------|
| 2026-10-24 | baseline_raw | shipped | 0.5972 | 0.0378 | 131 | 2.0 | Public window; far too many possessions vs GT 17 |
| 2026-10-24 | baseline_smooth25 | shipped | 0.6200 | 0.2868 | 36 | 7.6 | |
| 2026-10-24 | baseline_smooth50 | shipped | 0.6207 | 0.3256 | 26 | 10.8 | Full-match floor spellF1 **0.2433** (not this public number) |
| 2026-10-24 | baseline_smooth125 | shipped | 0.5812 | 0.4242 | 16 | 18.5 | Best on **public** spellF1; loses to smooth50 on full match |
| 2026-10-26 | sm_v1 | E1 | 0.5131 | 0.3000 | 27 | 8.8 | T21 state machine; poss << raw 131; validate OK |
| 2026-10-26 | sm_v1_1 | E1 | 0.6051 | 0.3111 | 28 | 9.3 | Tweak v1.1; validate OK; graded X down (1239→237) |
| 2026-10-28 | sm_v1_2 | E1 | 0.6051 | 0.3111 | 28 | 9.3 | Auto-version smoke (same model as v1.1) |
| 2026-10-28 | sm_v1_3 | E1 | 0.6162 | 0.2381 | 25 | 9.2 | T23 Smart Tagging priors; validate OK |
| 2026-10-28 | sm_v1_4 | E1 | 0.6271 | 0.3130 | 29 | 8.9 | Soft set-piece cap 50f + narrow phases; v1.1 rules |
| 2026-10-28 | sm_v1_5 | E1 | 0.6312 | 0.2800 | 23 | 11.2 | Same priors + control=37 / min_spell=25; spell down |
| 2026-10-28 | sm_v1_6 | E1 | 0.6271 | 0.3130 | 29 | 8.9 | T24 sandwich merge only; tie with sm_v1_4 |
| 2026-10-28 | sm_v1_7 | E1 | 0.6288 | 0.3083 | 31 | 8.3 | Oct archive T24 restart; **not from this branch** (`newrules#1`); do not use |
| 2026-10-28 | sm_v1_8 | E1 | 0.6178 | 0.3130 | 29 | 8.9 | T24 merge + no phase votes; macro down |
| 2026-10-29 | sm_v1_9 | E1 | 0.6341 | 0.4205 | 22 | 11.9 | Fill short D <50f; beats public smooth50 |
| 2026-10-29 | sm_v1_10 | E1 | 0.6406 | 0.3905 | 25 | 10.4 | + restart reset after D≥50f; spell down vs v1.9 |
| 2026-10-29 | sm_v2 | E1 | 0.6397 | 0.4649 | 20 | 13.1 | ARCHIVE: min_spell=50 + short-X; public-GT era |
| 2026-11-03 | sm_v1_restore | E1 | 0.5131 | 0.3000 | 27 | 8.8 | Bit-identical to archived sm_v1; clarification restore |
| 2026-11-03 | sm_v1_1_compliant | E1 | 0.6051 | 0.3111 | 28 | 9.3 | Compliant sm_v1_1 defaults; pipeline log only (not a keep driver) |
| 2026-11-03 | sm_v1_2_compliant | E1 | 0.6065 | 0.3111 | 28 | 9.3 | Compliant sm_v1_2; pipeline log only (not a keep driver) |
| 2026-11-03 | sm_v1_3_compliant | E1 | 0.6063 | 0.3022 | 28 | 9.3 | Compliant sm_v1_3 cluster majority; ≠ Oct archive sm_v1_3; log only |
| 2026-11-07 | sm_v1_4_compliant | E1 | 0.6189 | 0.3043 | 29 | 8.9 | Compliant soft ST; ≠ Oct archive sm_v1_4; log only; later archived vs sm_v1_3 structure |
| 2026-11-07 | sm_v1_5_compliant | E1 | 0.6113 | 0.3111 | 28 | 9.3 | Compliant missing-ball H/A (Metrica external); ≠ Oct archive sm_v1_5; log only |
| 2026-11-08 | sm_v1_6_compliant | E1 | 0.6113 | 0.3111 | 28 | 9.3 | Half-kickoff reset; ≠ Oct archive; log only |
| 2026-11-09 | sm_v1_7_compliant | E1 | 0.6090 | 0.3238 | 25 | 10.4 | Prior LATEST; public numbers log only |
| 2026-11-12 | sm_v1_8_compliant | E1 | — | — | — | — | Prior LATEST; T32 D-bridge<25f; local spells 359 / mean 13.8; Metrica spellF1 0.491; public log TBD |
| 2026-11-14 | sm_v1_9_compliant | Codex | — | — | — | — | Prior LATEST; protected learned D; local 331 spells / mean 16.84 / s12 3 / s25 8; external Metrica spellF1 0.4961; Sportec excluded-match spellF1 0.4143 |
| 2026-11-14 | sm_v1_10_compliant | Codex | — | — | — | — | Prior LATEST; calibrated boosted D; local 294 spells / mean 10.07 / s12 0 / s25 2; original Metrica suite 0.5716 (includes training Game 1); five excluded-match whole-timeline suite 0.4811 |
| 2026-11-14 | sm_v2_2_compliant | Codex | — | — | — | — | Prior LATEST; residual ownership; local 319 spells / mean 9.28 / s12 0 / s25 13; Metrica suite spellF1 0.6345 / hidden 0.7878 |
| 2026-11-14 | sm_v2_3_compliant | Codex | — | — | — | — | Prior LATEST; boundary ensemble; local 342 spells / mean 8.65 / s12 0 / s25 8; Metrica vs v2.2 spellF1 0.6443 / hidden 0.8014 (both games); SHA256 `597e3ba8…` |
| 2026-11-16 | sm_v2_7_compliant | Codex | — | — | — | — | **LATEST**; event-first touch model on v2.6; local 339 spells / mean 8.73 / s12 0 / s25 7; Metrica 0.6654 / 0.8014; Sportec 0.6218 / 0.8069; SkillCorner D 0.4381 / 0.7272; SHA256 `017b9546…` |
| 2026-11-16 | sm_v2_6_compliant | Codex | — | — | — | — | Prior LATEST; fit-only corrector (margin 2.5); local 342 / 8.65 / 0 / 8; Metrica 0.6474 / 0.8014; Sportec 0.6187 / 0.8069; SHA256 `c912639c…` |
| 2026-11-16 | sm_v2_5_compliant | Codex | — | — | — | — | Demoted (leaky corrector); margin-3 error corrector on v2.3; local 341 spells / mean 8.68 / s12 0 / s25 8; Metrica spellF1 0.6509 / hidden 0.8014; Sportec 0.6191 / 0.8069; SHA256 `8b5ff700…` |
| 2026-11-16 | sm_v2_4_compliant | Codex | — | — | — | — | Demoted; error corrector on v2.3; local 353 spells / mean 8.38 / s12 0 / s25 10; Metrica spellF1 0.6650 / hidden 0.8015; Sportec 0.6226 / 0.8069; SHA256 `cd684645…` |
| | | | | | | | |

Add a new blank row for every model CSV (T30). Pipeline metrics in Notes are logs, not keep/drop drivers.

---

## Setup / ops notes

| Date | Note |
|------|------|
| 2026-10-24 | T01 — Python 3.14.0 via `.venv/`; numpy 2.5.4, pandas 3.1.0; `import numpy,pandas` OK |
| 2026-10-24 | T02 — Data completeness: OK — tactical atd.txt/xml/homo/smart_tagging + broadcast atd.txt/xml/homo present (mp4 optional, skipped) |
| 2026-10-24 | T03 — Google account ready: yes |
| 2026-10-24 | T04 — `git check-ignore -v data/tactical/` → `.gitignore:32:data/*`; nothing under `data/` staged |
| 2026-10-24 | T11 — `starter/baseline.py` regenerated four `baseline_*.csv` (ball on 113488/199319 frames, 56.9%) |
| 2026-10-24 | T13r signed off 2026-10-24 |
| 2026-10-26 | T20/T21 — `starter/features.py` + `state_machine.py` + `predict_v1.py` → `evaluation/results/sm_v1.csv` |
| 2026-10-28 | T22 — `predict_v1.bat` and `predict_v1.sh` write `evaluation/results/sm_v1_1.csv`, validate, then score. Validate OK (199319 rows). Public score matches the 2026-10-26 sm_v1_1 row (spellF1 0.3111, poss 28, dur 9.3 s) |
| 2026-10-28 | T23 — `smart_priors.py` + state_machine SET PIECES→D / weak phase votes; `sm_v1_3.csv` |
| 2026-10-28 | Soft priors: set-piece dead capped at 50f; drop final-third/box phase votes; CLI `--full-set-piece-dead` / `--no-phase-votes`. `sm_v1_4` / `sm_v1_5`. |
| 2026-10-28 | T24 diagnose `sm_v1_4` @IoU0.5: ~10 missed GT spells; ~7 near-miss (lagged/shifted edges, lags often +19…+59f); ~3 zero same-team overlap; long GT spells often fragmented by predicted D. Spurious: short islands after D / between opposing runs. |
| 2026-10-28 | T24 code: sandwich-only `_merge_short_spells`; `--restart-control-frames` / `--restart-window-frames` (default window=0). `sm_v1_6`–`sm_v1_8`. |
| 2026-10-29 | Short-D fill: `_fill_short_dead` (`min_dead_frames=50`); CLI `--min-dead-frames`. Post-stoppage reset CLI `--restart-reset-dead-frames` / `--restart-establish-frames` (default reset=0). `sm_v1_9` / `sm_v1_10`. |
| 2026-10-29 | `DEFAULT_MIN_SPELL_FRAMES=50` + `_fill_short_abstain`; `sm_v2.csv` (VERSION 2.0). |
| 2026-11-03 | Organiser clarification adopted: no GT/video parameter tuning. Code restored to sm_v1 (`sm_v1_restore.csv` == `sm_v1.csv`). |
| 2026-11-03 | Merged upstream `c073f29` + `f8f2629`: public 5 min not training/tuning; no hand labels on this game; external data = other games only. |
| 2026-11-03 | Compliant sm_v1_1 defaults shipped; `sm_v1_1_compliant.csv`. Local checks: graded X 29754→237; X-after-D 151→0; graded D↑ (edge heuristics). |
| 2026-11-03 | Compliant sm_v1_2: establish-without-X + min_spell 18; `sm_v1_2_compliant.csv`. Local vs v1_1: graded X 237→177; short&lt;12f 4→6; mean dur/n_spells flat; X-after-D 0. |
| 2026-11-03 | Compliant sm_v1_3 (T25): contested majority-near-ball; `sm_v1_3_compliant.csv`. Local vs v1_2: structure flat (X 177, n_spells 428, short&lt;12f 6); ~1.7% frames differ (H/A rebalance). ≠ Oct archive sm_v1_3. |
| 2026-11-07 | Compliant sm_v1_4 (T26): missing-aware soft SET PIECES→D; `sm_v1_4_compliant.csv`. Explained missing (D∩soft ST prior) 10.7%→21.7% of graded missing; graded D↑; n_spells 428→456; mean dur 11.73→10.33; short&lt;12f 6→10. ≠ Oct archive sm_v1_4. |
| 2026-11-07 | Compliant sm_v1_5 (T28): backdate-switch + gap-bridge mid on sm_v1_3 base; Smart Tagging off. External Metrica: mean spellF1 0.314→0.370, hidden-frame team acc 0.505→0.510; LOGO both games. Local vs sm_v1_3: n_spells 428→421; mean dur 11.73→11.92; short&lt;25f 24→16; X/D flat. ≠ Oct archive sm_v1_5. |
| 2026-11-08 | Compliant sm_v1_6 (T29): clear last_live at HT; D on half-2 graded before published 2nd KO (113742); seed opposite of 1st KO team. Local vs sm_v1_5: D 33212→35660; n_sp 421→416; mean 11.92→11.83; s12/s25/X flat; ≪ smooth50. Sanity: 1st live A → 2nd KO H. |
| 2026-11-09 | `sm_v1_7_compliant.csv` on `newrules#1` (VC on visible + sm_v1_6 on missing + half-kickoff-reset). Oct `sm_v1_7.csv` archive-only, **not from this branch**. Local vs sm_v1_6: n_sp 416→406; mean 11.83→12.11; s12 6→4; s25 16→11; D flat 35660; X 177→281; ≪ smooth50. Metrica: spellF1 0.429 / hidden 0.512 vs sm_v1_5. **Promoted** to forward path. Assess: [../docs/ASSESS.md](../docs/ASSESS.md). |
| 2026-11-12 | T27 pass-travel fill (`_apply_pass_travel`) abandoned: force-fill worse; X-only no-op. |
| 2026-11-12 | T27 redesign `_confirm_control`. Track A ↑; Metrica spellF1 ↓. **Closed not promoted.** |
| 2026-11-12 | T32 `_bridge_short_dead` (max 25f, same-team D sandwich). Track A: spells 406→359, mean 12.11→13.76, s25 11→9. Track B: spellF1 0.429→0.491, hidden flat, both games. **Promoted** `sm_v1_8_compliant.csv` → LATEST. [../docs/algorithms/dead-bridge.md](../docs/algorithms/dead-bridge.md). |

---

## Experiment notes (E1 / E2)

Short bullets only — why a change was kept or dropped. **Keep? must cite challenge rule / physics / external data.** Do not keep because public spellF1 moved.

| Date | Change | Keep? | Why |
|------|--------|-------|-----|
| 2026-10-24 | Public vs full-match baseline order | note | smooth125 wins public spellF1 (0.4242); docs say smooth50 wins full match (0.2433). Do not pick windows on public 5 min alone. |
| 2026-10-26 | sm_v1 defaults: control_frames=15, control_dist_m=3.0, out_margin=0.02, min_spell_frames=8 | yes (v1) | Hold on missing ball; hysteresis vs flicker; out-of-pitch ≠ missing. Public poss 27 vs raw 131; spellF1 0.30. D recall still weak (margin-only dead-ball). Next: T22 docs / T23 tags. |
| 2026-10-26 | Tweak v1.1 → sm_v1_1: control_frames=25, min_spell=12, missing_near_edge=50, still_near_edge=25 @0.5m/s; keep last_live on D | yes (v1.1) | vs sm_v1: spellF1 0.30→0.31, macroF1 0.51→0.61, poss 27→28, graded X 1239→237. Slight spell gain; D/macro better; poss not closer to 17 yet. |
| 2026-10-28 | T23 Smart Tagging → sm_v1_3: SET PIECES force D; weak phase team votes; `--no-smart-tagging` to A/B | keep (tune later) | vs sm_v1_2: spellF1 0.31→0.24 (down), macroF1 0.61→0.62, poss 28→25, graded X→0. Set-piece windows over-long as D (~35s tags). |
| 2026-10-28 | Soft set-piece + narrow phases → sm_v1_4: cap=50f; drop ball-in-final-third / box votes; rules still v1.1 | yes (prior default) | vs sm_v1_1: spellF1 0.311→0.313, macroF1 0.605→0.627, D recall 0.31→0.34, abstain 237→19. vs sm_v1_3: spell recovered. Set-piece D frames 43987→2950. |
| 2026-10-28 | Rule bump → sm_v1_5: control=37, min_spell=25 + soft priors | **no** (defaults stay v1.1) | vs sm_v1_4: spellF1 0.313→0.280 (down); poss 29→23 / dur 8.9→11.2 closer to ~17/14 but spell lost. Soft priors kept; hysteresis knobs left at 25/12. |
| 2026-10-28 | T24 sandwich merge → sm_v1_6 | **yes** (correctness; public tie) | Same spell/macro/matched@0.5 as sm_v1_4 on public window. Safer merge (no prev-alone / post-D wipe). Keep as default merge. |
| 2026-10-28 | T24 asymmetric restart → sm_v1_7 (restart_control=12, window=50) | **no** | spellF1 0.313→0.308; matched@0.5 still 8 but spurious 21→23; poss 29→31. Default `restart_window_frames=0`. Oct `sm_v1_7.csv` **not from this branch**. |
| 2026-10-28 | T24 `--no-phase-votes` → sm_v1_8 | **no** | spellF1 tied 0.313; macro 0.627→0.618. Keep phase votes on. |
| 2026-10-29 | Fill short D <50f → sm_v1_9 | **yes (default)** | vs sm_v1_4: spellF1 0.313→0.421, macro 0.627→0.634, matched@0.5 8→9, poss 29→22 / dur 8.9→11.9. Beats public smooth50 (0.326). FULL-match sanity poss 449→366 / mean dur 11.1→13.8 s (plausible). Physical reason: GT stoppages ≥138f; SM short D blips are tracking noise splitting spells. Cutoff 25≈50 in prior probe (stable). |
| 2026-10-29 | Restart reset after D≥50f → sm_v1_10 | **no** (reset default stays 0) | vs sm_v1_9: spellF1 0.421→0.391; matched@0.5 tied 9 but spur 13→16; abstain 15→144; FULL poss 366→418 / dur 13.8→11.5. Macro up slightly; spell/structure worse. Keep `--restart-reset-dead-frames` behind flag. |
| 2026-10-29 | Sandwich min_spell 12→50 + short graded-X fill → sm_v2 | ARCHIVE (not forward path) | Public-GT era keep. Superseded by Nov clarification restore to sm_v1. |
| 2026-11-03 | Restore defaults to sm_v1 (15/3.0/0.02/8; no ST priors; no edge D knobs) | **yes (active baseline)** | Rule-motivated T21; bit-identical to archived sm_v1.csv. Clarification: stop GT keep/drop. |
| 2026-11-03 | Compliant sm_v1_1 (control=25, min_spell=12, edge D, freeze last_live on D) | yes (then superseded by sm_v1_2) | Challenge: controlled switch ~1s; touch≠turnover; dead near edge; keep last through stoppage. Local: graded X 29754→237; X-after-D 151→0. Pipeline numbers logged once — not used to keep/drop. |
| 2026-11-03 | Compliant sm_v1_2: establish-without-X + min_spell 12→18 | yes (then superseded by sm_v1_3) | CHALLENGE: live ball → H/A (not X while establishing); touch ≠ turnover (milder merge ~0.72 s). Local vs sm_v1_1: graded X 237→177; X-after-D 0; short&lt;12f 4→6 (near); mean dur 11.72→11.73; n_spells 428 flat. Pipeline log only — not used to keep/drop. |
| 2026-11-03 | Compliant sm_v1_3: contested majority near ball (radius 3 m, contest ≤1 m) | yes (structure reference; then sm_v1_5) | Physics: team presence in control circle, not only nearest track; override nearest only when duel contested. Local vs sm_v1_2: graded X/n_spells/short islands flat; short&lt;25f 25→24; ~1.7% frames H↔A. Pipeline log only — not used to keep/drop. Rollback: `--no-majority-near`. |
| 2026-11-07 | Compliant sm_v1_4: missing-aware soft SET PIECES → D (cap 50f live; full tag while ball missing; no phase votes) | archived (not default) | Challenge: set piece = dead. Local vs sm_v1_3: explained missing ↑ but mean dur 11.73→10.33 and short islands ↑ — worse structure than sm_v1_3. Kept reachable via `--smart-tagging`. |
| 2026-11-07 | Compliant sm_v1_5: backdate-switch + gap-bridge mid (long-gap D dropped) | yes (then sm_v1_6) | Link & Hoernig 2017: switch at first opposing control; TBP continues through void phases then places turnover in the dark when teams disagree across a gap. Chosen on Metrica Sample_Game_1/2 (event-derived GT + synthetic dropout) vs sm_v1_3; LOGO both games; long-gap→D hurt hidden-frame acc. Local vs sm_v1_3: structure flat/better (n_sp 421, mean 11.92, s25 16); ≪ smooth50. Rollback: `--no-backdate-switch --gap-bridge off`. |
| 2026-11-08 | Compliant sm_v1_6: half-kickoff-reset (HT clear; D pre-2nd KO; seed opposite) | **yes (default)** | Laws of the Game: other team kicks off 2nd half; play dead until restart. Uses published `KICKOFF_VIDEO_FRAMES` (FAQ/DATA), not GT. Local vs sm_v1_5: pre-KO gap all D; short islands flat; mean dur 11.83; ≪ smooth50. Rollback: `--no-half-kickoff-reset`. |
| 2026-11-09 | Vidal-Codina 2022 possession-zone gain as cand, on the sm_v1_5 rules | **no** | Paper §2.3; published radii only. Metrica spellF1↑ but hidden↓. Full table: [../docs/algorithms/vidal-codina.md](../docs/algorithms/vidal-codina.md) §1. |
| 2026-11-09 | VC gap edges at pz=1.0 m: reappear / edge | **no** | SpellF1↑ hidden↓ both games. [../docs/algorithms/vidal-codina.md](../docs/algorithms/vidal-codina.md) §2. |
| 2026-11-09 | **sm_v1_7**: Vidal-Codina on visible frames; missing frames stay sm_v1_5 / sm_v1_6 | **yes (default)** | Metrica vs sm_v1_5: spellF1 0.370→0.429, hidden 0.510→0.512, both games. Local vs sm_v1_6: spells 416→406, mean 11.83→12.11, short&lt;25f 16→11; ≪ smooth50. ≠ Oct archive `sm_v1_7.csv`. Rollback → `sm_v1_6`. [../docs/algorithms/vidal-codina.md](../docs/algorithms/vidal-codina.md) §3. |
| 2026-11-12 | T27 pass-travel post-fill (`--pass-travel`) | **no** | Hop fill abandoned. |
| 2026-11-12 | T27 control-confirm sandwich gate | **no** | Track B lost. Closed. |
| 2026-11-12 | **sm_v1_8** T32 short D-sandwich bridge (≤25f, same H/A) | yes (then superseded) | Tracking flicker ≠ stoppage. Track A + B beat sm_v1_7. Still in stack under later models. Rollback `--no-dead-bridge`. [../docs/algorithms/dead-bridge.md](../docs/algorithms/dead-bridge.md). |
| 2026-11-14 | **sm_v1_10** calibrated boosted live/dead | yes (then sm_v2_2) | External Sportec+Metrica fit. [../docs/algorithms/boosted-dead-ball.md](../docs/algorithms/boosted-dead-ball.md). |
| 2026-11-14 | **sm_v2_2** residual ownership | yes (then sm_v2_3) | [../docs/algorithms/temporal-ownership-research.md](../docs/algorithms/temporal-ownership-research.md). |
| 2026-11-14 | **sm_v2_3** boundary ownership ensemble | yes (then sm_v2_6/7) | Beats v2.2 on Metrica A+B. Still in stack. Rollback `--no-boundary-model`. [../docs/algorithms/boundary-ownership-research.md](../docs/algorithms/boundary-ownership-research.md). |
| 2026-11-16 | sm_v2_4 live-team error corrector | **no (demoted)** | Crossfit/B/C up vs v2.3 but Track A s25 8→10 / mean down; restore v2.3. Opt-in `--error-corrector-file`. [../docs/algorithms/push-090-natural.md](../docs/algorithms/push-090-natural.md). |
| 2026-11-16 | sm_v2_5 player-node corrector | **no (held)** | Crossfit spell 0.722 < then-v2.4 0.729; vis acc down; worst Δ −0.025. [../docs/algorithms/push-090-natural.md](../docs/algorithms/push-090-natural.md). |
| 2026-11-16 | sm_v2_5 sparse switch editor | **no (held)** | Crossfit spell 0.669 < v2.3 0.721; worst Δ −0.116. Keep v2.3. [../docs/algorithms/push-090-natural.md](../docs/algorithms/push-090-natural.md). |
| 2026-11-16 | insert-only contested switches | **no (held)** | Crossfit spell 0.719 < v2.3 0.721; safer inserts still flat. Keep v2.3. [../docs/algorithms/push-090-natural.md](../docs/algorithms/push-090-natural.md). |
| 2026-11-16 | **sm_v2_5** structure-preserving corrector (margin 3.0) | **no (demoted, leaky)** | Pareto vs v2.3 but corrector trained on eval matches. Replaced by sm_v2_6. [../docs/algorithms/push-090-natural.md](../docs/algorithms/push-090-natural.md). |
| 2026-11-16 | boundary transition retrain ladder | **no (held)** | Best rung_b 0.7208 < v2.5 0.723; hidden regresses. Keep then-v2.5. [../docs/algorithms/push-090-natural.md](../docs/algorithms/push-090-natural.md). |
| 2026-11-16 | **sm_v2_6** honest fit-only corrector (margin 2.5) | yes (then sm_v2_7) | Retrain on fit set only; Pareto vs v2.3 on A/B/C. Rollback `--no-error-corrector`. [../docs/algorithms/push-090-natural.md](../docs/algorithms/push-090-natural.md). |
| 2026-11-16 | **sm_v2_7** event-first touch model (+ SkillCorner) | **yes (default / LATEST)** | Pareto vs v2.6 on A/B/C/D. Rollback `--no-event-first`. [../docs/algorithms/push-090-natural.md](../docs/algorithms/push-090-natural.md). |
