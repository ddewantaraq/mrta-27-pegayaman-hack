# BAR-ATH video smoke v2 — boundaries (frozen before any BAR-ATH run)

Supersedes the v1 smoke only in detector choice, selected **externally**.
Nothing here may change after seeing BAR-ATH output.

## Detector selection (external only)

- Candidates: FootAndBall (v1, ISSIA-native) vs Roboflow `sports`
  football-ball-detection YOLO (fine-tuned on SoccerNet by its authors).
- Selection metric: ISSIA cameras 5+6, identical protocol to a-ext
  (threshold defaults of each repo, native resolution; YOLO at 1280).
- Winner is frozen for the smoke. BAR-ATH output must never re-open
  this choice.

## ROI gating (pre-registered)

- When ATD ball present: accept a ball detection only within **5 m**
  (pitch space, after unprojection) of the ATD ball. Gate value set
  from ATD jitter scale in ball_refine_sim (σ ≤ 0.6 m contested;
  5 m is a loose pipeline gate, not a fitted constant).
- When ATD missing: accept the best detection (fill role) — reported
  separately, not part of the agreement kill metric.

## Smoke (identical to v1)

- Segment 11185–13184, kill = median disagreement > 2 m on
  ATD-present compared frames (gated detections only; also report
  ungated present rate to expose gating effect).
- Aggregates only; no per-frame output leaves memory; no threshold,
  gate, segment or detector changes after output is seen.
- All other v1 boundaries (no hand labels, no visual checking,
  local only, no calibration on BAR-ATH) remain in force.
