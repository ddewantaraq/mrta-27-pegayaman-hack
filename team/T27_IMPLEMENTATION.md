# T27 — Conservative Same-Team Pass-Like Travel Detection

## 1. Objective

Identify conservative same-team ball-travel candidates using
observed ball positions and nearest-team information.

This module provides supplementary evidence only.
It does not confirm that a pass occurred.

## 2. Implementation Files

- starter/pass_travel_t27.py — standalone candidate detector
- starter/t27_integration.py — optional analysis and label filtering
- tests/ — synthetic unit tests

## 3. Detection Rules

A candidate must satisfy:

1. Two separate contact events are detected.
2. Both endpoint contacts belong to the same team.
3. Travel distance meets the minimum threshold.
4. Estimated travel speed is physically plausible.
5. The gap is within the configured frame limits.
6. No prohibited intermediate contacts are present.
7. No ungraded or dead-ball frames are crossed.

The optional integration additionally requires the predicted
possession labels throughout the candidate interval to match
the candidate team.

## 4. Default Parameters

| Parameter | Default |
|---|---|
| fps | 25.0 |
| contact_radius_m | 2.0 |
| min_distance_m | 3.0 |
| max_speed_mps | 35.0 |
| max_gap_frames | 50 |
| min_gap_frames | 1 |
| max_intermediate_contacts | 0 |

These are heuristic defaults, not empirically optimized values.

## 5. Graded Mask

The integration requires an explicit graded mask.

For actual match analysis, the caller must provide the correct
match-specific graded mask from the existing pipeline.

Synthetic tests use explicitly constructed boolean masks.

The integration must not infer graded frames from labels != "X".

## 6. Validation

Latest local synthetic unit-test result:

29 tests executed.
29 tests passed.
0 failures.
0 errors.

Test coverage includes:

- Same-team travel candidates
- Opposing-team rejection
- Dead-ball boundaries
- Ungraded boundaries
- Team-switch rejection
- Continuous contact handling
- Intermediate contact handling
- Invalid physical parameters
- Invalid mask lengths
- Prediction-label immutability

## 7. Limitations

Nearest-team features do not identify individual players.

A detected candidate is not proof of a completed pass.

Straight-line distance and estimated speed are approximations.

Synthetic tests do not establish real-match accuracy or
improvements in macroF1, spellF1, or possession statistics.

## 8. Baseline Safety

T27 is optional and does not modify existing V1.6 predictions.

The following baseline files remain unchanged:

- starter/state_machine.py
- starter/predict_v1.py

No production integration has been activated.

## 9. Git Status

T27 implementation files and tests remain untracked.

No git commit or git push has been performed.

## 10. Next Steps

1. Review remaining detector edge cases.
2. Verify integration contracts and match-specific masks.
3. Review code and documentation before staging.
4. Evaluate real-match performance only when separately authorized.

## 11. Contact Duration Regression Fix

The detector identifies continuous contact runs.

For each pair of consecutive runs:
- The final frame of the first run is the departure point.
- The first frame of the next run is the arrival point.
- The gap counts frames without detected contact.
- Distance and speed are calculated between these endpoints.
- Dead-ball and ungraded boundaries remain enforced.

This prevents continuous contact frames from being incorrectly
counted as intermediate contact events.

The max_intermediate_contacts parameter is retained for API
compatibility. The current implementation only considers
consecutive contact runs and does not support skipping
intermediate contact runs.

## 12. Updated Validation

Latest synthetic unit-test result:
- 30 tests executed
- 30 tests passed
- 0 failures
- 0 errors

The contact-duration regression test is included.

Real-match accuracy has not been evaluated.
