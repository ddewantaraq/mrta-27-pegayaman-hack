# Touch supervision from unused event files (P1, held)

Deep-research proposal P1: the fetch scripts download Sportec `events_raw`
XML and SkillCorner `dynamic_events.csv` / `phases_of_play.csv` but no
parser references them. This note records the parse-and-align probe and the
union-candidate head that followed. Verdict: **held** — detector recall
gains do not convert to spellF1.

## Parsers (`research/experiments/touch_supervision.py`)

- Sportec: event `EventTime` → nearest tracking frame N via the ball
  `FrameSet` N/T clock; `Play` → touch, `TacklingGame` → duel
  (`PossessionChange` kept); team via `matchinformation` home/away ids.
  `sportec_frame_offset` maps N to feature index (mirrors
  `extract_sportec`). 984 plays + 229 duels on J03WMX.
- SkillCorner: `player_possession` → touch (team = `team_id`),
  `on_ball_engagement` → duel. Verified on all rows of one match: duel
  `team_id` is the pressing side (never equals the possessor's team), so
  duel team is resolved via `player_in_possession_id` → `match.json`
  players. `frame_start` indexes tracking rows (10 fps);
  `frame25 = round(frame10 * 2.5)`. 959 touches + 878 duels on SC_1874553.
- Tests: `research/tests/test_touch_supervision.py` (5 cases, fixture
  XML/CSV in tempdir, no data reads).

## Alignment (`research/experiments/touch_align.py`)

Report: `evaluation/reports/touch_supervision_align.json`. J03WOH is the
fit-set comparison; J03WMX is readout-only (Track C); SC_1874553 is
SkillCorner train (never Track D).

| Match | rule recall @12 | union recall @12 | event-team agree |
|---|---|---|---|
| J03WOH (fit) | 0.560 | 0.686 | 0.90 play / 0.65 duel |
| J03WMX (readout) | 0.707 | 0.801 | 0.93 play / 0.68 duel |
| SC_1874553 (train) | 0.372 | 0.789 | 0.84 touch / 0.82 duel |

Accel detector: accel ≥ 10 m/s² or slowdown ≥ 3 m/s within 2 m of a
player (physics constants, untuned). Accel alone beats rule alone on SC
recall (0.72 vs 0.37) at equal precision — resampled ball motion defeats
turn/speed-jump thresholds. Events are touch supervision, not switch
supervision (event→switch recall ≤ 0.30 Sportec / 0.66 SC).

## Union head (`research/experiments/touch_union_experiment.py`)

Same HGB config, GT lead+3 labels, decoder grid as `event_first.py`;
only candidates change (rule 10,745 → union 20,881 labelled touches).
FIT LOMO natural spellF1: rule 0.7702 / hidden 0.5617 vs union 0.7635 /
0.5612, both selecting c0.0_m1.0. Dual-margin sweep (accel-only margin
1/2/3/4, rule fixed 1.0): 0.7635 / 0.7631 / 0.7631 / 0.7627 — gating
harder does not help. Reports: `touch_union_crossfit.json`,
`touch_union_margin.json`.

## Decision

**Hold.** Do not ship union candidates. Keep the parsers: event-team
labels (0.93/0.83) are validated supervision for future touch-team heads.
Next P1 follow-up if revisited: event-labeled soft-touch training targets
instead of denser candidates.
