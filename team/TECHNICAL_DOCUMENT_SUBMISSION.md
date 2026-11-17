# More than a Hack 2027 — Phase 1: Ball Possession
## Technical document — team review draft

**Team:** Pegayaman Hack
**Selected submission model:** `sm_v2_7_compliant.csv` (approved `LATEST`, accepted as final 17 November 2026; SHA256 `017b9546…`)
**Status:** Final technical document for submission. Covers exactly the six items required by `SUBMISSION.md` section 3.

### 1. Approach and rationale

We predict one of four labels for each video frame: `H` (home-team possession), `A` (away-team possession), `D` (dead ball), and `X` (no established possession/abstention). The challenge match contains 199,319 frames. The ranking metric, spellF1, rewards recovering coherent possession intervals rather than merely matching isolated frames. Automatically tracked ball locations are missing in a substantial fraction of frames, and individual tracking is imperfect.

Our approved `sm_v2_7` pipeline combines (1) a rule-based state machine and physically motivated possession logic, (2) a calibrated boosted live/dead-ball classifier, (3) a baseline-aware player-set residual ownership model, (4) an ensemble ownership and turnover-boundary head, (5) a fit-only live-team error corrector (margin 2.5), and (6) an event-first touch model (ball-touch candidates, boosted touch-team classifier, two-state Viterbi decode, conservative visible-frame overwrite at |logit| ≥ 1.0). Observed-ball evidence supports team ownership when available; temporal reasoning handles missing-ball stretches. The boundary head adjusts transition costs during missing frames, while the ensemble combines ownership logits. The pipeline retains explicit dead-ball/no-possession handling rather than forcing a team label at all times.

We chose this hybrid design because nearest-player-only voting is sensitive to tracking gaps, contested balls, and short-lived possession switches. Rules provide a reproducible foundation, while externally trained models improve live/dead separation and ownership under missing-ball conditions. The approved inference path uses packaged model weights and NumPy/pandas; retraining is not required to reproduce the CSV.

### 2. Assets used, and alternatives tested

**Challenge-provided assets — used:** The final `sm_v2_7` inference consumes **tactical ATD only** (player/ball tracking from the tactical recording). No other supplied modality is an inference input.

**Challenge-provided assets — tried and dropped:** Smart Tagging was tested as soft set-piece/phase priors and archived (phase intervals are not possessions; the shipped default has it off). Tactical homography is loaded only for debugging, never as a model input. The broadcast feed and its tracking were deferred throughout (separate unsynchronised timeline, 65.8% usable homography) and never fused. A tactical-video ball detector (FootAndBall) was evaluated as post-freeze research under frozen boundaries and killed on domain shift; it is not part of the submission.

**External assets:** Metrica sample match tracking and event-derived labels, Sportec Open DFL match data, and SkillCorner Open Data support model fitting and external assessment. Packaged learned parameters are stored in `starter/models/`. See `docs/algorithms/boosted-dead-ball.md`, `temporal-ownership-research.md`, `boundary-ownership-research.md`, and `push-090-natural.md` for method and source details.

**Tested but not selected as final defaults:** T27 pass-like travel/controlled-possession modifications were not promoted after external evaluation; `sm_v2_4` live-team error correction was demoted because possession structure deteriorated despite some external improvements; `sm_v2_5` was demoted after its corrector was found to be trained on evaluation matches (leakage), replaced by the honest fit-only corrector in `sm_v2_6`. Switch editors, insert-only switches, boundary retraining ladders, HSMM decoding, union touch candidates, and GNN/SSL probes were all tested under the same gates and held. The short dead-ball bridge remains part of the approved stack.

### 3. Training data and annotation provenance

Every labelled dataset used, by final component. **No labels, annotations, or human-produced data from the challenge BAR–ATH game were used for training, validation, or model selection in any component, and the public five-minute ground truth was used only as a one-off pipeline check, never for tuning.**

| Component | Fit matches | Label source and who produced it |
|---|---|---|
| Boosted live/dead (`sm_v1_10`) | Sportec J03WOH, Metrica Sample Game 1 (2 matches) | Sportec Open DFL frame-level `BallStatus` (provider tracking annotation, Bassek et al. 2025, CC BY 4.0); Metrica event-derived live/dead from Metrica's published sample-game events |
| Residual ownership (`sm_v2_2`) + boundary ensemble (`sm_v2_3`) | Sportec J03WOH, J03WN1, J03WOY + Metrica Sample Game 1 (4 matches) | Sportec `BallStatus`/`BallPossession` mapped frame-exactly to H/A/D; Metrica H/A/D derived from Metrica event types by documented code (`starter/external/metrica_gt.py`) |
| Error corrector v2 (`sm_v2_6`) | Sportec J03WOH, J03WN1, J03WOY + Metrica Sample Game 3 (4 matches, fit-only after leakage audit) | Same Sportec/Metrica label construction as above; margin 2.5 selected by leave-one-match-out over these fit matches only |
| Event-first touch model (`sm_v2_7`) | The 4 fit matches above + 16 SkillCorner Open Data training-split matches (20 matches total; 10,745 labelled rule-candidate touches in the fit protocol) | SkillCorner `possession.group` team labels (provider broadcast-tracking annotation, MIT, © SkillCorner via PySport, pinned commit + sha256 manifest); frames without a group (~45%) treated as unlabelled and ignored, never guessed |

Evaluation-only data (never fitted): Metrica Sample Game 2, Sportec J03WMX/J03WPY/J03WQQ/J03WR9, and the 4-match SkillCorner hold-out (SC_1953632, SC_2006229, SC_2007448, SC_2013725, lowest four by sha256 rule). Metrica Game 1 is a training source and is never described as held-out. Label caveats: Metrica labels are event-derived and noisier than Sportec frame-level labels; SkillCorner is 10 fps broadcast tracking resampled to 25 fps with no dead-ball label. Exact per-component training manifests ship in `starter/models/*.json` (`training_matches`) and are enforced by the assess leakage guard.

### 4. Possession rules

**Established control:** A team switch requires sustained control: an opposing nearest player within 3 m for 25 consecutive frames (~1 s), or equivalent learned-ownership evidence. A contested override uses cluster majority within 3 m when the nearest-player duel is within 1 m. A single touch, deflection, or lost aerial duel without subsequent control does not switch possession; contested/loose balls stay with the last team.

**Loose or missing ball:** Missing ball is never treated as dead by itself. The last live team is carried through gaps; if the confirmed team after a gap differs, the switch is placed at the gap midpoint and backdated to first opposing control. Learned boundary-dependent transition costs decode ownership during missing stretches. `X` appears only where possession was never established (e.g. cold start) or outside graded play.

**Challenges and turnovers:** Switches must clear layered evidence: rule hysteresis (above), the boundary ensemble's learned switch costs, the fit-only corrector (flips only on disagreement at |logit| ≥ 2.5), and the event-first touch decode (overwrites visible frames only on disagreement at |logit| ≥ 1.0). H/A islands shorter than 18 frames are merged to suppress flicker.

**Restarts and dead ball:** Dead ball combines geometric signals (out-of-pitch, near-edge still/missing) with the calibrated boosted live/dead classifier. At half-time the carried team is cleared, graded second-half frames before the published second kickoff are `D`, and the second half is seeded to the opposite kickoff team. Same-team `D` sandwiches of at most 25 frames are bridged so tracker flicker does not split a possession.

### 5. Evaluation and limitations

The approved `sm_v2_7` was promoted on external evidence and possession structure, not on the challenge's public five-minute labels, under a four-track assess (structure / Metrica / Sportec / SkillCorner hold-out) with a training-leakage guard. Final external scores: Metrica spellF1 **0.6654** / hidden accuracy **0.8014**; Sportec **0.6218 / 0.8069** (all four suite matches improve spellF1 over `sm_v2_6`); SkillCorner hold-out **0.4381 / 0.7272**. The local challenge output has **339 possession spells**, mean **8.73 s**, **zero** spells under 12 frames, and **seven** under 25 frames. These are external and structural diagnostics, **not** private-ground-truth scores; the private challenge score is unknown.

Limitations: (1) contested turnovers remain the dominant failure — an external switch audit still misses ~31% of ground-truth switches, 94.5% of them on contested peaks where the ball is airborne, occluded, or missing (41% of graded frames have no tracked ball); (2) tracking carries no player identities, so control in duels must be inferred from collective geometry alone; (3) domain shift between external providers and the challenge match, including a materially higher predicted dead-ball count than earlier variants; (4) some natural-condition external matches regress slightly even when aggregates improve. With more time we would pursue externally trained video ball refinement (a frozen-detector smoke was tested and killed on domain shift) and genuinely independent external validation of switch timing.

### 6. Reproduction

Requirements: Python 3.8+ (Python 3.11 is supported), official match assets obtained under the organiser's terms, and the repository's packaged model weights. The official match data are **not** redistributed in the public repository.

**Windows PowerShell (from the repository root):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-inference.txt
```

Place the official organiser-provided match bundle in `data/` according to `DATA.md`, or configure `MTAH_DATA_DIR`. Then run:

```powershell
python starter/predict_v1.py --out evaluation/results/sm_v2_7_compliant.csv
python evaluation/validate_submission.py evaluation/results/sm_v2_7_compliant.csv
```

The validator confirms the format and 199,319 rows. The default predict reproduces the submitted file byte-identically (SHA256 `017b9546…`); this was verified against the frozen artifact. Running the prediction requires access to the licensed challenge assets. For external assessment, see `docs/ASSESS.md`; Metrica/Sportec/SkillCorner downloads are optional for generating the submission CSV and are never redistributed.

### References and internal audit trail

- `SUBMISSION.md`, `RULES.md`, `DATA.md`, `README.md`
- `docs/ASSESS.md`, `team/RESULTS.md`, `team/TECH_DOC_DRAFT.md`
- `docs/algorithms/boosted-dead-ball.md`
- `docs/algorithms/temporal-ownership-research.md`
- `docs/algorithms/boundary-ownership-research.md`
- `docs/algorithms/controlled-possession.md`, `docs/algorithms/dead-bridge.md`
- External data sources: Metrica Sports sample-data repository; Sportec Open DFL (figshare DOI 10.6084/m9.figshare.28196177.v1, CC BY 4.0, Bassek et al. 2025); SkillCorner Open Data (MIT, © SkillCorner, distributed with PySport, pinned commit with sha256 manifest)

**Submission checklist:** Team name confirmed (Pegayaman Hack); training datasets/annotators/counts verified in Section 3; final asset usage and active rules confirmed in Sections 2 and 4; validate the exact CSV submitted; name the final `.pdf` after the registered team. Submission deadline: **17 November 2026**.
