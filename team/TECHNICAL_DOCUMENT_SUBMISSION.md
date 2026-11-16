# More than a Hack 2027 — Phase 1: Ball Possession
## Technical document — team review draft

**Team:** Pegayaman Hack
**Selected submission model:** `sm_v2_3_compliant.csv` (approved `LATEST` as of 16 November 2026)
**Status:** Draft for E1/E2 technical and data-provenance review. Not yet approved for submission.

### 1. Approach and rationale

We predict one of four labels for each video frame: `H` (home-team possession), `A` (away-team possession), `D` (dead ball), and `X` (no established possession/abstention). The challenge match contains 199,319 frames. The ranking metric, spellF1, rewards recovering coherent possession intervals rather than merely matching isolated frames. Automatically tracked ball locations are missing in a substantial fraction of frames, and individual tracking is imperfect.

Our approved `sm_v2_3` pipeline combines (1) a rule-based state machine and physically motivated possession logic, (2) a calibrated boosted live/dead-ball classifier, (3) a baseline-aware player-set residual ownership model, and (4) an ensemble ownership and turnover-boundary head. Observed-ball evidence supports team ownership when available; temporal reasoning handles missing-ball stretches. The boundary head adjusts transition costs during missing frames, while the ensemble combines ownership logits. The pipeline retains explicit dead-ball/no-possession handling rather than forcing a team label at all times.

We chose this hybrid design because nearest-player-only voting is sensitive to tracking gaps, contested balls, and short-lived possession switches. Rules provide a reproducible foundation, while externally trained models improve live/dead separation and ownership under missing-ball conditions. The approved inference path uses packaged model weights and NumPy/pandas; retraining is not required to reproduce the CSV.

### 2. Assets used, and alternatives tested

**Challenge-provided assets:** The official BAR–ATH match bundle includes automatic player/ball tracking, tactical and broadcast feeds, Smart Tagging, and camera-to-pitch geometry. The team's prediction pipeline uses tactical ATD-derived features; the other supplied assets are described in `DATA.md`. **Review needed:** confirm precisely which supplied modalities the final `sm_v2_3` inference actually consumes, rather than implying all were used.

**External assets:** Metrica sample match tracking and event-derived labels, and Sportec Open DFL match data, support model fitting and external assessment. Packaged learned parameters are stored in `starter/models/`. See `docs/algorithms/boosted-dead-ball.md`, `temporal-ownership-research.md`, and `boundary-ownership-research.md` for method and source details.

**Tested but not selected as final defaults:** T27 pass-like travel/controlled-possession modifications were not promoted after external evaluation; `sm_v2_4` live-team error correction was demoted because possession structure deteriorated despite some external improvements; `sm_v2_5` player-node correction was held after failing its cross-fit gate. Other boundary blending, missing-ball retraining, and temporal corrections were also tested without promotion. The short dead-ball bridge remains part of the approved stack.

### 3. Training data and annotation provenance — MUST VERIFY BEFORE SUBMISSION

The team's results log says the `sm_v2_3` boundary model and earlier ownership model were fitted using Sportec matches **J03WOH, J03WN1, J03WOY** and **Metrica Sample Game 1**. Earlier learned live/dead variants also used external Sportec/Metrica data. The documented excluded-fit evaluation set includes **Metrica Sample Game 2** and Sportec **J03WMX, J03WPY, J03WQQ, J03WR9**; Metrica Game 3 was used for additional checks and should not be described as an untouched test.

**Required E1/E2 confirmations:**

- Exact labelled datasets used to fit **each** final learned component (including any distinct training splits for live/dead and boundary models).
- Original dataset publishers, who generated each label (human annotators, official event providers, or automated event-derived procedures), and the exact number of labelled matches, frames, or events used for training.
- Label construction and mapping to H/A/D, including exclusions and quality caveats.
- Confirmation that **no labels or human-produced annotations from the challenge BAR–ATH game** were used for training or model selection, and that the public five-minute ground truth was not used for tuning.

**Do not submit this section with these fields unresolved.** The supplied summary confirms match IDs and broad provenance, but does not establish exact label counts or original annotator identities.

### 4. Possession rules

**Established control:** Team assignment uses observed ball proximity and player geometry, with temporal persistence to avoid interpreting every brief touch as a turnover. A contested ball does not automatically switch possession merely because a different player becomes nearest.

**Loose or missing ball:** During tracking gaps, the pipeline carries and reconciles plausible ownership evidence and uses temporal decoding to infer continuity or a turnover. It does not assume the ball is dead simply because the tracker loses it. The model can retain `X` when possession is not established.

**Challenges and turnovers:** The system requires evidence of a genuine change of control; a boundary-aware component adjusts the cost of switches, especially through missing-ball stretches. Short H/A fragments are treated conservatively to reduce flicker.

**Restarts and dead ball:** Dead-ball detection combines rule-based signals with the calibrated learned classifier. The half-time/kickoff logic resets prior possession and treats the pre-restart interval as dead. Short isolated D interruptions between same-team possession spells may be bridged under constrained rules, rather than splitting a continuous spell.

**Review needed:** Check the final active flags and precise thresholds against `starter/predict_v1.py`, `starter/state_machine.py`, and the final algorithm notes before asserting specific numeric rules.

### 5. Evaluation and limitations

The approved `sm_v2_3` was promoted on external evidence and possession structure, not on the challenge's public five-minute labels. In the documented five-match excluded-fit full-timeline suite, the mean external spellF1 improved **0.580146 → 0.599177**, and hidden-frame team accuracy improved **0.774845 → 0.791608**, compared with `sm_v2_2`. In the original Metrica protocol, spellF1 improved **0.634473 → 0.644263**, and hidden accuracy **0.787772 → 0.801381**; **Metrica Game 1 was part of training**, so this protocol must not be called wholly held-out. The local challenge output was reported as **342 possession spells**, mean **8.65 s**, **zero** spells under 12 frames, and **eight** under 25 frames. These are structural diagnostics, **not** private-ground-truth scores.

Limitations include missing/incorrect ball and player tracks, uncertain individual control in duels, imperfect restart/dead-ball timing, domain shift between external providers and the challenge match, and short-spell/turnover-boundary errors. Some natural-condition matches regress slightly even when aggregate external metrics improve. The private challenge score is unknown. Future work would target better calibration of live/dead transitions and ownership switches with genuinely independent external validation.

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
python starter/predict_v1.py --out evaluation/results/sm_v2_3_compliant.csv
python evaluation/validate_submission.py evaluation/results/sm_v2_3_compliant.csv
```

The validator should confirm the expected format and 199,319 rows. **These commands are documented for reproduction; they have not been executed as part of writing this draft.** Running them requires access to the licensed challenge assets. For external assessment, see `docs/ASSESS.md`; Metrica/Sportec downloads are optional for generating the submission CSV.

### References and internal audit trail

- `SUBMISSION.md`, `RULES.md`, `DATA.md`, `README.md`
- `docs/ASSESS.md`, `team/RESULTS.md`, `team/TECH_DOC_DRAFT.md`
- `docs/algorithms/boosted-dead-ball.md`
- `docs/algorithms/temporal-ownership-research.md`
- `docs/algorithms/boundary-ownership-research.md`
- `docs/algorithms/controlled-possession.md`, `docs/algorithms/dead-bridge.md`
- External data sources: Metrica Sports sample-data repository; Sportec Open DFL (figshare DOI 10.6084/m9.figshare.28196177.v1)

**Submission checklist:** Confirm team name; verify every training dataset/annotator/count; confirm exact final asset usage and active rules; have E1/E2 review; name final `.pdf` after the registered team; validate the exact CSV submitted. Submission deadline: **17 November 2026**.
