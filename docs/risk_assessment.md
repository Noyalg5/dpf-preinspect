# Risk assessment

Four parts: a process FMEA for introducing the gate, a system FMEA for the station and software, AI-specific risks mapped to the mitigations in the code, and station health-and-safety hazards. A project risk register follows.

Ratings use the AIAG & VDA FMEA Handbook (2019) scales: severity (S), occurrence (O) and detection (D), each 1 to 10. Action priority (AP: H, M, L) follows the handbook's logic, which weighs severity first, then occurrence, then detection [verify each AP against the handbook's table before use]. The risk priority number (RPN = S × O × D) is shown for reference only, as the handbook recommends. All ratings are first estimates by the project team, to be reviewed in a cross-functional FMEA session.

## 1. Process FMEA: introducing the pre-inspection gate

| Step | Failure mode | Effect | S | Cause | O | Current or planned control | D | AP | RPN | Action |
|---|---|---|---|---|---|---|---|---|---|---|
| Goods-in data | Wrong unit ID or part number | Wrong history and geometry; wrong route | 6 | Manual entry, label damage | 4 | Barcode scan; validation gate (unknown values, duplicate IDs) | 3 | M | 72 | Scan-only entry |
| Goods-in data | Temperature logged in °F | Thermal history misread | 4 | Mixed controller exports | 4 | Probable-Fahrenheit check blanks the field and flags it | 2 | L | 32 | Source-level unit metadata |
| Weighing | Scale out of calibration | Load misjudged; wrong EXTENDED_CLEAN decisions | 5 | Drift, damage | 3 | Daily reference weight; range checks | 4 | M | 60 | Calibration schedule |
| Flow bench | Sensor dropout or leak at adaptor | Wrong restriction estimate | 5 | Worn seal, sensor fault | 4 | Dropout check; curve-fit residual; image-quality and input gates refer the unit | 3 | M | 60 | Seal inspection at shift start |
| Imaging | Face obstructed or unit badly seated | Defects missed or invented | 8 | Clamp, glove, tool, label in view | 4 | Image-quality gate (obstruction, seating); anomaly guard; operator check | 3 | H | 96 | Fixture redesign so clamps cannot enter the field |
| Imaging | Lighting or camera drift | Systematic misreads | 7 | Lamp ageing, lens dirt, setting change | 3 | Reference unit each shift; embedding PSI drift monitor | 4 | M | 84 | Lamp hours log |
| Routing | Serviceable unit routed to QUARANTINE | Core scrapped in error if not caught | 8 | Model error, threshold choice | 3 | Wrongful-scrap cap in threshold search; a person confirms every QUARANTINE | 3 | H | 72 | Monthly review of confirmed and released quarantines |
| Routing | Scrap unit routed to PROCEED | Wasted clean (as today) | 4 | Missed crack or internal damage | 5 | Downstream tests still catch it; minimum scrap-capture constraint | 2 | L | 40 | Better leak detection at station |
| Routing | Heavily loaded unit routed to PROCEED | Rework loop (as today) | 3 | Ash under-estimated | 5 | Downstream test catches it | 2 | L | 30 | None |
| Operator decision | Recommendation accepted without checking (automation bias) | Errors pass unchallenged | 7 | Trust, time pressure | 5 | Rationale and plausible outcomes on screen; override rate monitored; training | 5 | H | 175 | Periodic blind-review audits |
| Operator decision | Unjustified override | System knowledge lost; inconsistency | 4 | Habit, disagreement | 4 | Reason required; overrides audited | 3 | L | 48 | Feedback in toolbox talks |
| Lane routing | Unit sent to the wrong lane | Wrong process | 5 | Labelling, handling | 3 | MES route queue; lane scan | 3 | M | 45 | Lane scanners |

## 2. System FMEA

| Element | Failure mode | Effect | S | Cause | O | Control | D | AP | RPN |
|---|---|---|---|---|---|---|---|---|---|
| Camera and lens | Focus shift | Blurred images, missed cracks | 7 | Vibration, knock | 3 | Sharpness check in image-quality gate | 2 | M | 42 |
| Lighting | LED failure or ageing | Under-exposed or uneven images | 6 | Lifetime | 3 | Brightness check; reference unit | 3 | M | 54 |
| Backlight | Ash dust on panel | False leak spots | 5 | Dust | 5 | Anomaly guard; cleaning schedule | 4 | M | 100 |
| Edge PC | Crash or slowdown | Station stops | 5 | Hardware, updates | 3 | Health endpoint; fallback to today's process | 2 | L | 30 |
| Models | Silent degradation | Rising error rates | 8 | Drift in inputs or population | 4 | PSI drift, p-chart, calibration-over-time monitoring | 4 | H | 128 |
| Models | Unknown defect type | Misclassified | 8 | New failure mode | 3 | Unexplained-anomaly guard routes to REFER | 4 | H | 96 |
| Data pipeline | Corrupt or altered training data | Wrong model | 8 | Storage fault, manual edits | 2 | SHA-256 manifest; training refuses failed integrity | 2 | L | 32 |
| Audit trail | Records altered or deleted | Loss of traceability | 7 | Insider, tool error | 2 | Append-only triggers; hash chain; `preinspect audit verify` | 2 | L | 28 |
| MES link | Route not delivered | Unit waits or follows default | 5 | Network | 3 | Route queue with status; manual lane override | 3 | M | 45 |

## 3. AI-specific risks and implemented mitigations

Framed after ISO/IEC 23894:2023 (risk management for AI).

| Risk | Mitigation in this repository |
|---|---|
| Data drift (population, camera, lighting) | PSI on metadata and image embedding, p-chart of REFER and QUARANTINE share, calibration over time: `service/monitoring.py`, Quality Engineer page |
| Label noise in recorded dispositions | Consistency check of disposition against test results in the validation gate (`data/validate.py`); headline metrics against truth; calibration and conformal sets on held-out data |
| Automation bias | Rationale, plausible outcomes and overlays shown with every route; overrides require a reason and are audited (`service/schemas.py`, `service/audit.py`); QUARANTINE always needs confirmation |
| Unseen defects | Unexplained-anomaly guard (`features/learned.py`, `models/patchcore.py`) routes to REFER |
| Lighting or acquisition change | Image-quality gate fitted on training images (`service/image_quality.py`); lighting randomisation in training; commissioning procedure (`validation/domain_shift.py`) |
| Over-confidence | Probability calibration with method chosen on held-out data (`models/calibration.py`); conformal prediction sets (`decision/conformal.py`) |
| Unsafe automatic decisions | Wrongful-scrap cap and minimum scrap capture in the threshold search (`decision/routing.py`); acceptance tests AC-01, AC-02, AC-06 |
| Opaque decisions | Concept route with physical features; template rationale (`decision/rationale.py`); explanations tested against ground truth (`explain/xai_tests.py`) |
| Traceability | Run provenance (git commit, data manifest hash) on every result; model and data versions in every audit record |
| Cyber and OT exposure | No external API exposure; edge PC in the OT zone; only route and status leave the station; dependencies pinned through `pyproject.toml` |

## 4. Station health and safety hazards

**Draft for review by a competent person.** This list supports, and does not replace, the site's risk assessment and COSHH assessment.

| Hazard | Who | Controls |
|---|---|---|
| Manual handling of heavy units (filters in cans can weigh tens of kilograms) | Operators | Lifting aid or roller conveyor to the cradle; two-person lift above the site limit; cradle at working height; manual-handling training |
| Pinch and crush points (turntable, cradle, enclosure door) | Operators, maintenance | Guarded turntable; interlocked door; hold-to-run motion; lock-off for maintenance |
| Bright light (dome, ring and backlight) | Operators | Light-tight enclosure; interlock switches the lights off when the door opens |
| Ash and soot dust (oil-additive ash contains metal compounds) | Operators, maintenance | COSHH assessment; enclosure with local exhaust ventilation; no compressed-air blowing at the station; HEPA vacuum for cleaning; suitable gloves and respiratory protection as determined by the assessment; washing facilities |
| Electrical (lighting, PC, turntable drive) | Maintenance | Low-voltage lighting; enclosure IP rating suited to dust; PAT and inspection schedule |
| Slips and trips (cables, dropped units) | All | Cable management; defined lanes; housekeeping |
| Display-screen use | Operators | Screen at suitable height; glove-friendly touch screen |

## 5. Project risk register

| ID | Risk | Likelihood | Impact | Owner | Mitigation |
|---|---|---|---|---|---|
| P1 | Real data slower to collect than planned | Medium | High | Associate, company supervisor | Start capture with a temporary rig in month 3; simulator keeps development moving |
| P2 | Labelling effort underestimated | High | Medium | Associate | Labelling guide; pre-labels from the model; label only REFER and sampled units later |
| P3 | Large sim-to-real gap | High | Medium | Associate, academic supervisor | Concept route; re-parameterise simulator from real data; fine-tune on real faces |
| P4 | Too few scrap examples of some failure modes | Medium | High | Quality | Longer capture window; anomaly guard for rare modes; teardown notes |
| P5 | Operator distrust or over-trust | Medium | High | Company supervisor | Involve operators in design; rationale; shadow mode; training |
| P6 | IT/OT security approval delays | Medium | Medium | IT/OT | Early engagement; edge-only design; documented data flows |
| P7 | Hardware lead times | Medium | Medium | Maintenance | Order long-lead items after requirements sign-off |
| P8 | Scope creep (new-build inspection, other products) | Medium | Medium | Steering group | Change control; retargeting kept as a configuration exercise after core delivery |
| P9 | Knowledge leaves with the associate | Medium | High | Company supervisor | Training manual, recorded sessions, engineer guide, handover period |
| P10 | Acceptance criteria not agreed | Low | High | Quality | Criteria workshop in WP1; illustrative targets as a starting point |
