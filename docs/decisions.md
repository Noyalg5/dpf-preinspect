# Decision log

Architecture decision records (ADRs) and assumptions, newest last. Each entry
states the context, the decision and its consequences.

## ADR-001 Python environment

- **Context.** The machine's pyenv build of CPython 3.11 lacks the `_lzma` module, so `import torchvision` fails.
- **Decision.** `make setup` creates the virtual environment from a uv-managed standalone CPython 3.11 when uv is installed, and falls back to `python3 -m venv` otherwise.
- **Consequences.** No Homebrew packages are needed. Users without uv need a Python build that includes `lzma`.

## ADR-002 Gradient boosting library

- **Context.** LightGBM and XGBoost wheels on macOS need an OpenMP runtime, usually installed through Homebrew.
- **Decision.** Use scikit-learn `HistGradientBoostingClassifier`/`Regressor` for every tabular model.
- **Consequences.** Pure-pip install; native missing-value support; SHAP uses the model-agnostic explainer where the tree explainer does not support the model.

## ADR-003 Run folders and provenance

- **Context.** Results must be traceable to code, configuration and data.
- **Decision.** Every training run creates `runs/<timestamp>_<profile>/` holding the resolved config, git commit (and dirty flag), the SHA-256 of the dataset manifest, library versions and all metrics. Reports are rendered only from these folders.
- **Consequences.** A number in any document can be traced back to a run folder and a dataset hash.

## ADR-004 Compute device and sizing

- **Context.** Development machine: Apple M2, 8 GB unified memory, 8 cores. Benchmark (ResNet-18, training step): about 89 images/s at 256 px on MPS, about 15 images/s on CPU.
- **Decision.** Device order cuda, mps, cpu with `PYTORCH_ENABLE_MPS_FALLBACK=1`; `PREINSPECT_DEVICE` forces a device (CI uses CPU). Images are loaded lazily from disk rather than held in memory. Dataset sizes, crop sizes and epochs per profile are set from this benchmark to meet the time budgets.
- **Consequences.** The demo profile trades some accuracy for a run time under 15 minutes; the full profile restores resolution and seeds.

## ADR-005 Pretrained weights

- **Context.** ImageNet weights help both the U-Net encoder and the PatchCore feature extractor, but the system must run offline.
- **Decision.** torchvision ResNet-18 ImageNet weights are used when available in the local cache or downloadable; otherwise, or when `PREINSPECT_OFFLINE_WEIGHTS=1`, the encoder is randomly initialised and the run records which was used.
- **Consequences.** CI is deterministic and offline; offline anomaly-guard results are weaker and flagged as such.

## ADR-006 Synthetic generator v1.0 frozen

- **Context.** Experiments must not be tuned against the data they are scored on, and every result must trace to one generator version.
- **Decision.** Generator parameters (the `generator` section of `configs/base.yaml`) were set from `docs/domain_primer.md` and adjusted only for scenario plausibility *before any model was trained*: ash accumulation rate (2.2 to 3.0 g/L per 1 000 h), residual-ash sensitivity of the post-clean ΔP ratio (0.035 to 0.05 per g/L), mean plug-loss rate (0.35 to 0.12 per face) and downstream-test noise (0.025 to 0.015). The targets were a class mix of roughly half PASS, a quarter each REWORK and SCRAP, plausible plug-loss prevalence, and about 30 % defect-free units. Version 1.0 was then frozen. Any later change bumps `generator.version` and re-runs every stage.
- **Consequences.** The manifest records the generator version and the SHA-256 of its configuration; run provenance records the manifest hash.

## ADR-007 Evaluation labels

- **Context.** Recorded dispositions contain downstream test noise near the specification limit (about 6 % disagree with the latent truth) on top of transcription errors.
- **Decision.** Train on recorded labels, which is all a real line would have. Report headline metrics against the latent true outcome, and report recorded-label metrics alongside where they differ.
- **Consequences.** Headline numbers measure the method rather than the label noise. A real deployment would not have the truth; `docs/real_data_plan.md` covers how to estimate label noise there.

## ADR-008 Field-level quarantine for ambiguous temperatures

- **Context.** A maximum regeneration temperature above 1 000 could be a Fahrenheit slip (a normal 550 to 700 °C reading becomes 1 022 to 1 292) or a genuine uncontrolled regeneration. Quarantining whole records removed most novel-defect units, which are exactly the units that matter.
- **Decision.** Such values are blanked for modelling (the field is optional and already missing on about 40 % of records), and the record is kept and listed for verification against the source log. Every other check quarantines the whole record.
- **Consequences.** Genuine extreme readings are lost to the model until verified. The real fix is source-level unit metadata on the controller export; see `docs/real_data_plan.md`.

## ADR-009 Mask conventions

- **Context.** Some defects are hard to see under some lighting (stains under backlight), and cracks are one to two pixels wide.
- **Decision.** Masks are physical ground truth, independent of lighting, except where a clamp occludes the face. Crack masks are dilated by one pixel as a labelling tolerance. Masks are stored as one 8-bit bitmask per image, with bit *k* for defect class *k*.
- **Consequences.** Lighting-dependent visibility shows up in per-lighting performance and the imaging DoE, not in the labels.

## ADR-010 Novel-defect set

- **Context.** Thermal melt fails the integrity test, so every novel-defect unit is SCRAP.
- **Decision.** The novel set evaluates the anomaly guard (image and pixel AUROC) and routing to REFER, not outcome classification.
- **Consequences.** A model that calls melted units SCRAP without the guard firing is still flagged in the report: the guard should route unseen defects to an inspector.

## ADR-011 Classical lattice estimation uses part metadata

- **Context.** The spectrum of a face has two candidate fundamental peaks: the checkerboard diagonal and the wall grid. Under low-angle light the wall grid can dominate, and a template correlation alone can prefer the wrong reading.
- **Decision.** The expected pitch from the part's datasheet (cell density and diameter in the metadata) chooses between the two readings. Frequency resolution comes from zero-padding and parabolic peak interpolation, because a one-degree orientation error already shifts edge cells by a third of a cell.
- **Consequences.** The classical pipeline needs correct part metadata. In the stylised simulator the datasheet relation is scaled by a documented constant (`classical.lattice.cells_per_sqrt_cpsi`); a real station would use `25.4 / sqrt(cpsi)` mm and a calibrated mm-per-pixel.

## ADR-012 Lighting mode is a station input

- **Context.** Several measurements invert under backlight: cracks and through-channels become bright, and deposits cannot be judged.
- **Decision.** The station records its lighting mode per image, and the classical pipeline selects its rules (radius landmark, ridge polarity, missing-plug test) accordingly. The lighting mode is also given to the boosting models as a categorical feature.
- **Consequences.** A mislabelled lighting mode degrades measurements; the input validator checks it is a known value.

## ADR-013 Feature caching

- **Context.** Classical feature extraction dominates training time on a laptop.
- **Decision.** Classical features are cached under `data/<profile>/cache/`, keyed by the manifest hash, the `classical` config section and the SHA-256 of the extractor source files. The cache folder is excluded from the data manifest.
- **Consequences.** Any change to data, config or extractor code forces re-extraction; a stale cache cannot be used silently.

## ADR-014 Batch and crop sizes under memory pressure

- **Context.** The development laptop shares 8 GB of memory with other applications and swaps heavily. Per-image MPS throughput fell several-fold at large batches.
- **Decision.** Batch size 8 for perception, fusion and inference; U-Nets train on random crops (they are fully convolutional) and run on full images; the demo profile uses smaller crops and images for the fusion network. The PatchCore coreset loop runs on the CPU to avoid a device synchronisation per iteration.
- **Consequences.** Training is slower per epoch than the clean benchmark suggested but no longer collapses; stage timings are recorded in every run.

## ADR-015 Cross-fitted vision models

- **Context.** Concepts measured on training units by models trained on those same units are unrealistically clean, so an outcome model fitted on them would over-trust them.
- **Decision.** Training units are split into folds, each with its own U-Net and anomaly bank; training units are described by the other folds' models. Calibration, test and deployment use the average of all folds.
- **Consequences.** About the same total training cost as one model; deployment runs two models per image.

## ADR-016 Routing alpha and safety constraints in the cost search

- **Context.** With three classes and imperfect accuracy, 90 % conformal sets contain SCRAP for most units, so REFER-on-ambiguity at a fixed alpha referred almost everything. The acceptance criteria also require a minimum scrap capture.
- **Decision.** Coverage is reported at the required levels (alpha 0.05 and 0.10), but the alpha used by the REFER rule is chosen with the thresholds by minimising expected cost. Constraints: the wrongful-scrap cap and a minimum scrap capture (both in `configs/costs.yaml`). If both cannot be met, capture is maximised within the cap.
- **Consequences.** The operating point is explicit about its price in referrals; the threshold search table is saved with every run.

## ADR-017 Image-quality gate

- **Context.** Opaque obstructions were read as deposits and drove serviceable units to QUARANTINE; the anomaly guard did not catch them because the U-Net named them.
- **Decision.** Per lighting mode, the acceptable range of face brightness, sharpness, out-of-range intensity area and seating is learned from training images only (percentiles plus a margin). An image outside it fails input validation and the unit is referred.
- **Consequences.** A small share of clean images is referred (false alarms); nothing in the gate was fitted to the corruptions used in the robustness study.

## ADR-018 The guard acts on unexplained anomaly

- **Context.** A memory bank of defect-free patches flags every known defect as anomalous; referring all of them would defeat the gate.
- **Decision.** The REFER rule uses the anomaly remaining outside (dilated) regions where the perception model predicts a known class. The full anomaly score remains a model feature.
- **Consequences.** Novel defects that the perception model partly names can be under-referred; the novel-set results measure this.

## ADR-019 Calibration data and method selection

- **Context.** One calibration split must serve probability calibration, threshold selection and conformal prediction.
- **Decision.** The calibration split is halved by alternating unit IDs. The first half fits calibration maps (method for boosting chosen by cross-validated log-loss among sigmoid, isotonic and temperature), the anomaly threshold and the routing policy; the second half fits conformal quantiles. Test ECE of every candidate method is reported but not used to choose.
- **Consequences.** Conformal guarantees hold on the second half; thresholds may be slightly optimistic because the first half is reused.

## ADR-020 Deploy the concept route

- **Context.** The end-to-end network and the concept route are close in macro-F1 on synthetic data.
- **Decision.** Deploy the concept route; keep the end-to-end network as a challenger evaluated in every run.
- **Consequences.** Rationale and explanation tests are possible, commissioning a new line is a perception-only task, and the decision layer sees physical quantities.

## ADR-021 Attribution methods

- **Context.** SHAP's tree explainer does not handle the categorical splits of scikit-learn histogram boosting; Integrated Gradients through the image encoder exhausted memory.
- **Decision.** Permutation SHAP with a small background sample for the boosting models; IG on flow and metadata with the image embedding computed once and held fixed (`FusionNet.fuse`).
- **Consequences.** SHAP values are approximate (sampling); IG attributions are exact for the fixed embedding.

## ADR-022 Audit trail and line integration

- **Context.** Every recommendation and decision must be traceable and tamper-evident without new infrastructure.
- **Decision.** SQLite with UPDATE and DELETE triggers that abort, and a SHA-256 hash chain over every record (content plus previous hash). Inspections, operator decisions and downstream results are separate record types. The MES is represented by an adapter interface with a SQLite route-queue mock.
- **Consequences.** Tampering through other tools is detected by `preinspect audit verify`; a production system would also replicate the log off the station.

## ADR-023 Drift monitoring

- **Context.** Drift in images is hard to see in metadata alone.
- **Decision.** PSI on numeric metadata and on eight principal components of the U-Net bottleneck, fitted on training units; p-chart of the REFER and QUARANTINE share per batch; calibration over time once downstream results are logged.
- **Consequences.** The image embedding reference ships with each run; conventional PSI bands (0.10, 0.25) are used as triggers.

## ADR-024 Line simulation scope

- **Context.** The process-impact estimate needs queueing effects, not just a cost per unit.
- **Decision.** SimPy model with oven slots running continuously, a test bench, an inspector and a pre-inspection station; routes drawn from the deployed policy's test-split route-given-outcome rates; common random numbers between baseline and gated scenarios.
- **Consequences.** No shifts, breakdowns or set-ups; all rates and costs are labelled assumptions in `configs/line_sim.yaml` and `configs/costs.yaml`.
