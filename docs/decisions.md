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
