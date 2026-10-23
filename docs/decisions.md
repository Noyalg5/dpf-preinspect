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
