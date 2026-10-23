# Project plan

The 24-month plan is added in Phase 8. This file currently holds the prototype delivery log.

## Prototype delivery log

| Date | Phase | Work done | Issues | Next |
|---|---|---|---|---|
| 2026-10-23 | 0 Scaffold | Package layout, CLI, configs (profiles, costs, acceptance, line simulation), run folders with provenance, CI, licence. | pyenv Python lacked `_lzma`; switched to uv-managed Python (ADR-001). | Domain primer, synthetic generator. |
| 2026-10-23 | 1 Synthetic data | Domain primer; virtual inspection cell (population, flow bench, outcome model, scene sampler, renderer with three lighting modes, camera profiles, faults, simulated inspectors); manifest with SHA-256 per file; validation gate with record- and field-level quarantine; data card and gallery; generator v1.0 frozen. Demo data: 1 300 units in about 1 to 2.5 min depending on memory pressure. | Ambiguous Fahrenheit readings were quarantining genuine extreme units (ADR-008); rounding made one record self-inconsistent (fixed by deciding on recorded values). | Classical image-processing baseline. |
