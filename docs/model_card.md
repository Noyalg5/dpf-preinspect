# Model card: DPF pre-inspection concept route

*Rendered from run `20261023-062255_demo` (profile `demo`) on 2026-10-23. Format after Mitchell et al. (2019).*

> **Trained and evaluated on synthetic data only.** Not validated on real filters; not for production use.

## Model details

- **Deployed model:** `learned_flow_meta`, gradient boosting (scikit-learn histogram boosting) over learned visual concepts, anomaly scores, flow features and metadata, with temperature calibration.
- **Perception:** 2 cross-fitted U-Nets (ResNet-18 encoder, ImageNet initialisation) for 7 defect classes.
- **Anomaly guard:** PatchCore-style memory bank of 5000 patch features per fold from defect-free training faces.
- **Decision layer:** split-conformal outcome sets, cost-based thresholds, template rationale.
- **Version:** run `20261023-062255_demo`, code `60efad0ef1`, data manifest `93ee938bc98e`, generator v1.0.
- **Licence:** MIT. Owner: Noyal George.

## Intended use

- Decision support at a pre-inspection station of a DPF cleaning line. The model recommends PROCEED, EXTENDED_CLEAN, QUARANTINE or REFER, with a rationale, before cleaning.
- A person confirms every QUARANTINE; the system never scraps a unit.
- **Out of scope:** real filters before validation on real data; new-build substrate inspection without retraining (see `docs/system_design.md`); any use as the sole basis for an emissions-compliance or safety decision.

## Factors

Lighting mode, camera profile, substrate material (cordierite, SiC), cell density, diameter, application and image quality. The robustness study and domain-shift study in `reports/validation_report.md` measure the effect of these.

## Metrics

Macro-F1, balanced accuracy, SCRAP recall and AUC, ΔP-ratio MAE, expected calibration error, conformal coverage. At the operating point: wrongful-scrap rate, scrap capture rate and REFER share.

## Evaluation data

The chronological test split (254 units), the novel-defect set (thermal melt, unseen in training) and the domain-shift set (new cell density and camera). Headline scores use the simulator's true outcome.

## Training data

The chronological training split after the validation gate. Labels are recorded downstream dispositions, about 6.2 % of which differ from the true outcome because of test noise. See `docs/data_card.md`.

## Quantitative analysis

| Metric (test) | Value |
|---|---|
| Macro-F1 | 0.649 |
| Balanced accuracy | 0.638 |
| Recall PASS / REWORK / SCRAP | 0.836 / 0.570 / 0.508 |
| AUC PASS / REWORK / SCRAP | 0.817 / 0.837 / 0.839 |
| ΔP-ratio MAE | 0.066 |
| ECE (calibrated) | 0.054 |
| Conformal coverage at 90 % (recorded labels) | 91.3 % |
| Wrongful-scrap rate at operating point | 1.5 % |
| Scrap capture at operating point | 59.3 % |
| REFER share at operating point | 18.9 % |

## Ethical and safety considerations

- **Automation bias.** Operators may accept recommendations without checking them. The UI shows the rationale and the plausible outcomes; overrides need a reason; override rates are monitored.
- **Wrongful scrap** destroys a serviceable core. It is capped in the threshold search, and QUARANTINE always needs confirmation by a person.
- **Missed scrap** is caught downstream by today's tests in any case; the gate saves cost but does not replace those tests.
- **Workforce.** The system is designed to support inspectors, not to replace the downstream tests or the inspector's judgement.

## Caveats and recommendations

- Retrain and revalidate on real data before any pilot. Commission each new product line using the procedure in `docs/training_manual.md`.
- Monitor input drift (PSI), REFER and QUARANTINE rates (p-chart) and calibration against downstream results; see the Quality Engineer dashboard.
