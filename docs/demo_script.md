# Demo script (10 minutes)

*Numbers below are rendered from run `20261023-062255_demo` (profile `demo`). Synthetic data throughout; say so at the start.*

**Before the session:** run `make demo`, then start `make api` and `make app` in two terminals and `make replay` once to fill the dashboard. Open `docs/data_card.md`, `README.md` and `reports/validation_report.md` in tabs.

| Time | Show | Say |
|---|---|---|
| 0:00 to 1:00 | README, "The problem" | Every unit is cleaned before tests decide its fate, so scrap uses oven time for nothing. The gate predicts the outcome *before* cleaning. All data here is synthetic. |
| 1:00 to 2:30 | Data card and gallery | 1,300 simulated units built from the domain primer: eight defect classes, nuisances, three lighting modes. Faults were injected on purpose and the validation gate caught them; 123 records were quarantined with reasons. |
| 2:30 to 4:00 | Headline table, ablation | Best macro-F1 0.649 (Learned concepts + flow + metadata). No single modality is enough: flow alone has a SCRAP AUC of 0.528, because cracks lower the pressure drop. I deploy the concept route because every decision is built from measurements an inspector can check. |
| 4:00 to 6:00 | Inspection Station page | Pick a test unit with a crack; toggle the defect outlines and anomaly heatmap; read the rationale aloud; override with a reason to show the audit trail. Pick a unit that is referred and explain why. |
| 6:00 to 7:00 | Quality Engineer page | Route mix, p-chart of REFER and QUARANTINE share, PSI drift tables. Re-run `make replay` with `--drift` to show drift appearing. |
| 7:00 to 8:30 | Validation report | 6 of 8 criteria pass. Point to the failures and what would fix them. Robustness curves; repeatability 80.9 %; %GRR 6.8 %. |
| 8:30 to 9:30 | Cost and simulation | Saving 9.94 GBP per unit under assumed costs, with 18.9 % of units referred. The tornado chart shows which assumption matters most. |
| 9:30 to 10:00 | Project plan | First steps in the KTP: agree acceptance criteria, collect real data, shadow-mode pilot. |

**Likely questions:** see `docs/faq.md`.
