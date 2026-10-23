# FAQ: the hard questions

*Answers are rendered from run `20261023-062255_demo` (profile `demo`). All data is synthetic.*

**Is any of this real data?**
No. Every unit, image, curve and label comes from the simulator in `src/preinspect/synth/`, designed from `docs/domain_primer.md` and frozen at v1.0 before any model was trained. The results show that the method works end to end and how it would be validated; they say nothing about real filters.

**Did you tune the simulator to make the models look good?**
No. Parameters were set for plausible class balance and defect prevalence before training (ADR-006), and later changes would bump the version and re-run everything. Some results are unfavourable and are reported as such (for example the acceptance criteria that fail).

**Which approach wins?**
Learned concepts + flow + metadata has the highest test macro-F1 (0.649). The deployed concept route differs from the end-to-end network by 0.122 macro-F1. At its cost-optimal operating point, the deployed model refers 18.9 % of units and saves 9.94 GBP per unit, more than any other approach.

**Why deploy the concept route rather than the end-to-end network?**
Its inputs are physical measurements (crack length, missing plugs, outlet soot, flow restriction against weight gain), so every decision has a rationale an inspector can check. Onboarding a new line only requires adapting the perception model. Its explanations can be tested against ground truth. The end-to-end network is kept as a challenger.

**Does fusing modalities help?**
In the concept route, all three modalities together score 0.129 macro-F1 above the best single group (metadata). The data was built so that no single modality is sufficient: vision sees damage, flow and weight see loading, and internal cracks are only visible indirectly.

**Does a pre-clean flow test earn its cycle time?**
Compare the ablation rows with and without F in `docs/technical_report.md` section 6.2. Flow on its own has a SCRAP AUC of 0.528, because leaks lower the pressure drop; its value is in combination with weight gain and images.

**How often is the system wrong in a way that matters?**
On the test split, 1.5 % of serviceable units are routed to QUARANTINE (a person still confirms), and 40.7 % of true scrap units go to cleaning unchecked (today's downstream tests still catch them).

**Are the probabilities trustworthy?**
After temperature calibration the expected calibration error is 0.054 (before: 0.142). Conformal sets at a nominal 90 % cover 91.3 % of recorded test outcomes.

**What happens with a defect the model has never seen?**
The anomaly guard flags it when it is not explained by a known defect class. Melted faces, absent from training, are separated from defect-free faces with image AUROC 0.976.

**What if the camera, lighting or line changes?**
On a new product line with an unseen cell density and camera, macro-F1 moves from 0.649 (main test) to 0.584, and 0.582 after commissioning on 50 units. Images outside the training envelope are referred to an inspector by the image-quality gate. Drift is monitored with PSI on metadata and on an image embedding.

**Is it repeatable?**
Re-imaging the same unit gives the same route 80.9 % of the time; the predicted ΔP ratio has %GRR 6.8 % and ndc 20. Route repeatability is limited by units near decision thresholds.

**Can an operator overrule it?**
Yes, always. An override needs a reason; both the recommendation and the override are written to a hash-chained audit trail, and overrides are counted on the dashboard.

**How fast is it?**
p95 per unit on CPU 0.65 s (cpu (ONNX Runtime U-Net)), well inside a station cycle of minutes.

**What would it save?**
Under the assumed costs in `configs/costs.yaml`, 9.94 GBP per unit on the test split. The line simulation shows oven hours falling from 10681 to 9209 over 20 days; the saving is most sensitive to `standard_clean_cost`. Real figures need real costs and a pilot.

**What would you do first in the KTP?**
Agree the decision rules and acceptance criteria with quality and operations, capture a labelled real dataset (`docs/real_data_plan.md`), and run the gate in shadow mode beside today's process.
