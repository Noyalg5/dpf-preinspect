# Acceptance test plan

Criteria and targets are rendered from `configs/acceptance.yaml` (status: **illustrative**). The targets are illustrative and must be agreed with quality, operations and IT/OT before a pilot. `make validate` evaluates every criterion and writes `reports/validation_report.md`, which records PASS or FAIL against these targets with links to the evidence.


| ID | Criterion | Requirement | Target | Method | Evidence |
|---|---|---|---|---|---|
| AC-01 | Wrongful-scrap rate | Share of serviceable units (true PASS or REWORK) routed to QUARANTINE on the test split. | `wrongful_scrap_rate` <= 0.020 | Route every test unit with the deployed policy; divide the serviceable units (true PASS or REWORK) routed to QUARANTINE by all serviceable units. | `reports/validation_report.md` |
| AC-02 | SCRAP capture | Share of true SCRAP units routed to QUARANTINE or REFER before cleaning, so they do not consume a clean. Read together with AC-08, since referring everything would satisfy it. | `scrap_capture_rate` >= 0.900 | Route every test unit; divide the true SCRAP units routed to QUARANTINE or REFER by all true SCRAP units. Read with AC-08. | `reports/validation_report.md` |
| AC-03 | p95 latency | End-to-end inference time per unit (two images, flow curve, metadata), CPU, 95th percentile. | `latency_p95_cpu_s` <= 2.000 | Time the full inspection path per unit (JPEG decode, inference, decision, rationale) on CPU, on the deployed runtime, after warm-up; report the 95th percentile. | `reports/validation_report.md` |
| AC-04 | Decision repeatability | Share of units whose route is identical across two re-imaging trials (MSA set). | `route_repeatability` >= 0.900 | Re-image each MSA unit twice with seating, exposure and noise jitter (and re-measure its flow curve); count the units whose route is the same on both trials. | `reports/validation_report.md` |
| AC-05 | Conformal coverage | Empirical coverage of 90 % outcome prediction sets on the test split, allowing finite-sample tolerance. | `coverage_alpha_0_10` >= 0.870 | Fit conformal sets on the conformal half of the calibration split; on the test split, count the units whose recorded outcome lies in its 90 % set. | `reports/validation_report.md` |
| AC-06 | Graceful degradation | At every corruption type and severity, confidently wrong automatic routes stay within a margin of the clean-input rate. | `max_wrong_auto_route_increase` <= 0.100 | Apply each corruption at five severities to a stratified test sample; compute the rate of unsafe automatic routes (wrongful quarantine or unchecked scrap); report the largest increase over clean inputs. | `reports/validation_report.md` |
| AC-07 | Audit-trail integrity | A tampered audit record is detected by hash-chain verification and an untampered chain verifies clean. | `audit_integrity_ok` == yes | Log decisions to a fresh audit trail and verify it; alter records directly in the database and verify again. Pass only if the clean chain verifies and the tampered one is detected. | `reports/validation_report.md` |
| AC-08 | Inspector workload | Share of units routed to REFER on the test split; the gate must automate most decisions to be worth having. | `refer_share` <= 0.350 | Route every test unit; divide the units routed to REFER by all units. | `reports/validation_report.md` |

## Test environment

- Data: the `demo` profile of the synthetic dataset (generator v1.0); the chronological test split, the MSA subset and corrupted copies of test images. A real pilot repeats the tests on real units under `docs/real_data_plan.md`.
- Software: the run folder under test (code commit, data manifest hash and configuration recorded in its provenance).
- Hardware: for AC-03, the station's edge PC; the prototype measures the development laptop's CPU.

## Procedure

1. `make data`, `make train` and `make evaluate` for the profile, or reuse an approved run.
2. `make validate`: runs the studies and evaluates the criteria.
3. Review `reports/validation_report.md` with quality. Any FAIL is either fixed and retested or accepted as a signed deviation with a mitigation and an owner.

## Sign-off

| Role | Name | Decision | Date |
|---|---|---|---|
| Quality engineer | | | |
| Operations manager | | | |
| IT/OT | | | |
| Company supervisor | | | |
