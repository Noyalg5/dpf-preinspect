# Data-quality report: demo dataset

> Synthetic data generated locally by the virtual inspection cell (generator v1.0, seed 20261023). It demonstrates the validation method, not the quality of any real data source.

- Validated: 2026-10-23T06:22:50
- Manifest SHA-256: `93ee938bc98ead8368081a23574a83aeeae0b83c31f577cc70c9a7c6b243ad73`
- Records checked: 1,308

## 1. Integrity (blocks training if it fails)

**Result: PASS.** 5,206 files listed in the manifest were re-hashed (SHA-256) and compared; unlisted files and missing schema columns also fail this check.

No missing, altered or unlisted files.

## 2. Record and field checks (failing records are quarantined, not dropped)

| Check | Level | Rule | Records flagged |
|---|---|---|---|
| `MISSING_REQUIRED` | record | A required field is blank. | 26 |
| `OUT_OF_RANGE` | record | A numeric field is outside its plausible range. | 13 |
| `UNKNOWN_CATEGORY` | record | A categorical field holds an unrecognised value. | 0 |
| `DUPLICATE_ID` | record | The unit ID appears on more than one record. | 16 |
| `PROBABLE_FAHRENHEIT` | field | Temperature may be a Fahrenheit value; field blanked, record kept. | 39 |
| `FLOW_DROPOUTS` | record | Flow curve has more sensor dropouts than can be imputed. | 32 |
| `DISPOSITION_MISMATCH` | record | Recorded disposition contradicts the recorded test results. | 39 |
| `IMAGE_UNREADABLE` | record | An image file is missing, unreadable or blank. | 0 |

- Records quarantined (any record-level check): **123** of 1,308. A record failing several checks is counted once here and once per check above.
- Fields blanked for verification at source (record kept): **39**. A temperature above the Fahrenheit threshold could be a unit slip or a genuine extreme regeneration; the record alone cannot tell, so the value is not used.
- Records accepted: **1,185**.
- Non-blocking: 117 flow curves have one or two dropouts, which are tolerated (the curve fit uses the remaining points).

Quarantined records are listed with every reason in `validation/quarantine.csv` for review and correction at source. They are excluded from training and evaluation.

### Examples

| Unit | Check | Detail |
|---|---|---|
| DPF-000103 | `MISSING_REQUIRED` | material is blank |
| DPF-000157 | `MISSING_REQUIRED` | material is blank |
| DPF-000426 | `MISSING_REQUIRED` | material is blank |
| DPF-000579 | `MISSING_REQUIRED` | material is blank |
| DPF-001087 | `MISSING_REQUIRED` | material is blank |
| DPF-000020 | `MISSING_REQUIRED` | application is blank |
| DPF-000367 | `MISSING_REQUIRED` | application is blank |
| DPF-000441 | `MISSING_REQUIRED` | application is blank |
| DPF-000697 | `MISSING_REQUIRED` | application is blank |
| DPF-000754 | `MISSING_REQUIRED` | application is blank |
| DPF-000998 | `MISSING_REQUIRED` | application is blank |
| DPF-001073 | `MISSING_REQUIRED` | application is blank |
| DPF-001100 | `MISSING_REQUIRED` | application is blank |
| DPF-000278 | `MISSING_REQUIRED` | operating_hours is blank |
| DPF-000922 | `MISSING_REQUIRED` | operating_hours is blank |
| DPF-000995 | `MISSING_REQUIRED` | operating_hours is blank |
| DPF-001248 | `MISSING_REQUIRED` | operating_hours is blank |
| DPF-000176 | `MISSING_REQUIRED` | weight_gain_g is blank |
| DPF-000518 | `MISSING_REQUIRED` | weight_gain_g is blank |
| DPF-000581 | `MISSING_REQUIRED` | weight_gain_g is blank |
| DPF-000713 | `MISSING_REQUIRED` | weight_gain_g is blank |
| DPF-000835 | `MISSING_REQUIRED` | weight_gain_g is blank |
| DPF-000870 | `MISSING_REQUIRED` | weight_gain_g is blank |
| DPF-001086 | `MISSING_REQUIRED` | weight_gain_g is blank |
| DPF-001157 | `MISSING_REQUIRED` | weight_gain_g is blank |

## 3. Accepted records by split

| Split | Accepted units |
|---|---|
| calib | 153 |
| novel | 53 |
| shift | 126 |
| test | 254 |
| train | 599 |
