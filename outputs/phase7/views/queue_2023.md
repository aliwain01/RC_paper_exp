# Queue summary · 2023 · `xgb`

Inputs: 28 harmonised CMS files under `data/interim` and 80 registered raw files (`data/raw/manifest.json`).

20936 top-flagged providers in the explained set (no individual drill-down).

| scheme group | leading indicator (share) | mean contribution |
|---|---|---|
| Upcoding | 0.03716 | 0.1847 |
| Overutilization | 0.3226 | 0.8192 |
| Atypical service mix | 0.1635 | 0.5646 |
| Charge inflation | 0.09004 | 0.5268 |
| Patient mix | 0.3065 | 0.8248 |
| DME referral intensity | 0.06668 | 0.2448 |
| DME referral concentration | 0.01347 | 0.1107 |

Precision at audit capacity (G3, out of fold):

| capacity k | precision |
|---|---|
| 0.001 | 0.00382 |
| 0.005 | 0.001337 |
| 0.01 | 0.001146 |
| 0.02 | 0.0008121 |
