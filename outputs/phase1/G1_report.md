# Gate G1 report: end of Phase 1 (data acquisition)

Generated 2026-10-20T08:54:38.971196+00:00 by `scenario-fraud phase1 report`. Regenerate after any change to `data/raw/manifest.json`, column maps or labels. This report states facts; the go/no-go decision belongs to the human (HANDOFF §6, G1).

## 1. Files acquired

| source | year | kind | file | MB | rows | cols | route | as_of | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| cms_prov | 2018 | data | MUP_PHY_R25_P07_V20_D18_Prov.csv.gz | 142.3 | 1121462 | 81 | other |  | 04a10009a91c |
| cms_prov | 2019 | data | MUP_PHY_R25_P07_V20_D19_Prov.csv.gz | 146.3 | 1155870 | 81 | other |  | bd789342af70 |
| cms_prov | 2020 | data | MUP_PHY_R25_P07_V20_D20_Prov.csv.gz | 145.7 | 1161542 | 81 | other |  | 6de6d53eaa8f |
| cms_prov | 2021 | data | MUP_PHY_R25_P07_V20_D21_Prov.csv.gz | 150.6 | 1198754 | 81 | other |  | 6b0a62810383 |
| cms_prov | 2022 | data | MUP_PHY_R25_P07_V20_D22_Prov.csv.gz | 153.7 | 1230293 | 81 | other |  | 5c34a49258ea |
| cms_prov | 2023 | data | MUP_PHY_R25_P05_V20_D23_Prov.csv.gz | 156.2 | 1259343 | 81 | other |  | 33dac951769e |
| cms_prov | 2024 | data | MUP_PHY_R26_P05_V10_D24_Prov.csv.gz | 159.9 | 1296739 | 81 | other |  | e9e779645b5d |
| cms_prov | 2024 | dictionary | MUP_PHY_RY25_20250312_DD_Prvdr_508.pdf | 0.4 |  |  | other |  | f8c4d5353bd9 |
| cms_prov | 2024 | methodology | MUP_PHY_RY25_202350312_Methodology_508.pdf | 0.3 |  |  | other |  | daf2ddb6ab19 |
| cms_ps | 2018 | data | MUP_PHY_R25_P04_V20_D18_Prov_Svc.csv.gz | 491.8 | 9961865 | 28 | other |  | 57d38e70030d |
| cms_ps | 2019 | data | MUP_PHY_R25_P04_V20_D19_Prov_Svc.csv.gz | 496.9 | 10140228 | 28 | other |  | 8543d0fa5634 |
| cms_ps | 2020 | data | MUP_PHY_R25_P05_V20_D20_Prov_Svc.csv.gz | 476.5 | 9449361 | 28 | other |  | 306f2c37809c |
| cms_ps | 2021 | data | MUP_PHY_R25_P05_V20_D21_Prov_Svc.csv.gz | 494.5 | 9886177 | 28 | other |  | 4e3d65744b60 |
| cms_ps | 2022 | data | MUP_PHY_R25_P05_V20_D22_Prov_Svc.csv.gz | 492.1 | 9755427 | 28 | other |  | 915eef8d6f09 |
| cms_ps | 2023 | data | MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv.gz | 483.1 | 9660647 | 28 | other |  | c6766dda7c66 |
| cms_ps | 2024 | data | PHY_R26_P05_V10_D24_Prov_Svc.csv.gz | 504.4 | 9781673 | 28 | other |  | 509dc7ce4cd0 |
| cms_ps | 2024 | dictionary | MUP_PHY_RY25_20250312_DD_PRV_SVC_508.pdf | 0.1 |  |  | other |  | 9e0b8ab93dcd |
| cms_ps | 2024 | methodology | MUP_PHY_RY25_202350312_Methodology_508.pdf | 0.3 |  |  | other |  | daf2ddb6ab19 |
| dme_p | 2018 | data | mup_dme_ry25_p05_v20_dy18_rfrr.csv.gz | 53.9 | 386039 | 97 | other |  | 6a66a3566d00 |
| dme_p | 2019 | data | mup_dme_ry25_p05_v20_dy19_rfrr.csv.gz | 53.9 | 389310 | 97 | other |  | a230559d11d6 |
| dme_p | 2020 | data | mup_dme_ry25_p05_v20_dy20_rfrr.csv.gz | 51.8 | 383488 | 97 | other |  | 056d3a22669e |
| dme_p | 2021 | data | mup_dme_ry25_p05_v20_dy21_rfrr.csv.gz | 50.9 | 384264 | 97 | other |  | ce2590bc47bd |
| dme_p | 2022 | data | mup_dme_ry25_p05_v20_dy22_rfrr.csv.gz | 50.1 | 385637 | 97 | other |  | 5b4a0ffbc3a3 |
| dme_p | 2023 | data | mup_dme_ry25_p05_v10_dy23_rfrr.csv.gz | 49.9 | 387352 | 97 | other |  | a5eab60cbb81 |
| dme_p | 2024 | data | mup_dme_ry26_p05_v10_dy24_rfrr.csv.gz | 48.8 | 381228 | 97 | other |  | c08099f34749 |
| dme_p | 2024 | dictionary | MUP_DME_RY24_Data-Dictionary_20240620_rfrr-DY17-22.pdf | 0.2 |  |  | other |  | a82304820fdb |
| dme_p | 2024 | methodology | MUP_DME_RY25_Methodology_20250617_508.pdf | 0.3 |  |  | other |  | d39751c80dd5 |
| dme_s | 2018 | data | mup_dme_ry25_p05_v20_dy18_rfrhpr.csv.gz | 91.3 | 1656180 | 33 | other |  | 19011eada857 |
| dme_s | 2019 | data | mup_dme_ry25_p05_v20_dy19_rfrhpr.csv.gz | 93.3 | 1656449 | 33 | other |  | 4917ec15cbe0 |
| dme_s | 2020 | data | mup_dme_ry25_p05_v20_dy20_rfrhpr.csv.gz | 92.3 | 1607435 | 33 | other |  | b1eb814537a5 |
| dme_s | 2021 | data | mup_dme_ry25_p05_v20_dy21_rfrhpr.csv.gz | 87.6 | 1516153 | 33 | other |  | 630e48fc4948 |
| dme_s | 2022 | data | mup_dme_ry25_p05_v20_dy22_rfrhpr.csv.gz | 84.1 | 1457378 | 33 | other |  | 529733aaf78e |
| dme_s | 2023 | data | mup_dme_ry25_p05_v10_dy23_rfrhpr.csv.gz | 81.3 | 1439587 | 33 | other |  | 4a8205394ea2 |
| dme_s | 2024 | data | mup_dme_ry26_p05_v10_dy24_rfrhpr.csv.gz | 77.2 | 1347255 | 33 | other |  | 09a335020ced |
| dme_s | 2024 | dictionary | MUP_DME_RY25_Data-Dictionary_20250620_rfrhpr_508.pdf | 0.1 |  |  | other |  | b663097a0d78 |
| dme_s | 2024 | methodology | MUP_DME_RY25_Methodology_20250617_508.pdf | 0.3 |  |  | other |  | d39751c80dd5 |
| leie | 2025 | supplement_excl | 2501excl.csv | 0.0 | 92 | 18 | oig | 2025-01 | 3a5a0afa932d |
| leie | 2025 | supplement_excl | 2502excl.csv | 0.0 | 215 | 18 | oig | 2025-02 | c205b63665d7 |
| leie | 2025 | supplement_excl | 2503excl.csv | 0.1 | 274 | 18 | oig | 2025-03 | 15274a2f6c9d |
| leie | 2025 | supplement_excl | 2504excl.csv | 0.1 | 273 | 18 | oig | 2025-04 | 20ef8d7140d0 |
| leie | 2025 | supplement_excl | 2505excl.csv | 0.1 | 278 | 18 | oig | 2025-05 | 1fa13969b415 |
| leie | 2025 | supplement_excl | 2506excl.csv | 0.0 | 244 | 18 | oig | 2025-06 | 6521897589ed |
| leie | 2025 | supplement_excl | 2507excl.csv | 0.0 | 242 | 18 | oig | 2025-07 | e970823bb2f1 |
| leie | 2025 | supplement_excl | 2508excl.csv | 0.0 | 108 | 18 | oig | 2025-08 | 33ae2e11202f |
| leie | 2025 | supplement_excl | 2509excl.csv | 0.0 | 195 | 18 | oig | 2025-09 | 2cc692c19d12 |
| leie | 2025 | supplement_excl | 2510excl.csv | 0.0 | 249 | 18 | oig | 2025-10 | aca30844138f |
| leie | 2025 | supplement_excl | 2511excl.csv | 0.0 | 140 | 18 | oig | 2025-11 | 08a36b4bf011 |
| leie | 2025 | supplement_excl | 2512excl.csv | 0.1 | 357 | 18 | oig | 2025-12 | d7f5163145d7 |
| leie | 2025 | supplement_rein | 2501rein.csv | 0.0 | 62 | 18 | oig | 2025-01 | 75ca45a4d723 |
| leie | 2025 | supplement_rein | 2502rein.csv | 0.0 | 30 | 18 | oig | 2025-02 | 7145cdc6130c |
| leie | 2025 | supplement_rein | 2503rein.csv | 0.0 | 63 | 18 | oig | 2025-03 | 1e33db45a054 |
| leie | 2025 | supplement_rein | 2504rein.csv | 0.0 | 57 | 18 | oig | 2025-04 | 3c9ac0f1ff7c |
| leie | 2025 | supplement_rein | 2505rein.csv | 0.0 | 60 | 18 | oig | 2025-05 | a6a4d0561960 |
| leie | 2025 | supplement_rein | 2506rein.csv | 0.0 | 45 | 18 | oig | 2025-06 | feb107dd71d3 |
| leie | 2025 | supplement_rein | 2507rein.csv | 0.0 | 54 | 18 | oig | 2025-07 | 5df6742a19b4 |
| leie | 2025 | supplement_rein | 2508rein.csv | 0.0 | 31 | 18 | oig | 2025-08 | 85c825ed1ae3 |
| leie | 2025 | supplement_rein | 2509rein.csv | 0.0 | 84 | 18 | oig | 2025-09 | 18a82d5088a8 |
| leie | 2025 | supplement_rein | 2510rein.csv | 0.0 | 59 | 18 | oig | 2025-10 | b0db633af6e8 |
| leie | 2025 | supplement_rein | 2511rein.csv | 0.0 | 42 | 18 | oig | 2025-11 | 8f39a554c51b |
| leie | 2025 | supplement_rein | 2512rein.csv | 0.0 | 35 | 18 | oig | 2025-12 | a8eb5af2472e |
| leie | 2026 | current | UPDATED.csv | 10.8 | 84093 | 18 | oig | 2026-10-19 | 7483d47bc487 |
| leie | 2026 | record_layout | leie_record_layout.pdf | 0.1 |  |  | oig |  | 7200f5c83c6a |
| leie | 2026 | supplement_excl | 2601excl.csv | 0.0 | 73 | 18 | oig | 2026-01 | 0b38db3e4390 |
| leie | 2026 | supplement_excl | 2602excl.csv | 0.0 | 95 | 18 | oig | 2026-02 | 70b8ab4af94b |
| leie | 2026 | supplement_excl | 2603excl.csv | 0.1 | 298 | 18 | oig | 2026-03 | d616e3aa8b36 |
| leie | 2026 | supplement_excl | 2604excl.csv | 0.1 | 322 | 18 | oig | 2026-04 | 49b5ca8acbde |
| leie | 2026 | supplement_excl | 2605excl.csv | 0.0 | 254 | 18 | oig | 2026-05 | c8f2fb4f21d8 |
| leie | 2026 | supplement_excl | 2606excl.csv | 0.0 | 245 | 18 | oig | 2026-06 | a8e14a0e0aa4 |
| leie | 2026 | supplement_excl | 2607excl.csv | 0.0 | 253 | 18 | oig | 2026-07 | 63b34460ca68 |
| leie | 2026 | supplement_excl | 2608excl.csv | 0.0 | 200 | 18 | oig | 2026-08 | cf5f901faa3a |
| leie | 2026 | supplement_excl | 2609excl.csv | 0.0 | 145 | 18 | oig | 2026-09 | c5f303ea3738 |
| leie | 2026 | supplement_rein | 2601rein.csv | 0.0 | 68 | 18 | oig | 2026-01 | 1cb77580556d |
| leie | 2026 | supplement_rein | 2602rein.csv | 0.0 | 59 | 18 | oig | 2026-02 | 009b9f6b589b |
| leie | 2026 | supplement_rein | 2603rein.csv | 0.0 | 46 | 18 | oig | 2026-03 | 0bbb38c2397e |
| leie | 2026 | supplement_rein | 2604rein.csv | 0.0 | 67 | 18 | oig | 2026-04 | f9c1d41788c5 |
| leie | 2026 | supplement_rein | 2605rein.csv | 0.0 | 41 | 18 | oig | 2026-05 | 026084d192a2 |
| leie | 2026 | supplement_rein | 2606rein.csv | 0.0 | 41 | 18 | oig | 2026-06 | ee3cff4d2094 |
| leie | 2026 | supplement_rein | 2607rein.csv | 0.0 | 76 | 18 | oig | 2026-07 | d1217770d31e |
| leie | 2026 | supplement_rein | 2608rein.csv | 0.0 | 41 | 18 | oig | 2026-08 | 0da85394c6a8 |
| leie | 2026 | supplement_rein | 2609rein.csv | 0.0 | 53 | 18 | oig | 2026-09 | 8a789b37e362 |

### Expected but not registered

| source | kind | missing_years |
|---|---|---|
| leie | snapshot | - |

### DMEPOS 2024

- `dme_s` 2024: registered, 1347255 rows, 77.2 MB
- `dme_p` 2024: registered, 381228 rows, 48.8 MB

## 2. Column maps (C2) and harmonised files

| dataset | year | map | harmonised_rows | bene_cc_cols |
|---|---|---|---|---|
| cms_ps | 2018 | reviewed | 9961865 |  |
| cms_ps | 2019 | reviewed | 10140228 |  |
| cms_ps | 2020 | reviewed | 9449361 |  |
| cms_ps | 2021 | reviewed | 9886177 |  |
| cms_ps | 2022 | reviewed | 9755427 |  |
| cms_ps | 2023 | reviewed | 9660647 |  |
| cms_ps | 2024 | reviewed | 9781673 |  |
| cms_prov | 2018 | reviewed | 1121462 | 25 |
| cms_prov | 2019 | reviewed | 1155870 | 25 |
| cms_prov | 2020 | reviewed | 1161542 | 25 |
| cms_prov | 2021 | reviewed | 1198754 | 25 |
| cms_prov | 2022 | reviewed | 1230293 | 25 |
| cms_prov | 2023 | reviewed | 1259343 | 25 |
| cms_prov | 2024 | reviewed | 1296739 | 25 |
| dme_s | 2018 | reviewed | 1656180 |  |
| dme_s | 2019 | reviewed | 1656449 |  |
| dme_s | 2020 | reviewed | 1607435 |  |
| dme_s | 2021 | reviewed | 1516153 |  |
| dme_s | 2022 | reviewed | 1457378 |  |
| dme_s | 2023 | reviewed | 1439587 |  |
| dme_s | 2024 | reviewed | 1347255 |  |
| dme_p | 2018 | reviewed | 386039 |  |
| dme_p | 2019 | reviewed | 389310 |  |
| dme_p | 2020 | reviewed | 383488 |  |
| dme_p | 2021 | reviewed | 384264 |  |
| dme_p | 2022 | reviewed | 385637 |  |
| dme_p | 2023 | reviewed | 387352 |  |
| dme_p | 2024 | reviewed | 381228 |  |

## 3. LEIE sources and EXCLTYPE coverage (C3)

- LEIE layout verified against the OIG record-layout PDF: **True**
- EXCLTYPE map verified: **True**
- Snapshot date (label horizon): 2026-10-19
- Exclusion events after union: 85156; not in the current file (recovered): 1124

| excltype_raw | authority | events | with_npi | tiers |
|---|---|---|---|---|
| 1128b4 | 1128(b)(4) | 33930 | 2761 | prior_work |
| 1128a1 | 1128(a)(1) | 26361 | 3503 | primary,prior_work,primary_b1,broad |
| 1128a2 | 1128(a)(2) | 8205 | 409 | prior_work |
| 1128a3 | 1128(a)(3) | 5995 | 709 | primary,prior_work,primary_b1,broad |
| 1128a4 | 1128(a)(4) | 3651 | 1043 | broad |
| 1128b14 | 1128(b)(14) | 2173 | 198 | - |
| 1128b8 | 1128(b)(8) | 1484 | 96 | broad |
| 1128b1 | 1128(b)(1) | 957 | 50 | primary_b1 |
| 1128b5 | 1128(b)(5) | 817 | 91 | broad |
| 1128b7 | 1128(b)(7) | 769 | 205 | primary,prior_work,primary_b1,broad |
| 1128b3 | 1128(b)(3) | 337 | 22 | broad |
| 1128Aa | 1128A(a) | 174 | 32 | - |
| 1128b6 | 1128(b)(6) | 77 | 9 | primary,primary_b1,broad |
| 1128b2 | 1128(b)(2) | 67 | 9 | broad |
| 1156 | 1156 | 53 | 1 | broad |
| 1128b15 | 1128(b)(15) | 39 | 2 | broad |
| BRCH SA | breach-SA | 29 | 21 | - |
| BRCH CIA | breach-CIA | 14 | 9 | - |
| 1128b11 | 1128(b)(11) | 13 | 2 | - |
| 1160 | 1160 | 8 | 0 | - |
| 1128b16 | 1128(b)(16) | 2 | 1 | primary,primary_b1,broad |
| 1128b12 | 1128(b)(12) | 1 | 0 | - |

## 4. Positive NPI counts per year × label tier

Counts over all by-Provider NPI-years, after the ≥ 50 non-drug services rule where `cms_ps` is harmonised. W = 3.

| tier | year | pos | neg | dropped | pos_after_volume_rule |
|---|---|---|---|---|---|
| primary | 2018 | 118 | 1120184 | 1160 | 105 |
| primary | 2019 | 130 | 1154825 | 915 | 114 |
| primary | 2020 | 127 | 1160721 | 694 | 114 |
| primary | 2021 | 151 | 1198096 | 507 | 130 |
| primary | 2022 | 124 | 1229833 | 336 | 107 |
| primary | 2023 | 99 | 1259046 | 198 | 87 |
| primary | 2024 | 51 | 1296566 | 122 | 44 |
| prior_work | 2018 | 173 | 1120184 | 1105 | 155 |
| prior_work | 2019 | 218 | 1154825 | 827 | 190 |
| prior_work | 2020 | 222 | 1160721 | 599 | 193 |
| prior_work | 2021 | 280 | 1198096 | 378 | 223 |
| prior_work | 2022 | 261 | 1229833 | 199 | 209 |
| prior_work | 2023 | 256 | 1259046 | 41 | 216 |
| prior_work | 2024 | 147 | 1296566 | 26 | 122 |
| primary_b1 | 2018 | 120 | 1120184 | 1158 | 107 |
| primary_b1 | 2019 | 133 | 1154825 | 912 | 117 |
| primary_b1 | 2020 | 130 | 1160721 | 691 | 116 |
| primary_b1 | 2021 | 157 | 1198096 | 501 | 136 |
| primary_b1 | 2022 | 127 | 1229833 | 333 | 110 |
| primary_b1 | 2023 | 102 | 1259046 | 195 | 90 |
| primary_b1 | 2024 | 51 | 1296566 | 122 | 44 |
| broad | 2018 | 178 | 1120184 | 1100 | 154 |
| broad | 2019 | 181 | 1154825 | 864 | 158 |
| broad | 2020 | 168 | 1160721 | 653 | 152 |
| broad | 2021 | 194 | 1198096 | 464 | 167 |
| broad | 2022 | 157 | 1229833 | 303 | 137 |
| broad | 2023 | 125 | 1259046 | 172 | 110 |
| broad | 2024 | 59 | 1296566 | 114 | 50 |

**Unique positive NPIs over 2018–2023** (DESIGN §14 threshold ≈ 300):

| tier | W | unique_positive_npis | after_volume_rule | threshold |
|---|---|---|---|---|
| primary | 3 | 427 | 383 | 300 |
| prior_work | 3 | 839 | 726 | 300 |
| primary_b1 | 3 | 435 | 391 | 300 |
| broad | 3 | 574 | 513 | 300 |

W sensitivity (primary tier):

| tier | W | unique_positive_npis | positive_rows |
|---|---|---|---|
| primary | 1 | 147 | 147 |
| primary | 2 | 290 | 402 |
| primary | 3 | 427 | 749 |
| primary | 5 | 564 | 1443 |

## 5. NPI-match losses

Exclusion events with 2018 < exclusion year ≤ 2026, by tier. `npis_in_part_b` needs harmonised `cms_prov` years.

| tier | events | no_npi | invalid_npi | unique_npis | npis_in_part_b | npis_not_in_part_b |
|---|---|---|---|---|---|---|
| primary | 9977 | 7440 | 0 | 2524 | 622 | 1902 |
| prior_work | 17134 | 13037 | 0 | 4065 | 1162 | 2903 |
| primary_b1 | 10198 | 7629 | 0 | 2556 | 633 | 1923 |
| broad | 11347 | 8177 | 0 | 3149 | 851 | 2298 |

## 6. Reinstatement recovery: Internet Archive gaps and loss profile

- Full-file snapshots used: 2026-10-19
- Reinstatement supplement months: 2025-01, 2025-02, 2025-03, 2025-04, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11, 2025-12, 2026-01, 2026-02, 2026-03, 2026-04, 2026-05, 2026-06, 2026-07, 2026-08, 2026-09
- Uncovered days in window 2019-01-01 → 2026-10-19: 2848

Uncovered intervals (an exclusion that starts and is reinstated inside one is invisible):

| from | to | days | uncovered_months | bounded_by_snapshot |
|---|---|---|---|---|
| 2019-01-01 | 2026-10-19 | 2848 | 73 | True |

Reinstated records by authority (from supplements and snapshots):

| authority | n_reinstated | median_days | p10_days | share_shorter_than_longest_gap | mandatory | in_primary | in_prior_work | in_primary_b1 | in_broad |
|---|---|---|---|---|---|---|---|---|---|
| 1128(b)(4) | 473 | 2816.0 | 134.0 | 0.5031712473572939 | False | False | True | False | False |
| 1128(a)(1) | 331 | 2903.0 | 1833.0 | 0.486404833836858 | True | True | True | True | True |
| 1128(a)(3) | 112 | 2528.5 | 1830.0 | 0.5357142857142857 | True | True | True | True | True |
| 1128(a)(2) | 85 | 3138.0 | 1828.0 | 0.43529411764705883 | True | False | True | False | False |
| 1128(a)(4) | 59 | 2497.0 | 1830.0 | 0.576271186440678 | True | False | False | False | True |
| 1128(b)(1) | 14 | 3192.5 | 740.0 | 0.42857142857142855 | False | False | False | True | False |
| 1128(b)(7) | 14 | 1718.0 | 1099.0 | 0.7142857142857143 | False | True | True | True | True |
| 1128(b)(3) | 7 | 1153.0 | 366.0 | 1.0 | False | False | False | False | True |
| 1128(b)(14) | 7 | 6430.0 | 4957.0 | 0.14285714285714285 | False | False | False | False | False |
| 1128(b)(5) | 6 | 3323.5 | 2316.0 | 0.5 | False | False | False | False | True |
| 1128(b)(15) | 2 | 6067.0 | 4823.0 | 0.0 | False | False | False | False | True |
| 1128(b)(2) | 1 | 1127.0 | 1127.0 | 1.0 | False | False | False | False | True |
| 1128A(a) | 1 | 3653.0 | 3653.0 | 0.0 | False | False | False | False | False |
| breach-SA | 1 | 186.0 | 186.0 | 1.0 | False | False | False | False | False |
| breach-CIA | 1 | 1931.0 | 1931.0 | 1.0 | False | False | False | False | False |

## 7. Label horizon

| year | years_observed | W | complete |
|---|---|---|---|
| 2018 | 3 | 3 | True |
| 2019 | 3 | 3 | True |
| 2020 | 3 | 3 | True |
| 2021 | 3 | 3 | True |
| 2022 | 3 | 3 | True |
| 2023 | 2.8 | 3 | False |
| 2024 | 1.8 | 3 | False |
