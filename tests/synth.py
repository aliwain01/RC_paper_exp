"""Synthetic harmonised CMS files and LEIE file for tests of Phases 2-7.

NOTHING HERE IS REAL DATA, AND NOTHING COMPUTED FROM IT IS A RESULT. Files land only under the throw-away
project root a test passes in (the `proj` fixture), never in the repository's data/ or outputs/.

    truth = make_synthetic(proj, n_npi=1200, years=range(2018, 2025), seed=0)

writes, per year, data/interim/{cms_ps,cms_prov,dme_s,dme_p}/{year}.parquet with exactly the harmonised
columns of configs/column_maps/_schema.yaml (all required, sensible optional ones, `year` Int32), each with
a `{year}.meta.json` marked `synthetic: true`; then an LEIE file in the tests/test_labels.py layout,
registered as source leie / year 2026 / kind current / as_of 2026-10-12, and runs labels.run_labels.

What it imitates (so code meets the situations real files produce):
- 12 specialties with distinct HCPCS profiles (one rare), entity types I and O, states in all 9 Census
  divisions plus territories, military and foreign codes, RUCA codes with decimals and nulls.
- Office E&M 99211-99215, new 99202-99205 (99201 only before 2021), ED 99281-99285 with the 2021 office
  and 2023 ED coding shifts; ~50 procedure codes; drug J-codes (`hcpcs_drug_ind` Y, fractional units);
  place of service F and O; 2020 volume dip.
- DMEPOS rows with RBCS level/id, brace L-codes, K0800-K0899 power mobility, E- and A-codes, rental Y/N;
  many providers refer no DME; a few DME-only referrers with no Part B rows.
- CMS suppression (DESIGN §5.2): service rows and DME service rows with <= 10 beneficiaries dropped,
  by-Provider and DME by-Provider rows with <= 10 beneficiaries omitted, sub-group counts < 11 null.
  Summary totals come from the unsuppressed rows, so they exceed the visible service rows.
- Planted signal: in the 3 years before a primary-tier exclusion, most excluded providers run one or two
  schemes (SCHEMES); a few never-excluded providers do too (unlabelled positives, PU setting).

RBCS ids are placeholders shaped like real ones; dollar amounts are rough magnitudes, not fee schedules.
Generation is vectorised over providers (loops only over specialties and code tables).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from scenario_fraud import acquire, labels
from scenario_fraud.harmonise import DTYPES, _target_spec, load_schema
from scenario_fraud.paths import Paths
from tests.test_labels import HDR

AS_OF = "2026-10-12"
SUPPRESS_MAX = 10                  # CMS: records from <= 10 beneficiaries are not published (DESIGN §5.2)
SCHEMES = ("upcoding", "overutilization", "atypical_mix", "charge_inflation", "dme_mill")
# Scheme groups (DESIGN §7.5) each planted scheme should move. dme_mill moves both DME groups.
SCHEME_GROUPS = {
    "upcoding": ("upcoding",),
    "overutilization": ("overutilization",),
    "atypical_mix": ("atypical_mix",),
    "charge_inflation": ("charge_inflation",),
    "dme_mill": ("dme_intensity", "dme_concentration"),
}
SCHEME_WEIGHT = np.array([0.30, 0.25, 0.15, 0.15, 0.15])
RAMP = np.array([0.0, 1.0, 0.85, 0.7])   # scheme intensity by years before exclusion (index 1-3)

# ------------------------------------------------------------------------------------------- code tables

# kind: (P(place = F) before the provider's own propensity, price factor when billed at F)
KINDS = {
    "em": (0.10, 0.72), "ed": (1.0, 1.0), "surg": (1.0, 1.0), "img": (0.40, 0.45), "proc": (0.10, 0.75),
    "pt": (0.02, 1.0), "amb": (0.0, 1.0), "lab": (0.05, 1.0), "drug": (0.05, 1.0), "rare": (0.15, 0.80),
}
# code: (description, kind, allowed $ per unit, national, 2018)
CODES: dict[str, tuple[str, str, float]] = {
    "99211": ("Established patient office visit, level 1", "em", 23.0),
    "99212": ("Established patient office visit, level 2", "em", 46.0),
    "99213": ("Established patient office visit, level 3", "em", 76.0),
    "99214": ("Established patient office visit, level 4", "em", 110.0),
    "99215": ("Established patient office visit, level 5", "em", 148.0),
    "99201": ("New patient office visit, level 1", "em", 46.0),
    "99202": ("New patient office visit, level 2", "em", 77.0),
    "99203": ("New patient office visit, level 3", "em", 109.0),
    "99204": ("New patient office visit, level 4", "em", 167.0),
    "99205": ("New patient office visit, level 5", "em", 211.0),
    "99281": ("Emergency department visit, level 1", "ed", 22.0),
    "99282": ("Emergency department visit, level 2", "ed", 43.0),
    "99283": ("Emergency department visit, level 3", "ed", 64.0),
    "99284": ("Emergency department visit, level 4", "ed", 121.0),
    "99285": ("Emergency department visit, level 5", "ed", 178.0),
    "93000": ("Electrocardiogram, complete", "proc", 17.0),
    "93010": ("Electrocardiogram, interpretation", "proc", 9.0),
    "93015": ("Cardiovascular stress test", "proc", 75.0),
    "93306": ("Echocardiography, complete", "img", 220.0),
    "93458": ("Left heart catheterisation", "surg", 1000.0),
    "20610": ("Arthrocentesis, major joint", "proc", 55.0),
    "20550": ("Injection, tendon sheath", "proc", 45.0),
    "27447": ("Total knee arthroplasty", "surg", 1350.0),
    "27130": ("Total hip arthroplasty", "surg", 1380.0),
    "29881": ("Knee arthroscopy with meniscectomy", "surg", 520.0),
    "64721": ("Carpal tunnel release", "surg", 580.0),
    "26055": ("Trigger finger release", "proc", 380.0),
    "25607": ("Open treatment distal radius fracture", "surg", 900.0),
    "73560": ("X-ray knee, 1-2 views", "img", 33.0),
    "73030": ("X-ray shoulder, 2 views", "img", 30.0),
    "71046": ("X-ray chest, 2 views", "img", 30.0),
    "71045": ("X-ray chest, 1 view", "img", 25.0),
    "74177": ("CT abdomen and pelvis with contrast", "img", 300.0),
    "70450": ("CT head without contrast", "img", 110.0),
    "77067": ("Screening mammography, bilateral", "img", 140.0),
    "72148": ("MRI lumbar spine without contrast", "img", 230.0),
    "76700": ("Ultrasound, abdomen complete", "img", 110.0),
    "97110": ("Therapeutic exercise, each 15 min", "pt", 30.0),
    "97140": ("Manual therapy, each 15 min", "pt", 28.0),
    "97112": ("Neuromuscular re-education, each 15 min", "pt", 32.0),
    "97530": ("Therapeutic activities, each 15 min", "pt", 35.0),
    "97161": ("Physical therapy evaluation, low complexity", "pt", 100.0),
    "97162": ("Physical therapy evaluation, moderate complexity", "pt", 100.0),
    "G0283": ("Electrical stimulation, unattended", "pt", 13.0),
    "11721": ("Debridement of nails, 6 or more", "proc", 45.0),
    "11720": ("Debridement of nails, 1-5", "proc", 35.0),
    "11055": ("Paring of benign lesion, single", "proc", 25.0),
    "11042": ("Debridement, subcutaneous tissue", "proc", 70.0),
    "G0127": ("Trimming of dystrophic nails", "proc", 15.0),
    "A0425": ("Ground mileage, per statute mile", "amb", 7.5),
    "A0427": ("Ambulance service, ALS emergency", "amb", 430.0),
    "A0429": ("Ambulance service, BLS emergency", "amb", 400.0),
    "A0428": ("Ambulance service, BLS non-emergency", "amb", 230.0),
    "36415": ("Routine venipuncture", "lab", 3.0),
    "81002": ("Urinalysis, non-automated", "lab", 3.5),
    "80053": ("Comprehensive metabolic panel", "lab", 11.0),
    "85025": ("Complete blood count", "lab", 8.0),
    "G0439": ("Annual wellness visit, subsequent", "proc", 115.0),
    "G0438": ("Annual wellness visit, initial", "proc", 170.0),
    "99497": ("Advance care planning, first 30 min", "proc", 85.0),
    "99490": ("Chronic care management, 20 min", "proc", 42.0),
    "96372": ("Therapeutic injection, SC/IM", "proc", 14.0),
    "96413": ("Chemotherapy administration, IV, first hour", "proc", 130.0),
    "96365": ("IV infusion, first hour", "proc", 70.0),
    "J1100": ("Injection, dexamethasone sodium phosphate, 1 mg", "drug", 0.10),
    "J3301": ("Injection, triamcinolone acetonide, 10 mg", "drug", 1.60),
    "J1030": ("Injection, methylprednisolone acetate, 40 mg", "drug", 3.00),
    "J1885": ("Injection, ketorolac tromethamine, 15 mg", "drug", 0.60),
    "J0897": ("Injection, denosumab, 1 mg", "drug", 18.0),
    "J9271": ("Injection, pembrolizumab, 1 mg", "drug", 50.0),
    "J9312": ("Injection, rituximab, 10 mg", "drug", 85.0),
    "J2505": ("Injection, pegfilgrastim, 6 mg", "drug", 3500.0),
    "J7325": ("Hyaluronan, intra-articular, 1 mg", "drug", 4.0),
    # rare everywhere: billed by very few providers of any specialty
    "G0483": ("Drug test, definitive, 22 or more classes", "rare", 245.0),
    "81408": ("Molecular pathology procedure, level 9", "rare", 2000.0),
    "81479": ("Unlisted molecular pathology procedure", "rare", 1500.0),
    "64999": ("Unlisted procedure, nervous system", "rare", 350.0),
    "95810": ("Polysomnography, attended", "rare", 560.0),
    "0275T": ("Percutaneous laminotomy, lumbar", "rare", 900.0),
    "G0277": ("Hyperbaric oxygen, per 30 min", "rare", 30.0),
    "99091": ("Collection and interpretation of physiologic data", "rare", 56.0),
}
CODE_LIST = list(CODES)
CODE_ID = {c: i for i, c in enumerate(CODE_LIST)}
CODE_KIND = np.array([CODES[c][1] for c in CODE_LIST])
CODE_PRICE = np.array([CODES[c][2] for c in CODE_LIST])
CODE_DRUG = CODE_KIND == "drug"
CODE_PF = np.array([KINDS[k][0] for k in CODE_KIND])
CODE_SITE = np.array([KINDS[k][1] for k in CODE_KIND])
EST = ("99211", "99212", "99213", "99214", "99215")
NEW = ("99201", "99202", "99203", "99204", "99205")
ED = ("99281", "99282", "99283", "99284", "99285")
NEW_SHARES = (0.01, 0.06, 0.40, 0.38, 0.15)
EM_2021 = {"99211": 0.98, "99212": 1.23, "99213": 1.20, "99214": 1.17, "99215": 1.22, "99202": 0.96, "99205": 1.02}
# Codes rare in every specialty: low billing probability and penetration, so mostly suppressed.
RARE = tuple(c for c in CODE_LIST if CODES[c][1] == "rare")
RARE_UPD = {c: (4.0 if c == "G0277" else 1.0) for c in RARE}
RARE_PROFILE = tuple((c, 0.015, 0.015, 1.3, RARE_UPD[c]) for c in RARE)

# DME: code: (description, RBCS_Lvl, RBCS_Id, rental Y/N/B, allowed $ per service, claims per bene,
#             services per claim, supplier rate, purchase price multiple for B codes, high-risk)
DME: dict[str, tuple] = {
    "L0650": ("LSO, sagittal-coronal control, prefabricated", "POS", "DO001N", "N", 1000.0, 1.05, 1.0, 0.30, 1, True),
    "L0648": ("LSO, sagittal control, prefabricated", "POS", "DO001N", "N", 950.0, 1.05, 1.0, 0.30, 1, True),
    "L1833": ("KO, adjustable knee joints, prefabricated", "POS", "DO002N", "N", 480.0, 1.10, 1.0, 0.30, 1, True),
    "L1851": ("KO, single upright, prefabricated", "POS", "DO002N", "N", 520.0, 1.10, 1.0, 0.30, 1, True),
    "L3960": ("SEWHO, abduction positioning, prefabricated", "POS", "DO003N", "N", 420.0, 1.05, 1.0, 0.30, 1, True),
    "L3916": ("WHFO, wrist extension control, prefabricated", "POS", "DO003N", "N", 120.0, 1.10, 1.0, 0.30, 1, True),
    "L1902": ("AFO, ankle gauntlet, prefabricated", "POS", "DO004N", "N", 55.0, 1.10, 1.0, 0.30, 1, True),
    "A5500": ("Diabetic shoe, off-the-shelf", "POS", "DS003N", "N", 80.0, 1.20, 2.0, 0.40, 1, False),
    "A4351": ("Intermittent urinary catheter", "POS", "DS002N", "N", 1.9, 4.0, 50.0, 0.40, 1, False),
    "K0823": ("Power wheelchair, group 2 standard", "DME", "DW002N", "B", 420.0, 6.0, 1.0, 0.50, 9, True),
    "K0856": ("Power wheelchair, group 3 standard", "DME", "DW002N", "B", 600.0, 6.0, 1.0, 0.50, 9, True),
    "K0800": ("Power operated vehicle, group 1", "DME", "DW002N", "B", 150.0, 6.0, 1.0, 0.50, 8, True),
    "K0001": ("Standard wheelchair", "DME", "DW001N", "Y", 60.0, 6.0, 1.0, 0.50, 1, False),
    "E0601": ("Continuous positive airway pressure device", "DME", "DR001N", "Y", 85.0, 8.0, 1.0, 0.40, 1, False),
    "E0260": ("Hospital bed, semi-electric", "DME", "DB001N", "Y", 140.0, 7.0, 1.0, 0.40, 1, False),
    "E1390": ("Oxygen concentrator", "DME", "DX001N", "Y", 175.0, 10.0, 1.0, 0.40, 1, False),
    "E0143": ("Walker, folding, wheeled", "DME", "DA001N", "N", 75.0, 1.0, 1.0, 0.50, 1, False),
    "E0114": ("Crutches, underarm", "DME", "DA001N", "N", 35.0, 1.0, 1.0, 0.50, 1, False),
    "A4253": ("Blood glucose test strips, per 50", "DME", "DS001N", "N", 9.0, 4.0, 2.5, 0.35, 1, False),
    "A4259": ("Lancets, per box of 100", "DME", "DS001N", "N", 4.0, 3.5, 1.5, 0.35, 1, False),
    "A7030": ("CPAP full face mask", "DME", "DR002N", "N", 95.0, 2.5, 1.0, 0.40, 1, False),
    "J7613": ("Albuterol inhalation solution, per 1 mg", "Drug", "DD001N", "N", 0.04, 4.0, 120.0, 0.40, 1, False),
}
DME_LIST = list(DME)
DME_ID = {c: i for i, c in enumerate(DME_LIST)}
DME_COL = {k: np.array([DME[c][i] for c in DME_LIST]) for i, k in enumerate(
    ("desc", "lvl", "rbcs", "rent", "price", "cpb", "spc", "srate", "buy", "high"))}
HIGH_RISK_DME = tuple(c for c in DME_LIST if DME[c][9])
MILL_CODES = np.array([DME_ID[c] for c in ("L0650", "L0648", "L1833", "L1851", "L3960", "K0823", "K0856")])
GP_DME = (("A4253", 0.6, 0.06), ("A4259", 0.5, 0.05), ("E0601", 0.3, 0.02), ("A7030", 0.3, 0.02),
          ("E1390", 0.35, 0.02), ("E0143", 0.4, 0.02), ("E0260", 0.2, 0.01), ("K0001", 0.2, 0.01),
          ("A4351", 0.2, 0.01), ("J7613", 0.2, 0.01), ("L1833", 0.05, 0.005), ("K0823", 0.03, 0.003))


@dataclass(frozen=True)
class Spec:
    name: str
    weight: float                     # share of NPIs
    p_org: float                      # P(entity type O)
    benes: float                      # median distinct beneficiaries, individuals (organisations x4)
    risk: float                       # mean HCC risk score of the panel
    codes: tuple                      # (code, p_bill, penetration, bene-days per bene, units per bene-day)
    est: tuple | None = None          # (p_bill, penetration, visits per bene, level shares 99211-99215)
    new: tuple | None = None          # (p_bill, penetration)
    ed: tuple | None = None           # (p_bill, penetration, visits per bene, level shares 99281-99285)
    p_dme: float = 0.0                # P(the provider refers any DME)
    dme: tuple = ()                   # (code, p_refer, penetration)
    schemes: tuple = SCHEMES          # schemes that can be planted


_ED_GEN = (0.02, 0.08, 0.35, 0.35, 0.20)
_GP = (("36415", 0.55, 0.35, 2.0, 1), ("93000", 0.5, 0.15, 1.2, 1), ("81002", 0.3, 0.1, 1.5, 1),
       ("85025", 0.25, 0.2, 1.5, 1), ("80053", 0.25, 0.2, 1.5, 1), ("G0439", 0.6, 0.3, 1.0, 1),
       ("G0438", 0.3, 0.05, 1.0, 1), ("99497", 0.15, 0.05, 1.0, 1), ("99490", 0.2, 0.08, 6.0, 1),
       ("96372", 0.35, 0.08, 2.0, 1), ("20610", 0.1, 0.03, 1.5, 1), ("J1100", 0.15, 0.04, 1.5, 8),
       ("J3301", 0.1, 0.03, 1.2, 4), ("J1030", 0.15, 0.04, 1.2, 1))
SPECS: tuple[Spec, ...] = (
    Spec("Internal Medicine", 0.19, 0.04, 450, 1.65, _GP + (("J0897", 0.05, 0.03, 2.0, 60),),
         est=(0.97, 0.78, 2.8, (0.01, 0.06, 0.40, 0.43, 0.10)), new=(0.80, 0.10), ed=(0.04, 0.05, 1.1, _ED_GEN),
         p_dme=0.7, dme=GP_DME),
    Spec("Family Practice", 0.17, 0.04, 380, 1.35,
         _GP + (("11055", 0.1, 0.02, 1.5, 1), ("J1885", 0.1, 0.03, 1.0, 2)),
         est=(0.97, 0.80, 2.6, (0.02, 0.08, 0.45, 0.38, 0.07)), new=(0.80, 0.12), ed=(0.06, 0.05, 1.1, _ED_GEN),
         p_dme=0.7, dme=GP_DME),
    Spec("Nurse Practitioner", 0.17, 0.0, 220, 1.5,
         (("G0439", 0.5, 0.3, 1.0, 1), ("99490", 0.2, 0.08, 6.0, 1), ("96372", 0.35, 0.08, 2.0, 1),
          ("36415", 0.4, 0.3, 2.0, 1), ("81002", 0.2, 0.1, 1.5, 1), ("99497", 0.1, 0.05, 1.0, 1),
          ("11721", 0.05, 0.1, 3.0, 1), ("J1030", 0.1, 0.04, 1.2, 1)),
         est=(0.92, 0.75, 2.4, (0.03, 0.10, 0.50, 0.32, 0.05)), new=(0.60, 0.12), ed=(0.05, 0.05, 1.1, _ED_GEN),
         p_dme=0.55, dme=GP_DME),
    Spec("Cardiology", 0.07, 0.03, 800, 1.95,
         (("93000", 0.9, 0.5, 1.5, 1), ("93010", 0.6, 0.4, 2.0, 1), ("93306", 0.8, 0.25, 1.1, 1),
          ("93015", 0.6, 0.1, 1.0, 1), ("93458", 0.4, 0.04, 1.0, 1), ("36415", 0.2, 0.2, 2.0, 1),
          ("85025", 0.15, 0.2, 2.0, 1)),
         est=(0.95, 0.70, 2.1, (0.01, 0.05, 0.35, 0.47, 0.12)), new=(0.85, 0.20),
         p_dme=0.45, dme=(("E1390", 0.5, 0.03), ("E0260", 0.2, 0.01), ("E0143", 0.3, 0.02), ("K0823", 0.03, 0.002))),
    Spec("Orthopedic Surgery", 0.06, 0.03, 600, 1.25,
         (("20610", 0.85, 0.25, 2.2, 1), ("27447", 0.5, 0.03, 1.0, 1), ("27130", 0.4, 0.02, 1.0, 1),
          ("29881", 0.4, 0.02, 1.0, 1), ("20550", 0.5, 0.06, 1.3, 1), ("73560", 0.7, 0.2, 1.2, 1),
          ("73030", 0.6, 0.12, 1.2, 1), ("J3301", 0.6, 0.15, 1.6, 4), ("J7325", 0.3, 0.05, 1.5, 48),
          ("J1100", 0.2, 0.05, 1.2, 8)),
         est=(0.95, 0.60, 2.0, (0.01, 0.10, 0.55, 0.30, 0.04)), new=(0.92, 0.35),
         p_dme=0.8, dme=(("L1833", 0.6, 0.03), ("L1851", 0.4, 0.02), ("L3960", 0.3, 0.015), ("L0650", 0.4, 0.02),
                         ("L3916", 0.3, 0.01), ("E0143", 0.4, 0.02), ("E0114", 0.5, 0.03), ("K0001", 0.2, 0.01))),
    Spec("Physical Therapist in Private Practice", 0.09, 0.15, 160, 1.3,
         (("97110", 0.98, 0.9, 10.0, 2.5), ("97140", 0.85, 0.6, 8.0, 1.5), ("97112", 0.7, 0.4, 6.0, 1.5),
          ("97530", 0.75, 0.5, 7.0, 1.5), ("97161", 0.8, 0.3, 1.0, 1), ("97162", 0.8, 0.45, 1.0, 1),
          ("G0283", 0.5, 0.3, 6.0, 1)),
         schemes=("overutilization", "atypical_mix", "charge_inflation")),
    Spec("Podiatry", 0.05, 0.02, 380, 1.55,
         (("11721", 0.9, 0.5, 3.5, 1), ("11720", 0.5, 0.1, 2.0, 1), ("11055", 0.8, 0.3, 3.0, 1),
          ("11042", 0.4, 0.05, 3.0, 1), ("G0127", 0.6, 0.25, 3.0, 1)),
         est=(0.90, 0.50, 2.5, (0.02, 0.25, 0.55, 0.16, 0.02)), new=(0.75, 0.15),
         p_dme=0.6, dme=(("A5500", 0.6, 0.06), ("L1902", 0.3, 0.01), ("E0114", 0.1, 0.005))),
    Spec("Diagnostic Radiology", 0.07, 0.30, 1800, 1.6,
         (("71046", 0.9, 0.25, 1.2, 1), ("71045", 0.7, 0.15, 1.3, 1), ("74177", 0.85, 0.12, 1.1, 1),
          ("70450", 0.8, 0.1, 1.1, 1), ("77067", 0.6, 0.2, 1.0, 1), ("72148", 0.75, 0.05, 1.0, 1),
          ("76700", 0.7, 0.06, 1.0, 1), ("73560", 0.7, 0.08, 1.1, 1), ("73030", 0.6, 0.06, 1.1, 1)),
         schemes=("overutilization", "atypical_mix", "charge_inflation")),
    Spec("Ambulance Service Provider", 0.04, 0.92, 600, 2.4,
         (("A0425", 1.0, 0.95, 1.6, 14), ("A0427", 0.9, 0.4, 1.4, 1), ("A0429", 0.95, 0.5, 1.3, 1),
          ("A0428", 0.8, 0.35, 2.5, 1)),
         schemes=("overutilization", "charge_inflation")),
    Spec("Emergency Medicine", 0.06, 0.05, 900, 1.9,
         (("93010", 0.6, 0.3, 1.2, 1), ("96372", 0.3, 0.05, 1.0, 1), ("36415", 0.1, 0.05, 1.0, 1),
          ("J1885", 0.3, 0.05, 1.0, 2)),
         ed=(1.0, 0.97, 1.2, (0.01, 0.04, 0.20, 0.35, 0.40)),
         p_dme=0.25, dme=(("E0114", 0.6, 0.02), ("E0143", 0.3, 0.01)),
         schemes=("upcoding", "overutilization", "atypical_mix", "charge_inflation")),
    Spec("Hematology-Oncology", 0.03, 0.05, 350, 2.3,
         (("96413", 0.9, 0.4, 5.0, 1), ("96365", 0.6, 0.2, 3.0, 1), ("96372", 0.7, 0.3, 3.0, 1),
          ("36415", 0.8, 0.6, 6.0, 1), ("85025", 0.8, 0.6, 5.0, 1), ("J9271", 0.6, 0.12, 6.0, 200),
          ("J9312", 0.4, 0.05, 4.0, 70), ("J2505", 0.6, 0.15, 4.0, 1), ("J0897", 0.5, 0.15, 2.0, 120),
          ("J1100", 0.8, 0.4, 5.0, 12)),
         est=(0.97, 0.85, 4.0, (0.03, 0.04, 0.25, 0.48, 0.20)), new=(0.85, 0.20),
         p_dme=0.35, dme=(("E0260", 0.3, 0.01), ("E1390", 0.3, 0.02))),
    Spec("Hand Surgery", 0.006, 0.0, 250, 1.15,   # the rare specialty: a handful of providers
         (("64721", 0.9, 0.1, 1.0, 1), ("26055", 0.8, 0.1, 1.2, 1), ("25607", 0.5, 0.03, 1.0, 1),
          ("20550", 0.7, 0.15, 1.3, 1), ("J3301", 0.6, 0.2, 1.2, 4)),
         est=(0.95, 0.60, 2.0, (0.01, 0.15, 0.55, 0.26, 0.03)), new=(0.90, 0.40),
         p_dme=0.6, dme=(("L3916", 0.6, 0.04), ("L3960", 0.2, 0.01))),
)

# Geography. Weights ~ population (millions); every Census division, territories, military, foreign.
STATES = {
    "CA": 39.5, "TX": 29.1, "FL": 21.5, "NY": 20.2, "PA": 13.0, "IL": 12.8, "OH": 11.8, "GA": 10.7, "NC": 10.4,
    "MI": 10.1, "NJ": 9.3, "VA": 8.6, "WA": 7.7, "AZ": 7.2, "MA": 7.0, "TN": 6.9, "IN": 6.8, "MD": 6.2,
    "MO": 6.2, "WI": 5.9, "CO": 5.8, "MN": 5.7, "SC": 5.1, "AL": 5.0, "LA": 4.7, "KY": 4.5, "OR": 4.2,
    "OK": 4.0, "CT": 3.6, "UT": 3.3, "IA": 3.2, "NV": 3.1, "AR": 3.0, "MS": 3.0, "KS": 2.9, "NM": 2.1,
    "NE": 2.0, "ID": 1.8, "WV": 1.8, "HI": 1.5, "NH": 1.4, "ME": 1.4, "RI": 1.1, "MT": 1.1, "DE": 1.0,
    "SD": 0.9, "ND": 0.8, "AK": 0.7, "DC": 0.7, "VT": 0.6, "WY": 0.6,
    "PR": 3.0, "VI": 0.1, "GU": 0.15, "AE": 0.15, "AP": 0.1, "AA": 0.05, "ZZ": 0.05,
}
NO_RUCA_STATES = ("AE", "AP", "AA", "ZZ")
FORCED_STATES = ("PR", "AE", "GU")       # guaranteed present, whatever the sample size
RUCA = {"1": 0.52, "1.1": 0.03, "2": 0.08, "2.1": 0.01, "3": 0.02, "4": 0.07, "4.1": 0.02, "5": 0.03,
        "5.1": 0.005, "6": 0.01, "7": 0.03, "7.1": 0.01, "7.2": 0.005, "8": 0.01, "9": 0.005, "10": 0.03,
        "10.1": 0.005, "10.3": 0.003, "99": 0.002, None: 0.02}
# by-Provider chronic-condition percentages: (harmonised name, base logit)
CC = (("bene_cc_ph_afib_v2_pct", -2.2), ("bene_cc_ph_cancer6_v2_pct", -2.0), ("bene_cc_ph_ckd_v2_pct", -1.3),
      ("bene_cc_ph_copd_v2_pct", -1.9), ("bene_cc_ph_diabetes_v2_pct", -0.9),
      ("bene_cc_ph_hf_nonihd_v2_pct", -1.8), ("bene_cc_ph_hyperlipidemia_v2_pct", 0.3),
      ("bene_cc_ph_hypertension_v2_pct", 0.6), ("bene_cc_ph_ischemicheart_v2_pct", -1.0),
      ("bene_cc_bh_depress_v1_pct", -1.1), ("bene_cc_bh_alz_nonalzdem_v2_pct", -2.0),
      ("bene_cc_ph_arthritis_v2_pct", -0.4))
# LEIE authorities (raw EXCLTYPE spelling, weight). Primary tier ~2/3; the rest other tiers or removed.
AUTHORITIES = (("1128b7", 0.30), ("1128a1", 0.22), ("1128a3", 0.05), ("1128b6", 0.04), ("1128b16", 0.03),
               ("1128(b)(1)(A)", 0.02), ("1128b4", 0.20), ("1128b14", 0.05), ("1128a2", 0.04),
               ("1128b5", 0.03), ("1128b8", 0.02))
PRIMARY_RAW = ("1128b7", "1128a1", "1128a3", "1128b6")


# ------------------------------------------------------------------------------------------- helpers

def make_npis(rng: np.random.Generator, n: int) -> np.ndarray:
    """n unique 10-digit NPIs starting with 1, with a valid Luhn check digit (prefix 80840)."""
    base = rng.choice(100_000_000, size=n, replace=False).astype(np.int64) + 100_000_000
    digits = (base[:, None] // 10 ** np.arange(8, -1, -1)) % 10
    dbl = digits[:, 0::2] * 2                      # positions 1, 3, ..., 9: doubled (rightmost first)
    total = (dbl - 9 * (dbl > 9)).sum(1) + digits[:, 1::2].sum(1) + 24
    check = (10 - total % 10) % 10
    return np.char.mod("%010d", base * 10 + check)


def _dates(rng: np.random.Generator, years: np.ndarray) -> np.ndarray:
    """A random date within each year, never after AS_OF."""
    start = (years - 1970).astype("datetime64[Y]").astype("datetime64[D]")
    end = np.minimum((years + 1 - 1970).astype("datetime64[Y]").astype("datetime64[D]"),
                     np.datetime64(AS_OF) + np.timedelta64(1, "D"))
    span = (end - start).astype(np.int64)
    return start + (rng.random(len(years)) * span).astype(np.int64).astype("timedelta64[D]")


def _gamma_rows(rng: np.random.Generator, alpha: np.ndarray) -> np.ndarray:
    g = rng.gamma(np.maximum(alpha, 1e-6))
    return g / g.sum(1, keepdims=True)


def _gather(values, idx: np.ndarray, name: str) -> pl.Series:
    return pl.Series(name, list(values), dtype=pl.String).gather(idx)


# ------------------------------------------------------------------------------------------- population

@dataclass
class _Pop:
    npi: np.ndarray        # (n,) str
    spec: np.ndarray       # (n,) specialty index
    ent: np.ndarray        # (n,) "I" / "O"
    state: np.ndarray
    ruca: np.ndarray       # object, None = missing
    part: np.ndarray       # "Y" / "N"
    zip5: np.ndarray
    nb: np.ndarray         # base panel size
    risk: np.ndarray
    markup: np.ndarray     # submitted / allowed
    price: np.ndarray      # provider price noise
    fac: np.ndarray        # propensity to bill at facility
    dual: np.ndarray       # P(dual eligible)
    geo: np.ndarray        # geographic price factor
    first: np.ndarray      # first active year
    last: np.ndarray       # last active year
    stop: np.ndarray       # first year barred (exclusion), 9999 = never
    stop_partial: np.ndarray  # still bills in the exclusion year (excluded in the second half)
    est: np.ndarray        # (n, 5) base established-visit level shares
    eds: np.ndarray        # (n, 5) base ED level shares
    new: np.ndarray        # (n, 5) base new-visit level shares


def _population(rng: np.random.Generator, n: int, years: list[int], npis: np.ndarray) -> _Pop:
    w = np.array([s.weight for s in SPECS])
    spec = rng.choice(len(SPECS), size=n, p=w / w.sum())
    p_org = np.array([s.p_org for s in SPECS])[spec]
    ent = np.where(rng.random(n) < p_org, "O", "I")
    st_names = np.array(list(STATES))
    st_w = np.array(list(STATES.values()))
    state = rng.choice(st_names, size=n, p=st_w / st_w.sum())
    missing = [s for s in FORCED_STATES if s not in set(state)]
    if missing:
        state[rng.choice(n, size=len(missing), replace=False)] = missing
    ruca_v = np.array(list(RUCA), dtype=object)
    ruca_w = np.array(list(RUCA.values()))
    ruca = rng.choice(ruca_v, size=n, p=ruca_w / ruca_w.sum())
    ruca[np.isin(state, NO_RUCA_STATES)] = None
    geo_by_state = {s: f for s, f in zip(st_names, rng.uniform(0.9, 1.15, len(st_names)))}
    geo_by_state["PR"] = 0.85
    base_nb = np.array([s.benes for s in SPECS])[spec] * np.where(ent == "O", 4.0, 1.0)
    part_time = rng.random(n) < 0.10                     # small panels: suppression and the volume rule bite
    nb = base_nb * rng.lognormal(0, 0.8, n) * np.where(part_time, rng.uniform(0.03, 0.12, n), 1.0)
    risk = np.array([s.risk for s in SPECS])[spec] * rng.lognormal(0, 0.18, n)
    dual = rng.beta(2, 6, n) + np.where(state == "PR", 0.3, 0.0)
    y0, y1 = min(years), max(years)
    entrant = (rng.random(n) < 0.2) & (y1 > y0)          # starts billing after the first year
    first = np.where(entrant, rng.integers(y0 + 1, max(y1, y0 + 1) + 1, n), 0)
    leaver = (rng.random(n) < 0.08) & (y1 > y0)          # stops billing before the last year
    last = np.where(leaver, rng.integers(y0, max(y1, y0 + 1), n), 9999)

    def shares(field: str) -> np.ndarray:
        """Provider-specific level shares around the specialty's (or a generic) profile."""
        default = NEW_SHARES if field == "new" else _ED_GEN
        base = np.array([getattr(s, field)[3] if field != "new" and getattr(s, field) else default
                         for s in SPECS])[spec]
        return _gamma_rows(rng, 40.0 * base)

    return _Pop(
        npi=npis, spec=spec, ent=ent, state=state, ruca=ruca,
        part=np.where(rng.random(n) < 0.96, "Y", "N"), zip5=np.char.mod("%05d", rng.integers(1000, 99950, n)),
        nb=nb, risk=risk, markup=rng.lognormal(np.log(2.2), 0.3, n), price=rng.lognormal(0, 0.04, n),
        fac=rng.beta(1.2, 6, n), dual=np.clip(dual, 0.01, 0.95),
        geo=np.array([geo_by_state[s] for s in state]), first=first, last=last,
        stop=np.full(n, 9999), stop_partial=np.zeros(n, bool),
        est=shares("est"), eds=shares("ed"), new=shares("new"),
    )


@dataclass
class _Profile:
    """Static per-specialty code propensities, so a provider bills similar codes year after year."""
    members: np.ndarray    # provider indices of this specialty
    codes: np.ndarray      # code ids
    p_bill: np.ndarray
    pen: np.ndarray
    spb: np.ndarray
    upd: np.ndarray
    u: np.ndarray          # (m, K) billing propensity
    pen_noise: np.ndarray
    spb_noise: np.ndarray
    ue: np.ndarray         # (m, 3) billing propensity for the established / ED / new E&M groups
    pen_e: np.ndarray      # (m, 3) penetration noise for the E&M groups
    refers: np.ndarray     # (m,) refers DME at all
    dcodes: np.ndarray
    dp: np.ndarray
    dpen: np.ndarray
    du: np.ndarray
    dpen_noise: np.ndarray


def _profiles(rng: np.random.Generator, pop: _Pop) -> list[_Profile]:
    out = []
    for s_idx, s in enumerate(SPECS):
        members = np.flatnonzero(pop.spec == s_idx)
        prof = s.codes + RARE_PROFILE
        m, k, kd = len(members), len(prof), len(s.dme)
        out.append(_Profile(
            members=members,
            codes=np.array([CODE_ID[c[0]] for c in prof]),
            p_bill=np.array([c[1] for c in prof]), pen=np.array([c[2] for c in prof]),
            spb=np.array([c[3] for c in prof]), upd=np.array([c[4] for c in prof], dtype=float),
            u=rng.random((m, k)), pen_noise=rng.lognormal(0, 0.5, (m, k)), spb_noise=rng.lognormal(0, 0.25, (m, k)),
            ue=rng.random((m, 3)), pen_e=rng.lognormal(0, 0.25, (m, 3)),
            refers=rng.random(m) < s.p_dme,
            dcodes=np.array([DME_ID[c[0]] for c in s.dme], dtype=np.int64),
            dp=np.array([c[1] for c in s.dme]), dpen=np.array([c[2] for c in s.dme]),
            du=rng.random((m, kd)), dpen_noise=rng.lognormal(0, 0.5, (m, kd)),
        ))
    return out


# ------------------------------------------------------------------------------------------- planting

@dataclass
class _Plan:
    events: pl.DataFrame   # LEIE rows with an NPI (Part B and non-Part B)
    no_npi: int
    schemes: np.ndarray    # (n, 5) bool
    win_first: np.ndarray  # first scheme year (9999 = none)
    win_last: np.ndarray
    intensity: np.ndarray
    source: np.ndarray     # "" / "excluded" / "unlabelled"


def _plan(rng: np.random.Generator, pop: _Pop, years: list[int], excluded_frac: float,
          extra_npis: np.ndarray, cfg: labels.LabelConfig) -> _Plan:
    n = len(pop.npi)
    n_exc = max(1, round(excluded_frac * n))
    exc = np.sort(rng.choice(n, size=n_exc, replace=False))
    auth_raw = np.array([a for a, _ in AUTHORITIES])
    auth_w = np.array([w for _, w in AUTHORITIES])
    raw = rng.choice(auth_raw, size=n_exc, p=auth_w / auth_w.sum())
    yrs = rng.integers(2017, 2027, n_exc)
    ev_idx, ev_raw, ev_year = [exc], [raw], [yrs]
    second = exc[rng.random(n_exc) < 0.06]            # a later, second exclusion under a primary authority
    if second.size:
        y2 = np.minimum(yrs[np.searchsorted(exc, second)] + rng.integers(1, 4, second.size), 2026)
        ev_idx.append(second)
        ev_raw.append(rng.choice(np.array(PRIMARY_RAW), size=second.size))
        ev_year.append(y2)
    ev_idx, ev_raw, ev_year = np.concatenate(ev_idx), np.concatenate(ev_raw), np.concatenate(ev_year)
    ev_date = _dates(rng, ev_year)
    primary = np.isin([cfg.excltype_map[labels.normalise_excltype(r)] for r in ev_raw], cfg.tiers["primary"])

    # Barred from billing after the first exclusion; still billing in that year if excluded after June.
    first_date = np.full(n, np.datetime64("2100-01-01"), dtype="datetime64[D]")
    np.minimum.at(first_date, ev_idx, ev_date)
    hit = first_date < np.datetime64("2100-01-01")
    pop.stop[hit] = first_date[hit].astype("datetime64[Y]").astype(int) + 1970
    pop.stop_partial[hit] = (first_date[hit] - first_date[hit].astype("datetime64[Y]")).astype(int) >= 181

    # Schemes: in the 3 years before the first primary-tier exclusion, most such providers run 1-2 schemes.
    yp = np.full(n, 9999)
    np.minimum.at(yp, ev_idx[primary], ev_year[primary])
    cand = np.flatnonzero((yp < 9999) & (rng.random(n) < 0.85))
    never = np.setdiff1d(np.arange(n), ev_idx)
    unl = rng.choice(never, size=max(1, round(0.005 * n)), replace=False)   # unlabelled scheme providers
    planted = np.concatenate([cand, unl])
    applic = np.array([[sch in SPECS[s].schemes for sch in SCHEMES] for s in range(len(SPECS))])[pop.spec[planted]]
    keys = np.where(applic, np.log(SCHEME_WEIGHT) + rng.gumbel(size=applic.shape), -np.inf)
    order = np.argsort(-keys, axis=1)
    schemes = np.zeros((n, len(SCHEMES)), bool)
    schemes[planted, order[:, 0]] = True
    two = (rng.random(len(planted)) < 0.3) & (applic.sum(1) >= 2)
    schemes[planted[two], order[two, 1]] = True
    win_first, win_last = np.full(n, 9999), np.full(n, 9999)
    win_first[cand], win_last[cand] = yp[cand] - 3, yp[cand] - 1
    start = rng.integers(min(years) - 1, max(years), len(unl))   # 3-year window overlapping the data
    win_first[unl], win_last[unl] = start, start + 2
    source = np.full(n, "", dtype=object)
    source[cand], source[unl] = "excluded", "unlabelled"

    # LEIE rows with an NPI: Part B providers, plus excluded parties who never billed Part B.
    n_out = len(extra_npis)
    out_raw = rng.choice(np.array(PRIMARY_RAW + ("1128b4",)), size=n_out)
    out_date = _dates(rng, rng.integers(2017, 2027, n_out))
    events = pl.DataFrame({
        "npi": np.concatenate([pop.npi[ev_idx], extra_npis]),
        "excltype": np.concatenate([ev_raw, out_raw]),
        "excl_date": np.concatenate([ev_date, out_date]),
        "is_org": np.concatenate([pop.ent[ev_idx] == "O", np.zeros(n_out, bool)]),
        "state": np.concatenate([pop.state[ev_idx], np.full(n_out, "TX")]),
        "specialty": np.concatenate([np.array([SPECS[s].name for s in pop.spec[ev_idx]], dtype=object),
                                     np.full(n_out, "Nurse Aide", dtype=object)]),
        "in_part_b": np.concatenate([np.ones(len(ev_idx), bool), np.zeros(n_out, bool)]),
    })
    return _Plan(events=events, no_npi=max(3, round(0.003 * n)), schemes=schemes, win_first=win_first,
                 win_last=win_last, intensity=rng.uniform(0.7, 1.0, n), source=source)


# ------------------------------------------------------------------------------------------- Part B rows

def _em_rows(rng, idx, nb, pen, vpb, shares, codes, up, ou):
    """Visits split over 5 level codes. Upcoding moves share 0.6*up of levels 1-3 to levels 4-5
    (65 % to level 5); overutilization multiplies visits by 1 + 1.2*ou, not beneficiaries."""
    benes_tot = rng.binomial(nb, np.clip(pen, 0, 0.98))
    visits = benes_tot * vpb * rng.lognormal(0, 0.12, len(idx))
    shares = shares.copy()
    low = shares[:, :3] * (0.6 * up)[:, None]
    shares[:, :3] -= low
    shares[:, 3] += 0.35 * low.sum(1)
    shares[:, 4] += 0.65 * low.sum(1)
    base = np.rint(visits[:, None] * shares)
    benes = np.minimum(benes_tot[:, None], np.ceil(base / (1 + (vpb - 1) * 0.6)))
    srv = np.rint(base * (1 + 1.2 * ou)[:, None])
    benes = np.where(srv > 0, np.maximum(benes, 1), 0)
    srv = np.maximum(srv, benes)
    keep = srv > 0
    return (np.broadcast_to(idx[:, None], srv.shape)[keep], np.broadcast_to(codes, srv.shape)[keep],
            benes[keep], srv[keep], srv[keep])


def _plain_rows(rng, pr: _Profile, pos, idx, nb, ou):
    n, k = len(pos), len(pr.codes)
    bill = (pr.u[pos] + rng.normal(0, 0.03, (n, k))) < pr.p_bill
    pen = np.clip(pr.pen * pr.pen_noise[pos] * rng.lognormal(0, 0.1, (n, k)), 0, 0.98)
    benes = rng.binomial(nb[:, None], pen) * bill
    drug = CODE_DRUG[pr.codes][None, :]
    ouk = np.where(drug, 0.0, ou[:, None])
    days = np.maximum(benes, np.rint(benes * pr.spb * pr.spb_noise[pos] * rng.lognormal(0, 0.08, (n, k))
                                     * (1 + 0.7 * ouk)))
    units = days * pr.upd * rng.lognormal(0, 0.05, (n, k)) * (1 + 0.5 * ouk)
    srv = np.where(drug, np.maximum(np.round(units, 1), 0.1), np.maximum(np.rint(units), days))
    keep = benes > 0
    return (np.broadcast_to(idx[:, None], benes.shape)[keep], np.broadcast_to(pr.codes, benes.shape)[keep],
            benes[keep], srv[keep], days[keep])


def _atypical_rows(rng, idx, nb, at):
    """2-3 codes from the rare pool, with enough beneficiaries to survive suppression."""
    sel = (at > 0) & (nb >= 15)
    idx, nb, at = idx[sel], nb[sel], at[sel]
    n = len(idx)
    if not n:
        return None
    pick = np.argsort(rng.random((n, len(RARE))), axis=1)[:, :3]
    use = np.arange(3)[None, :] < (2 + (rng.random(n) < 0.5))[:, None]
    codes = np.array([CODE_ID[c] for c in RARE])[pick]
    upd = np.array([RARE_UPD[c] for c in RARE])[pick]
    benes = np.minimum(nb[:, None], np.maximum(SUPPRESS_MAX + 1, np.rint(rng.uniform(14, 40, (n, 3)) * (0.6 + at)[:, None])))
    days = np.rint(benes * rng.uniform(1.5, 3.5, (n, 3)))
    return (np.broadcast_to(idx[:, None], (n, 3))[use], codes[use], benes[use], (days * upd)[use], days[use])


def _part_b(rng, year: int, pop: _Pop, profs: list[_Profile], active: np.ndarray, nb: np.ndarray,
            sch: np.ndarray) -> tuple[pl.DataFrame, pl.DataFrame]:
    """All (unsuppressed) service rows of the year, priced; returns (visible cms_ps, cms_prov)."""
    up, ou, at, ch = (sch[:, i] for i in range(4))
    est_shift = np.array([0.8, 0.75, 0.88, 1.15, 1.4]) if year >= 2021 else np.ones(5)
    ed_shift = np.array([0.9, 0.9, 0.95, 1.0, 1.08]) if year >= 2023 else np.ones(5)
    parts = []
    for s_idx, (s, pr) in enumerate(zip(SPECS, profs)):
        pos = np.flatnonzero(active[pr.members])
        if not pos.size:
            continue
        idx = pr.members[pos]
        new = (s.new[0], s.new[1], 1.05) if s.new else None
        for j, (g, base, codes, shift) in enumerate(((s.est, pop.est, EST, est_shift), (s.ed, pop.eds, ED, ed_shift),
                                                      (new, pop.new, NEW, np.ones(5)))):
            if g is None:
                continue
            sh = base[idx] * shift
            if codes is NEW and year >= 2021:   # 99201 deleted from 2021: its share moves to 99202
                sh[:, 1] += sh[:, 0]
                sh[:, 0] = 0.0
            sh = _gamma_rows(rng, 300.0 * sh / sh.sum(1, keepdims=True))
            b = pr.ue[pos, j] < g[0]
            parts.append(_em_rows(rng, idx[b], nb[idx[b]], g[1] * pr.pen_e[pos[b], j], g[2], sh[b],
                                  np.array([CODE_ID[c] for c in codes]), up[idx[b]], ou[idx[b]]))
        parts.append(_plain_rows(rng, pr, pos, idx, nb[idx], ou[idx]))
    act = np.flatnonzero(active)
    extra = _atypical_rows(rng, act, nb[act], at[act])
    if extra is not None:
        parts.append(extra)
    p, c, b, s_, d = (np.concatenate([x[i] for x in parts]) for i in range(5))
    p, c = p.astype(np.int64), c.astype(np.int64)

    # Place of service: fixed for some kinds, else provider propensity; some rows split over F and O.
    pf = CODE_PF[c]
    mixed = (pf > 0) & (pf < 1)
    f = rng.random(len(p)) < np.where(mixed, np.clip(pf + pop.fac[p] * (1 - pf), 0, 1), pf)
    split = mixed & (b >= 2) & (rng.random(len(p)) < 0.15)
    r = rng.uniform(0.25, 0.75, len(p))
    b1 = np.where(split, np.clip(np.rint(b * r), 1, b - 1), b)
    frac = b1 / b
    s1 = np.where(CODE_DRUG[c], np.round(s_ * frac, 1), np.rint(s_ * frac))
    d1 = np.rint(d * frac)
    p = np.concatenate([p, p[split]])
    c = np.concatenate([c, c[split]])
    f = np.concatenate([f, ~f[split]])
    b = np.concatenate([b1, (b - b1)[split]])
    s_ = np.concatenate([s1, (s_ - s1)[split]])
    d = np.concatenate([d1, (d - d1)[split]])
    s_ = np.where(CODE_DRUG[c], np.maximum(s_, 0.1), np.maximum(s_, b))
    d = np.maximum(d, b)
    # One row per (npi, code, place): merge collisions (e.g. an atypical code the provider already billed).
    key = (p * len(CODE_LIST) + c) * 2 + f
    uk, inv = np.unique(key, return_inverse=True)
    p, c, f = uk // 2 // len(CODE_LIST), uk // 2 % len(CODE_LIST), (uk % 2).astype(bool)
    b = np.minimum(np.bincount(inv, b), nb[p])
    s_ = np.bincount(inv, s_)
    d = np.bincount(inv, d)
    s_ = np.where(CODE_DRUG[c], np.round(s_, 1), s_)

    # Prices.
    yf = 1.012 ** (year - 2018) * (np.array([EM_2021.get(x, 1.0) for x in CODE_LIST])[c] if year >= 2021 else 1.0)
    site = np.where(f, CODE_SITE[c], 1.0)
    n = len(p)
    allowed = (CODE_PRICE[c] * yf * pop.geo[p] * site * pop.price[p] * rng.lognormal(0, 0.03, n)
               * np.where(pop.part[p] == "N", 0.95, 1.0) * (1 + 0.15 * ch[p]))
    stdzd = CODE_PRICE[c] * yf * site * rng.lognormal(0, 0.02, n) * (1 + 0.15 * ch[p])
    pymt = allowed * rng.uniform(0.76, 0.80, n)
    mk = np.where(CODE_DRUG[c], 1 + (pop.markup[p] - 1) * 0.3, pop.markup[p])
    sbmtd = allowed * mk * rng.lognormal(0, 0.05, n) * (1 + 1.5 * ch[p])

    # by-Provider totals from all rows (before suppression).
    m = len(pop.npi)
    tot = {k: np.bincount(p, v, minlength=m) for k, v in (
        ("srv", s_), ("sb", s_ * sbmtd), ("al", s_ * allowed), ("py", s_ * pymt), ("sd", s_ * stdzd), ("b", b))}
    ncodes = np.bincount(np.unique(p * len(CODE_LIST) + c) // len(CODE_LIST), minlength=m)
    tot_benes = np.minimum(nb, tot["b"]).astype(np.int64)

    vis = b > SUPPRESS_MAX
    pv, cv = p[vis], c[vis]
    ps = pl.DataFrame({
        "npi": _gather(pop.npi, pv, "npi"),
        "ent_cd": _gather(pop.ent, pv, "ent_cd"),
        "state": _gather(pop.state, pv, "state"),
        "ruca": pl.Series("ruca", list(pop.ruca), dtype=pl.String).gather(pv),
        "prvdr_type": _gather([s.name for s in SPECS], pop.spec[pv], "prvdr_type"),
        "mdcr_prtcptg_ind": _gather(pop.part, pv, "mdcr_prtcptg_ind"),
        "hcpcs_cd": _gather(CODE_LIST, cv, "hcpcs_cd"),
        "hcpcs_drug_ind": np.where(CODE_DRUG[cv], "Y", "N"),
        "place_of_srvc": np.where(f[vis], "F", "O"),
        "tot_benes": b[vis].astype(np.int64),
        "tot_srvcs": s_[vis],
        "tot_bene_day_srvcs": d[vis],
        "avg_sbmtd_chrg": np.round(sbmtd[vis], 6),
        "avg_mdcr_alowd_amt": np.round(allowed[vis], 6),
        "avg_mdcr_pymt_amt": np.round(pymt[vis], 6),
        "avg_mdcr_stdzd_amt": np.round(stdzd[vis], 6),
        "hcpcs_desc": _gather([CODES[x][0] for x in CODE_LIST], cv, "hcpcs_desc"),
        "zip5": _gather(pop.zip5, pv, "zip5"),
    })

    # by-Provider file: providers with > 10 beneficiaries and at least one service.
    keep = np.flatnonzero(active & (tot_benes > SUPPRESS_MAX) & (tot["srv"] > 0))
    tb = tot_benes[keep]
    risk = pop.risk[keep] * rng.lognormal(0, 0.04, len(keep))
    dual = rng.binomial(tb, pop.dual[keep])
    dual_ok = (dual > SUPPRESS_MAX) & (tb - dual > SUPPRESS_MAX) & (rng.random(len(keep)) > 0.01)
    cols = {
        "npi": _gather(pop.npi, keep, "npi"),
        "ent_cd": _gather(pop.ent, keep, "ent_cd"),
        "state": _gather(pop.state, keep, "state"),
        "ruca": pl.Series("ruca", list(pop.ruca), dtype=pl.String).gather(keep),
        "prvdr_type": _gather([s.name for s in SPECS], pop.spec[keep], "prvdr_type"),
        "mdcr_prtcptg_ind": _gather(pop.part, keep, "mdcr_prtcptg_ind"),
        "tot_hcpcs_cds": ncodes[keep].astype(np.int64),
        "tot_benes": tb,
        "tot_srvcs": np.round(tot["srv"][keep], 1),
        "tot_sbmtd_chrg": np.round(tot["sb"][keep], 2),
        "tot_mdcr_alowd_amt": np.round(tot["al"][keep], 2),
        "tot_mdcr_pymt_amt": np.round(tot["py"][keep], 2),
        "tot_mdcr_stdzd_amt": np.round(tot["sd"][keep], 2),
        "bene_avg_risk_scre": np.round(risk, 4),
        "bene_dual_cnt": pl.Series(dual.astype(np.int64)).scatter(np.flatnonzero(~dual_ok), None),
        "bene_ndual_cnt": pl.Series((tb - dual).astype(np.int64)).scatter(np.flatnonzero(~dual_ok), None),
    }
    for name, logit in CC:
        pct = np.rint(100 / (1 + np.exp(-(logit + 1.2 * (risk - 1.4) + rng.normal(0, 0.3, len(keep))))))
        cnt = pct * tb / 100
        bad = (cnt < SUPPRESS_MAX + 1) | (tb - cnt < SUPPRESS_MAX + 1) | (rng.random(len(keep)) < 0.01)
        cols[name] = pl.Series(pct.astype(float)).scatter(np.flatnonzero(bad), None)
    return ps, pl.DataFrame(cols)


# ------------------------------------------------------------------------------------------- DMEPOS rows

def _dme(rng, pop: _Pop, profs: list[_Profile], active: np.ndarray, nb: np.ndarray, dm: np.ndarray,
         dme_only: tuple[np.ndarray, np.ndarray]) -> tuple[pl.DataFrame, pl.DataFrame]:
    parts = []
    for s, pr in zip(SPECS, profs):
        pos = np.flatnonzero(active[pr.members] & pr.refers)
        if not pos.size or not len(pr.dcodes):
            continue
        idx = pr.members[pos]
        k = len(pr.dcodes)
        bill = pr.du[pos] < pr.dp
        pen = np.clip(pr.dpen * pr.dpen_noise[pos] * rng.lognormal(0, 0.1, (len(pos), k)), 0, 0.5)
        benes = rng.binomial(nb[idx][:, None], pen) * bill
        keep = benes > 0
        parts.append((np.broadcast_to(idx[:, None], benes.shape)[keep], np.broadcast_to(pr.dcodes, benes.shape)[keep],
                      benes[keep], np.zeros(keep.sum(), bool)))
    # DME mill: 2-3 high-risk codes, many claims per patient, filled by 1-2 suppliers.
    mill = np.flatnonzero(active & (dm > 0) & (nb >= 15))
    if mill.size:
        pick = MILL_CODES[np.argsort(rng.random((mill.size, len(MILL_CODES))), axis=1)[:, :3]]
        use = np.arange(3)[None, :] < (2 + (rng.random(mill.size) < 0.5))[:, None]
        benes = np.minimum(nb[mill][:, None], np.maximum(SUPPRESS_MAX + 1, np.rint(
            rng.uniform(20, 55, (mill.size, 3)) * (0.6 + dm[mill])[:, None])))
        parts.append((np.broadcast_to(mill[:, None], (mill.size, 3))[use], pick[use], benes[use],
                      np.ones(use.sum(), bool)))
    # DME-only referrers (no Part B rows): their own index space after the Part B population.
    o_npi, o_nb = dme_only
    gp = np.array([DME_ID[c] for c, _, _ in GP_DME])
    gpen = np.array([pen for _, _, pen in GP_DME])
    ob = rng.binomial(o_nb[:, None].astype(np.int64), np.clip(gpen * 3, 0, 0.5)) * (rng.random((len(o_npi), len(gp))) < 0.5)
    keep = ob > 0
    parts.append((len(pop.npi) + np.broadcast_to(np.arange(len(o_npi))[:, None], ob.shape)[keep],
                  np.broadcast_to(gp, ob.shape)[keep], ob[keep], np.zeros(keep.sum(), bool)))

    p, c, b, ml = (np.concatenate([x[i] for x in parts]) for i in range(4))
    p, c, b = p.astype(np.int64), c.astype(np.int64), b.astype(np.int64)
    rent_mode = DME_COL["rent"][c]
    rent = np.where(rent_mode == "Y", True, np.where(rent_mode == "B", (rng.random(len(p)) < 0.75) & ~ml, False))
    split = (rent_mode == "B") & ~ml & (b >= 2) & (rng.random(len(p)) < 0.2)
    b1 = np.where(split, np.clip(np.rint(b * 0.6), 1, b - 1), b).astype(np.int64)
    p, c, ml = np.concatenate([p, p[split]]), np.concatenate([c, c[split]]), np.concatenate([ml, ml[split]])
    rent = np.concatenate([rent, ~rent[split]])
    b = np.concatenate([b1, (b - b1)[split]])
    key = (p * len(DME_LIST) + c) * 2 + rent
    uk, inv = np.unique(key, return_inverse=True)
    p, c, rent = uk // 2 // len(DME_LIST), uk // 2 % len(DME_LIST), (uk % 2).astype(bool)
    b = np.bincount(inv, b).astype(np.int64)
    ml = np.bincount(inv, ml) > 0
    n = len(p)
    all_nb = np.concatenate([nb, o_nb])
    b = np.minimum(b, np.maximum(all_nb[p], 1).astype(np.int64))
    dmp = np.concatenate([dm, np.zeros(len(o_npi))])[p]
    buy = (DME_COL["rent"][c] == "B") & ~rent             # purchase of a usually-rented item
    cpb = np.where(buy, 1.05, DME_COL["cpb"][c])
    clms = np.maximum(b, np.rint(b * cpb * rng.lognormal(0, 0.15, n) * np.where(ml, 1.5 + dmp, 1.0))).astype(np.int64)
    srv = np.maximum(clms, np.rint(clms * DME_COL["spc"][c] * rng.lognormal(0, 0.1, n)))
    suplrs = np.where(ml, np.minimum(b, 1 + (rng.random(n) < 0.4)), 1 + rng.binomial(np.maximum(b - 1, 0), DME_COL["srate"][c]))
    geo = np.concatenate([pop.geo, np.ones(len(o_npi))])[p]
    price = DME_COL["price"][c] * np.where(buy, DME_COL["buy"][c], 1)
    allowed = price * (0.95 + 0.1 * (geo - 0.9) / 0.25) * rng.lognormal(0, 0.03, n)
    pymt = allowed * rng.uniform(0.78, 0.80, n)
    stdzd = price * rng.lognormal(0, 0.01, n)
    sbmtd = allowed * rng.lognormal(np.log(1.9), 0.2, n) * np.where(ml, 1.3, 1.0)

    names = np.concatenate([pop.npi, o_npi])
    specs = np.concatenate([np.array([SPECS[s].name for s in pop.spec], dtype=object),
                            np.full(len(o_npi), "Nurse Practitioner", dtype=object)])
    vis = b > SUPPRESS_MAX
    pv, cv = p[vis], c[vis]
    ds = pl.DataFrame({
        "npi": _gather(names, pv, "npi"),
        "rbcs_lvl": _gather(DME_COL["lvl"], cv, "rbcs_lvl"),
        "rbcs_id": _gather(DME_COL["rbcs"], cv, "rbcs_id"),
        "hcpcs_cd": _gather(DME_LIST, cv, "hcpcs_cd"),
        "suplr_rentl_ind": np.where(rent[vis], "Y", "N"),
        "tot_suplrs": suplrs[vis].astype(np.int64),
        "tot_suplr_benes": b[vis],
        "tot_suplr_clms": clms[vis],
        "tot_suplr_srvcs": srv[vis].astype(float),
        "avg_suplr_sbmtd_chrg": np.round(sbmtd[vis], 6),
        "avg_suplr_mdcr_alowd_amt": np.round(allowed[vis], 6),
        "avg_suplr_mdcr_pymt_amt": np.round(pymt[vis], 6),
        "avg_suplr_mdcr_stdzd_amt": np.round(stdzd[vis], 6),
        "hcpcs_desc": _gather(DME_COL["desc"], cv, "hcpcs_desc"),
        "prvdr_type": _gather(specs, pv, "prvdr_type"),
    })

    # DME by-Provider: distinct beneficiaries ~ largest row + 35 % of the rest; counts 1-10 suppressed.
    m = len(names)
    lvl = DME_COL["lvl"][c]

    def distinct(mask: np.ndarray) -> np.ndarray:
        s = np.bincount(p[mask], b[mask], minlength=m)
        mx = np.zeros(m)
        np.maximum.at(mx, p[mask], b[mask])
        return np.rint(mx + 0.35 * (s - mx))

    tot_b = np.minimum(distinct(np.ones(n, bool)), np.maximum(all_nb, 1))
    dme_b = np.minimum(distinct(lvl == "DME"), tot_b)
    pos_b = np.minimum(distinct(lvl == "POS"), tot_b)
    pay = srv * pymt
    dme_pay = np.bincount(p, np.where(lvl == "DME", pay, 0), minlength=m)
    pos_pay = np.bincount(p, np.where(lvl == "POS", pay, 0), minlength=m)
    keep = np.flatnonzero(tot_b > SUPPRESS_MAX)

    def sup(cnt: np.ndarray, amt: np.ndarray | None = None) -> pl.Series:
        hidden = (cnt > 0) & (cnt <= SUPPRESS_MAX)
        vals = (cnt if amt is None else amt).astype(float)
        return pl.Series(np.where(hidden, np.nan, vals)).fill_nan(None)

    dp = pl.DataFrame({
        "npi": _gather(names, keep, "npi"),
        "tot_suplr_benes": tot_b[keep].astype(np.int64),
        "dme_tot_suplr_benes": sup(dme_b[keep]).cast(pl.Int64),
        "prvdr_type": _gather(specs, keep, "prvdr_type"),
        "cat_dme_suplr_mdcr_pymt_amt": sup(dme_b[keep], np.round(dme_pay[keep], 2)),
        "cat_pos_tot_suplr_benes": sup(pos_b[keep]),
        "cat_pos_suplr_mdcr_pymt_amt": sup(pos_b[keep], np.round(pos_pay[keep], 2)),
    })
    return ds, dp


# ------------------------------------------------------------------------------------------- writing

def _conform(df: pl.DataFrame, schema_ds: dict, year: int) -> pl.DataFrame:
    """Cast to the harmonised dtypes and append `year`; refuse any column the schema does not know."""
    exprs = []
    for col in df.columns:
        spec = _target_spec(schema_ds, col)
        if spec is None:
            raise ValueError(f"synthetic column {col!r} is not in _schema.yaml")
        exprs.append(pl.col(col).cast(DTYPES[spec["dtype"]]))
    missing = [c for c, s in schema_ds["columns"].items() if s.get("required") and c not in df.columns]
    if missing:
        raise ValueError(f"synthetic frame lacks required columns {missing}")
    return df.select(exprs).with_columns(pl.lit(year, dtype=pl.Int32).alias("year"))


def _write(paths: Paths, dataset: str, year: int, df: pl.DataFrame, seed: int) -> Path:
    out = paths.interim / dataset / f"{year}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    meta = {"dataset": dataset, "year": year, "synthetic": True, "generator": "tests/synth.py", "seed": seed,
            "n_rows": df.height, "note": "synthetic test data; not CMS data; never a result"}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def _leie_csv(rng, plan: _Plan, path: Path) -> Path:
    ev = plan.events
    n_ev, k = ev.height, plan.no_npi
    org = ev["is_org"].to_numpy()
    i = np.arange(n_ev)
    blank = np.full(n_ev, "", dtype=object)
    date_s = np.char.replace(np.datetime_as_string(ev["excl_date"].to_numpy(), unit="D"), "-", "")
    cols = {
        "LASTNAME": np.where(org, "", np.char.add("SYNTH", i.astype(str))),
        "FIRSTNAME": np.where(org, "", "TEST"),
        "MIDNAME": blank,
        "BUSNAME": np.where(org, np.char.add("SYNTHETIC ORG ", i.astype(str)), ""),
        "GENERAL": np.where(org, "BUSINESS", "IND- LIC HC SERV PRO"),
        "SPECIALTY": ev["specialty"].str.to_uppercase().to_numpy(),
        "UPIN": blank,
        "NPI": ev["npi"].to_numpy(),
        "DOB": np.where(org, "", "19600101"),
        "ADDRESS": np.full(n_ev, "1 MAIN ST", dtype=object),
        "CITY": np.full(n_ev, "TOWN", dtype=object),
        "STATE": ev["state"].to_numpy(),
        "ZIP": np.full(n_ev, "00001", dtype=object),
        "EXCLTYPE": ev["excltype"].to_numpy(),
        "EXCLDATE": date_s,
        "REINDATE": np.full(n_ev, "00000000", dtype=object),   # current file: exclusions in effect only
        "WAIVERDATE": np.full(n_ev, "00000000", dtype=object),
        "WVRSTATE": blank,
    }
    j = np.arange(k)
    nonpi = {h: np.full(k, "", dtype=object) for h in HDR}
    nonpi.update({
        "LASTNAME": np.char.add("NONPI", j.astype(str)), "FIRSTNAME": np.full(k, "TEST", dtype=object),
        "GENERAL": np.full(k, "IND- LIC HC SERV PRO", dtype=object), "SPECIALTY": np.full(k, "NURSE AIDE", dtype=object),
        "NPI": np.full(k, "0000000000", dtype=object), "DOB": np.full(k, "19700101", dtype=object),
        "STATE": np.full(k, "FL", dtype=object), "EXCLTYPE": rng.choice(np.array(["1128b7", "1128a1", "1128b4"]), k),
        "EXCLDATE": np.char.replace(np.datetime_as_string(_dates(rng, rng.integers(2017, 2027, k)), unit="D"), "-", ""),
        "REINDATE": np.full(k, "00000000", dtype=object), "WAIVERDATE": np.full(k, "00000000", dtype=object),
    })
    df = pl.concat([
        pl.DataFrame({h: pl.Series(h, cols[h].astype(str), dtype=pl.String) for h in HDR}),
        pl.DataFrame({h: pl.Series(h, nonpi[h].astype(str), dtype=pl.String) for h in HDR}),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    return path


# ------------------------------------------------------------------------------------------- entry point

def make_synthetic(proj: Paths, *, n_npi: int = 1200, years: Iterable[int] = range(2018, 2025), seed: int = 0,
                   excluded_frac: float = 0.04) -> dict:
    """Write synthetic harmonised interim files + LEIE (registered) under `proj`, run labels, return the truth.

    Returned dict: npis, dme_only_npis, leie_only_npis, years, providers (static attributes), exclusions (LEIE
    rows with an NPI), planted (one row per planted provider-year: npi, year, source, schemes), schemes
    {npi: tuple}, scheme_years {npi: tuple}, excluded_npis (Part B NPIs in the LEIE file), primary_npis,
    unlabelled_npis, n_leie_no_npi, files {dataset: [paths]}, labels (run_labels summary)."""
    years = sorted(int(y) for y in years)
    if n_npi < 50 or not years:
        raise ValueError("need n_npi >= 50 and at least one year")
    rng = np.random.default_rng([seed, 0])
    cfg = labels.LabelConfig.load(proj)
    n_dme_only, n_leie_only = max(2, round(0.02 * n_npi)), max(2, round(0.003 * n_npi))
    all_npis = make_npis(rng, n_npi + n_dme_only + n_leie_only)
    pop = _population(rng, n_npi, years, all_npis[:n_npi])
    dme_only = (all_npis[n_npi:n_npi + n_dme_only], np.rint(rng.lognormal(np.log(150), 0.5, n_dme_only)))
    profs = _profiles(rng, pop)
    plan = _plan(rng, pop, years, excluded_frac, all_npis[n_npi + n_dme_only:], cfg)
    schema = load_schema(proj)

    files: dict[str, list[str]] = {d: [] for d in ("cms_ps", "cms_prov", "dme_s", "dme_p")}
    planted_rows = []
    for y in years:
        yr = np.random.default_rng([seed, 1, y])
        active = ((pop.first <= y) & (y <= pop.last) & (yr.random(n_npi) > 0.03)
                  & ((y < pop.stop) | ((y == pop.stop) & pop.stop_partial)))
        vol = 0.82 if y == 2020 else 1 + 0.01 * (y - 2018)
        nb = np.maximum(1, np.rint(pop.nb * vol * yr.lognormal(0, 0.12, n_npi))).astype(np.int64)
        in_win = (plan.win_first <= y) & (y <= plan.win_last)
        ramp = RAMP[np.clip(plan.win_last + 1 - y, 1, 3)]
        inten = np.where(in_win, plan.intensity * ramp, 0.0)
        sch = plan.schemes * inten[:, None]
        hit = np.flatnonzero(in_win & active & plan.schemes.any(1))
        if hit.size:
            planted_rows.append(pl.DataFrame({
                "npi": pop.npi[hit], "year": np.full(hit.size, y, dtype=np.int32),
                "source": plan.source[hit].astype(str),
                "schemes": [[SCHEMES[j] for j in np.flatnonzero(r)] for r in plan.schemes[hit]],
                "intensity": inten[hit],
            }))
        ps, prov = _part_b(yr, y, pop, profs, active, nb, sch)
        ds, dp = _dme(yr, pop, profs, active, nb, sch[:, 4], dme_only)
        for name, df, grain in (("cms_ps", ps, ["npi", "hcpcs_cd", "place_of_srvc"]), ("cms_prov", prov, ["npi"]),
                                ("dme_s", ds, ["npi", "hcpcs_cd", "suplr_rentl_ind"]), ("dme_p", dp, ["npi"])):
            out = _write(proj, name, y, _conform(df.sort(grain), schema[name], y), seed)
            files[name].append(proj.rel(out))

    src = proj.root / "synthetic_src" / "UPDATED.csv"
    _leie_csv(np.random.default_rng([seed, 2]), plan, src)
    acquire.register(proj, src, "leie", 2026, "current", as_of=AS_OF, move=True,
                     notes="synthetic test data (tests/synth.py); not the OIG file")
    lab = labels.run_labels(proj, cfg)

    planted = (pl.concat(planted_rows) if planted_rows else
               pl.DataFrame(schema={"npi": pl.String, "year": pl.Int32, "source": pl.String,
                                    "schemes": pl.List(pl.String), "intensity": pl.Float64}))
    by_npi = planted.group_by("npi").agg(pl.col("schemes").first(), pl.col("year").sort()).sort("npi")
    ev = plan.events
    primary_raw = [r for r, _ in AUTHORITIES if cfg.excltype_map[labels.normalise_excltype(r)] in cfg.tiers["primary"]]
    providers = pl.DataFrame({
        "npi": pop.npi, "prvdr_type": [SPECS[s].name for s in pop.spec], "ent_cd": pop.ent,
        "state": pop.state, "ruca": pl.Series(list(pop.ruca), dtype=pl.String),
    })
    return {
        "seed": seed, "years": years, "n_npi": n_npi,
        "npis": sorted(pop.npi.tolist()),
        "dme_only_npis": sorted(dme_only[0].tolist()),
        "leie_only_npis": sorted(all_npis[n_npi + n_dme_only:].tolist()),
        "providers": providers,
        "exclusions": ev.select("npi", "excltype", "excl_date", "in_part_b"),
        "excluded_npis": sorted(set(ev.filter("in_part_b")["npi"].to_list())),
        "primary_npis": sorted(set(ev.filter(pl.col("in_part_b") & pl.col("excltype").is_in(primary_raw))["npi"].to_list())),
        "unlabelled_npis": sorted(pop.npi[plan.source == "unlabelled"].tolist()),
        "planted": planted,
        "schemes": {r["npi"]: tuple(r["schemes"]) for r in by_npi.to_dicts()},
        "scheme_years": {r["npi"]: tuple(r["year"]) for r in by_npi.to_dicts()},
        "n_leie_no_npi": plan.no_npi,
        "files": files,
        "labels": lab,
    }
