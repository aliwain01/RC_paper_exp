import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from vizlib import *

OUT = sys.argv[1]

# ---------------------------------------------------------------- Fig 1: system architecture
def fig1():
    W, H = 21, 10.3
    fig, ax = canvas(W, H, "Fig. 1 — End-to-end architecture",
                     "Public Medicare data → provider-year table → glass-box and black-box models → scenario-level attribution → evaluation → audience-specific views")
    lanes = [
        ("1  Public sources", C["blue"]),
        ("2  Preparation", C["aqua"]),
        ("3  Analysis datasets", C["yellow"]),
        ("4  Models", C["violet"]),
        ("5  Scenario attribution", C["orange"]),
        ("6  Evaluation", C["green"]),
        ("7  Outputs", C["magenta"]),
    ]
    lw, gap, x0 = 2.62, 0.30, 0.35
    xs = [x0 + i * (lw + gap) for i in range(len(lanes))]
    top = H - 1.35
    for (lab, col), x in zip(lanes, xs):
        lane_header(ax, x, top - 0.48, lw, lab, col)
        ax.add_patch(FancyBboxPatch((x - 0.06, 0.62), lw + 0.12, top - 0.6 - 0.62,
                                    boxstyle="round,pad=0,rounding_size=0.1", fc="none",
                                    ec=GRID, lw=1.0, zorder=1))
    y0 = top - 0.68
    cw = lw - 0.12

    def col(i, items, spacing=0.16, down_arrows=False):
        y = y0
        x = xs[i] + 0.06
        bottoms = []
        for t, lines, *rest in items:
            dashed = rest[0] if rest else False
            yb = card(ax, x, y, cw, t, lines, lanes[i][1], dashed=dashed, fs=8.1, tfs=9.0)
            bottoms.append((y, yb))
            if down_arrows and len(bottoms) > 1:
                pass
            y = yb - spacing
        if down_arrows:
            for (t1, b1), (t2, b2) in zip(bottoms, bottoms[1:]):
                arrow(ax, (x + cw / 2, b1), (x + cw / 2, t2), color=lanes[i][1], lw=1.2, ms=9)
        return bottoms

    col(0, [
        ("Part B — by Provider & Service", ["NPI × HCPCS × place of service", "× year: services, patients,", "charges, allowed, paid", "2013 – 2024"]),
        ("Part B — by Provider", ["NPI × year summary:", "totals + patient mix", "(age, dual, chronic conds,", "avg. HCC risk score)"]),
        ("DMEPOS — Referring Provider", ["(+ & Service): referring NPI ×", "HCPCS × rental × year, plus per-", "NPI summary (distinct patients)", "2013/14 – 2024*"]),
        ("OIG LEIE + archives", ["current file (in-effect only),", "monthly supplements (≈12 mo),", "Internet Archive snapshots"]),
        ("NPPES registry (optional)", ["enumeration date, entity type,", "taxonomy, address"], True),
    ])
    col(1, [
        ("Ingest & version", ["raw CSV → Parquet", "download date + checksum", "verify data dictionaries"]),
        ("Cohort & join", ["unit = NPI-year", "min-volume threshold", "peer-group hierarchy"]),
        ("Label from LEIE", ["fraud-related authorities", "positive: excluded within", "W = 3 yrs after billing year", "reinstatement recovery + gaps"]),
        ("Peer-relative features", ["peer cell: specialty × entity", "× year (× region)", "label-free stats, per year"]),
        ("Assign scenario groups", ["7 scheme groups + context", "fixed a priori, 1 group/feature"]),
    ], down_arrows=True)
    col(2, [
        ("Real provider-year table", ["~1.1–1.3 M providers / year", "~35–50 features", "positive count = Phase 1", "go / no-go gate"]),
        ("Temporal × provider split", ["5-fold provider cross-fitting", "train 2018–21 · test 2022–23", "showcase 2024 (immature labels)", "+ rolling-origin repeats"]),
        ("Scheme-injected copy", ["schemes applied to raw rows", "of clean providers; totals", "updated, suppression re-applied", "→ known scheme (ground truth)"]),
    ])
    col(3, [
        ("Logistic regression", ["regularised, scenario features", "exact contributions"]),
        ("EBM — main effects", ["GAM: one shape per feature", "exact contributions"]),
        ("EBM — GA²M", ["+ selected pairwise terms", "exact, named interactions"]),
        ("Gradient boosting", ["XGBoost / LightGBM", "± monotone constraints", "calibrated (isotonic)"]),
        ("Shared settings", ["one-hot categoricals; same", "negative subsample for all"], True),
    ])
    col(4, [
        ("A  Summed SHAP", ["Σ feature SHAP per group", "cross-group interactions split ½"]),
        ("B  Group Shapley", ["exact over 8 groups from the", "coalition table (256 subsets)"]),
        ("C  Group interaction matrix", ["Shapley interaction index from", "same table: pure + compound"]),
        ("D  Glass-box native", ["LR / EBM terms grouped", "reference explanation"]),
        ("E  Peer counterfactual", ["v(N) − v(N∖G) from the same", "table; Δ risk (non-additive)"]),
        ("Baseline = peers", ["≤200 same-year, same-peer-cell", "training-fold providers"], True),
    ], spacing=0.13)
    col(5, [
        ("Detect", ["precision@k, recall@k (audit", "capacity), PR-AUC, ROC-AUC", "paired NPI bootstrap CIs"]),
        ("Correct", ["injected scheme recovered?", "top-1 / top-3 / pair recovery"]),
        ("Consistent", ["GBM vs glass box: sign,", "rank, top-3 overlap, share cos."]),
        ("Faithful", ["group-deletion curves", "vs random order"]),
        ("Stable", ["seeds / bootstraps / grouping", "variants / context ablation"]),
    ])
    col(6, [
        ("Analyst / investigator view", ["risk rank, scenario indicators,", "compound pairs, peer stats,", "drill-down to features"]),
        ("Decision record", ["investigator decision + rationale", "linked to model version and", "attribution snapshot"]),
        ("Provider / member notice", ["ADR-style request, policy", "reason, peer comparison (CBR", "format); no score / detection logic"]),
        ("Oversight & paper", ["global scenario importance,", "fairness audit, model card,", "results tables and figures"]),
    ])

    # inter-lane chevrons
    for x in xs[1:]:
        for yy in (7.3, 4.6, 2.2):
            chevron(ax, x - gap + 0.04, yy, color=MUTED, size=0.13)

    # feedback loop
    arrow(ax, (xs[6] + lw / 2, 0.85), (xs[1] + lw / 2, 0.85), color=C["magenta"], ls=(0, (4, 3)), lw=1.2, rad=0.0)
    ax.text((xs[1] + xs[6]) / 2 + lw / 2, 0.91, "future work: confirmed investigation outcomes become labels (not available in public data)",
            ha="center", va="bottom", fontsize=8.2, color=INK2, style="italic")
    footnote(ax, W, "* DMEPOS 2024 listed in the data.gov catalogue (Jul 2026); re-checked in Phase 1. Dashed = optional / shared setting. ADR = Additional Documentation Request; CBR = CMS Comparative Billing Report.", y=0.18)
    save(fig, f"{OUT}/fig01_system_architecture.png")


# ---------------------------------------------------------------- Fig 2: data model
def fig2():
    W, H = 19, 11.2
    fig, ax = canvas(W, H, "Fig. 2 — Data model: from public tables to one row per provider-year",
                     "Every source is reduced to the NPI-year grain before joining. Field names are indicative — confirm against each year's CMS data dictionary.")
    xs, xm, xr = 0.35, 6.9, 12.9
    ws, wm, wr = 5.9, 5.3, 5.75
    ax.text(xs, H - 1.35, "SOURCE TABLE  (native grain)", fontsize=9.5, color=MUTED, fontweight="bold")
    ax.text(xm, H - 1.35, "REDUCTION TO NPI-YEAR", fontsize=9.5, color=MUTED, fontweight="bold")
    ax.text(xr, H - 1.35, "ANALYSIS TABLE  (NPI × year)", fontsize=9.5, color=MUTED, fontweight="bold")
    srcs = [
        ("Part B by Provider & Service", C["blue"], ["grain: NPI × HCPCS × POS(F/O) × year",
         "Rndrng_NPI, Rndrng_Prvdr_Type, _Ent_Cd,", "_State_Abrvtn, _RUCA, HCPCS_Cd,",
         "HCPCS_Drug_Ind, Place_Of_Srvc, Tot_Benes,", "Tot_Srvcs, Tot_Bene_Day_Srvcs, Avg_Sbmtd_Chrg,",
         "Avg_Mdcr_Alowd_Amt, Avg_Mdcr_Pymt_Amt,", "Avg_Mdcr_Stdzd_Amt   (~9–10 M rows / yr)"],
         ["E&M level shares (99211–15, 99202–05,", "99281–85) · services / patient ·",
          "HCPCS mix vector → divergence vs specialty ·", "charge ratio · standardised pay / patient"]),
        ("Part B by Provider", C["blue"], ["grain: NPI × year",
         "Tot_HCPCS_Cds, Tot_Benes, Tot_Srvcs, Tot_*_Amt,", "Bene_Avg_Age, Bene_Dual_Cnt, Bene_Feml_Cnt,",
         "Bene_CC_*_Pct (chronic conditions),", "Bene_Avg_Risk_Scre   (~1.2–1.3 M rows / yr)"],
         ["patient acuity (risk score, chronic %)", "intensity residual vs acuity · dual share ·", "practice size · data coverage (context)"]),
        ("DMEPOS by Referring Provider & Service  (+ summary file)", C["blue"], ["grain: referring NPI × HCPCS × rental × year",
         "Rfrg_NPI, RBCS_Id, HCPCS_Cd, Suplr_Rentl_Ind,", "Tot_Suplrs, Tot_Suplr_Benes, Tot_Suplr_Clms,",
         "Tot_Suplr_Srvcs, Avg_Suplr_Mdcr_Pymt_Amt …", "+ by Referring Provider: distinct Tot_Suplr_Benes"],
         ["DME paid / DME patient · share of panel", "referred · high-risk category share ·",
          "claims per supplier (approx. concentration)", "· rental share · no-DME indicator"]),
        ("OIG LEIE  (+ supplements ≈12 mo, + archive snapshots)", C["red"], ["grain: excluded person / entity",
         "NPI, EXCLTYPE, EXCLDATE, REINDATE,", "name / address fields (OIG record layout)"],
         ["keep fraud-related EXCLTYPE · NPI match", "label(NPI, t) = 1 if excluded in (t, t+W]",
          "drop t ≥ excl. year and t < excl. year − W"]),
    ]
    y = H - 1.6
    mids = []
    for name, col, src_lines, red_lines in srcs:
        yb = card(ax, xs, y, ws, name, src_lines, col, fs=7.9, tfs=9.3, mono=False)
        ym_top = y - 0.05
        yb2 = card(ax, xm, ym_top, wm, "→ " + ("label" if "LEIE" in name else "features"), red_lines,
                   C["red"] if "LEIE" in name else C["aqua"], fs=8.2, tfs=9.0)
        arrow(ax, (xs + ws, (y + yb) / 2), (xm, (ym_top + yb2) / 2), color=INK2)
        mids.append(((ym_top + yb2) / 2))
        y = min(yb, yb2) - 0.28

    # analysis table
    blocks = [
        ("Keys", C["gray"], ["NPI · year · split tag (train / test / showcase)"]),
        ("Context — inputs, never reasons", C["gray"], ["specialty · entity type · Census division · RUCA", "size · tenure (NPPES, opt.) · data coverage"]),
        ("Scheme features (7 groups)", C["orange"], ["Upcoding · Overutilization · Atypical service mix", "Charge inflation · Patient-mix mismatch",
                                                    "DME referral intensity · DME referral concentration", "≈ 35–50 peer-relative features (label-free, per year)"]),
        ("Label", C["red"], ["y ∈ {0,1} (positive-unlabeled) · exclusion type", "label-horizon completeness flag"]),
        ("Injection metadata (copy only)", C["yellow"], ["injected ∈ {0,1} · scheme(s) · intensity"]),
    ]
    yt = H - 1.6
    ax.add_patch(FancyBboxPatch((xr - 0.12, 2.55), wr + 0.24, yt - 2.55 + 0.12, boxstyle="round,pad=0,rounding_size=0.1",
                                fc="none", ec=INK2, lw=1.3, zorder=1))
    for t, col, lines in blocks:
        yt = card(ax, xr, yt - 0.06, wr, t, lines, col, fs=8.2, tfs=9.0, dashed=(t.startswith("Context") or t.startswith("Injection"))) - 0.12
    for m in mids:
        arrow(ax, (xm + wm, m), (xr - 0.12, m), color=INK2)
    footnote(ax, W, "By-Provider 2024 = 1,296,739 rows (CMS API page); other sizes measured in Phase 1. CMS excludes rows derived from ≤10 beneficiaries, so rare codes are invisible.", y=0.3)
    save(fig, f"{OUT}/fig02_data_model.png")


# ---------------------------------------------------------------- Fig 3: label timeline
def fig3():
    W, H = 18, 8.4
    fig, ax = canvas(W, H, "Fig. 3 — Data availability, label construction and temporal split",
                     "Exclusions arrive years after the billing they punish, so the newest years look artificially clean. Accuracy is claimed only on years with a (near-)complete label horizon.")
    yr0, yr1 = 2013, 2027
    xL, xR = 4.3, W - 0.6
    X = lambda yr: xL + (yr - yr0) / (yr1 - yr0) * (xR - xL)
    # year axis
    for yr in range(yr0, yr1 + 1):
        ax.plot([X(yr)] * 2, [1.0, H - 1.35], color=GRID, lw=0.8, zorder=0)
        ax.text(X(yr) + (X(yr + 1) - X(yr)) / 2 if yr < yr1 else X(yr), H - 1.3, str(yr) if yr < yr1 else "",
                ha="center", fontsize=8.6, color=INK2)
    rowy = lambda i: H - 1.95 - i * 0.62 - (0.32 if i >= 3 else 0) - (0.32 if i >= 5 else 0)

    def bar(i, a, b, col, label, txt=None, hatch=None, alpha_amt=0.55):
        y = rowy(i)
        ax.add_patch(FancyBboxPatch((X(a) + 0.03, y - 0.19), X(b) - X(a) - 0.06, 0.38,
                                    boxstyle="round,pad=0,rounding_size=0.06", fc=tint(col, alpha_amt), ec=col, lw=1.0,
                                    hatch=hatch, zorder=2))
        if txt:
            ax.text((X(a) + X(b)) / 2, y, txt, ha="center", va="center", fontsize=8.2, color=INK, zorder=3)

    def rlabel(i, t, bold=False):
        ax.text(0.35, rowy(i), t, va="center", fontsize=9.2, color=INK, fontweight="bold" if bold else None)

    ax.text(0.35, rowy(0) + 0.42, "DATA AVAILABILITY", fontsize=8.6, color=MUTED, fontweight="bold")
    rlabel(0, "Part B (both files)"); bar(0, 2013, 2025, C["blue"], "", "published 2013 – 2024")
    rlabel(1, "DMEPOS by Referring"); bar(1, 2013, 2024, C["blue"], "", "published 2013/14 – 2023"); bar(1, 2024, 2025, C["blue"], "", "2024*", alpha_amt=0.75)
    rlabel(2, "LEIE snapshot")
    ax.plot([X(2026.78)], [rowy(2)], marker="v", ms=11, color=C["red"], zorder=4)
    ax.text(X(2026.78) - 0.12, rowy(2), "current file ≈ Oct 2026 + supplements (≈12 mo) + Internet Archive snapshots (2017, 2019 confirmed)", ha="right", va="center", fontsize=8.2, color=INK2)
    for yy in (2017.07, 2019.03):
        ax.plot([X(yy)], [rowy(2)], marker="o", ms=6, color=C["red"], zorder=4)

    ax.text(0.35, rowy(3) + 0.42, "STUDY DESIGN", fontsize=8.6, color=MUTED, fontweight="bold")
    rlabel(3, "Primary split", True)
    bar(3, 2018, 2022, C["violet"], "", "TRAIN 2018–2021 (5-fold provider cross-fit)")
    bar(3, 2022, 2024, C["green"], "", "TEST  2022 – 2023")
    bar(3, 2024, 2025, C["yellow"], "", "SHOWCASE")
    rlabel(4, "Label horizon (W = 3)")
    bar(4, 2018, 2023, C["green"], "", "complete: exclusions through t+3 observed", alpha_amt=0.75)
    bar(4, 2023, 2024, C["yellow"], "", "≈ 2.8 y", alpha_amt=0.7)
    bar(4, 2024, 2025, C["red"], "", "≈ 1.8 y", alpha_amt=0.7)

    ax.text(0.35, rowy(5) + 0.42, "LABEL RULE — EXAMPLES", fontsize=8.6, color=MUTED, fontweight="bold")
    rlabel(5, "Provider A (excluded Mar 2023)")
    bar(5, 2016, 2020, C["gray"], "", "dropped: ambiguous (t < yE − W)", alpha_amt=0.85)
    bar(5, 2020, 2023, C["red"], "", "POSITIVE  t = 2020 – 2022")
    ax.plot([X(2023.2)], [rowy(5)], marker="X", ms=11, color=C["red"], zorder=4)
    ax.text(X(2023.2) + 0.15, rowy(5), "exclusion (EXCLTYPE fraud-related); t ≥ 2023 dropped", va="center", fontsize=8.2, color=INK2)
    rlabel(6, "Provider B (never excluded)")
    bar(6, 2016, 2025, C["blue"], "", "NEGATIVE = unlabeled, not proven clean (positive-unlabeled setting)", alpha_amt=0.8)
    rlabel(7, "Provider C (scheme starts 2024)")
    bar(7, 2018, 2024, C["blue"], "", "negative (true)", alpha_amt=0.8)
    bar(7, 2024, 2025, C["yellow"], "", "looks clean", alpha_amt=0.6)
    ax.text(X(2025) + 0.12, rowy(7), "exclusion ≈ 2027+\n→ not yet in LEIE", va="center", fontsize=8.2, color=INK2)

    ax.text(0.35, 0.62, "Rule:  row (NPI, t) is positive if the provider's first fraud-related exclusion year yE satisfies  t < yE ≤ t + W.   Rows with t ≥ yE are dropped (barred / partial year);  rows with t < yE − W are dropped (fraud onset unknown).",
            fontsize=8.6, color=INK)
    ax.text(0.35, 0.3, "W = 3 is an assumption (onset unobservable); sensitivity W ∈ {1, 2, 5}. Rolling origins: 2018–20 → 2021, 2018–21 → 2022, 2018–22 → 2023. 2020 flagged (COVID). * DMEPOS 2024 per data.gov catalogue; re-checked in Phase 1.",
            fontsize=8.2, color=MUTED)
    save(fig, f"{OUT}/fig03_label_timeline.png")


if __name__ == "__main__":
    which = sys.argv[2:] or ["1", "2", "3"]
    for w in which:
        globals()[f"fig{w}"]()
