import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from vizlib import *
import numpy as np

OUT = sys.argv[1]

GROUPS = [
    ("1  Upcoding", "billing higher-paying codes than delivered",
     ["share of top-level office visits (99215 | 99211–15)", "share of top-level new-patient visits (99205)",
      "share of top-level ED visits (99285)", "allowed $ per E&M service vs peers"], "Part B · P&S"),
    ("2  Overutilization", "more services than the patients need",
     ["non-drug services per patient vs peers", "beneficiary-days per patient", "units per beneficiary-day",
      "distinct HCPCS per patient · volume (log)"], "Part B · P&S + summary"),
    ("3  Atypical service mix", "billing what the specialty rarely bills",
     ["divergence of HCPCS mix from peer cell (JS)", "share of services in codes rare for peers",
      "concentration of billing in few codes (HHI)", "drug-code payment share vs peers"], "Part B · P&S"),
    ("4  Charge inflation", "prices out of line with peers",
     ["submitted ÷ allowed charge ratio vs peers", "standardised payment per patient vs peers",
      "standardised payment per service vs peers", "code-adjusted price index"], "Part B · P&S + summary"),
    ("5  Patient-mix mismatch", "intensity not explained by how sick patients are",
     ["avg. HCC risk score vs peers", "services per patient ÷ risk score (residual)",
      "payment per patient ÷ risk score (residual)", "dual-eligible share vs peers"], "Part B · summary"),
    ("6  DME referral intensity", "orders more equipment than peers",
     ["DME paid per DME patient vs peers", "share of panel referred for DME",
      "high-risk category share (orthotics, power", "  mobility) · claims / patient · no-DME flag"], "DMEPOS · service + summary files"),
    ("7  DME referral concentration", "referrals funnelled to few suppliers",
     ["claims per supplier vs peers (approximate:", "  supplier identities are not public)",
      "rental share (direction learned)"], "DMEPOS · service file"),
]


def fig4():
    W, H = 19, 8.6
    fig, ax = canvas(W, H, "Fig. 4 — Scenario taxonomy: seven scheme groups plus a context group",
                     "Each feature belongs to exactly one group, fixed before modelling. Scheme groups can be shown as reasons; the context group is a model input that is never shown as a reason.")
    cw, gx = 4.45, 0.22
    cols = [C["orange"], C["orange"], C["orange"], C["orange"], C["orange"], C["blue"], C["blue"]]
    pos = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)]
    ytops = [H - 1.35, H - 1.35 - 2.75 - 0.25]
    for (name, meaning, feats, src), col, (cx, cy) in zip(GROUPS, cols, pos):
        x = 0.35 + cx * (cw + gx)
        lines = ["!" + meaning, ""] + ["· " + f if not f.startswith("  ") else "  " + f.strip() for f in feats] + ["", "source: " + src]
        card(ax, x, ytops[cy], cw, name, [l.lstrip("!") if False else l for l in lines], col, fs=8.4, tfs=10.5, h=2.75)
    # context
    x = 0.35 + 3 * (cw + gx)
    card(ax, x, ytops[1], cw, "Context  (input, never a reason)",
         ["!defines the peer group; absorbs structural", "!differences so scheme features stay clean", "",
          "· specialty · entity type (individual / org.)", "· Census division · RUCA (urban–rural)",
          "· Medicare participation · practice size", "· tenure (NPPES, opt.) · data coverage", "  (share surviving suppression)",
          "", "matched traits: attribution = 0 (peer baseline)"], C["gray"], fs=8.4, tfs=10.5, h=2.75, dashed=True)
    ax.text(0.35, 0.95, "Direction: higher = more suspicious for most scheme features (monotone candidates; exceptions in catalog). All features are peer-relative (robust z within specialty × entity × year, × region if ≥200 peers), using label-free per-year statistics.",
            fontsize=8.6, color=INK)
    ax.text(0.35, 0.55, "Orange = Part B-derived scheme groups; blue = DMEPOS-derived. Feature list is the v1.1 proposal — see FEATURE_CATALOG.md for formulas, data fields and rationale.",
            fontsize=8.2, color=MUTED)
    save(fig, f"{OUT}/fig04_scenario_taxonomy.png")


def fig5():
    W, H = 21, 10.8
    fig, ax = canvas(W, H, "Fig. 5 — Five ways to compute a scenario-level attribution",
                     "They answer different questions, so the study computes all five and measures which is correct, consistent, faithful and stable. Mini-plots are illustrative, not results.")
    cw, gx = 3.92, 0.2
    meths = [
        ("A  Summed SHAP", C["orange"], [
            "!Q: how much of the score does", "!   each group's features carry?",
            "", "Φ_G = Σ_{j∈G} φ_j", "within-group interactions → G", "cross-group interactions → ½ each",
            "", "additive: yes (sums to f(x) − E[f])", "cost: one TreeSHAP pass", "role: current baseline method"]),
        ("B  Group Shapley", C["orange"], [
            "!Q: what is each group worth when", "!   groups are the players?",
            "", "coalition table: v(S) for all 256 subsets", "v(S) = mean_B f(x_S, z_¬S)", "Φ_G = exact Shapley over 8 groups",
            "", "additive: yes", "cost: 256 × |B| predictions / provider", "role: principled group attribution"]),
        ("C  Group interaction matrix", C["orange"], [
            "!Q: which groups act alone and", "!   which act together?",
            "", "I_GH: Shapley interaction index (same table)", "M_GH = I_GH/2;  M_GG = Φ_G − Σ_H M_GH", "diag = pure, off-diag = compound",
            "", "additive: yes (all cells sum)", "cost: free once table exists", "role: analyst output"]),
        ("D  Glass-box native", C["violet"], [
            "!Q: what does a model that is exact", "!   by construction say?",
            "", "LR: β_j (x_j − x̄_j^peer)", "EBM: f_j(x_j) − E_peer[f_j]", "pair terms: in-group → G; cross → cell",
            "", "additive: yes, exactly", "cost: free", "role: reference for consistency"]),
        ("E  Peer counterfactual", C["green"], [
            "!Q: how much would risk drop if this", "!   group looked like a typical peer?",
            "", "Δ_G = v(N) − v(N∖{G}) (same table)", "group G alone takes peer values", "report in probability units",
            "", "additive: no", "cost: free once table exists", "role: actionable / analyst view"]),
    ]
    yt = H - 1.35
    for i, (name, col, lines) in enumerate(meths):
        x = 0.35 + i * (cw + gx)
        card(ax, x, yt, cw, name, [l.lstrip("!") for l in lines], col, fs=8.4, tfs=10.3, h=3.15)
        # mini plot area
        px, py, pw, ph = x + 0.25, 1.75, cw - 0.5, 3.6
        ax.add_patch(FancyBboxPatch((x, 1.45), cw, 4.25, boxstyle="round,pad=0,rounding_size=0.08", fc=SURFACE, ec=GRID, lw=1))
        ax.text(x + 0.15, 5.55, "illustrative", fontsize=7.8, color=MUTED, style="italic", va="top")
        gl = ["Upc", "Ovr", "Mix", "Chg", "Pmx", "DMEi", "DMEc"]
        if i in (0, 1, 3):
            vals = {0: [1.3, .5, .2, .3, .15, -.05, .15], 1: [1.15, .45, .25, .3, .2, -.05, .2], 3: [1.0, .6, .15, .35, .1, -.1, .1]}[i]
            mx = 1.5
            for k, (g, v) in enumerate(zip(gl, vals)):
                yy = 5.05 - k * 0.47
                ax.text(px + 0.55, yy, g, ha="right", va="center", fontsize=7.8, color=INK2)
                x0 = px + 1.55
                ax.plot([x0, x0], [1.85, 5.25], color=GRID, lw=1)
                wbar = v / mx * (pw - 1.7)
                ax.add_patch(FancyBboxPatch((x0 + min(0, wbar), yy - 0.15), abs(wbar), 0.3, boxstyle="round,pad=0,rounding_size=0.04",
                                            fc=C["red"] if v > 0 else C["blue"], ec="none"))
        elif i == 2:
            M = np.array([[.7, .35, 0, .05, .1, 0, 0], [.35, .2, 0, 0, .05, 0, 0], [0, 0, .15, 0, 0, 0, 0], [.05, 0, 0, .25, 0, 0, 0],
                          [.1, .05, 0, 0, .05, 0, 0], [0, 0, 0, 0, 0, -.05, .1], [0, 0, 0, 0, 0, .1, .05]])
            cs = (pw - 0.7) / 7
            for r in range(7):
                ax.text(px + 0.55, 5.0 - r * cs - cs / 2 + 0.15, gl[r], ha="right", va="center", fontsize=7.4, color=INK2)
                ax.text(px + 0.65 + r * cs + cs / 2, 5.25, gl[r], ha="center", va="bottom", fontsize=6.6, color=INK2, rotation=0)
                for c in range(7):
                    v = M[r, c]
                    colr = tint(C["red"], 1 - min(abs(v) / .7, 1) * 0.9) if v >= 0 else tint(C["blue"], 1 - abs(v) / .7 * 0.9)
                    ax.add_patch(Rectangle((px + 0.65 + c * cs, 5.15 - (r + 1) * cs), cs - 0.03, cs - 0.03, fc=colr, ec="none"))
            ax.text(px + pw / 2 + 0.3, 1.65, "Upcoding × Overutilization = compound", ha="center", fontsize=7.6, color=INK2)
        elif i == 4:
            bars = [("current risk", 0.047, INK2), ("Upcoding → peer", 0.012, C["green"]), ("Overutil. → peer", 0.031, C["green"]), ("Charge → peer", 0.040, C["green"])]
            for k, (lab, v, colr) in enumerate(bars):
                yy = 5.0 - k * 0.8
                ax.text(px, yy + 0.25, lab, fontsize=7.8, color=INK2, va="bottom")
                ax.add_patch(FancyBboxPatch((px, yy - 0.12), v / 0.05 * (pw - 0.6), 0.3, boxstyle="round,pad=0,rounding_size=0.04", fc=colr, ec="none"))
                ax.text(px + v / 0.05 * (pw - 0.6) + 0.08, yy + 0.03, f"{v:.1%}", fontsize=7.8, color=INK, va="center")
    ax.text(0.35, 1.0, "Shared design choices:  (1) background = ≤200 same-year, same-peer-cell, training-fold providers (interventional);  (2) all additive methods compared in log-odds, communicated in probability;",
            fontsize=8.6, color=INK)
    ax.text(0.35, 0.68, "(3) B, C, E and deletion curves share one coalition table, computed on the explained set (top 2 %, random 5,000, all injected).   Red = raises risk, blue = lowers risk.",
            fontsize=8.6, color=INK)
    save(fig, f"{OUT}/fig05_attribution_methods.png")


def fig6():
    W, H = 18, 8.6
    fig, ax = canvas(W, H, "Fig. 6 — Why the baseline matters: global vs. peer reference (illustrative numbers)",
                     "Same provider, same model, same final score. Against all providers, 'being a cardiologist' looks like a reason; against cardiology peers, context drops out and only scheme behaviour remains.")
    names = ["Context", "Upcoding", "Overutilization", "Charge inflation", "Atypical mix", "Patient-mix", "DME intensity", "DME concentration"]
    left = [0.95, 1.35, 0.55, 0.30, 0.20, 0.15, -0.05, 0.05]
    right = [0.05, 1.30, 0.50, 0.30, 0.20, 0.15, -0.05, 0.15]
    sig = lambda z: 1 / (1 + np.exp(-z))
    for pi, (base, vals, ttl) in enumerate([(-6.5, left, "Baseline = average of ALL providers"), (-5.6, right, "Baseline = cardiology peers, same region")]):
        x0 = 0.35 + pi * 9.0
        pw = 8.4
        ax.text(x0, H - 1.45, ttl, fontsize=11, fontweight="bold", color=INK, va="top")
        lo, hi = -7.0, -1.9
        X = lambda v: x0 + 2.0 + (v - lo) / (hi - lo) * (pw - 2.3)
        for t in np.arange(-7, -1.95, 1.0):
            ax.plot([X(t)] * 2, [1.35, H - 2.05], color=GRID, lw=0.8, zorder=0)
            ax.text(X(t), 1.12, f"{t:.0f}", ha="center", fontsize=8, color=INK2)
        ax.text(X(-4.75), 0.78, "log-odds of fraud-related exclusion", ha="center", fontsize=8.2, color=INK2)
        cum = base
        order = sorted(range(8), key=lambda k: -abs(vals[k]))
        for r, k in enumerate(order):
            v = vals[k]; y = H - 2.35 - r * 0.62
            colr = C["gray"] if names[k] == "Context" else (C["red"] if v > 0 else C["blue"])
            ax.add_patch(FancyBboxPatch((X(min(cum, cum + v)), y - 0.2), max(abs(X(cum + v) - X(cum)), 0.02), 0.4,
                                        boxstyle="round,pad=0,rounding_size=0.04", fc=colr, ec="none", zorder=3))
            ax.text(x0 + 1.9, y, names[k], ha="right", va="center", fontsize=8.8, color=INK, fontweight="bold" if names[k] == "Context" else None)
            ax.text(X(max(cum, cum + v)) + 0.08, y, f"{v:+.2f}", va="center", fontsize=8.2, color=INK2)
            if r < 7:
                ax.plot([X(cum + v)] * 2, [y - 0.2, y - 0.42], color=MUTED, lw=0.8)
            cum += v
        ax.plot([X(base)] * 2, [1.35, H - 2.05], color=INK2, ls=(0, (3, 3)), lw=1)
        ax.plot([X(cum)] * 2, [1.35, H - 2.05], color=INK, lw=1.3)
        ax.text(X(base) - 0.05, H - 1.95, f"E[f] = {base:.1f}  (p = {sig(base):.2%})", ha="right", fontsize=8.2, color=INK2)
        ax.text(X(cum) + 0.05, H - 1.95, f"f(x) = {cum:.1f}  (p = {sig(cum):.1%})", ha="left", fontsize=8.2, color=INK)
    ax.text(0.35, 0.3, "Gray = context group, red = raises risk, blue = lowers risk. Context is reported to investigators as 'peer group: cardiology, TX' — never as a reason.",
            fontsize=8.2, color=MUTED)
    save(fig, f"{OUT}/fig06_peer_baseline.png")


def fig7():
    W, H = 20, 8.6
    fig, ax = canvas(W, H, "Fig. 7 — Scheme injection: creating ground truth for explanation correctness",
                     "Real providers, real interactions — with a documented scheme applied to their raw aggregates, so the 'right answer' for the explanation is known.")
    steps = [
        ("Clean pool", C["blue"], ["never-excluded providers,", "scheme-applicable, by fold", "(stratified by specialty)"]),
        ("Assign scheme(s)", C["yellow"], ["single scheme, or", "two at once (compound)", "intensity: low / med / high"]),
        ("Edit RAW rows", C["orange"], ["service rows + summary totals", "updated consistently; CMS", "suppression re-applied"]),
        ("Re-run pipeline", C["aqua"], ["same per-year peer stats", "(injected excluded), same", "feature code and groups"]),
        ("Explain", C["orange"], ["methods A–E on every", "injected provider"]),
        ("Score recovery", C["green"], ["top-1 / top-3 hit vs injected", "pair recovery (method C)", "vs chance = 1/7"]),
    ]
    cw, gx = 3.0, 0.3
    yt = H - 1.4
    for i, (t, col, lines) in enumerate(steps):
        x = 0.35 + i * (cw + gx)
        yb = card(ax, x, yt, cw, t, lines, col, fs=8.5, tfs=10, h=1.2)
        if i:
            arrow(ax, (x - gx + 0.02, yt - 0.6), (x - 0.02, yt - 0.6), color=INK2, lw=1.4)
    # two uses
    card(ax, 0.35, yt - 1.45, 9.6, "Use (a) — detection stress test",
         ["model trained on LEIE labels only; does it rank injected providers highly?",
          "informative but may legitimately fail if LEIE fraud ≠ injected patterns"], C["violet"], fs=8.5, tfs=9.8, h=1.05)
    card(ax, 10.2, yt - 1.45, 9.45, "Use (b) — explanation correctness (primary)",
         ["injection model: injected = 1 vs clean = 0 (LEIE positives removed); train folds 2018–21 → test folds 2022–23",
          "the injected scheme is the answer key: a necessary, not sufficient, test"], C["green"], fs=8.5, tfs=9.8, h=1.05)

    # table
    ty = yt - 2.8
    hdr = ["Scheme", "Raw-aggregate edit (examples)", "Expected primary group", "Expected spill-over"]
    cx = [0.35, 3.4, 11.2, 15.0]
    for h_, x in zip(hdr, cx):
        ax.text(x, ty, h_, fontsize=9.2, fontweight="bold", color=INK, va="top")
    ax.plot([0.35, W - 0.35], [ty - 0.32, ty - 0.32], color=INK2, lw=0.9)
    rows = [
        ("Upcoding", "move p% of 99212–13 services to 99214–15; recompute allowed / paid", "Upcoding", "Charge inflation (payment per service)"),
        ("Overutilization", "scale services per patient by ×m on existing codes", "Overutilization", "Charge (payment / patient), Patient-mix"),
        ("Atypical services", "add services in codes rare for the specialty", "Atypical service mix", "Overutilization"),
        ("Charge inflation", "scale submitted charges by ×c (allowed unchanged)", "Charge inflation", "—"),
        ("DME mill referrals", "add high-risk DME rows concentrated on 1–2 suppliers", "DME intensity + concentration", "—"),
        ("Compound", "two of the above together (e.g. Upcoding + Overutilization)", "both groups + the pair cell in C", "as above"),
    ]
    for r, row in enumerate(rows):
        y = ty - 0.5 - r * 0.42
        if r % 2 == 0:
            ax.add_patch(Rectangle((0.3, y - 0.3), W - 0.6, 0.42, fc=tint(C["gray"], 0.93), ec="none", zorder=0))
        for v, x in zip(row, cx):
            ax.text(x, y, v, fontsize=8.5, color=INK if x == cx[0] else INK2, va="top")
    ax.text(0.35, 0.62, "Circularity guard: edits go through the real feature code, so schemes have spill-over. Methods are differentiated mainly by compound, spill-over and distractor cases.",
            fontsize=8.6, color=INK)
    ax.text(0.35, 0.3, "Injected providers never appear in the real-data performance results. Intensities and rates are parameters reported with every result.",
            fontsize=8.2, color=MUTED)
    save(fig, f"{OUT}/fig07_scheme_injection.png")


def fig8():
    W, H = 20, 10.4
    fig, ax = canvas(W, H, "Fig. 8 — Evaluation framework and decision logic",
                     "Performance decides which model is worth explaining; four explanation tests decide whether its explanations can be trusted.")
    qs = [
        ("Detect — is the black box worth it?", C["violet"], ["precision@k, recall@k at k = 0.1, 0.5, 1, 2 % of providers", "PR-AUC (primary), ROC-AUC, lift over base rate",
                                                             "5-fold provider cross-fitting; paired NPI bootstrap", "(1,000×), Holm-corrected; per origin + pooled"]),
        ("Correct — does it name the true scheme?", C["green"], ["injected set: top-1 accuracy, top-3 hit, MRR", "compound: both in top-3; method-C pair hit",
                                                                 "distractor cases reported separately; chance 1/7"]),
        ("Consistent — does GBM agree with glass box?", C["orange"], ["per provider: sign agreement, Spearman ρ, top-3 Jaccard", "magnitude: cosine of share vectors s = a / Σ|a|",
                                                                      "by flag agreement; vs permutation baseline"]),
        ("Faithful — does it reflect the model?", C["aqua"], ["group-deletion curve read from the coalition table", "in attribution order; area under score curve",
                                                              "vs random order and vs other methods"]),
        ("Stable — does it survive small changes?", C["blue"], ["10 seeds + bootstrap retrains: top-1 agreement, Kendall τ", "grouping variants (data-driven clusters, moved features)",
                                                                "context ablation: Δ performance, Δ explanations"]),
    ]
    yt = H - 1.35
    for t, col, lines in qs:
        yt = card(ax, 0.35, yt, 9.3, t, lines, col, fs=8.6, tfs=10) - 0.17

    # decision flow on the right
    x0 = 10.6
    def dbox(x, y, w, h, txt, col, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", fc=tint(col, 0.85), ec=col, lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.8, color=INK, zorder=3, fontweight="bold" if bold else None, linespacing=1.35)
    dbox(x0 + 2.4, H - 2.35, 4.2, 0.9, "Is GBM significantly better than\nthe best glass box (EBM / LR)?", C["violet"], True)
    dbox(x0, H - 4.35, 4.0, 0.95, "YES → explain the GBM\n(methods A, B, C, E)", C["violet"])
    dbox(x0 + 5.0, H - 4.35, 4.0, 0.95, "NO → recommend the glass box\n(method D is exact)", C["gray"])
    arrow(ax, (x0 + 3.6, H - 2.35), (x0 + 2.0, H - 3.4), color=INK2)
    arrow(ax, (x0 + 5.4, H - 2.35), (x0 + 7.0, H - 3.4), color=INK2)
    dbox(x0, H - 6.25, 4.0, 0.95, "Do GBM explanations agree with\nthe glass box AND recover\ninjected schemes?", C["orange"], True)
    arrow(ax, (x0 + 2.0, H - 4.35), (x0 + 2.0, H - 5.3), color=INK2)
    dbox(x0 - 0.1, H - 8.25, 2.2, 1.15, "agree + correct\n→ performance\nwithout losing\nexplanation validity", C["green"])
    dbox(x0 + 2.25, H - 8.25, 2.3, 1.15, "disagree, GBM\ncorrect → glass box\nmisses structure\n(report which)", C["yellow"])
    dbox(x0 + 4.7, H - 8.25, 2.2, 1.15, "GBM incorrect\n→ performance\nis not\nexplainable", C["red"])
    arrow(ax, (x0 + 1.0, H - 6.25), (x0 + 1.0, H - 7.1), color=INK2)
    arrow(ax, (x0 + 2.5, H - 6.25), (x0 + 3.4, H - 7.1), color=INK2)
    arrow(ax, (x0 + 3.6, H - 6.25), (x0 + 5.8, H - 7.1), color=INK2)
    dbox(x0 + 5.0, H - 6.25, 4.0, 0.95, "Scenario framework + method\ncomparison still reported\n(publishable either way)", C["gray"])
    arrow(ax, (x0 + 7.0, H - 4.35), (x0 + 7.0, H - 5.3), color=INK2)
    ax.text(x0, 1.2, "Every branch is a reportable result. The paper's thesis is stated as a hypothesis\n(H1: GBM > glass box at audit-relevant k;  H2: GBM scenario attributions are consistent and correct),\nnot as an assumption.",
            fontsize=8.8, color=INK, va="bottom", linespacing=1.4)
    save(fig, f"{OUT}/fig08_evaluation_framework.png")


def fig9():
    W, H = 21, 10.6
    fig, ax = canvas(W, H, "Fig. 9 — Audiences and explanation provenance across the decision lifecycle",
                     "One internal record (scenario attributions + decision), rendered into audience-specific views. Every outward-facing reason is traceable back to a human decision and a model snapshot.")
    lanes = [("System", C["violet"]), ("Analyst (UPIC / SIU)", C["orange"]), ("Decision record", C["gray"]),
             ("Provider", C["blue"]), ("Beneficiary / member", C["aqua"]), ("Oversight / regulator", C["green"])]
    lx, lw_ = 0.35, 2.3
    top = H - 1.3
    lh = 1.33
    for i, (lab, col) in enumerate(lanes):
        y = top - (i + 1) * lh
        ax.add_patch(Rectangle((lx, y + 0.06), W - 0.7 - 2.25, lh - 0.12, fc=tint(col, 0.94), ec="none", zorder=0))
        ax.add_patch(Rectangle((lx, y + 0.06), 0.07, lh - 0.12, fc=col, ec="none", zorder=1))
        ax.text(lx + 0.2, y + lh / 2, lab, va="center", fontsize=10, fontweight="bold", color=INK)
    def node(lane, x, w, txt, col):
        y = top - (lane + 1) * lh + 0.2
        ax.add_patch(FancyBboxPatch((x, y), w, lh - 0.4, boxstyle="round,pad=0,rounding_size=0.08", fc=SURFACE, ec=col, lw=1.2, zorder=3))
        ax.text(x + w / 2, y + (lh - 0.4) / 2, txt, ha="center", va="center", fontsize=8.3, color=INK, zorder=4, linespacing=1.3)
        return (x, y, w, lh - 0.4)
    n1 = node(0, 2.9, 2.9, "score providers (annual /\nrolling) · rank · compute\nscenario attributions", C["violet"])
    n2 = node(1, 6.2, 3.1, "triage queue: risk rank,\ntop scenarios, compound pairs,\npeer stats, feature drill-down", C["orange"])
    n3 = node(1, 9.7, 2.8, "review records / request\ndocumentation / audit /\nrefer or close", C["orange"])
    n4 = node(2, 9.7, 2.8, "decision + rationale +\nmodel version + attribution\nsnapshot (immutable)", C["gray"])
    n5 = node(3, 12.9, 3.2, "ADR / documentation request\nor review outcome; factual\npeer comparison (CBR format)", C["blue"])
    n6 = node(3, 16.35, 1.9, "response /\nappeal with\nevidence", C["blue"])
    n7 = node(4, 12.9, 3.2, "claim status (MSN / EOB):\npending / paid / denied, policy\nreason; no fraud language", C["aqua"])
    n8 = node(5, 6.2, 3.1, "global scenario importance,\nfairness audit by specialty /\nregion, model card, drift", C["green"])
    n9 = node(5, 12.9, 3.2, "audit trail: sample decisions,\ntrace notice → decision →\nattribution → model", C["green"])
    c = lambda n, side: {"r": (n[0] + n[2], n[1] + n[3] / 2), "l": (n[0], n[1] + n[3] / 2), "b": (n[0] + n[2] / 2, n[1]), "t": (n[0] + n[2] / 2, n[1] + n[3])}[side]
    arrow(ax, c(n1, "r"), c(n2, "t"), color=INK2, rad=-0.2)
    arrow(ax, c(n2, "r"), c(n3, "l"), color=INK2)
    arrow(ax, c(n3, "b"), c(n4, "t"), color=INK2)
    arrow(ax, c(n4, "r"), c(n5, "l"), color=INK2, rad=0.15)
    arrow(ax, c(n4, "r"), c(n7, "l"), color=INK2, rad=0.25)
    arrow(ax, c(n5, "r"), c(n6, "l"), color=INK2)
    arrow(ax, c(n6, "t"), (n3[0] + n3[2] - 0.3, n3[1] + n3[3]), color=C["blue"], ls=(0, (4, 3)), rad=0.35)
    arrow(ax, c(n1, "b"), c(n8, "t"), color=INK2, rad=0.0)
    arrow(ax, c(n4, "b"), c(n9, "l"), color=INK2, ls=(0, (4, 3)), rad=0.3)
    # rules box
    rx = 18.95
    ax.text(rx, top - 0.05, "Rules", fontsize=10, fontweight="bold", color=INK, va="top")
    rules = ["human makes every", "adverse decision", "", "context never shown", "as a reason", "", "no score or detection", "logic leaves the SIU",
             "", "external reasons map", "to the decision record,", "not to raw SHAP", "", "member is rarely the", "subject of provider", "fraud: status only"]
    for i, r in enumerate(rules):
        ax.text(rx, top - 0.45 - i * 0.26, r, fontsize=8.2, color=INK2, va="top")
    ax.text(0.35, 0.35, "CBR = CMS Comparative Billing Report (confidential, educational peer comparison; paused from Jan 2024), a format precedent only. ADR = Additional Documentation Request; MSN = Medicare Summary Notice. Stages beyond scoring are a design proposal.",
            fontsize=8.2, color=MUTED)
    save(fig, f"{OUT}/fig09_audiences_lifecycle.png")


if __name__ == "__main__":
    for w in (sys.argv[2:] or ["4", "5", "6", "7", "8", "9"]):
        globals()[f"fig{w}"]()
