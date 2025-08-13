"""
Generate SCC plots for the combinations in the manuscript.

Usage
-----
python scc_rank.py <config.yml> [--outdir FIG_DIR]

The YAML *config.yml* must define the same keys as before (dp4_df, acd_df, ...),
plus a root *data_dir*.

"""

import argparse
import os
import yaml
from scipy.integrate import trapezoid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  
plt.style.use(["science", "nature", "high-vis"])

# plot specifications
main_text = [
    {"name": "ACD_DP4*_IR_ROC", "type": "roc"},
    {"name": "ACD_manual_auto_DP4*", "type": "scc"},
    {"name": "IR_alone", "type": "scc"},
    {"name": "IR_DP4*_ACD_combination", "type": "scc"},
]

supporting_information = [
    {"name": "ACD_DP4*_combination", "type": "scc"},
    {"name": "IR_high_low_combination", "type": "scc"},
    {"name": "IR_lb", "type": "scc"},
    {"name": "IR_low_DP4*_ACD", "type": "scc"},
]

DEFAULT_LB = 12
LB_VALUES = (12, 10, 8)
HIGH_LOW = ("high", "low")

EPS = 1e-8  

def fused_percentile(v1, v2, eps=EPS):
    """Percentile‑rank average + tiny jitter to break residual ties."""
    n = len(v1)
    pct = ((n - v1.rank(ascending=False, method="min")) +
           (n - v2.rank(ascending=False, method="min"))) / (2 * (n - 1))
    z = ((v1 - v1.mean()) / v1.std(ddof=0) + (v2 - v2.mean()) / v2.std(ddof=0)) / 2
    return pct + eps * z

def norm_gap(s0, sj):
    denom = s0 + sj
    return np.nan if denom <= 0 or not np.isfinite(denom) else (s0 - sj) / denom

THRESHOLDS = np.concatenate([
    np.arange(0, 1, 0.0005),
    np.array([0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999]),
])

def build_curves(
    dp4_df,
    acd_df,
    acd_auto_df,
    ir_df,
):
    diffs = {
        "DP4*": [],
        "ACD":        [],
        "ACD_auto":   [],
        "IR.Cai":         [],
        "IR.Cai+DP4*": [],
        "IR.Cai+ACD": [],
        "ACD+DP4*": [],
    }

    raw = {
        "DP4*": {"pos": [], "neg": []},
        "ACD": {"pos": [], "neg": []},
        "ACD_auto": {"pos": [], "neg": []},
        "IR.Cai": {"pos": [], "neg": []}
    }

    for _, dp4_row in dp4_df.iterrows(): # iterate over the dp4_df
        mol = dp4_row["Molecule"]
        ir_row        = ir_df[ir_df["Molecule"] == mol]
        acd_row       = acd_df[acd_df["Molecule"] == mol]
        acd_auto_row  = acd_auto_df[acd_auto_df["Molecule"] == mol]
        if ir_row.empty or acd_row.empty:
            continue

        dp4      = dp4_row.drop(["Molecule", "Comparison"]).astype(float)
        ir       = ir_row.iloc[0].drop(["Molecule", "Comparison"]).astype(float)
        acd      = acd_row.iloc[0].drop(["Molecule", "Comparison"]).astype(float)
        acd_auto = acd_auto_row.iloc[0].drop(["Molecule", "Comparison"]).astype(float)

        for s in (dp4, ir, acd, acd_auto):
            s.index = s.index.astype(int)
        idx = sorted(set(dp4.index) | set(ir.index) | set(acd.index))
        dp4, ir, acd, acd_auto = (s.reindex(idx) for s in (dp4, ir, acd, acd_auto))

        fused_dp4_ir = fused_percentile(dp4, ir)
        fused_acd_ir = fused_percentile(acd, ir)
        fused_dp4_acd = fused_percentile(dp4, acd)

        s0_dp4, s0_acd, s0_acd_auto, s0_ir = dp4[0], acd[0], acd_auto[0], ir[0]
        s0_fd, s0_fa, s0_fd_acd = fused_dp4_ir[0], fused_acd_ir[0], fused_dp4_acd[0]

        # raw positives
        raw["DP4*"]["pos"].append(dp4[0])
        raw["ACD"]["pos"].append(acd[0])
        raw["ACD_auto"]["pos"].append(acd_auto[0])
        raw["IR.Cai"]["pos"].append(ir[0])

        # pairwise differences
        for j in idx:
            if j == 0:
                continue
            diffs["IR.Cai"].append(s0_ir - ir[j])
            diffs["ACD"].append(s0_acd - acd[j])
            diffs["ACD_auto"].append(s0_acd_auto - acd_auto[j])

            gap_dp4 = norm_gap(s0_dp4, dp4[j])
            if np.isfinite(gap_dp4):
                diffs["DP4*"].append(gap_dp4)

            diffs["IR.Cai+DP4*"].append(s0_fd - fused_dp4_ir[j])
            diffs["IR.Cai+ACD"].append(s0_fa - fused_acd_ir[j])
            diffs["ACD+DP4*"].append(s0_fd_acd - fused_dp4_acd[j])

            # raw negatives
            raw["DP4*"]["neg"].append(dp4[j])
            raw["ACD"]["neg"].append(acd[j])
            raw["ACD_auto"]["neg"].append(acd_auto[j])
            raw["IR.Cai"]["neg"].append(ir[j])

    # convert raw gaps to SCC & ROC curves
    def _scc(gaps):
        gaps = np.asarray(gaps, float)
        gaps = gaps[np.isfinite(gaps)]
        total = len(gaps)
        frac, tpr = [], []
        for tau in THRESHOLDS:
            mask = np.abs(gaps) > tau
            n = mask.sum()
            if n == 0:
                frac.append(0.0)
                tpr.append(np.nan)
                continue
            correct = (gaps[mask] > tau).sum()
            incorrect = (gaps[mask] < -tau).sum()
            frac.append(1 - n / total)
            tpr.append(correct / (correct + incorrect) if correct + incorrect else np.nan)
        return np.asarray(frac), np.asarray(tpr)

    ROC_POINTS = 2001

    def _roc(pos_scores, neg_scores, n_points=ROC_POINTS):
        pos = np.asarray(pos_scores, float)
        neg = np.asarray(neg_scores, float)
        pos = pos[np.isfinite(pos)]
        neg = neg[np.isfinite(neg)]
        if len(pos) == 0 or len(neg) == 0:
            return np.array([np.nan]), np.array([np.nan])

        lo = np.nanmin([pos.min(initial=np.inf), neg.min(initial=np.inf)])
        hi = np.nanmax([pos.max(initial=-np.inf), neg.max(initial=-np.inf)])
        if not np.isfinite(lo) or not np.isfinite(hi):
            return np.array([np.nan]), np.array([np.nan])

        # Higher score => more likely positive
        taus = np.linspace(lo - 1e-12, hi + 1e-12, n_points)

        # Vectorised counts
        pos_sorted = np.sort(pos)
        neg_sorted = np.sort(neg)
        # For threshold tau, TP = count(pos >= tau) = len(pos) - idx_first_gt
        # Use searchsorted on sorted arrays for speed/stability
        tpr = (pos.size - np.searchsorted(pos_sorted, taus, side="left")) / max(pos.size, 1)
        fpr = (neg.size - np.searchsorted(neg_sorted, taus, side="left")) / max(neg.size, 1)
        return fpr, tpr

    # scc curves
    curves = {}
    for name, gaps in diffs.items():
        curves[name] = _scc(gaps)
    
    # roc curves
    for name, raw_data in raw.items():
        curves[name + " (ROC)"] = _roc(raw_data["pos"], raw_data["neg"])

    return curves

COLOUR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def plot_curves(
    curves,
    curve_names,
    kind="scc",  # "scc" | "roc"
    title=None,
    out_file=None,
):
    plt.figure(figsize=(4, 2))

    def extrapolate_left(x_first, y_first, slope=0.5, n=25):
        """
        Generate (x, y) points for a straight‑line extension from
        (x_first, y_first) back to the y‑axis with the given slope.
        """
        x_extra = np.linspace(0, x_first, n)           # 0 → x_first
        y_extra = y_first + slope * (x_extra - x_first)
        return x_extra, y_extra

    for idx, cname in enumerate(curve_names):
        if kind == "roc" and "(ROC)" not in cname:
            cname_plot = cname + " (ROC)"
        else:
            cname_plot = cname
        if cname_plot not in curves:
            raise ValueError(f"Curve '{cname_plot}' not available in the cache")
        x, y = curves[cname_plot]
        finite = np.isfinite(y)
        x, y = x[finite], y[finite]
        # find the area under each curve and put it in the legend
        # add the point (1,1) to each curve
        if kind == "scc":
            x = np.concatenate([x, [1]])
            y = np.concatenate([y, [1]])
        if kind == "scc" and (cname == 'ACD' or cname == 'ACD_auto'):
            x0, y0 = x[0], y[0]
            x_extra, y_extra = extrapolate_left(x0, y0)
            plt.plot(x_extra, y_extra, color=COLOUR_CYCLE[idx % len(COLOUR_CYCLE)], linewidth=2, linestyle=':')
            # add the extrapolated y-intercept to the curve
            x_add = x_extra[0]
            y_add = y_extra[0]
            x_area = np.concatenate([[x_add], x])
            y_area = np.concatenate([[y_add], y])
        else:
            x_area = x
            y_area = y
        
        if kind == "roc":
            area = -trapezoid(y_area, x_area)
            plt.plot(
                x,
                y,
                label=f'{cname.replace(" (ROC)", "")}; AUC = {area:.3f}',
                color=COLOUR_CYCLE[idx % len(COLOUR_CYCLE)],
                linewidth=2,
            )
        else:
            area = trapezoid(y_area, x_area)
            plt.plot(
                x,
                y,
                label=f'{cname.replace(" (ROC)", "")}; CA = {area:.3f}',
                color=COLOUR_CYCLE[idx % len(COLOUR_CYCLE)],
                linewidth=2,
            )
        

    
    if kind == "scc":
        plt.xlabel("Proportion of compound pairs classified as unsolved", fontsize=10)
        plt.ylabel("True positive rate", fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0.7, 1.01)
    else:  # ROC
        plt.xlabel("False positive rate", fontsize=10)
        plt.ylabel("True positive rate", fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1.01)
        plt.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)  # random line

    if title:
        # plt.title(title)
        pass
    plt.legend(fontsize="small", loc="lower right")
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300)
        print(f"Saved figure to {out_file}")
    else:
        plt.show()

# figure recipes
PLOT_RECIPES = {
    "ACD_DP4*_IR_ROC": {
        "curves": ["IR.Cai", "DP4*", "ACD"],
        "ir_variant": ("high", DEFAULT_LB),
        "kind": "roc",
    },
    "ACD_manual_auto_DP4*": {
        "curves": ["DP4*", "ACD", "ACD_auto"],
        "ir_variant": ("high", DEFAULT_LB),
        "kind": "scc",
    },
    "IR_alone": {
        "curves": ["IR.Cai"],
        "ir_variant": ("high", DEFAULT_LB),
        "kind": "scc",
    },
    "IR_DP4*_ACD_combination": {
        "curves": ["IR.Cai", "DP4*", "ACD", "IR.Cai+DP4*", "IR.Cai+ACD"],
        "ir_variant": ("high", DEFAULT_LB),
        "kind": "scc",
    },
    "ACD_DP4*_combination": {
        "curves": ["DP4*", "ACD", "ACD+DP4*"],
        "ir_variant": ("high", DEFAULT_LB),
        "kind": "scc",
    },
    "IR_high_low_combination": {
        "curves": [
            "IR.Cai",  # plotted multiple times with different variants below
        ],
        "ir_variant": None,  # special handling
        "kind": "scc",
    },
    "IR_lb": {
        "curves": [
            "IR.Cai",  # idem – handled specially
        ],
        "ir_variant": None,
        "kind": "scc",
    },
    "IR_low_DP4*_ACD": {
        "curves": ["IR.Cai", "DP4*", "ACD", "IR.Cai+DP4*", "IR.Cai+ACD"],
        "ir_variant": ("low", DEFAULT_LB),
        "kind": "scc",
    },
}

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate SCC/ROC plots.")
    parser.add_argument("config", help="YAML file with paths to all CSV tables")
    parser.add_argument("--outdir", default="./rank_fusion", help="Directory to save figures")
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dp4_df      = pd.read_csv(os.path.join(cfg["data_dir"], cfg["dp4_df"]))
    acd_df      = pd.read_csv(os.path.join(cfg["data_dir"], cfg["acd_df"]))
    acd_auto_df = pd.read_csv(os.path.join(cfg["data_dir"], cfg["acd_auto_df"]))

    ir_tables = {}
    for lb in LB_VALUES:
        for hl in HIGH_LOW:
            key = f"ir_{hl}_{lb}_df"
            ir_tables[(hl, lb)] = pd.read_csv(os.path.join(cfg["data_dir"], cfg[key]))

    curve_cache = {}

    def _get_curves(hl: str, lb: int):
        if (hl, lb) not in curve_cache:
            curve_cache[(hl, lb)] = build_curves(
                dp4_df=dp4_df,
                acd_df=acd_df,
                acd_auto_df=acd_auto_df,
                ir_df=ir_tables[(hl, lb)],
            )
        return curve_cache[(hl, lb)]

    os.makedirs(args.outdir, exist_ok=True)

    for spec in main_text + supporting_information:
        name, kind = spec["name"], spec["type"]

        if name == "IR_high_low_combination":
            lb = DEFAULT_LB
            curves_to_plot, curve_names = {}, []

            # individual high-12 and low-12 curves
            for hl in ("high", "low"):
                tag = f"IR.Cai ({hl}_{lb})"
                curves_to_plot[tag] = _get_curves(hl, lb)["IR.Cai"]
                curve_names.append(tag)

            # build a fused "high+low" IR score table
            ir_high = ir_tables[("high", lb)]
            ir_low  = ir_tables[("low",  lb)]
            ir_comb = ir_high.copy()            # same index / meta columns
            score_cols = [c for c in ir_comb.columns
                        if c not in ("Molecule", "Comparison")]

            for i in ir_comb.index:
                v1 = ir_high.loc[i, score_cols].astype(float)
                v2 = ir_low .loc[i, score_cols].astype(float)
                ir_comb.loc[i, score_cols] = fused_percentile(v1, v2).values

            curves_comb = build_curves(dp4_df, acd_df, acd_auto_df, ir_comb)
            combo_tag = "IR.Cai (combined)"
            curves_to_plot[combo_tag] = curves_comb["IR.Cai"]
            curve_names.append(combo_tag)

            plot_curves(curves_to_plot, curve_names, kind="scc",
                        title=name,
                        out_file=os.path.join(args.outdir, f"{name}.png"))
            continue

        if name == "IR_alone":
            lb = DEFAULT_LB
            curves_to_plot, curve_names = {}, []
            for hl in ("high", "low"):
                tag = f"IR.Cai ({hl}_{lb})"
                curves_to_plot[tag] = _get_curves(hl, lb)["IR.Cai"]
                curve_names.append(tag)

            plot_curves(curves_to_plot, curve_names, kind="scc",
                        title=name,
                        out_file=os.path.join(args.outdir, f"{name}.png"))
            continue

        if name == "IR_lb":
            # special case: show all high/low + all line broadenings
            curve_names, curves_to_plot = [], {}
            for lb in LB_VALUES:
                for hl in HIGH_LOW:
                    tag = f"IR.Cai ({hl}_{lb})"
                    curves_variant = _get_curves(hl, lb)
                    curves_to_plot[tag] = curves_variant["IR.Cai"]  # SCC already
                    curve_names.append(tag)
            # Inject into plotter
            plot_curves(curves_to_plot, curve_names, kind="scc",
                        title=name, out_file=os.path.join(args.outdir, f"{name}.png"))
            continue

        recipe = PLOT_RECIPES[name]
        hl, lb = recipe["ir_variant"] if recipe["ir_variant"] else ("high", DEFAULT_LB)
        curves = _get_curves(hl, lb)
        plot_curves(curves,
                    recipe["curves"],
                    kind=recipe["kind"],
                    title=name,
                    out_file=os.path.join(args.outdir, f"{name}.png"))


    print("All figures written to", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
