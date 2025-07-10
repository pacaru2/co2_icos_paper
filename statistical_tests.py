
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Title       : Statistical Comparison of Forecast Models
# Description : Robust summary statistics, confidence intervals, pairwise 
#               comparisons, and Critical Difference (CD) diagrams for 
#               CO₂ forecasting model evaluation.
# Author      : Pablo Catret
# =============================================================================

# =============================================================================
# Import libraries
# =============================================================================
import warnings, glob, re, itertools, os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.weightstats import DescrStatsW
import scikit_posthocs as sp
from scipy.stats import median_abs_deviation, trim_mean
import matplotlib.pyplot as plt
import seaborn as sns  
import matplotlib as mpl
from scipy.stats import studentized_range
from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONSTANTS & FILE PATHS
# =============================================================================
CLASS_CSV = Path("data/ClasificacionesEstacion.csv")
MET_DIR   = Path("EntrenamientoModelos/metricas")
OUT_G     = Path("AnalisisAgrupaciones")
OUT_T     = Path("AnalisisTests")
OUT_G.mkdir(exist_ok=True)
OUT_T.mkdir(exist_ok=True)

KEEP = {
 "BIR","GAT","HPB","HTM","IPR","JFJ","JUE","KIT","KRE","LMP","LUT",
 "NOR","OPE","OXK","PAL","PUI","PUY","RGL","SAC","SMR","SNO","SSL",
 "STE","TOH","TRN","UTO","WES","ZSF"
}

METRICS = ["MAE","MSE","RMSE","MAPE","sMAPE","RMSLE"]

# =============================================================================
# 1. LOAD STATION CLASSIFICATION & BINNING
# =============================================================================

cls = pd.read_csv(CLASS_CSV, sep=";", decimal=",").rename(columns=str.strip)
cls = cls[cls.Abbreviation.isin(KEEP)].copy()
cls["Latitud"]  = cls.Latitud

cls["ElevBin"] = pd.cut(cls.Level, [0,200,500,1000,2000,1e6],
                        right=True, labels=["0-200","200-500","500-1000","1000-2000",">2000"])
cls["LatBin"]  = pd.cut(cls.Latitud, [34,43,50,57,64,90],
                        right=True, labels=["34-43","43-50","50-57","57-64",">64"])
cls["ESAxKoppen"] = cls.ESA + "_" + cls.Koppen

# =============================================================================
# 2. MERGE ALL METRIC FILES
# =============================================================================

dfs=[]
for f in glob.glob(str(MET_DIR / "*_METRICS.csv")):
    df = pd.read_csv(f)
    abbr = df.Station.str[:3].iloc[0]
    if abbr in KEEP:
        dfs.append(df)
metrics = pd.concat(dfs, ignore_index=True)
metrics["Station"] = metrics["Station"].str.replace(" ", "_", regex=False)
metrics = metrics.merge(cls, left_on=metrics.Station.str[:3], right_on="Abbreviation")

# =============================================================================
# 3. COMPUTE ROBUST STATISTICS BY GROUP
# =============================================================================

def robust_summary(group_field: str, out_name: str) -> None:
    """
    Guarda un CSV con:
        <Metric>_median  <Metric>_p25  <Metric>_p75
        <Metric>_MAD     <Metric>_IQR  <Metric>_trim10
    para cada grupo en group_field.
    """
    def one_row(g):                                  # g es un sub-DataFrame
        res = {}
        for m in METRICS:
            vals = g[m].dropna().to_numpy()
            if vals.size == 0:
                continue
            res.update({
                f"{m}_median" : np.median(vals),
                f"{m}_p25"    : np.percentile(vals, 25),
                f"{m}_p75"    : np.percentile(vals, 75),
                f"{m}_MAD"    : median_abs_deviation(vals),
                f"{m}_IQR"    : np.percentile(vals, 75) - np.percentile(vals, 25),
                f"{m}_trim10" : trim_mean(vals, 0.10),
            })
        return pd.Series(res)

    out = (metrics.groupby(group_field, dropna=False)
                   .apply(one_row)
                   .reset_index())
    out.to_csv(OUT_G / f"robust_{out_name}.csv", index=False)

robust_summary("Model",       "Model")
robust_summary("ESA",         "ESA")
robust_summary("Koppen",      "Koppen")
robust_summary("ESAxKoppen",  "ESAxKoppen")
robust_summary("ElevBin",     "ElevBin")
robust_summary("LatBin",      "LatBin")



# =============================================================================
# 4. BOOTSTRAP CONFIDENCE INTERVALS (p25, MEDIAN, p75)
# =============================================================================


B      = 50_000                          
alpha  = 0.05                           
rng    = np.random.default_rng(0)
boots  = []                              
plots  = {"p25": [], "median": [], "p75": []} 

for metric in METRICS:
    piv = metrics.pivot_table(index="Station", columns="Model", values=metric)

    for mdl in piv.columns:
        vals = piv[mdl].dropna().to_numpy()
        if len(vals) < 5:
            continue                    

        reps = rng.choice(vals, size=(B, len(vals)), replace=True)
        qs   = np.percentile(reps, (25, 50, 75), axis=1)

        ci25_low, ci25_hi = np.percentile(qs[0], [100*alpha/2, 100*(1-alpha/2)])
        ci50_low, ci50_hi = np.percentile(qs[1], [100*alpha/2, 100*(1-alpha/2)])
        ci75_low, ci75_hi = np.percentile(qs[2], [100*alpha/2, 100*(1-alpha/2)])

        p25    = np.percentile(vals, 25)
        median = np.median(vals)
        p75    = np.percentile(vals, 75)

        boots.append({
            "Metric": metric, "Model": mdl,
            "p25": p25, "ci25_low": ci25_low, "ci25_hi": ci25_hi,
            "median": median, "ci50_low": ci50_low, "ci50_hi": ci50_hi,
            "p75": p75, "ci75_low": ci75_low, "ci75_hi": ci75_hi,
            "N": len(vals)
        })

        plots["p25"].append( (metric, mdl, p25, ci25_low, ci25_hi) )
        plots["median"].append( (metric, mdl, median, ci50_low, ci50_hi) )
        plots["p75"].append( (metric, mdl, p75, ci75_low, ci75_hi) )

pd.DataFrame(boots).to_csv(OUT_T / "bootstrap_IC_valores.csv", index=False)

# =============================================================================
# 5. PLOT CONFIDENCE INTERVALS FOR MAPE
# =============================================================================

def plot_ic(data, stat_name, out_png):
    """
    data : tuple list (metric, model, valor, ci_low, ci_hi)
    """
    data_sorted = sorted(data, key=lambda x: (x[0], x[2]))

    labels = [f"{metric} | {model}" for metric, model, *_ in data_sorted]

    vals = np.array([v for _, _, v, _, _ in data_sorted])

    err_low = vals - np.array([lo for _, _, _, lo, _ in data_sorted])
    err_high = np.array([hi for _, _, _, _, hi in data_sorted]) - vals
    err = np.vstack([err_low, err_high])          
    plt.figure(figsize=(10, 0.4 * len(vals) + 1))
    y = np.arange(len(vals))
    plt.errorbar(vals, y, xerr=err, fmt="o", capsize=3, linewidth=1)
    plt.yticks(y, labels, fontsize=9)
    plt.xlabel(f"{stat_name}")
    plt.title(f"IC 95 % of {stat_name}")
    plt.grid(axis="x", ls=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# =============================================================================
# 6. PAIRWISE COMPARISON: IS MODEL A SIGNIFICANTLY BETTER THAN B?
# =============================================================================

for stat_key, stat_label in [
        ("median", "median"),
        ("p25",    "25% percentile"),
        ("p75",    "75% percentile")]:
    
    data_mape = [t for t in plots[stat_key] if t[0] == "MAPE"]
    if not data_mape:
        print(f"No hay datos de {stat_key} para MAPE")
        continue
    
    plot_ic(
        data_mape,
        f"MAPE {stat_label}",
        OUT_T / f"IC_{stat_key}_MAPE.png"
    )

print("Bootstrap de valores e IC guardados en:", OUT_T)

diffs = {}
for metric in METRICS:
    piv = metrics.pivot_table(index="Station", columns="Model", values=metric)
    for q, label in zip([25,50,75], ["p25","median","p75"]):
        ci = np.zeros((len(piv.columns), len(piv.columns)), bool) 
        for i,a in enumerate(piv.columns):
            for j,b in enumerate(piv.columns):
                if i >= j:        
                    continue
                delta = piv[a] - piv[b]  
                delta = delta.dropna().to_numpy()
                reps = rng.choice(delta, size=(B, len(delta)), replace=True).mean(1)
                lo, hi = np.percentile(reps, [2.5, 97.5])
                ci[i,j] = hi < 0    
                ci[j,i] = lo > 0     
        diffs[(metric,label)] = pd.DataFrame(ci, index=piv.columns, columns=piv.columns)           

for (metric,label), M in diffs.items():
    plt.figure(figsize=(8,6))
    sns.heatmap(M.astype(int), cmap="Greens", cbar=False,
                linewidth=.5, linecolor="white",
                annot=False, square=True)
    plt.title(f"Paired superiority – {metric} – {label}")
    plt.xlabel("Compared Model")
    plt.ylabel("Reference Model")
    plt.tight_layout()
    plt.savefig(OUT_T/f"super_{metric}_{label}.png", dpi=150)

# =============================================================================
# 7. POST-HOC TEST (NEMENYI) – SIGNIFICANCE HEATMAPS
# =============================================================================

for metric in METRICS:
    for q, q_name in [(0.25, "p25"), (0.50, "median"), (0.75, "p75")]:
        piv_q = (
            metrics
            .groupby(["Station", "Model"])[metric]
            .quantile(q)
            .unstack()
        )
        piv_q = piv_q.dropna(how="any")
        if piv_q.shape[1] < 3 or piv_q.empty:
            print(f"{metric} {q_name}: datos insuficientes")
            continue

        try:
            pvals = sp.posthoc_nemenyi_friedman(piv_q)
            labels = list(pvals.columns)

            fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(labels)), 8))

            cmap = sns.color_palette("Greens", 4)
            bounds = [0, 0.001, 0.01, 0.05, 1.0]
            norm = plt.matplotlib.colors.BoundaryNorm(bounds, len(cmap))
            cmap = ["#FBEFEF"] + cmap

            sns.heatmap(
                pvals,
                ax=ax,
                cmap=cmap,
                norm=norm,
                cbar_kws={
                    "ticks": [0.0005, 0.005, 0.03, 0.5],
                    "label": "p-Value"
                },
                square=True,
                linewidths=0.5,
                linecolor="white",
                annot=False,
                xticklabels=labels,
                yticklabels=labels
            )

            ax.set_title(f"Nemenyi Test– {metric} – {q_name}", fontsize=13)
            ax.set_xlabel("Compared Model")
            ax.set_ylabel("Reference Model")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(OUT_T / f"CD_{metric}_{q_name}_alt.png", dpi=150)
            plt.close()
            print(f"Heatmap de p-values guardado para {metric} {q_name}")

        except Exception as e:
            print(f"Error en heatmap para {metric} {q_name}: {e}")

# =============================================================================
# 8. CRITICAL DIFFERENCE (CD) DIAGRAMS (DEMSAR STYLE)
# =============================================================================

def cd_diagram(av_ranks: pd.Series, CD: float,
               title: str, fname: Path,
               best_left: bool = True) -> None:
    """
    Draws a Critical Difference diagram (Demšar style).
    """
    av_ranks = av_ranks.sort_values(ascending=best_left)
    xs = av_ranks.values
    labels = av_ranks.index.to_list()
    k = len(xs)

    fig_width = max(8, 0.5 * k + 2)
    fig_height = 4.5               
    plt.figure(figsize=(fig_width, fig_height))
    y0 = 1.0                       

    plt.hlines(y0, xs.min() - 0.5, xs.max() + 0.5, lw=1.2, color="black")

    for i, (x, label) in enumerate(zip(xs, labels)):
        plt.plot(x, y0, "ko", ms=5)
        if i % 2 == 0:
            plt.text(x, y0 + 0.10, label, rotation=90,
                     ha="center", va="bottom", fontsize=9)
        else:
            plt.text(x, y0 - 0.10, label, rotation=90,
                     ha="center", va="top", fontsize=9)

    cd_y = y0 - 1.0
    plt.hlines(cd_y, xs.min(), xs.min() + CD, lw=3, color="black")
    plt.text(xs.min() + CD / 2, cd_y - 0.1,
             f"CD = {CD:.2f}", ha="center", va="top", fontsize=10)

    groups = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and xs[j + 1] - xs[i] <= CD:
            j += 1
        groups.append((i, j))
        i = j + 1

    g_y = y0 - 0.6
    step = -0.25
    for g0, g1 in groups:
        if g1 > g0:
            plt.hlines(g_y, xs[g0], xs[g1], lw=2.5, color="steelblue")
            g_y += step

    plt.xticks(fontsize=10)
    plt.yticks([])
    plt.xlabel("Average rank (↓ better)", fontsize=11)
    plt.title(title, fontsize=12)
    plt.grid(False)

    y_top = y0 + 0.6   # top margin
    y_bot = cd_y - 0.4 # bottom margin
    plt.ylim(y_bot, y_top)

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()



# =============================================================================
# MAIN LOOP: GENERATE CD DIAGRAMS BY METRIC & QUANTILE
# =============================================================================
alpha = 0.05
for metric in METRICS:
    for q, q_name in [(0.25, "p25"), (0.50, "median"), (0.75, "p75")]:
        piv_q = (metrics
                 .groupby(["Station", "Model"])[metric]
                 .quantile(q)
                 .unstack())
        piv_q = piv_q.dropna(how="any")
        if piv_q.shape[1] < 3 or piv_q.empty:
            print(f"{metric} {q_name}: datos insuficientes")
            continue

        ranks = piv_q.rank(axis=1, method="average")
        av_rk = ranks.mean()
        k, N = len(av_rk), len(ranks)

        q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)
        CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

        cd_diagram(av_rk, CD,
                   title=f"Critical Difference – {metric} – {q_name}",
                   fname=OUT_T / f"CD_{metric}_{q_name}.png")

        print(f"CD {metric} {q_name} guardado en AnalisisTests/")


# =============================================================================
# 9. WHERE DOES EACH GROUP WIN? - MAPE + CONTINUOUS HEATMAP
# =============================================================================

HYBRID_LIST = {"ProphetTCN", "ProphetLGB", "ProHiTS", "MediaSimple", "Meta"}
metrics["Family"] = metrics["Model"].apply(
    lambda m: "Hybrid" if m in HYBRID_LIST else "Traditional"
)

TARGET_METRIC = "MAPE"
best_idx = (
    metrics
    .groupby(["Station", "Family"])[TARGET_METRIC]
    .idxmin()
    .dropna()
    .astype(int)
)
best = metrics.loc[best_idx].copy()

valid = (best.groupby(["Station", "Family"]).size()
             .unstack(fill_value=0)
             .query("Hybrid>0 and Traditional>0")
             .index)
best = best[best.Station.isin(valid)]

if "LatBin" not in best.columns:
    best["LatBin"] = pd.cut(best.Latitud,
                            [34,43,50,57,64,90],
                            labels=["34-43","43-50","50-57","57-64",">64"],
                            right=True)
if "ElevBin" not in best.columns:
    best["ElevBin"] = pd.cut(best.Level,
                             [0,200,500,1000,2000,1e6],
                             labels=["0-200","200-500","500-1000",
                                     "1000-2000",">2000"],
                             right=True)

def diff_by(var: str) -> pd.DataFrame:
    rows=[]
    for lvl, g in best.groupby(var, dropna=False):
        med = g.groupby("Family")[TARGET_METRIC].median()
        if {"Hybrid","Traditional"}.issubset(med.index):
            delta = med["Traditional"] - med["Hybrid"]
            rows.append((lvl, med["Hybrid"], med["Traditional"], delta))
    return pd.DataFrame(rows, columns=[var,"Hybrid_med","Trad_med","Delta"])

GROUP_VARS = ["LatBin","ElevBin","Koppen","ESA"]
out_dir = Path("AnalisisWins")
out_dir.mkdir(exist_ok=True)

summary=[]
for var in GROUP_VARS:
    df = diff_by(var).sort_values("Delta", ascending=False)
    df.to_csv(out_dir / f"diff_{TARGET_METRIC}_{var}.csv", index=False)
    summary.append(df.rename(columns={var:"Level"}).assign(Variable=var))

rows = {}
for var in GROUP_VARS:  # ["ESA", "Koppen", "LatBin", "ElevBin"]
    df = diff_by(var)
    if df.empty:
        print(f"{var}: sin datos.")
        continue

    if var == "LatBin":
        df[var] = df[var].astype(str).str.replace(">", "≥")
        df[var] = df[var].str.replace(r"(\d+)", r"\1º", regex=True)

    row = df.set_index(var)["Delta"].to_frame().T
    rows[var] = row

if not rows:
    raise RuntimeError("No hay datos para ningún heatmap")

max_abs = max(abs(df.values).max() for df in rows.values())
max_abs = np.round(max_abs + 0.01, 2)   
cmap    = "RdBu_r"
norm    = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

n_rows   = len(rows)
fig_w    = max(8, 1.6 * max(df.shape[1] for df in rows.values()))
fig_h    = 3.6 * n_rows
fig, ax_arr = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False)
ax_arr = ax_arr.ravel()
letters = ['a)', 'b)', 'c)', 'd)']

for i, (ax, (var, row)) in enumerate(zip(ax_arr, rows.items())):
    sns.heatmap(row,
                cmap=cmap, norm=norm,
                annot=True, fmt=".2f", annot_kws={"size": 20},
                linewidths=.5, linecolor="white",
                cbar=False, ax=ax)

    ax.set_ylabel(f"{letters[i]}", fontsize=20, rotation=0, labelpad=40)
    ax.set_xlabel("")
    ax.set_yticklabels([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=15)

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)

cbar.set_label("ΔMAPE  (Best Simple − Best Hybrid)", fontsize=22)
cbar.ax.tick_params(labelsize=18)
cbar.set_ticks([-max_abs, 0, max_abs])
cbar.ax.set_yticklabels([f"{-max_abs:.2f}", "0", f"{max_abs:.2f}"])

plt.subplots_adjust(left=0.20, right=0.9, top=0.96, bottom=0.05, hspace=0.45)
out_png = out_dir / "grid_heatmap_delta_MAPE.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Imagen final guardada en: {out_png}")
