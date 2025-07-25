# -*- coding: utf-8 -*-
# =============================================================================
# Title       : Drift Analysis – Station LMP 8.0
# Description : 
#   • Kernel density + histogram overlay with multiple styles for historical data
#   • Red highlight for test data
#   • Time series overlay plot separating training and test sets
#   • Statistical tests: Welch t-test, KS test, Mann–Whitney U test
# Author      : Pablo Catret
# =============================================================================

# =============================================================================
# Import libraries
# =============================================================================

from pathlib import Path
from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import ks_2samp, mannwhitneyu

# =============================================================================
# Configuration
# =============================================================================
STATION_NAME      = "LMP 8.0"
DATA_ROOT         = Path("data")
OLD_PATHS_FILE    = "paths_datos_antiguos_2024.txt"
NEW_PATHS_FILE    = "paths_datos_nuevos_2024.txt"
STATIONS_FILE     = "estaciones_alturas.txt"

OUT_DIR           = Path("analysisLMP")
OUT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_COLOUR      = "#9e9e9e"
TEST_COLOUR       = "#d62728"
DASHES            = [
    (0, (1, 0)),          
    (0, (6, 2)),          
    (0, (3, 2, 1, 2)),    
    (0, (1, 2)),          
    (0, (8, 2, 2, 2)),    
]

plt.rcParams.update({
    "font.size":       14,
    "axes.labelsize":  14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi":      120,
})
sns.set_theme(style="whitegrid")

# =============================================================================
# Step 1: Locate raw data files for target station
# =============================================================================
with open(DATA_ROOT / STATIONS_FILE, encoding="utf-8") as f:
    stations = [ln.strip() for ln in f]
idx = stations.index(STATION_NAME)

def nth_line(txt: str, n: int) -> str:
    with open(DATA_ROOT / txt, encoding="utf-8") as f:
        return [ln.strip() for ln in f][n]

raw_files = {
    "old": DATA_ROOT / nth_line(OLD_PATHS_FILE, idx),
    "new": DATA_ROOT / nth_line(NEW_PATHS_FILE, idx),
}

# =============================================================================
# Step 2: Read and preprocess daily interpolated series
# =============================================================================
def read_noaa(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        header = next(i for i, ln in enumerate(fh) if not ln.startswith("#"))
    df = pd.read_csv(path, sep=";", comment="#", header=header)
    cols = ["Site","SamplingHeight","Year","Month","Day","Hour","Minute",
            "DecimalDate","co2"] + df.columns.tolist()[9:]
    df.columns = cols[: len(df.columns)]
    df["TIMESTAMP"] = pd.to_datetime(df[["Year","Month","Day","Hour","Minute"]])
    df.drop(columns=["Year","Month","Day","Hour","Minute"], inplace=True)
    df.replace({-999.99: np.nan, -9.99: np.nan}, inplace=True)
    return df

raw = pd.concat([read_noaa(p) for p in raw_files.values()], ignore_index=True)

daily = raw.set_index("TIMESTAMP")[["co2"]].resample("D").mean()

# trim last NaN gap >60 d
nan_runs = daily["co2"].isna().astype(int)
pts = np.flatnonzero(np.diff(np.r_[False, nan_runs, False]))
runs = pts.reshape(-1, 2)
long_nans = runs[(runs[:, 1] - runs[:, 0]) > 60]
if long_nans.size:
    daily = daily.iloc[long_nans[-1, 1]:]

# IQR filter + PCHIP interpolation
q1, q3 = daily["co2"].quantile([.25, .75])
bounds = (q1 - 1.5*(q3 - q1), q3 + 1.5*(q3 - q1))
daily.loc[~daily["co2"].between(*bounds), "co2"] = np.nan
daily["value"] = daily["co2"].interpolate(method="pchip")
daily = daily[["value"]].dropna().reset_index()

# =============================================================================
# Step 3: Split into complete 365-day blocks (latest = test)
# =============================================================================
packs, tmp = [], daily.sort_values("TIMESTAMP")
while len(tmp) >= 365:
    packs.insert(0, tmp.iloc[-365:])
    tmp = tmp.iloc[:-365]
n_packs = len(packs)
print(f"{n_packs} complete 365-day packs found.")

# =============================================================================
# Step 4: KDE + histogram overlay by pack
# =============================================================================
fig_hist, ax_hist = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

n_hist = len(packs) - 1
dash_cycle = cycle(DASHES)
color_list = sns.color_palette("muted", n_colors=n_hist)

max_density = 0.0

# Historical (training) packs: grey or varied colors/dashes
for pack, color in zip(packs[:-1][::-1], color_list):
    dash = next(dash_cycle)
    label = f"{pack['TIMESTAMP'].min().date()}–{pack['TIMESTAMP'].max().date()}"
    sns.histplot(
        pack["value"], ax=ax_hist,
        stat="density", bins="auto",
        element="step", fill=False,
        linewidth=1.2, color=color, linestyle="--",
        label=label
    )
    sns.kdeplot(
        pack["value"], ax=ax_hist,
        linewidth=1.2, color=color, linestyle="--"
    )
    try:
        kde_y = ax_hist.lines[-1].get_ydata().max()
        max_density = max(max_density, kde_y)
    except Exception:
        pass

# Test pack (latest): solid red
test_pack = packs[-1]
label = f"{test_pack['TIMESTAMP'].min().date()}–{test_pack['TIMESTAMP'].max().date()} (test)"
sns.histplot(
    test_pack["value"], ax=ax_hist,
    stat="density", bins="auto",
    element="step", fill=False,
    linewidth=1.5, color=TEST_COLOUR, linestyle='-',
    label=label
)
sns.kdeplot(
    test_pack["value"], ax=ax_hist,
    linewidth=1.5, color=TEST_COLOUR, linestyle='-'
)
try:
    kde_y = ax_hist.lines[-1].get_ydata().max()
    max_density = max(max_density, kde_y)
except Exception:
    pass

# Axis and layout
ax_hist.set_ylabel("Density")
ax_hist.set_xlabel("CO2 concentration (ppm)")
ax_hist.set_ylim(0, 0.18)
handles, labels = ax_hist.get_legend_handles_labels()

# Invertir solo los primeros 3 elementos
n_hist = len(packs) - 1
handles = handles[:n_hist][::-1] + handles[n_hist:]
labels = labels[:n_hist][::-1] + labels[n_hist:]

# Aplicar la leyenda reordenada
ax_hist.legend(handles, labels, title=None, frameon=True, ncol=1, loc="upper left")
# =============================================================================
# Step 4.1: Statistical drift tests (Welch, KS, Mann-Whitney)
# =============================================================================

latest   = test_pack["value"]
historic = pd.concat([p["value"] for p in packs[:-1]], ignore_index=True)
tests = {
    "Welch t-test":       ttest_ind(historic, latest, alternative="two-sided", usevar="unequal")[:2],
    "KS test":            ks_2samp(historic, latest)[:2],
    "Mann-Whitney":       mannwhitneyu(historic, latest, alternative="two-sided")[:2],
}
annotation = "\n".join(
    f"{name}: p = {pval:0.3g}"
    for name, (_, pval) in tests.items()
)
ax_hist.text(
    0.98, 0.96, annotation,
    transform=ax_hist.transAxes, ha='right', va='top',
    fontsize=12,
    bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round,pad=0.4", alpha=0.9)
)
ax_hist.set_xlim((405,435))
fig_hist.savefig(OUT_DIR / "LMP_hist_kde_overlay.png", dpi=300, bbox_inches="tight")

# =============================================================================
# Step 5: Overlay of time series (train vs test)
# =============================================================================

fig_ts, ax_ts = plt.subplots(figsize=(7.5, 3.5), constrained_layout=True)

for pack in packs[:-1]:
    ax_ts.plot(pack["TIMESTAMP"], pack["value"], color=TRAIN_COLOUR,
               lw=1.1, alpha=0.9)
ax_ts.plot(test_pack["TIMESTAMP"], test_pack["value"],
           color=TEST_COLOUR, lw=1.4)

ax_ts.set_xlabel("Date")
ax_ts.set_ylabel("CO2 concentration (ppm)")
ax_ts.set_ylim(405, 435)
ax_ts.legend(
    handles=[
        plt.Line2D([], [], color=TRAIN_COLOUR, lw=2, label="Training set"),
        plt.Line2D([], [], color=TEST_COLOUR, lw=2, label="Test set"),
    ],
)

fig_ts.savefig(OUT_DIR / "LMP_train_test_timeseries.png",
               dpi=300, bbox_inches="tight")

# =============================================================================
# Step 6: Save drift test results to CSV
# =============================================================================

pd.DataFrame(
    [{"test": k, "statistic": v[0], "p_value": v[1]} for k, v in tests.items()]
).assign(significant=lambda d: d["p_value"] < 0.05) \
 .to_csv(OUT_DIR / "LMP_drift_tests.csv", index=False)

print("\nAll outputs saved in", OUT_DIR.resolve())
