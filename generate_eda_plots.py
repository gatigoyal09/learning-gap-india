import os
import tempfile
# Set writable cache dir for matplotlib before importing it
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
input_csv = Path("data/clean/nas_district_wide_v2.csv")
output_dir = Path("outputs/eda")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_csv)

# Style configuration
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# ==============================================================================
# (1) HISTOGRAM OF c8_avg
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
c8_data = df["c8_avg"].dropna()
mean_val = c8_data.mean()
median_val = c8_data.median()
std_val = c8_data.std()

n_bins = 25
counts, bins, patches = ax.hist(
    c8_data, bins=n_bins, color="#2b5c8f", edgecolor="white", alpha=0.85, rwidth=0.95
)

ax.axvline(mean_val, color="#d95f02", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
ax.axvline(median_val, color="#7570b3", linestyle="-.", linewidth=2, label=f"Median: {median_val:.2f}")

ax.set_title("Distribution of Class 8 Average Score (c8_avg) Across Districts", fontsize=14, pad=12, fontweight="bold")
ax.set_xlabel("Class 8 Average Score", fontsize=12)
ax.set_ylabel("Number of Districts", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)

# Annotations box
stats_text = (
    f"N = {len(c8_data)}\n"
    f"Mean = {mean_val:.2f}\n"
    f"Std = {std_val:.2f}\n"
    f"Median = {median_val:.2f}\n"
    f"Min = {c8_data.min():.2f}\n"
    f"Max = {c8_data.max():.2f}"
)
ax.text(
    0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top",
    horizontalalignment="right", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc")
)

plt.tight_layout()
hist_path = output_dir / "c8_avg_histogram.png"
fig.savefig(hist_path)
plt.close(fig)
print(f"Saved: {hist_path}")


# ==============================================================================
# (2) BOXPLOT OF c8_avg BY STATE (SORTED BY MEDIAN)
# ==============================================================================
state_medians = df.groupby("state")["c8_avg"].median().sort_values(ascending=True)
sorted_states = state_medians.index.tolist()
grouped_data = [df[df["state"] == s]["c8_avg"].dropna().values for s in sorted_states]

fig, ax = plt.subplots(figsize=(11, 14), dpi=300)
bp = ax.boxplot(
    grouped_data,
    vert=False,
    patch_artist=True,
    widths=0.65,
    flierprops=dict(marker="o", markersize=4, markerfacecolor="#e7298a", alpha=0.6, markeredgecolor="none"),
    medianprops=dict(color="#d95f02", linewidth=2),
    boxprops=dict(facecolor="#c6dbef", edgecolor="#2b5c8f", linewidth=1.2),
    whiskerprops=dict(color="#2b5c8f", linewidth=1.2),
    capprops=dict(color="#2b5c8f", linewidth=1.2),
)

# Set tick labels explicitly for full version compatibility
ax.set_yticks(range(1, len(sorted_states) + 1))
ax.set_yticklabels([f"{s} (n={len(d)})" for s, d in zip(sorted_states, grouped_data)], fontsize=9)

# Reference national average line
ax.axvline(mean_val, color="#d95f02", linestyle=":", linewidth=1.5, alpha=0.7, label=f"National District Mean ({mean_val:.1f})")

ax.set_title("Class 8 Average Score (c8_avg) by State / UT (Sorted by Median)", fontsize=14, pad=12, fontweight="bold")
ax.set_xlabel("Class 8 Average Score", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.4, axis="x")
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

plt.tight_layout()
box_path = output_dir / "c8_avg_boxplot_by_state.png"
fig.savefig(box_path)
plt.close(fig)
print(f"Saved: {box_path}")


# ==============================================================================
# (3) SCATTER PLOT OF c3_avg VS c8_avg
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
valid_scatter = df.dropna(subset=["c3_avg", "c8_avg"])
x = valid_scatter["c3_avg"]
y = valid_scatter["c8_avg"]

corr = np.corrcoef(x, y)[0, 1]

ax.scatter(x, y, color="#1f78b4", alpha=0.55, edgecolors="none", s=36, label=f"Districts (N={len(valid_scatter)})")

poly_coeffs = np.polyfit(x, y, 1)
poly_fn = np.poly1d(poly_coeffs)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, poly_fn(x_line), color="#e31a1c", linewidth=2, label=f"Linear Fit (r = {corr:.3f}, R² = {corr**2:.3f})")

ax.set_title("District Learning Progression: Class 3 vs Class 8 Average Scores", fontsize=13, pad=12, fontweight="bold")
ax.set_xlabel("Class 3 Average Score (c3_avg)", fontsize=12)
ax.set_ylabel("Class 8 Average Score (c8_avg)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11, loc="upper left")

plt.tight_layout()
scatter_path = output_dir / "c3_vs_c8_scatter.png"
fig.savefig(scatter_path)
plt.close(fig)
print(f"Saved: {scatter_path}")


# ==============================================================================
# (4) CORRELATION HEATMAP OF 15 SUBJECT SCORES
# ==============================================================================
subject_cols = [
    "c3_lang", "c3_math", "c3_evs",
    "c5_lang", "c5_math", "c5_evs",
    "c8_lang", "c8_math", "c8_sci", "c8_sst",
    "c10_lang", "c10_math", "c10_sci", "c10_sst", "c10_eng"
]

corr_matrix = df[subject_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
cax = ax.imshow(corr_matrix, cmap="YlGnBu", vmin=0.2, vmax=1.0)

cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson Correlation Coefficient", fontsize=11)

ax.set_xticks(range(len(subject_cols)))
ax.set_yticks(range(len(subject_cols)))
ax.set_xticklabels(subject_cols, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(subject_cols, fontsize=10)

for i in range(len(subject_cols)):
    for j in range(len(subject_cols)):
        val = corr_matrix.iloc[i, j]
        text_color = "white" if val > 0.65 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

ax.set_title("Correlation Heatmap of the 15 NAS Subject Scores Across Districts", fontsize=13, pad=14, fontweight="bold")
plt.tight_layout()
heatmap_path = output_dir / "subject_correlation_heatmap.png"
fig.savefig(heatmap_path)
plt.close(fig)
print(f"Saved: {heatmap_path}")


# ==============================================================================
# PRINT RANKINGS & VARIANCE DECOMPOSITION
# ==============================================================================
print("\n" + "=" * 80)
print("10 HIGHEST DISTRICTS BY CLASS 8 AVERAGE SCORE (c8_avg)")
print("=" * 80)
highest_10 = df.sort_values(by="c8_avg", ascending=False)[["state", "district", "c8_avg"]].head(10).reset_index(drop=True)
for idx, r in highest_10.iterrows():
    print(f"  {idx+1:>2}. {r['district']:<28} | {r['state']:<25} | c8_avg = {r['c8_avg']:.2f}")

print("\n" + "=" * 80)
print("10 LOWEST DISTRICTS BY CLASS 8 AVERAGE SCORE (c8_avg)")
print("=" * 80)
lowest_10 = df.sort_values(by="c8_avg", ascending=True)[["state", "district", "c8_avg"]].head(10).reset_index(drop=True)
for idx, r in lowest_10.iterrows():
    print(f"  {idx+1:>2}. {r['district']:<28} | {r['state']:<25} | c8_avg = {r['c8_avg']:.2f}")

def variance_decomposition(series, groups):
    valid = series.notna() & groups.notna()
    y = series[valid]
    g = groups[valid]
    N = len(y)
    K = g.nunique()

    grand_mean = y.mean()
    group_means = y.groupby(g).mean()
    group_counts = y.groupby(g).count()

    ss_total = ((y - grand_mean) ** 2).sum()
    ss_between = ((group_counts * (group_means - grand_mean) ** 2)).sum()
    ss_within = ss_total - ss_between

    df_b = K - 1
    df_w = N - K
    ms_b = ss_between / df_b
    ms_w = ss_within / df_w

    n0 = (N - (group_counts ** 2).sum() / N) / (K - 1)
    var_w = ms_w
    var_b = max(0, (ms_b - ms_w) / n0)
    icc = var_b / (var_b + var_w)

    return {
        "ss_total": ss_total,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "share_between_ss": ss_between / ss_total,
        "share_within_ss": ss_within / ss_total,
        "var_between": var_b,
        "var_within": var_w,
        "icc": icc
    }

print("\n" + "=" * 80)
print("SCORE VARIANCE DECOMPOSITION (BETWEEN STATES vs WITHIN STATES)")
print("=" * 80)

decomp_c8 = variance_decomposition(df["c8_avg"], df["state"])
print("Class 8 District Average (c8_avg):")
print(f"  - Total Sum of Squares (SS_total)     : {decomp_c8['ss_total']:>12.2f}")
print(f"  - Between-State SS (SS_between)       : {decomp_c8['ss_between']:>12.2f} ({decomp_c8['share_between_ss']*100:.2f}%)")
print(f"  - Within-State SS (SS_within)         : {decomp_c8['ss_within']:>12.2f} ({decomp_c8['share_within_ss']*100:.2f}%)")
print(f"  - Intraclass Correlation (ICC / Var%) : {decomp_c8['icc']*100:.2f}% between states, {(1-decomp_c8['icc'])*100:.2f}% within states")

decomp_avg = variance_decomposition(df["avg_score"], df["state"])
print("\nOverall District Average (avg_score):")
print(f"  - Total Sum of Squares (SS_total)     : {decomp_avg['ss_total']:>12.2f}")
print(f"  - Between-State SS (SS_between)       : {decomp_avg['ss_between']:>12.2f} ({decomp_avg['share_between_ss']*100:.2f}%)")
print(f"  - Within-State SS (SS_within)         : {decomp_avg['ss_within']:>12.2f} ({decomp_avg['share_within_ss']*100:.2f}%)")
print(f"  - Intraclass Correlation (ICC / Var%) : {decomp_avg['icc']*100:.2f}% between states, {(1-decomp_avg['icc'])*100:.2f}% within states")
print("=" * 80)
