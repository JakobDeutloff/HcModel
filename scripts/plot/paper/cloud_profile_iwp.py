
# %% import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from src.read_data import load_cre
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('WebAgg')

# %%
def control_plot(ax):
    ax.set_xlim(1e-5, 10)
    ax.set_xticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.spines[["top", "right"]].set_visible(False)


# %% load data
cre_binned, cre_mean = load_cre()
ds = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/interp_cf.nc")
ds_monsoon = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc")
result = pd.read_pickle("/work/bm1183/m301049/iwp_framework/mons/model_output/prefinal_2.pkl")
sokol_result = xr.open_dataset(
    "/work/bm1183/m301049/iwp_framework/blaz_adam/rcemip_iwp-resolved_statistics.nc"
)
gasparini_result = xr.open_dataset(
    "/work/bm1183/m301049/iwp_framework/blaz_adam/gasparini_twp_cre.nc"
)

# %% bin by IWP and average
IWP_bins_cf = np.logspace(-5, np.log10(30), 50)
IWP_bins_cre = np.logspace(-5, 1, 50)
IWP_points_cf = (IWP_bins_cf[:-1] + np.diff(IWP_bins_cf)) / 2
IWP_points_cre = (IWP_bins_cre[:-1] + np.diff(IWP_bins_cre)) / 2
ds_binned = ds.groupby_bins("IWP", IWP_bins_cf).mean()

# %% calculate IWP Hist
n_cells = len(ds_monsoon.lat) * len(ds_monsoon.lon)
hist, edges = np.histogram(ds_monsoon["IWP"].where(ds_monsoon["mask_height"]), bins=IWP_bins_cre)
hist = hist / n_cells


# %% plot cloud occurence vs IWP percentiles
def custom_fmt(x, pos):
    return f"{x:.0e}".replace("e+0", "e").replace("e-0", "e-")


fig, axes = plt.subplots(4, 1, figsize=(10, 10), height_ratios=[3, 2, 1, 1], sharex=True)

# plot cloud fraction
cf = axes[0].contourf(
    IWP_points_cf,
    ds.pressure_lev / 100,
    ds_binned["cf"].T,
    cmap="Blues",
    levels=np.arange(0.1, 1.1, 0.1),
)
axes[0].invert_yaxis()
axes[0].set_ylabel("Pressure / hPa")
axes[0].set_yticks([1000, 600, 200])

# contour icecond and liqcond
contour_liqcond = axes[0].contour(
    IWP_points_cf,
    ds.pressure_lev / 100,
    ds_binned["liqcond"].T,
    colors="k",
    levels=np.logspace(np.log10(2e-5), -3, 3),
    linestyles="--",
    linewidths=0.9,
    label="liqcond",
)
axes[0].clabel(
    contour_liqcond,
    inline=True,
    fontsize=8,
    fmt=ticker.FuncFormatter(custom_fmt),
    inline_spacing=0,
    rightside_up=False,
)

contour_icecond = axes[0].contour(
    IWP_points_cf,
    ds.pressure_lev / 100,
    ds_binned["icecond"].T,
    colors="k",
    levels=np.logspace(np.log10(1e-7), -3, 3),
    linestyles="-",
    linewidths=0.9,
    label="icecond",
)
axes[0].clabel(
    contour_icecond, inline=True, fontsize=8, fmt=ticker.FuncFormatter(custom_fmt), inline_spacing=0
)

labels = ["Frozen / kg m$^{-3}$", "Liquid / kg m$^{-3}$"]
linestyles = ["-", "--"]
for i, linestyle in enumerate(linestyles):
    axes[0].plot(
        [0, 0],
        [0, 0],
        color="k",
        linestyle=linestyle,
        label=labels[i],
    )
axes[0].legend(loc="lower left", frameon=False)

# plot CRE
axes[1].axhline(0, color="grey", linestyle="-", linewidth=0.5)
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_sw"],
    label="SW",
    color="blue",
    linestyle="-",
)
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_lw"],
    label="LW",
    color="red",
    linestyle="-",
)
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_net"],
    label="net",
    color="k",
    linestyle="-",
)
axes[1].plot(
    np.linspace(1 - 6, int(cre_mean.IWP.min()), 100),
    np.ones(100) * cre_mean["connected_net"][0].values,
    color="k",
)

# Sokol et al 2024 and Gasparini et al 2019
axes[1].plot(sokol_result.fwp / 1e3, sokol_result.ncre.mean("model").sel(SST=300), color="grey", linestyle="--", label="Sokol et al. (2019)")
axes[1].plot(gasparini_result.iwp_points / 1e3, gasparini_result.net_cre, color="grey", linestyle="-.", label='Gasparini et al. (2019)')

axes[1].set_ylabel("$C(I)$ / W m$^{-2}$")
axes[1].set_yticks([-200, 0, 200])

# plot IWP dist
axes[2].stairs(hist, edges, label="IWP", color="black")
axes[2].set_ylabel("$P(I)$")
axes[2].set_yticks([0, 0.02])

# plot P time C
P_times_C = hist * cre_mean["connected_net"]
axes[3].stairs(P_times_C, IWP_bins_cre, color="k", fill=True, alpha=0.5)
axes[3].stairs(P_times_C, IWP_bins_cre, color="k")
axes[3].set_ylabel("$P(I) ~ \cdot ~ C(I)$ / W m$^{-2}$")

# add colorbar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.84, 0.59, 0.02, 0.3])
cb = fig.colorbar(cf, cax=cax, label="Cloud Cover")
cb.set_ticks([0.1, 0.4, 0.7, 1])

# add legend for axes[1]
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(
    labels=labels,
    handles=handles,
    loc='lower left',
    frameon=False,
    ncols=2
)

# format axes
labels = ["a", "b", "c", "d"]
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-5, 10)
    ax.set_xticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.set_xscale("log")
    ax.text(
        0.05,
        1.12,
        labels.pop(0),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )

axes[3].set_xlabel("$I$ / kg m$^{-2}$")
fig.savefig("plots/paper/cloud_profile_iwp_mons.png", dpi=500, bbox_inches="tight")

# %% calculate numbers for text

# max net CRE
max_net_cre = cre_mean["connected_net"].max().values
iwp_max_net_cre = cre_mean.IWP[cre_mean["connected_net"].argmax()].values
print(f"Max net CRE: {max_net_cre:.2} at {iwp_max_net_cre:.2e}")
# max IWP distribution
iwp_max_hist = cre_mean.IWP[hist.argmax()].values
print(f"Max IWP distribution: {iwp_max_hist}")
# net HCRE at max IWP distribution
net_hcre_max_hist = cre_mean["connected_net"][hist.argmax()].values
print(f"Net HCRE at max IWP distribution: {net_hcre_max_hist}")

# %% plot just high cloud fraction and CRE for poster
fig, axes = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[2, 1], sharex=True)

# plot cloud fraction
cf = axes[0].contourf(
    IWP_points_cf,
    ds.pressure_lev / 100,
    ds_binned["cf"],
    cmap="Blues",
    levels=np.arange(0.1, 1.1, 0.1),
)
axes[0].invert_yaxis()
axes[0].set_ylabel("Pressure / hPa")
axes[0].set_yticks([1000, 600, 200])

# plot CRE
axes[1].axhline(0, color="grey", linestyle="--")
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_sw"],
    label="SW",
    color="blue",
    linestyle="-",
)
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_lw"],
    label="LW",
    color="red",
    linestyle="-",
)
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_net"],
    label="net",
    color="k",
    linestyle="-",
)

axes[1].set_ylabel("HCRE / W m$^{-2}$")
axes[1].set_yticks([-200, 0, 200])
axes[1].set_ylim([-280, 200])


for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-5, 10)
    ax.set_xticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.set_xscale("log")

# add colorbar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.84, 0.41, 0.02, 0.48])
cb = fig.colorbar(cf, cax=cax, label="Cloud Cover")
cb.set_ticks([0.1, 0.4, 0.7, 1])

# add legend for axes[1]
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    labels=labels,
    handles=handles,
    bbox_to_anchor=(0.9, 0.31),
    frameon=False,
)
axes[1].set_xlabel("Ice Water Path / kg m$^{-2}$")
fig.savefig("plots/presentation/cloud_profile_iwp_mons.png", dpi=500, bbox_inches="tight")
# %% make cre_comparison
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cre_mean.IWP, cre_mean["connected_net"], label="Net CRE", color="k")
ax.plot(cre_mean.IWP, cre_mean["connected_sw"], label="SW CRE", color="blue")
ax.plot(cre_mean.IWP, cre_mean["connected_lw"], label="LW CRE", color="red")

# sokol
ax.plot(
    sokol_result.fwp / 1e3,
    sokol_result.ncre.mean("model").sel(SST=300),
    color="k",
    linestyle="--",
    label="Sokol",
)
ax.plot(
    sokol_result.fwp / 1e3,
    sokol_result.swcre.mean("model").sel(SST=300),
    color="blue",
    linestyle="--",
    label="Sokol",
)
ax.plot(
    sokol_result.fwp / 1e3,
    sokol_result.lwcre.mean("model").sel(SST=300),
    color="red",
    linestyle="--",
    label="Sokol",
)

# gasparini
ax.plot(
    gasparini_result.iwp_points / 1e3,
    gasparini_result.net_cre,
    color="k",
    linestyle="-.",
    label="Gasparini",
)
ax.plot(
    gasparini_result.iwp_points / 1e3,
    gasparini_result.sw_cre,
    color="blue",
    linestyle="-.",
    label="Gasparini",
)
ax.plot(
    gasparini_result.iwp_points / 1e3,
    gasparini_result.lw_cre,
    color="red",
    linestyle="-.",
    label="Gasparini",
)

ax.set_xscale("log")
control_plot(ax)
ax.set_ylabel("$C(I)$ / W m$^{-2}$")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.axhline(0, color="grey", linestyle="-")

# make fake legend entries
labels = ["net", "SW", "LW", "ARTS", "Sokol et al. (2024)", "Gasparini et al. (2019)"]
handles = [
    plt.Line2D([0], [0], color="k", linestyle="-"),
    plt.Line2D([0], [0], color="blue", linestyle="-"),
    plt.Line2D([0], [0], color="red", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
    plt.Line2D([0], [0], color="grey", linestyle="-."),
]
fig.legend(
    labels=labels,
    handles=handles,
    bbox_to_anchor=(0.7, -0.02),
    ncols=2
)
fig.savefig('plots/paper/cre_comparison.png', dpi=500, bbox_inches="tight")

# %%
