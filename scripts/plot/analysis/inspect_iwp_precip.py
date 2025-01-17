# %% import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

# %% load data
path = "/work/bm1183/m301049/iwp_framework/mons/data/"
cre_allsky = xr.open_dataset(path + "cre_mean.nc")
cre_allsky_std = xr.open_dataset(path + "cre_std.nc")
path = "/work/bm1183/m301049/iwp_framework/ciwp/data/"
cre_nosnow = xr.open_dataset(path + "cre_mean.nc")
cre_nosnow_std = xr.open_dataset(path + "cre_std.nc")
ds_monsoon = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc")

# %% plot CRE vs IWP
fig, ax = plt.subplots()
cre_allsky["connected_sw"].plot(ax=ax, color="blue")
cre_allsky["connected_lw"].plot(ax=ax, color="red")
cre_allsky["connected_net"].plot(ax=ax, color="k")
#cre_nosnow["connected_sw"].plot(ax=ax, color="blue", linestyle="--")
#cre_nosnow["connected_lw"].plot(ax=ax, color="red", linestyle="--")
#cre_nosnow["connected_net"].plot(ax=ax, color="k", linestyle="--")
ax.set_xscale("log")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("$C(I)$ / W m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)
handles = [
    plt.Line2D([0], [0], color="k"),
    plt.Line2D([0], [0], color="blue"),
    plt.Line2D([0], [0], color="red"),
    plt.Line2D([0], [0], color="grey", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
]
labels = ["Net", "SW", "LW", "Allsky", "Nosnow"]
ax.legend(handles, labels)
fig.savefig("plots/iwp_inspection/cre_vs_iwp.png", dpi=300)


# %% calculate IWP  without precipitate
iwp_nosnow = (ds_monsoon["IWC"] * ds_monsoon["dzghalf"]).sum("height")
precip_path = ((ds_monsoon["snow"] + ds_monsoon["graupel"]) * ds_monsoon["dzghalf"]).sum("height")

# %% correlate iwp and iwp_nosnow

bins = np.logspace(-6, 1, 100)
bin_centers = (bins[:-1] + bins[1:]) / 2
binned_iwp_nosnow = iwp_nosnow.groupby_bins(ds_monsoon["IWP"], bins, labels=bin_centers).mean()
std_binned_iwp_nosnow = iwp_nosnow.groupby_bins(ds_monsoon["IWP"], bins).std()
flat_iwp = ds_monsoon["IWP"].values.flatten()
flat_iwp_nosnow = iwp_nosnow.values.flatten()

# %%
fig, ax = plt.subplots()
rand_idx = np.random.randint(0, len(flat_iwp), 10000)
ax.scatter(flat_iwp[rand_idx], flat_iwp_nosnow[rand_idx], s=0.5, marker='o', color='k')
ax.plot(bin_centers, binned_iwp_nosnow, color="red", label="mean")
ax.fill_between(
    bin_centers,
    binned_iwp_nosnow.values - std_binned_iwp_nosnow.values,
    binned_iwp_nosnow.values + std_binned_iwp_nosnow.values,
    alpha=0.5,
    color="red",
    label='$\pm$ $\sigma$'
)
ax.plot(np.logspace(-8, 1, 100), np.logspace(-8, 1, 100), color="grey", linestyle="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('IWP / kg m$^{-2}$')
ax.set_ylabel('CIWP / kg m$^{-2}$')
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(1e-6, 10)
ax.set_ylim(1e-6, 10)
ax.legend(frameon=False)    
fig.savefig("plots/iwp_inspection/iwp_vs_iwp_nosnow.png", dpi=300)

# %% calculate IWP Hists
IWP_bins_cre = cre_allsky["IWP_bins"]
n_cells = len(ds_monsoon.lat) * len(ds_monsoon.lon)
hist_allsky, edges = np.histogram(
    ds_monsoon["IWP"].where(ds_monsoon["mask_height"]), bins=IWP_bins_cre
)
hist_allsky = hist_allsky / n_cells
hist_nosnow, edges = np.histogram(iwp_nosnow.where(ds_monsoon["mask_height"]), bins=IWP_bins_cre)
hist_nosnow = hist_nosnow / n_cells
hist_precip, edges = np.histogram(precip_path.where(ds_monsoon["mask_height"]), bins=IWP_bins_cre)
hist_precip = hist_precip / n_cells

# %% plot iwp distribution with and without precipitate
fig, ax = plt.subplots()
ax.stairs(hist_allsky, edges, label="All", color="black")
ax.stairs(hist_nosnow, edges, label="Cloud Ice", color="red")
ax.stairs(hist_precip, edges, label="Precip", color="blue")
ax.set_xscale("log")
ax.set_ylabel("$P(I)$")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_yticks([0, 0.02])
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
fig.savefig("plots/iwp_inspection/iwp_distribution.png", dpi=300)


# %% calculate fraction of precipitate contained in IWP for every bin
precip_fraction = precip_path / (iwp_nosnow + precip_path)
mean_precip_fraction = precip_fraction.groupby_bins(ds_monsoon["IWP"], IWP_bins_cre).mean()
std_precip_fraction = precip_fraction.groupby_bins(ds_monsoon["IWP"], IWP_bins_cre).std()
bin_centers = IWP_bins_cre[:-1].values + IWP_bins_cre[1:].values / 2


# %% plot fraction of precipitate contained in IWP for every bin
fig, ax = plt.subplots()
ax.plot(bin_centers, mean_precip_fraction.values, color="k")
ax.fill_between(
    bin_centers,
    mean_precip_fraction.values - std_precip_fraction.values,
    mean_precip_fraction.values + std_precip_fraction.values,
    alpha=0.5,
    color="k",
)
ax.set_xscale("log")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("Fraction of precipitate in $I$")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("plots/iwp_inspection/precip_fraction.png", dpi=300)

# %% show noise intruduced by precipitate
