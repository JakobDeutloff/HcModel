# %% import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from src.read_data import load_cre

# %% load data
cre_binned, cre_mean = load_cre()
ds = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/interp_cf.nc")
ds_monsoon = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc")
result = pd.read_pickle("/work/bm1183/m301049/iwp_framework/mons/model_output/prefinal.pkl")

# %% bin by IWP and average
IWP_bins_cf = np.logspace(-5, np.log10(30), 50)
IWP_bins_cre = np.logspace(-5, 1, 50)
IWP_points_cf = (IWP_bins_cf[:-1] + np.diff(IWP_bins_cf)) / 2
IWP_points_cre = (IWP_bins_cre[:-1] + np.diff(IWP_bins_cre)) / 2
ds_binned = ds.groupby_bins("IWP", IWP_bins_cf).mean()

# %% find the two model levels closest to temperature and interpolate the pressure_lev coordinate between them
temps = [273.15]
levels = pd.DataFrame(index=ds_binned.IWP_bins.values, columns=temps)
for temp in temps:
    for iwp in ds_binned.IWP_bins.values:
        try:
            ta_vals = ds_binned.sel(IWP_bins=iwp, pressure_lev=slice(10000, 100000))["temperature"]
            temp_diff = np.abs(ta_vals - temp)
            level1_idx = temp_diff.argmin("pressure_lev").values
            level1 = ta_vals.pressure_lev.values[level1_idx]
            if ta_vals.sel(pressure_lev=level1) > temp:
                level2 = ta_vals.pressure_lev.values[level1_idx - 1]
                level_interp = np.interp(
                    temp,
                    [
                        ta_vals.sel(pressure_lev=level2).values,
                        ta_vals.sel(pressure_lev=level1).values,
                    ],
                    [level2, level1],
                )
            else:
                level2 = ta_vals.pressure_lev.values[level1_idx + 1]
                level_interp = np.interp(
                    temp,
                    [
                        ta_vals.sel(pressure_lev=level1).values,
                        ta_vals.sel(pressure_lev=level2).values,
                    ],
                    [level1, level2],
                )
            levels.loc[iwp, temp] = level_interp
        except:
            levels.loc[iwp, temp] = np.nan

# %% plot cloud occurence vs IWP percentiles
fig, axes = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[2, 1], sharex=True)

# plot cloud fraction
cf = axes[0].contourf(
    IWP_points_cf,
    ds.pressure_lev / 100,
    ds_binned["cf"].T,
    cmap="Blues",
    levels=np.arange(0.1, 1.1, 0.1),
)
axes[0].plot(IWP_points_cf, levels[273.15].values / 100, color="grey", linestyle="--")
axes[0].text(2e-6, 590, "0°C", color="grey", fontsize=11)
axes[0].invert_yaxis()
axes[0].set_ylabel("Pressure / hPa")

# plot CRE
axes[1].axhline(0, color="grey", linestyle="--")
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_sw"],
    label="SW",
    color="blue",
)
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_lw"],
    label="LW",
    color="red",
)
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_net"],
    label="Net",
    color="k",
)
axes[1].plot(np.linspace(1 - 6, cre_mean.IWP.min(), 100), np.zeros(100), color="k")
axes[1].set_ylabel("HCRE / W m$^{-2}$")
axes[1].set_xlabel("Ice Water Path / kg m$^{-2}$")

# add colorbar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.41, 0.02, 0.48])
fig.colorbar(cf, cax=cax, label="Cloud Cover")

# add legend for axes[1]
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    labels=labels,
    handles=handles,
    bbox_to_anchor=(0.92, 0.3),
    frameon=False,
)

# format axes
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-6, 10)
    ax.set_xticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.set_xscale("log")

fig.savefig("plots/presentation/cloud_profile.png", dpi=500, bbox_inches="tight")


# %%
