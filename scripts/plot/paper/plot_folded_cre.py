# %% import
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pickle
from src.read_data import load_cre

# %% load data
ds_monsoon = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc")
cre, cre_mean = load_cre()
path = "/work/bm1183/m301049/iwp_framework/mons/model_output/"
run = "prefinal"
with open(path + run + ".pkl", "rb") as f:
    result = pickle.load(f)

# %% multiply hist with cre result
IWP_bins = np.logspace(-5, 1, num=50)

n_profiles = ds_monsoon['IWP'].count().values
hist, edges = np.histogram(ds_monsoon['IWP'].where(ds_monsoon['mask_height']), bins=IWP_bins)
hist = hist / n_profiles

cre_sw_weighted = hist * result['SW_cre']
cre_lw_weighted = hist * result['LW_cre']
cre_net_weighted = cre_sw_weighted + cre_lw_weighted
cre_arts_sw_weighted = hist * cre_mean['connected_sw']
cre_arts_lw_weighted = hist * cre_mean['connected_lw']
cre_arts_net_weighted = cre_arts_sw_weighted + cre_arts_lw_weighted

# %% plot just P * C for paper
fig, ax = plt.subplots(figsize=(10, 4))
ax.stairs(cre_sw_weighted, IWP_bins, color='blue', label='SW')
ax.stairs(cre_lw_weighted, IWP_bins, color='red', label='LW')
ax.stairs(cre_net_weighted, IWP_bins, color='k', fill=True, alpha=0.5)
ax.stairs(cre_net_weighted, IWP_bins, color='k', label='Net')
ax.set_xscale('log')
ax.set_xlabel('$I$ / kg m$^{-2}$')
ax.set_ylabel(r"$C(I) \cdot P(I) ~ / ~ \mathrm{W ~ m^{-2}}$")
ax.set_yticks([-1, 0, 1])
ax.set_xlim(1e-5, 10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
fig.savefig("plots/paper/concept_cre.png", dpi=500, bbox_inches='tight')

# %% plot C and P and folded for presentation
fig, axes = plt.subplots(1, 3, figsize=(28, 4), sharex='row')

# plot IWP hist
axes[0].stairs(hist, edges, color='k')
axes[0].set_yticks([0, 0.02])
axes[0].set_ylabel('$P(I)$')
axes[0].set_xscale('log')

# plot cre 
axes[1].axhline(0, color='grey', linestyle='--')
axes[1].plot(result.index, result['SW_cre'], label='SW', color='blue')
axes[1].plot(result.index, result['LW_cre'], label='LW', color='red')
axes[1].plot(result.index, result['SW_cre'] + result['LW_cre'], label='Net', color='k')
axes[1].plot(result.index, cre_mean['connected_sw'], linestyle='--', color='blue')
axes[1].plot(result.index, cre_mean['connected_lw'], linestyle='--', color='red')
axes[1].plot(result.index, cre_mean['connected_sw'] + cre_mean['connected_lw'], linestyle='--', color='k')
axes[1].set_yticks([-200, 0, 200])
axes[1].set_ylabel('$C(I)$ / W m$^{-2}$')


# plot cre weighted IWP hist
axes[2].stairs(cre_sw_weighted, IWP_bins, color='blue', label='SW')
axes[2].stairs(cre_lw_weighted, IWP_bins, color='red', label='LW')
axes[2].stairs(cre_net_weighted, IWP_bins, color='k', fill=True, alpha=0.5)
axes[2].stairs(cre_net_weighted, IWP_bins, color='k', label='Net')
axes[2].stairs(cre_arts_sw_weighted, IWP_bins, color='blue', linestyle='--', label='SW (arts)')
axes[2].stairs(cre_arts_lw_weighted, IWP_bins, color='red', linestyle='--', label='LW (arts)')
axes[2].stairs(cre_arts_net_weighted, IWP_bins, color='k', linestyle='--', label='Net (arts)')
axes[2].set_ylabel(r"$ C(I) \cdot P(I) ~ / ~ \mathrm{W ~ m^{-2}}$")
axes[2].set_yticks([-1, 0, 1])


for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('$I$ / kg m$^{-2}$')

# legend at bottom with labels from ax 0

fig.subplots_adjust(bottom=0.1)
handles = [
    plt.Line2D([0], [0], color="grey", linestyle="--"),
    plt.Line2D([0], [0], color="grey"),
    plt.Line2D([0], [0], color="red", linestyle="-"),
    plt.Line2D([0], [0], color="blue", linestyle="-"),
    plt.Line2D([0], [0], color="black", linestyle="-"),
]
labels = ["ARTS", "Conceptual Model", "LW", "SW", "Net"]
fig.legend(
    handles,
    labels,
    bbox_to_anchor=(0.65, -0.08),
    loc='center',
    frameon=True,
    ncols=5,
)

fig.savefig("plots/presentation/iwp_cre_folded.png", dpi=500, bbox_inches='tight')

# %%
