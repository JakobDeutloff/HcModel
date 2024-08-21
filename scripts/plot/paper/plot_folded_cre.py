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

# %% plot C and P for presentation
fig, axes = plt.subplots(2, 1, figsize=(7, 5), height_ratios=[2, 1], sharex=True)

# plot cre 
axes[0].axhline(0, color='grey', linestyle='--')
axes[0].plot(result.index, result['SW_cre'], label='SW', color='blue')
axes[0].plot(result.index, result['LW_cre'], label='LW', color='red')
axes[0].plot(result.index, result['SW_cre'] + result['LW_cre'], label='Net', color='k')
axes[0].plot(result.index, cre_mean['connected_sw'], linestyle='--', color='blue')
axes[0].plot(result.index, cre_mean['connected_lw'], linestyle='--', color='red')
axes[0].plot(result.index, cre_mean['connected_sw'] + cre_mean['connected_lw'], linestyle='--', color='k')
axes[0].set_yticks([-200, 0, 200])
axes[0].set_ylabel('$C(I)$ / W m$^{-2}$')

# plot IWP hist
axes[1].stairs(hist, edges, color='k')
axes[1].set_yticks([0, 0.02])
axes[1].set_ylabel('$P(I)$')
axes[1].set_xlabel('$I$ / kg m$^{-2}$')
axes[1].set_xscale('log')


for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
    bbox_to_anchor=(0.5, -0.05),
    loc='center',
    frameon=True,
    ncols=5,
)

fig.savefig("plots/presentation/cre_with_iwp.png", dpi=500, bbox_inches='tight')


# %% plot P * C for presentation
fig, ax = plt.subplots(figsize=(7, 3))

# plot cre weighted IWP hist
ax.stairs(cre_sw_weighted, IWP_bins, color='blue', label='SW')
ax.stairs(cre_lw_weighted, IWP_bins, color='red', label='LW')
ax.stairs(cre_net_weighted, IWP_bins, color='k', fill=True, alpha=0.5)
ax.stairs(cre_net_weighted, IWP_bins, color='k', label='Net')

# plot arts as well (for presentation)
ax.stairs(cre_arts_sw_weighted, IWP_bins, color='blue', linestyle='--', label='SW (arts)')
ax.stairs(cre_arts_lw_weighted, IWP_bins, color='red', linestyle='--', label='LW (arts)')
ax.stairs(cre_arts_net_weighted, IWP_bins, color='k', linestyle='--', label='Net (arts)')

ax.set_xscale('log')
ax.set_xlabel(r'$I$ / kg m$^{-2}$')
ax.set_ylabel(r"$ C(I) \cdot P(I) ~ / ~ \mathrm{W ~ m^{-2}}$")
ax.set_yticks([-1, 0, 1])
ax.set_xlim(1e-5, 10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig("plots/presentation/c_times_p.png", dpi=500, bbox_inches='tight')

# %%
