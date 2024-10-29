# %% import
import numpy as np
from src.calc_variables import calc_cf
import xarray as xr
from scipy import interpolate
from tqdm import tqdm
import matplotlib.pyplot as plt

# %% Load icon sample data
ds = xr.open_dataset(
    "/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc"
).sel(lat=slice(-30, 30))

#%% subsample data to 1e6 profiles 
ds_flat = ds.stack(idx=("lat", "lon"))
ds_flat = ds_flat.reset_index("idx")

#%%
random_idx = np.random.choice(ds_flat["idx"], size=int(2e6), replace=False)
ds_rand = ds_flat.sel(idx=random_idx).squeeze()

# %% calculate cloud fraction
ds_rand = ds_rand.assign(cf=calc_cf(ds_rand))
ds_rand = ds_rand.assign(liqcond=ds_rand['LWC'] + ds_rand['rain'])
ds_rand = ds_rand.assign(icecond=ds_rand['IWC'] + ds_rand['snow'] + ds_rand['graupel'])
IWP_bins = np.logspace(-5, 1, 70)
IWP_points = (IWP_bins[:-1] + np.diff(IWP_bins)) / 2
ds_binned = ds_rand.groupby_bins("IWP", IWP_bins).mean('idx')
# %% test plot 
fig, ax = plt.subplots()
# plot cloud fraction
cf = ax.contourf(
    IWP_points,
    ds.height,
    ds_binned["cf"].T,
    cmap="Blues",
    levels=np.arange(0.1, 1.1, 0.1),
)

ax.invert_yaxis()
ax.set_xscale("log")

# %% create a new coordinate for pressure
pressure_levels = np.linspace(7000, ds["pressure"].max().values, num=80)

#%% create a new dataset with the same dimensions as ds, but with level_full replaced by pressure
ds_interp = xr.Dataset(
    coords={"pressure_lev": pressure_levels, "idx": ds_rand["idx"].values}
)
ds_interp["IWP"] = ds_rand["IWP"]


cf = np.zeros((len(pressure_levels), len(ds_rand["idx"])))
liqcond = cf.copy()
icecond = cf.copy()
values = {"cf": cf, "liqcond": liqcond, "icecond": icecond}

#%% interpolate every profile

for j in tqdm(range(len(ds_rand["idx"]))):
    for var in ["cf", "liqcond", "icecond"]:
        values[var][:, j] = interpolate.interp1d(
            ds_rand["pressure"].isel(idx=j),
            ds_rand[var].isel(idx=j),
            fill_value="extrapolate",
        )(pressure_levels)


ds_interp["cf"] = xr.DataArray(values["cf"], dims=["pressure_lev", "idx"])
ds_interp["liqcond"] = xr.DataArray(values["liqcond"], dims=["pressure_lev", "idx"])
ds_interp["icecond"] = xr.DataArray(values["icecond"], dims=["pressure_lev", "idx"])

# %% save data
ds_interp.to_netcdf("/work/bm1183/m301049/iwp_framework/mons/data/interp_cf.nc")
# %%
