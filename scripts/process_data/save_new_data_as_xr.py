# %%
import numpy as np 
import xarray as xr

# %% load data 
path = '/work/um0878/users/mbrath/StarARTS/results/processed_fluxes'
run = 'flux_monsun_Nf10000_1deg_nofrozen'
results_ref = xr.open_dataset(f"{path}/{run}/fluxes_{run}.nc")


# %%get lat and lon
lat_idx=[i for i, x in enumerate(results_ref.quantities.data) if str(x) == 'latitude'][0]
lon_idx=[i for i, x in enumerate(results_ref.quantities.data) if str(x) == 'longitude'][0]

lat=np.sort(np.unique(results_ref.auxiliary_variables.data[:,lat_idx]))
lon=np.sort(np.unique(results_ref.auxiliary_variables.data[:,lon_idx]))

cases=['clearsky_thermal','allsky_thermal','clearsky_solar','allsky_solar']

p_grid=results_ref.pressure.to_numpy()

results={}
for case_i in cases:

    fluxes=results_ref[case_i].to_numpy()
    fluxes = np.reshape(
        fluxes, (len(lat), len(lon),
                    np.size(fluxes, 1), np.size(fluxes, 2)))

    results[case_i]=fluxes
# %%
fluxes_3d = xr.Dataset(
    {
        "allsky_sw_up": (["lat", "lon", "pressure"], results['allsky_solar'][:, :, :, 1]),
        "allsky_sw_down": (["lat", "lon", "pressure"], results['allsky_solar'][:, :, :, 0]),
        "allsky_lw_up": (["lat", "lon", "pressure"], results['allsky_thermal'][:, :, :, 1]),
        "allsky_lw_down": (["lat", "lon", "pressure"], results['allsky_thermal'][:, :, :, 0]),
        "clearsky_sw_up": (["lat", "lon", "pressure"], results['clearsky_solar'][:, :, :, 1]),
        "clearsky_sw_down": (["lat", "lon", "pressure"], results['clearsky_solar'][:, :, :, 0]),
        "clearsky_lw_up": (["lat", "lon", "pressure"], results['clearsky_thermal'][:, :, :, 1]),
        "clearsky_lw_down": (["lat", "lon", "pressure"], results['clearsky_thermal'][:, :, :, 0]),
        "pressure_levels": (["pressure"], p_grid),
    },
    coords={"lat": lat, "lon": lon, "pressure": p_grid},
)
# %% save 
path = '/work/bm1183/m301049/iwp_framework/mons/raw_data/'
fluxes_3d.to_netcdf(f"{path}/{run}/fluxes_3d.nc")