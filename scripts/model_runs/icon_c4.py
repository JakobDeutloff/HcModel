# %%
import numpy as np
from src.hc_model import run_model
from src.read_data import (
    load_atms_and_fluxes,
    load_parameters,
    load_average_lc_parameters,
)
from src.calc_variables import cut_data
import pickle
import xarray as xr

# %% load data
sample, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
parameters = load_parameters()
const_lc_quantities = load_average_lc_parameters()
sample = xr.open_dataset("/work/bm1183/m301049/nextgems_profiles/cycle4/representative_sample_c4_conn3.nc")

# %% calculate constants used in the model
albedo_cs = cut_data(fluxes_3d["albedo_clearsky"]).mean()
R_t_cs = cut_data(fluxes_3d['clearsky_lw_up']).isel(pressure=-1).mean()
SW_in = cut_data(fluxes_3d["clearsky_sw_down"]).isel(pressure=-1).mean()

# %% mask out high clouds with tops below 350 hPa
mask = sample['mask_height']

# %% run model
result = run_model(
    IWP_bins = np.logspace(-5, np.log10(50), 60),
    albedo_cs = albedo_cs, 
    R_t_cs = R_t_cs,
    SW_in = SW_in,
    T_hc = sample["hc_temperature"].where(mask),
    LWP = sample['LWP'].where(mask),
    IWP = sample['IWP'].where(mask),
    connectedness=sample['connected'].where(mask),
    parameters = parameters,
    const_lc_quantities=const_lc_quantities,
    prescribed_lc_quantities=None
)

# %% save result 
path = '/work/bm1183/m301049/cm_results/'
with open(path + 'icon_c4.pkl', 'wb') as f:
    pickle.dump(result, f)

# %%
