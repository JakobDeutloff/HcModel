# %% import
import xarray as xr
import pandas as pd


# %% load data
def load_atms_and_fluxes():
    """
    Load the data from the ICON-ARTS runs.

    Returns
    -------
    atms : xarray.Dataset
        Atmospheric variables.
    fluxes_3d : xarray.Dataset
        Fluxes.
    fluxes_3d_noice : xarray.Dataset
        Fluxes without ice.
    """

    path = "/work/bm1183/m301049/iwp_framework/ciwp/data/"
    fluxes_noice = xr.open_dataset(path + "fluxes_noice_proc.nc")
    fluxes_allsky = xr.open_dataset(path + "fluxes_nosnow_proc.nc")
    atms = xr.open_dataset(path + "atms_proc.nc")
    

    return atms, fluxes_allsky, fluxes_noice


def load_cre():
    """
    Load the cloud radiative effect.

    Returns
    -------
    cre_binned : xarray.Dataset
        CRE binned by IWP and SW radiation.
    cre_interpolated : xarray.Dataset
        CRE interpolated over IWP and SW radiation bins.
    cre_average : xarray.Dataset
        CRE_interpolated averaged over SW radiation bins."""

    path = "/work/bm1183/m301049/iwp_framework/mons/data/"
    cre_binned = xr.open_dataset(path + "cre_binned.nc")
    cre_average = xr.open_dataset(path + "cre_mean.nc")
    return cre_binned, cre_average


def load_derived_vars():
    """
    Load the gridded derived variables.

    Returns
    -------
    lw_vars : xarray.Dataset
        Longwave variables.
    sw_vars : xarray.Dataset
        Shortwave variables.
    lc_vars : xarray.Dataset
        Low cloud variables.
    """

    path = "/work/bm1183/m301049/iwp_framework/mons/data/"
    lw_vars = xr.open_dataset(path + "lw_vars.nc")
    sw_vars = xr.open_dataset(path + "sw_vars.nc")
    lc_vars = xr.open_dataset(path + "lower_trop_vars.nc")
    return lw_vars, sw_vars, lc_vars

def load_mean_derived_vars():
    """
    Load the mean derived variables.

    Returns
    -------
    lw_vars : xarray.Dataset
        Longwave variables.
    sw_vars : xarray.Dataset
        Shortwave variables.
    lc_vars : xarray.Dataset
        Low cloud variables.
    """

    path = "/work/bm1183/m301049/iwp_framework/mons/data/"
    lw_vars = pd.read_pickle(path + "lw_vars_mean.pkl")
    sw_vars = pd.read_pickle(path + "sw_vars_mean.pkl")
    lc_vars = pd.read_pickle(path + "lower_trop_vars_mean.pkl")
    return lw_vars, sw_vars, lc_vars

def load_parameters(experiment='mons'):
    """
    Load the parameters needed for the model.

    Returns
    -------
    hc_albedo : dict
        Parameters for the high cloud albedo.
    hc_emissivity : dict
        Parameters for the high cloud emissivity.
    """

    path = f"/work/bm1183/m301049/iwp_framework/{experiment}/parameters/"
    hc_albedo = pd.read_pickle(path + "hc_albedo_params.pkl")
    hc_emissivity = pd.read_pickle(path + "hc_emissivity_params.pkl")
    c_h2o = pd.read_pickle(path + "C_h2o_params.pkl")
    lower_trop_params = pd.read_pickle(path + "lower_trop_params.pkl")

    return {
        "alpha_hc": hc_albedo,
        "em_hc": hc_emissivity,
        "c_h2o": c_h2o,
        "R_l": lower_trop_params["R_l"],
        "R_cs": lower_trop_params["R_cs"],
        "f": lower_trop_params["f"],
        "a_l": lower_trop_params["a_l"],
        "a_cs": lower_trop_params["a_cs"],
    }


# %%
