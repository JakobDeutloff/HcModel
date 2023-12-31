# %% import
import numpy as np
import pandas as pd


# define functions to calculate model quantities


def calc_constants(fluxes_3d):
    """
    Calculates constants for the model.

    PARAMETERS:
    ---------------------------
    fluxes_3d: xarray.Dataset
        3D fluxes data.

    RETURNS:
    ---------------------------
    albedo_cs: float
        Clearsky albedo.
    R_t_cs: float
        Clearsky R_t."""

    albedo_cs = fluxes_3d["albedo_clearsky"].mean()
    R_t_cs = fluxes_3d["clearsky_lw_up"].isel(pressure=-1).mean()
    SW_in = fluxes_3d["clearsky_sw_down"].isel(pressure=-1).mean().values
    return albedo_cs, R_t_cs, SW_in


def logisic(x, L, x0, k, j):
    """
    Logistic function.

    PARAMETERS:
    ---------------------------
    x: array-like
        Input data.
    L: float
        Maximum value.
    x0: float
        Midpoint.
    k: float
        Steepness.
    j: float
        Offset.

    RETURNS:
    ---------------------------
    y: array-like
        Output data.
    """
    return L / (1 + np.exp(-k * (x - x0))) + j


def calc_hc_temperature(IWP_bins, lw_vars, atms):
    """
    Calculates the cloud top temperature.
    
    PARAMETERS:
    ---------------------------
    IWP_bins: array-like
        Bins for IWP.
    lw_vars: xarray.Dataset
        Longwave variables.
    atms: xarray.Dataset
        Atmosphere data.
    
    RETURNS:
    ---------------------------
    T_hc_binned: pd.Series
        Cloud top temperature.
    """
    T_hc_binned = (
        lw_vars["h_cloud_temperature"].groupby_bins(atms["IWP"], IWP_bins).mean()
    )
    return T_hc_binned


def calc_LWP(IWP_bins, atms):
    """
    Calculates the liquid water path.
    
    PARAMETERS:
    ---------------------------
    IWP_bins: array-like
        Bins for IWP.
    atms: xarray.Dataset
        Atmosphere data.
    
    RETURNS:
    ---------------------------
    LWP_binned: pd.Series
        Liquid water path.
    """
    LWP_binned = (
        atms["LWP"].where(atms["LWP"] > 1e-6).groupby_bins(atms["IWP"], IWP_bins).mean()
    )
    return LWP_binned


def calc_lc_fraction(IWP_bins, atms):
    """
    Calculates the low cloud fraction.
    
    PARAMETERS:
    ---------------------------
    IWP_bins: array-like
        Bins for IWP.
    atms: xarray.Dataset
        Atmosphere data.
    
    RETURNS:
    ---------------------------
    lc_fraction_binned: pd.Series
        low cloud fraction.
    """
    lc_fraction_binned = atms["lc_fraction"].groupby_bins(atms["IWP"], IWP_bins).mean()
    return lc_fraction_binned


def calc_hc_albedo(IWP, alpha_hc_params):
    """
    Calculates the high-cloud albedo.

    PARAMETERS:
    ---------------------------
    IWP: array-like
        Input data.
    alpha_hc_params: tuple
        Parameters for the logistic function.

    RETURNS:
    ---------------------------
    fitted_vals: array-like
        high cloud albedo.
    """
    fitted_vals = logisic(np.log10(IWP), *alpha_hc_params, 0)
    return fitted_vals


def calc_hc_emissivity(IWP, em_hc_params):
    """
    Calculates the high-cloud emissivity.
    
    PARAMETERS:
    ---------------------------
    IWP: array-like
        Input data.
    em_hc_params: tuple
        Parameters for the logistic function.
        
    RETURNS:
    ---------------------------
    fitted_vals: array-like
        high cloud emissivity.
    """
    fitted_vals = logisic(np.log10(IWP), *em_hc_params, 0)
    return fitted_vals


def calc_alpha_t(LWP, lc_fraction, albedo_cs, alpha_t_params, const_lc_quantities):
    if const_lc_quantities is not None:
        lc_value = const_lc_quantities["alpha_t"]
    else:
        lc_value = logisic(np.log10(LWP), *alpha_t_params)
    cs_value = albedo_cs
    avg_value = lc_fraction * lc_value + (1 - lc_fraction) * cs_value
    return avg_value


def calc_R_t(LWP, lc_fraction, R_t_cs, R_t_params, const_lc_quantities):
    if const_lc_quantities is not None:
        lc_value = const_lc_quantities["R_t"]
    else:
        lc_value = R_t_params.slope * LWP + R_t_params.intercept
    lc_value[lc_value < R_t_cs] = R_t_cs
    avg_value = lc_fraction * lc_value + (1 - lc_fraction) * R_t_cs
    return avg_value


def hc_sw_cre(alpha_hc, alpha_t, SW_in):
    return -SW_in * (
        alpha_hc + (alpha_t * (1 - alpha_hc) ** 2) / (1 - alpha_t * alpha_hc) - alpha_t
    )


def hc_lw_cre(em_hc, T_h, R_t, sigma):
    return em_hc * ((-1 * sigma * T_h**4) - R_t)


# model function


def run_model(IWP_bins, fluxes_3d, atms, lw_vars, parameters, const_lc_quantities=None):
    """
    Runs the HC Model with given input data and parameters.

    INPUT:
    ---------------------------
    IWP_bins: array-like
        Bins for IWP.
    fluxes_3d: xarray.Dataset
        3D fluxes data.
    atms: xarray.Dataset
        Atmosphere data.
    lw_vars: xarray.Dataset
        Longwave variables.
    parameters: dict
        Parameters for the model.
    const_lc_quantities: dict, optional
        Constant values for lc quantities.

    RETURNS:
    ---------------------------
    results: pd.DataFrame
        Results of the model."""

    IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2

    # calculate constants
    albedo_cs, R_t_cs, SW_in = calc_constants(fluxes_3d)

    # calculate model quantities
    T_hc = calc_hc_temperature(IWP_bins, lw_vars, atms)
    LWP = calc_LWP(IWP_bins, atms)
    lc_fraction = calc_lc_fraction(IWP_bins, atms)
    alpha_t = calc_alpha_t(
        LWP, lc_fraction, albedo_cs, parameters["alpha_t"], const_lc_quantities
    )
    R_t = calc_R_t(LWP, lc_fraction, R_t_cs, parameters["R_t"], const_lc_quantities)
    alpha_hc = calc_hc_albedo(IWP_points, parameters["alpha_hc"])
    em_hc = calc_hc_emissivity(IWP_points, parameters["em_hc"])

    # calculate HCRE
    SW_cre = hc_sw_cre(alpha_hc, alpha_t, SW_in)
    LW_cre = hc_lw_cre(em_hc, T_hc, R_t, sigma=5.67e-8)

    # build results df
    results = pd.DataFrame()
    results.index = IWP_points
    results.index.name = "IWP"
    results["T_hc"] = T_hc
    results["LWP"] = LWP
    results["lc_fraction"] = lc_fraction
    results["alpha_t"] = alpha_t
    results["R_t"] = R_t
    results["alpha_hc"] = alpha_hc
    results["em_hc"] = em_hc
    results["SW_cre"] = SW_cre
    results["LW_cre"] = LW_cre

    return results
