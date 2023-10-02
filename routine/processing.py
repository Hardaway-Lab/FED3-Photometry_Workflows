import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.linear_model import HuberRegressor

from .utilities import exp2


def photobleach_correction(data, rois, baseline_sig="415nm"):
    # fit baseline
    base_roi = rois[0]  # TODO: make this flexible
    dat_fit = data.loc[data["signal"] == baseline_sig].copy()
    dat_fit["signal"] = baseline_sig + "-fit"
    dat_fit[list(set(rois) - set([base_roi]))] = np.nan
    x = np.linspace(0, 1, len(dat_fit))
    baseline = fit_exp2(dat_fit[base_roi], x)
    dat_fit[base_roi] = baseline
    sig_df_ls = [
        data[data["signal"] == sig].copy()
        for sig in set(np.unique(data["signal"])) - set([baseline_sig])
    ]
    for sig_df in sig_df_ls:
        sig_df["signal"] = sig_df["signal"] + "-norm"
    for roi in rois:
        for sig_df in sig_df_ls:
            model = HuberRegressor()
            model.fit(baseline.reshape((-1, 1)), sig_df[roi])
            sig_df[roi] = sig_df[roi] - model.predict(baseline.reshape((-1, 1)))
    data_norm = pd.concat([data, dat_fit] + sig_df_ls, ignore_index=True)
    return data_norm


def fit_exp2(a, x):
    dmax, dmin = a[:50].median(), a[-50:].median()
    drg = dmax - dmin
    p0 = (drg, -10, drg, 0.1, dmin - drg)
    try:
        popt, pcov = curve_fit(exp2, x, a, p0=p0, method="trf", ftol=1e-6, maxfev=1e4)
    except:
        warnings.warn("Biexponential fit failed")
        popt = p0
    return exp2(x, *popt)


def compute_dff(data, rois, sigs=["415nm", "470nm"]):
    if sigs is not None:
        data = data[data["signal"].isin(sigs)].copy()
    res_ls = []
    for sig, dat_sig in data.groupby("signal"):
        dat_fit = dat_sig.copy()
        dat_dff = dat_sig.copy()
        dat_fit["signal"] = sig + "-fit"
        dat_dff["signal"] = sig + "-dff"
        x = np.linspace(0, 1, len(dat_sig))
        for roi in rois:
            dat = dat_sig[roi]
            popt, pcov = curve_fit(
                exp2,
                x,
                dat,
                p0=(1.0, 0, 1.0, 0, dat.mean()),
                bounds=(
                    np.array([-np.inf, -np.inf, -np.inf, -np.inf, dat.min()]),
                    np.array([np.inf, np.inf, np.inf, np.inf, dat.max()]),
                ),
            )
            cur_fit = exp2(x, *popt)
            dat_fit[roi] = cur_fit
            dat_dff[roi] = 100 * (dat - cur_fit) / cur_fit
        res_ls.extend([dat_fit, dat_dff])
    return pd.concat([data] + res_ls, ignore_index=True)


def find_pks(data, rois, prominence, freq_wd, sigs=None):
    if sigs is not None:
        data = data[data["signal"].isin(sigs)].copy()
    res_ls = []
    for sig, dat_sig in data.groupby("signal"):
        for roi in rois:
            dat = dat_sig[roi]
            pks, props = find_peaks(dat, prominence=prominence)
            pvec = np.zeros_like(dat, dtype=bool)
            pvec[pks] = 1
            dat_sig[roi + "-pks"] = pvec
            dat_sig[roi + "-freq"] = dat_sig[roi + "-pks"].rolling(freq_wd).sum()
        res_ls.append(dat_sig)
    return pd.concat(res_ls, ignore_index=True)
