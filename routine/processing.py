import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.linear_model import HuberRegressor

from .utilities import exp2, min_transform


def photobleach_correction(data, rois, baseline_sig="415nm", min_trans=False):
    dat_base = data[data["signal"] == baseline_sig].copy()
    x = np.linspace(0, 1, len(dat_base))
    dat_fit = dat_base.copy()
    dat_fit["signal"] = baseline_sig + "-fit"
    sig_df_ls = [
        data[data["signal"] == sig].copy()
        for sig in set(np.unique(data["signal"])) - set([baseline_sig])
    ]
    for sig_df in sig_df_ls:
        sig_df["signal"] = sig_df["signal"] + "-norm"
    for roi in rois:
        dmax, dmin = dat_base[roi][:50].median(), dat_base[roi][-50:].median()
        drg = dmax - dmin
        p0 = (drg, -10, drg, 0.1, dmin - drg)
        try:
            popt, pcov = curve_fit(
                exp2, x, dat_base[roi], p0=p0, method="trf", ftol=1e-4, maxfev=50000
            )
        except:
            warnings.warn("Biexponential fit failed")
            popt = p0
        fit_415 = exp2(x, *popt)
        dat_fit[roi] = fit_415
        for sig_df in sig_df_ls:
            model = HuberRegressor()
            model.fit(fit_415.reshape((-1, 1)), sig_df[roi])
            sig_df[roi] = sig_df[roi] - model.predict(fit_415.reshape((-1, 1)))
            if min_trans:
                sig_df[roi] = min_transform(sig_df[roi])
    data_norm = pd.concat([data, dat_fit] + sig_df_ls, ignore_index=True)
    return data_norm


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


def moving_average_filter(x, wnd, mode="same"):
    return np.convolve(x, np.ones(wnd) / wnd, mode=mode)
