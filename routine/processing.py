import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import HuberRegressor

from .utilities import exp2


def photobleach_correction(data, rois, baseline_sig="415nm"):
    dat_base = data[data["signal"] == baseline_sig].copy()
    x = np.linspace(0, 1, len(dat_base))
    dat_fit = dat_base.copy()
    dat_fit["signal"] = "415nm-fit"
    sig_df_ls = [
        data[data["signal"] == sig].copy()
        for sig in set(np.unique(data["signal"])) - set(["415nm"])
    ]
    for roi in rois:
        popt, pcov = curve_fit(
            exp2, x, dat_base[roi], p0=(1.0, -1.0, 1.0, -1.0, dat_base[roi].mean())
        )
        fit_415 = exp2(x, *popt)
        dat_fit[roi] = fit_415
        for sig_df in sig_df_ls:
            sig_df["signal"] = sig_df["signal"] + "-norm"
            model = HuberRegressor()
            model.fit(fit_415.reshape((-1, 1)), sig_df[roi])
            sig_df[roi] = sig_df[roi] - model.predict(fit_415.reshape((-1, 1)))
    data_norm = pd.concat([data, dat_fit] + sig_df_ls, ignore_index=True)
    return data_norm
