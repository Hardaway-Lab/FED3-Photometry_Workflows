import itertools as itt

import numpy as np
import pandas as pd


def cut_df(df, nrow, sortby="Timestamp"):
    return df.sort_values(sortby).iloc[:nrow]


def exp2(x, a, b, c, d, e):
    return a * np.exp(b * x) + c * np.exp(d * x) + e


def load_data(data_file, ts_file, discard_nfm, led_dict):
    data = pd.read_csv(data_file)
    ts = pd.read_csv(ts_file, names=["Timestamp", "Key", "Time"])
    data = data[data["FrameCounter"] > discard_nfm].copy()
    data["signal"] = data["LedState"].map(led_dict)
    nfm = data.groupby("signal").size().min()
    data = (
        data.groupby("signal", group_keys=False)
        .apply(cut_df, nrow=nfm)
        .reset_index(drop=True)
    )
    return data, ts


def pool_events(ts, data, evt_range, rois, event_name="Key"):
    ts["event"] = ts[event_name].astype(str)
    ts["evt_id"] = ts["Timestamp"].astype(str) + "-" + ts["event"]
    data_join = data.merge(ts, on="Timestamp", how="left")
    max_fm = data_join["FrameCounter"].max()
    evt_df = []
    for _, dat_sig in data_join.groupby("signal"):
        for idx, row in dat_sig[dat_sig["evt_id"].notnull()].iterrows():
            fm = row["FrameCounter"]
            fm_range = tuple((np.array(evt_range) + fm).clip(0, max_fm))
            dat_sub = dat_sig[dat_sig["FrameCounter"].between(*fm_range)].copy()
            dat_sub["evt_fm"] = dat_sub["FrameCounter"] - fm
            dat_sub["event"] = row["event"]
            dat_sub["evt_id"] = row["evt_id"]
            for roi in rois:
                mean = dat_sub.loc[dat_sub["evt_fm"] < 0, roi].mean()
                std = dat_sub.loc[dat_sub["evt_fm"] < 0, roi].std()
                if std > 0:
                    dat_sub[roi] = (dat_sub[roi] - mean) / std
                else:
                    dat_sub[roi] = 0
            evt_df.append(dat_sub)
    evt_df = pd.concat(evt_df, ignore_index=True)
    return evt_df


def enumerated_product(*args):
    yield from zip(itt.product(*(range(len(x)) for x in args)), itt.product(*args))
