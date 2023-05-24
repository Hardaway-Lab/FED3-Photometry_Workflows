import itertools as itt

import numpy as np
import pandas as pd
import pandas.api.types as pdt


def cut_df(df, nrow, sortby="Timestamp"):
    return df.sort_values(sortby).iloc[:nrow]


def exp2(x, a, b, c, d, e):
    return a * np.exp(b * x) + c * np.exp(d * x) + e


def load_data(data_file, discard_nfm, led_dict, roi_dict):
    if isinstance(data_file, pd.DataFrame):
        data = data_file
    else:
        data = pd.read_csv(data_file)
    data = data[data["FrameCounter"] > discard_nfm].copy()
    data["signal"] = data["LedState"].map(led_dict)
    nfm = data.groupby("signal").size().min()
    data = (
        data.groupby("signal", group_keys=False)
        .apply(cut_df, nrow=nfm)
        .reset_index(drop=True)
        .rename(columns=roi_dict)
    )
    return data


def load_ts(ts_file):
    if isinstance(ts_file, pd.DataFrame):
        ts = ts_file
    else:
        ts = pd.read_csv(ts_file, header=None)
    if len(ts.columns) == 1:
        if ts.iloc[0, 0] == "ToString()":
            ts = ts.iloc[1:].infer_objects().copy()
        ts = df_to_numeric(ts)
        ts.columns = ["ts"]
        ts_type = "ts_behav"
    elif len(ts.columns) == 2:
        if ts.iloc[0, 0] == "Item1" and ts.iloc[0, 1] == "Item2":
            ts = ts.iloc[1:].infer_objects().copy()
        ts = df_to_numeric(ts)
        if pdt.is_integer_dtype(ts[0]) and pdt.is_float_dtype(ts[1]):
            ts.columns = ["fm_fp", "ts"]
            ts_type = "ts_fp"
        elif pdt.is_integer_dtype(ts[0]) and pdt.is_object_dtype(ts[1]):
            ts.columns = ["fm_behav", "event"]
            ts["event_type"] = "user"
            ts_type = "ts_events"
        else:
            raise ValueError("Don't know how to handle TS")
    elif len(ts.columns) == 3:
        ts = df_to_numeric(ts)
        ts.columns = ["ts_fp", "event", "time"]
        ts["event_type"] = "keydown"
        ts_type = "ts_keydown"
    else:
        raise ValueError("Don't know how to handle TS")
    return ts, ts_type


def pool_events(ts, data, evt_range, rois, event_name="Key"):
    ts["event"] = ts[event_name].astype(str)
    ts["evt_id"] = ts["Timestamp"].astype(str) + "-" + ts["event"]
    evt_df = []
    for _, dat_sig in data.groupby("signal"):
        ts_sig = pd.merge_asof(ts, dat_sig, on="Timestamp")
        dat_sig = dat_sig.merge(
            ts_sig[["FrameCounter", "event", "evt_id"]], on="FrameCounter", how="left"
        )
        max_fm = dat_sig["FrameCounter"].max()
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


def df_to_numeric(df):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df
