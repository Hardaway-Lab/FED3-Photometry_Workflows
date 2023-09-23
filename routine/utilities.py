import itertools as itt

import numpy as np
import pandas as pd
import pandas.api.types as pdt


def cut_df(df, nrow, sortby="SystemTimestamp"):
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


def load_ts(ts):
    ts = df_to_numeric(ts)
    if len(ts.columns) == 2:
        if pdt.is_object_dtype(ts[0]) and pdt.is_float_dtype(ts[1]):
            ts.columns = ["event", "ts"]
            ts["event_type"] = "keydown"
            ts_type = "ts_keydown"
        elif pdt.is_integer_dtype(ts[0]) and pdt.is_float_dtype(ts[1]):
            ts.columns = ["fm_behav", "ts"]
            ts_type = "ts_behav"
        elif pdt.is_integer_dtype(ts[0]) and (
            pdt.is_object_dtype(ts[1]) or pdt.is_bool_dtype(ts[1])
        ):
            ts.columns = ["fm_behav", "event"]
            ts["event_type"] = "user"
            ts_type = "ts_events"
        else:
            raise ValueError("Don't know how to handle TS")
    elif len(ts.columns) == 5:
        if ts.iloc[0, 0] == "DigitalIOName":
            ts = ts.iloc[1:].copy()
        ts = df_to_numeric(ts)
        ts.columns = ["io_name", "io_flag", "io_state", "ts_fp", "ts"]
        ts["event"] = (
            ts["io_name"].astype(str)
            + "-"
            + ts["io_flag"].astype(str)
            + "-"
            + ts["io_state"].astype(str)
        )
        ts["event_type"] = "arduino"
        ts_type = "ts_arduino"
    else:
        raise ValueError("Don't know how to handle TS")
    return ts, ts_type


def pool_events(data, evt_range, rois, norm=True):
    assert "event" in data.columns, "Please align event timestamps first!"
    evt_idx = data[data["event"].notnull()].index
    data.loc[evt_idx, "event"] = data.loc[evt_idx, "event"].astype(str)
    data.loc[evt_idx, "evt_id"] = (
        data.loc[evt_idx, "event"] + "-" + data.loc[evt_idx, "fm_fp"].astype(str)
    )
    max_fm = data["fm_fp"].max()
    evt_df = []
    for idx, row in data[data["evt_id"].notnull()].iterrows():
        fm = row["fm_fp"]
        fm_range = tuple((np.array(evt_range) + fm).clip(0, max_fm))
        dat_sub = data[data["fm_fp"].between(*fm_range)].copy()
        dat_sub["fm_evt"] = dat_sub["fm_fp"] - fm
        dat_sub["event"] = row["event"]
        dat_sub["evt_id"] = row["evt_id"]
        if norm:
            for roi in rois:
                mean = dat_sub.loc[dat_sub["fm_evt"] < 0, roi].mean()
                std = dat_sub.loc[dat_sub["fm_evt"] < 0, roi].std()
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
