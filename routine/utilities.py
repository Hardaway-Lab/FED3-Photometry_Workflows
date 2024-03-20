import itertools as itt

import numpy as np
import pandas as pd
import pandas.api.types as pdt

PULSE_DICT = {
    0.001: "pellet retrieval",
    0.002: "left poke",
    0.003: "right poke",
    0.004: "initiation of FED motor to deliver a pellet",
    0.005: "fall of pellet into magazine",
    0.006: "auditory cue",
    0.007: "LED strip activated",
}


def cut_df(df, nrow, sortby="SystemTimestamp"):
    return df.sort_values(sortby).iloc[:nrow]


def exp2(x, a, b, c, d, e):
    return a * np.exp(b * x) + c * np.exp(d * x) + e


def load_data(data_file, discard_time, led_dict, roi_dict):
    if isinstance(data_file, pd.DataFrame):
        data = data_file
    else:
        data = pd.read_csv(data_file)
    data = data[
        data["SystemTimestamp"] > data["SystemTimestamp"].min() + discard_time
    ].copy()
    data["signal"] = data["LedState"].map(led_dict)
    nfm = data.groupby("signal").size().min()
    data = (
        data.groupby("signal", group_keys=False)
        .apply(cut_df, nrow=nfm)
        .reset_index(drop=True)
        .rename(columns=roi_dict)
    )
    return data


def load_ts(ts, fps=30):
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
    elif len(ts.columns) == 3:
        if ts.iloc[0, 2] == "PulseWidth":
            ts = df_to_numeric(ts.iloc[1:].copy())
            ts.columns = ["event", "ts_fp", "pulsewidth"]
            ts["event_type"] = "fed"
            ts["pulsewidth"] = ts["pulsewidth"].round(3)
            ts["event"] = ts["pulsewidth"].map(PULSE_DICT)
            ts_type = "ts_fed"
        else:
            ts = df_to_numeric(ts)
            ts.columns = ["ts_fp", "event", "time"]
            ts["event_type"] = "keydown"
            ts_type = "ts_keydown"
    elif len(ts.columns) == 5 and ts.iloc[0, 2] == "DigitalIOState":
        ts_raw = df_to_numeric(ts.iloc[1:].copy())
        ts_raw.columns = [
            "DigitalIOName",
            "DigitalIOFlag",
            "DigitalIOState",
            "ts_fp",
            "ts",
        ]
        ts_raw["DigitalIOState"] = ts_raw["DigitalIOState"].map(
            {"True": True, "False": False}
        )
        assert len(ts_raw) % 2 == 0, "Opto timestamps must have even number of rows"
        ts_df = []
        for irow in range(0, len(ts_raw), 2):
            ts_seg = ts_raw.iloc[irow : irow + 2, :]
            state, col_fp, col_ts = (
                ts_seg["DigitalIOState"],
                ts_seg["ts_fp"],
                ts_seg["ts"],
            )
            assert (
                state.iloc[0] == True and state.iloc[-1] == False
            ), "Corrupted timestamp at row {}:\n{}".format(irow, ts_seg)
            nrow = max(int(round(np.ptp(col_fp) * fps)), 1)
            cur_ts = pd.DataFrame(
                {
                    "ts_fp": np.linspace(col_fp.iloc[0], col_fp.iloc[-1], nrow),
                    "ts": np.linspace(col_ts.iloc[0], col_ts.iloc[-1], nrow),
                }
            )
            cur_ts["event"] = "opto_stim"
            cur_ts["DigitalIOName"] = ts_seg["DigitalIOName"].unique().item()
            cur_ts["DigitalIOFlag"] = ts_seg["DigitalIOFlag"].unique().item()
            ts_df.append(cur_ts)
        ts = pd.concat(ts_df, ignore_index=True)
        ts["event_type"] = "opto"
        ts_type = "ts_opto"
    else:
        raise ValueError("Don't know how to handle TS")
    return ts, ts_type


def enumerated_product(*args):
    yield from zip(itt.product(*(range(len(x)) for x in args)), itt.product(*args))


def df_to_numeric(df):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def min_transform(a):
    return a - np.nanmin(a)


def compute_fps(df, fm_col="fm_fp", tcol="ts_fp", ledcol="LedState", mul_fac=1):
    nled = (df[ledcol].count() > 5).sum()
    mdf = df[tcol].diff().mean()
    mfm = df[fm_col].diff().mean()
    return float(mfm / mdf * mul_fac * nled)
