import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import label

from .utilities import load_ts


def align_ts(data, ts_files) -> None:
    data = data.rename(columns={"Timestamp": "ts_fp", "FrameCounter": "fm_fp"})
    ts_dict = dict()
    for dname, dat in ts_files.items():
        dat, ts_type = load_ts(dat.copy())
        print("Interpreting {} as {}".format(dname, ts_type))
        if ts_type == "ts_behav" or "ts_fp":
            if ts_type in ts_dict.keys():
                raise ValueError(
                    "Multiple {} supplied but only one expected.".format(ts_type)
                )
        ts_dict[ts_type] = dat
    if "ts_keydown" in ts_dict:
        ts_key = ts_dict.pop("ts_keydown")
        ts_key = pd.merge_asof(
            ts_key, data[["fm_fp", "ts_fp"]], on="ts_fp", direction="nearest"
        )
        data = data.merge(
            ts_key[["fm_fp", "event", "event_type"]], on="fm_fp", how="outer"
        )
    try:
        ts_fp = ts_dict.pop("ts_fp")
    except KeyError:
        warnings.warn("No FP TS supplied, returning data without further alignment")
        return data, ts_dict
    fm_diff = len(ts_fp) - len(data)
    if fm_diff != 0:
        diff_txt = (
            "{} frames more".format(abs(fm_diff))
            if fm_diff > 0
            else "{} frames less".format(abs(fm_diff))
        )
        warnings.warn("FP timestamp file has {} than data file".format(diff_txt))
    data = data.merge(ts_fp, on="fm_fp", how="left", validate="one_to_one")
    try:
        ts_behav = ts_dict.pop("ts_behav")
    except KeyError:
        warnings.warn(
            "No Behavior TS supplied, returning data without further alignment"
        )
        return data, ts_dict
    ts_behav["fm_behav"] = np.arange(len(ts_behav))
    ts_behav = pd.merge_asof(
        ts_behav, data[["fm_fp", "ts"]], on="ts", direction="nearest"
    ).rename(columns={"ts": "ts_behav"})
    ts_behav_dup = ts_behav[ts_behav["fm_fp"].duplicated(keep=False)]
    if len(ts_behav_dup) > 0:
        warnings.warn(
            "Multiple Behavior frames mapped to the same FP frame\n" + str(ts_behav_dup)
        )
    data = (
        data.merge(ts_behav, on="fm_fp", how="outer")
        .sort_values("fm_fp")
        .reset_index(drop=True)
    )
    evts = []
    for dname, dat in ts_dict.items():
        if "ts" in dat.columns:
            dat = pd.merge_asof(
                dat, ts_behav, left_on="ts", right_on="ts_behav", direction="nearest"
            )
        evts.append(dat[["fm_behav", "event", "event_type"]])
        print("aligned {}".format(dname))
    evts = pd.concat(evts)
    evts_dup = evts[evts["fm_behav"].duplicated(keep=False)]
    if len(evts_dup) > 0:
        warnings.warn(
            "Multiple events mapped to the same Behavior frame\n" + str(evts_dup)
        )
    data = (
        data.merge(evts, on="fm_behav", how="outer")
        .sort_values("fm_fp")
        .reset_index(drop=True)
    )
    return data, ts_dict


def label_bout(data, name) -> pd.DataFrame:
    lb, nlb = label(data[name])
    data[name + "_label"] = lb
    return data
