import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import label
from sklearn.cluster import KMeans

from .utilities import compute_fps, load_ts


def align_ts(data, ts_files, add_cols=["pulsewidth"]) -> None:
    data = data.rename(
        columns={
            "SystemTimestamp": "ts_fp",
            "FrameCounter": "fm_fp",
            "ComputerTimestamp": "ts",
        }
    )
    fps = compute_fps(data)
    print("Assuming FPS of {}".format(fps))
    # load input ts
    ts_dict = dict()
    for dname, dat in ts_files.items():
        dat, ts_type = load_ts(dat.copy(), fps=fps)
        if ts_type == "ts_fed":
            dat["event"] = dname[:4] + "-" + dat["event"]
        print("Interpreting {} as {}".format(dname, ts_type))
        if ts_type == "ts_behav" or ts_type == "ts_fp":
            if ts_type in ts_dict.keys():
                raise ValueError(
                    "Multiple {} supplied but only one expected.".format(ts_type)
                )
            ts_dict[ts_type] = dat
        else:
            ts_dict[dname] = dat
    # add ts to fm_fp
    if "ts_fp" in ts_dict:
        ts_fp = ts_dict.pop("ts_fp")
        fm_diff = int(ts_fp["fm_fp"].max()) - int(data["fm_fp"].max())
        if fm_diff != 0:
            diff_txt = (
                "{} frames more".format(abs(fm_diff))
                if fm_diff > 0
                else "{} frames less".format(abs(fm_diff))
            )
            warnings.warn("FP timestamp file has {} than data file".format(diff_txt))
        data = data.merge(ts_fp, on="fm_fp", how="left", validate="one_to_one")
    # align fm_behav basd on ts
    if "ts_behav" in ts_dict:
        ts_behav = ts_dict.pop("ts_behav")
        if "ts" in data.columns:
            ts_behav = pd.merge_asof(
                ts_behav, data[["fm_fp", "ts"]], on="ts", direction="nearest"
            ).rename(columns={"ts": "ts_behav"})
            ts_behav_dup = ts_behav[ts_behav["fm_fp"].duplicated(keep=False)]
            if len(ts_behav_dup) > 0:
                warnings.warn(
                    "Multiple Behavior frames mapped to the same FP frame\n"
                    + str(ts_behav_dup)
                )
            data = (
                data.merge(ts_behav, on="fm_fp", how="outer")
                .sort_values("fm_fp")
                .reset_index(drop=True)
            )
        else:
            warnings.warn("No FP TS supplied, cannot align Behavior frames")
    # align custom events
    evts = []
    for dname, dat in ts_dict.items():
        acols = list(set(add_cols).intersection(set(dat.columns)))
        if "fm_fp" in dat.columns:
            evts.append(dat[["fm_fp", "event", "event_type"]])
        elif "fm_behav" in dat.columns:
            if "fm_behav" not in data.columns:
                warnings.warn(
                    "No Behavior frames supplied, cannot align {}".format(dname)
                )
                continue
            dat = dat.merge(data[["fm_fp", "fm_behav"]], on="fm_behav", how="left")
            evts.append(dat[["fm_fp", "fm_behav", "event", "event_type"] + acols])
        elif "ts_fp" in dat.columns:
            if "ts_fp" not in data.columns:
                warnings.warn(
                    "No FP timestamps supplied, cannot align {}".format(dname)
                )
                continue
            dat = pd.merge_asof(
                dat, data[["fm_fp", "ts_fp"]], on="ts_fp", direction="nearest"
            )
            evts.append(dat[["fm_fp", "ts_fp", "event", "event_type"] + acols])
        elif "ts" in dat.columns:
            if "ts" not in data.columns:
                warnings.warn(
                    "No computer timestamps supplied, cannot align {}".format(dname)
                )
                continue
            dat = pd.merge_asof(
                dat, data[["fm_fp", "ts"]], on="ts", direction="nearest"
            )
            evts.append(dat[["fm_fp", "ts", "event", "event_type"] + acols])
        print("aligned {}".format(dname))
    if not len(evts) > 0:
        return data, ts_dict
    evts = pd.concat(evts, ignore_index=True).sort_values("fm_fp")
    evts_dup = evts[evts["fm_fp"].duplicated(keep=False)]
    if len(evts_dup) > 0:
        warnings.warn("Multiple events mapped to the same FP frame\n" + str(evts_dup))
        evts = evts.drop_duplicates(subset="fm_fp")
    data = (
        data.merge(
            evts[["fm_fp", "event", "event_type"] + acols], on="fm_fp", how="outer"
        )
        .sort_values("fm_fp")
        .reset_index(drop=True)
    )
    return data, ts_dict


def label_bout(data, name) -> pd.DataFrame:
    lb, nlb = label(data[name])
    data[name + "_label"] = lb
    return data


def classify_opto(
    ts: pd.DataFrame,
    ts_col="SystemTimestamp",
    filter_col="DigitalIOState",
    nstim=1,
    add_lab={"PulseWidth": 0.01},
) -> pd.DataFrame:
    ts = ts[ts[filter_col]].copy()
    t_diff = ts[ts_col].diff()[1:]
    classifier = KMeans(n_clusters=nstim + 1)
    t_diff_cls = classifier.fit_predict(t_diff.values.reshape((-1, 1))).squeeze()
    cls_intvs = classifier.cluster_centers_.squeeze()
    lab_dict = {
        i: "{}Hz".format(np.around(1 / intv)) for i, intv in enumerate(cls_intvs)
    }
    gap_cls = cls_intvs.argmax()
    gap_idxs = np.where(t_diff_cls == gap_cls)[0]
    gap_idxs = np.concatenate([[-1], gap_idxs, [len(ts) + 1]])
    out_ts = []
    for s_start, s_stop in zip(gap_idxs[:-1], gap_idxs[1:]):
        t_start = ts[ts_col].iloc[s_start + 1]
        labs = pd.Series(t_diff_cls[(s_start + 1) : s_stop]).map(lab_dict)
        lab_vals = labs.value_counts()
        if len(lab_vals) > 1:
            ilab = lab_vals.argmax()
            cur_lab = lab_vals.index[ilab]
            warnings.warn(
                "Inconsistent stimulus period detected.\nValue Counts:\n{}\nOriginal Timestamp:\n{}".format(
                    lab_vals.rename("count").reset_index(),
                    ts.iloc[(s_start + 1) : s_stop],
                )
            )
        else:
            cur_lab = lab_vals.index.item()
        out_ts.append(
            pd.DataFrame([{"EventFlag": cur_lab, ts_col: t_start, **add_lab}])
        )
    out_ts = pd.concat(out_ts, ignore_index=True)
    return out_ts
