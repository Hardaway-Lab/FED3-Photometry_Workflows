import io
import itertools as itt
import os
import warnings

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
from ipyfilechooser import FileChooser
from IPython.display import display
from ipywidgets import Layout, widgets
from plotly.colors import qualitative

from .plotting import (
    construct_cmap,
    plot_agg_polled,
    plot_events,
    plot_peaks,
    plot_polled_signal,
    plot_signals,
)
from .polling import agg_polled_events, poll_events
from .processing import find_pks, moving_average_filter, photobleach_correction
from .ts_alignment import align_ts, label_bout
from .utilities import compute_fps, load_data


class NPMBase:
    def __init__(self, fig_path="./figs/process", out_path="./output/process") -> None:
        self.wgt_opts = {
            "style": {"description_width": "initial"},
            "layout": Layout(width="80%"),
        }
        self.data = None
        self.fig_path = fig_path
        self.out_path = out_path
        self.prefix = None
        os.makedirs(self.fig_path, exist_ok=True)
        os.makedirs(self.out_path, exist_ok=True)

    def set_data(self, dpath: str = None, source: str = "local") -> None:
        if dpath is None:
            if source == "local":
                lab = widgets.Label("Select Data: ", layout=Layout(width="75px"))
                fc = FileChooser(".", **self.wgt_opts)
                fc.register_callback(self.on_set_data_local)
                display(widgets.HBox([lab, fc]))
            elif source == "remote":
                w_data = widgets.FileUpload(
                    accept=".csv",
                    multiple=False,
                    description="Upload Data File",
                    tooltip="Select data file to analyze",
                    **self.wgt_opts,
                )
                w_data.observe(self.on_set_data_remote, names="value")
                display(w_data)
        else:
            self.prefix = os.path.basename(dpath).split("_")[0]
            self.data = pd.read_csv(dpath)
            print("Using '{}' as output prefix".format(self.prefix))

    def on_set_data_remote(self, change) -> None:
        dat = change["new"][0]["content"].tobytes()
        self.data = pd.read_csv(io.BytesIO(dat), encoding="utf8")

    def on_set_data_local(self, fc) -> None:
        self.prefix = os.path.basename(fc.selected).split("_")[0]
        self.data = pd.read_csv(fc.selected)
        print("Using '{}' as output prefix".format(self.prefix))

    def set_paths(self, fig_path=None, out_path=None) -> None:
        if fig_path is None:
            lab = widgets.Label("Figure Path: ", layout=Layout(width="75px"))
            fc = FileChooser(self.fig_path, show_only_dirs=True, **self.wgt_opts)
            fc.register_callback(self.on_figpath)
            display(widgets.HBox([lab, fc]))
        else:
            self.fig_path = fig_path
            os.makedirs(fig_path, exist_ok=True)
        if out_path is None:
            lab = widgets.Label("Output Path: ", layout=Layout(width="75px"))
            fc = FileChooser(self.out_path, show_only_dirs=True, **self.wgt_opts)
            fc.register_callback(self.on_outpath)
            display(widgets.HBox([lab, fc]))
        else:
            self.out_path = out_path
            os.makedirs(out_path, exist_ok=True)

    def on_figpath(self, fc) -> None:
        self.fig_path = fc.selected_path
        os.makedirs(self.fig_path, exist_ok=True)

    def on_outpath(self, fc) -> None:
        self.out_path = fc.selected_path
        os.makedirs(self.out_path, exist_ok=True)

    def set_output_prefix(self, prefix: str = None) -> None:
        if prefix is None:
            w_pre = widgets.Text(
                value=self.prefix,
                description="Output Prefix",
                **self.wgt_opts,
            )
            w_pre.observe(self.on_prefix, names="value")
            display(w_pre)
        else:
            self.prefix = prefix

    def on_prefix(self, change) -> None:
        self.prefix = change["new"]


class NPMProcess(NPMBase):
    def __init__(self, fig_path="./figs/process", out_path="./output/process") -> None:
        super().__init__(fig_path, out_path)
        self.param_discard_time = None
        self.param_pk_prominence = None
        self.param_led_dict = {7: "initial", 1: "415nm", 2: "470nm", 4: "560nm"}
        self.param_roi_dict = None
        self.param_base_sig = None
        self.param_ma_wnd = None
        self.param_base_med_wnd = None
        self.data_norm = None
        print("Process initialized")

    def set_discard_time(self, discard_time: float = None) -> None:
        assert self.data is not None, "Please set data first!"
        if discard_time is None:
            w_txt = widgets.Label(
                "Number of Seconds to Discard from Beginning of Recording"
            )
            w_nfm = widgets.FloatSlider(
                min=0,
                value=0,
                max=10,
                step=0.01,
                tooltip="Cropping data points at the beginning of the recording can improve curve fitting.",
                **self.wgt_opts,
            )
            self.param_discard_time = 0
            w_nfm.observe(self.on_discard, names="value")
            display(widgets.VBox([w_txt, w_nfm]))
        else:
            self.param_discard_time = discard_time

    def on_discard(self, change) -> None:
        self.param_discard_time = float(change["new"])

    def set_pk_prominence(self, prom: int = None) -> None:
        if prom is None:
            w_txt = widgets.Label("Peak Prominence")
            w_pk = widgets.FloatSlider(
                min=0,
                value=0.1,
                max=3,
                step=0.001,
                **self.wgt_opts,
            )
            self.param_pk_prominence = 0.1
            w_pk.observe(self.on_pk_prominence, names="value")
            display(widgets.VBox([w_txt, w_pk]))
        else:
            self.param_pk_prominence = prom

    def on_pk_prominence(self, change) -> None:
        self.param_pk_prominence = change["new"]

    def set_roi(self, roi_dict: dict = None) -> None:
        assert self.data is not None, "Please set data first!"
        if roi_dict is None:
            w_txt = widgets.Label("ROIs to analyze (CTRL/CMD click to Select Multiple)")
            w_roi = widgets.SelectMultiple(
                options=self.data.columns,
                tooltip="Region1G Region2R etc",
                **self.wgt_opts,
            )
            w_roi.observe(self.on_roi, names="value")
            display(widgets.VBox([w_txt, w_roi]))
        else:
            self.param_roi_dict = roi_dict

    def on_roi(self, change) -> None:
        rois = change["new"]
        self.param_roi_dict = {r: r for r in rois}

    def set_roi_names(self, roi_dict: dict = None) -> None:
        assert self.param_roi_dict is not None, "Please set roi first!"
        if roi_dict is None:
            rois = list(self.param_roi_dict.keys())
            w_rois = [
                widgets.Text(
                    value=r,
                    placeholder=r,
                    description="Region or Animal Corresponding to {}".format(r),
                    **self.wgt_opts,
                )
                for r in rois
            ]
            for w in w_rois:
                w.observe(self.on_roi_name, names="value")
                display(w)
        else:
            self.param_roi_dict = {k: v.replace("-", "_") for k, v in roi_dict.items()}

    def on_roi_name(self, change) -> None:
        k, v = change["owner"].placeholder, change["new"]
        self.param_roi_dict[k] = v.replace("-", "_")

    def set_baseline(self, base_sig: dict = None):
        assert self.data is not None, "Please set data first!"
        if base_sig is None:
            rois = list(self.param_roi_dict.values())
            sigs = list(set(self.param_led_dict.values()) - set(["initial"]))
            roi_sig = list(itt.product(rois, sigs))
            for key_r, key_s in roi_sig:
                opts = [("-".join(rs), {(key_r, key_s): rs}) for rs in roi_sig]
                opts = opts + [("No correction", {(key_r, key_s): None})]
                w_base = widgets.Dropdown(
                    description="{}-{}: ".format(key_r, key_s),
                    options=opts,
                    value={(key_r, key_s): None},
                    **self.wgt_opts,
                )
                w_base.observe(self.on_baseline, names="value")
                self.param_base_sig = dict()
                display(w_base)
        else:
            self.param_base_sig = base_sig

    def on_baseline(self, change) -> None:
        self.param_base_sig.update(change["new"])
        self.param_base_sig = {
            k: v for k, v in self.param_base_sig.items() if v is not None
        }

    def set_ma_wnd(self, wnd: int = None) -> None:
        if wnd is None:
            w_txt = widgets.Label("Filter window size")
            w_wnd = widgets.IntSlider(
                min=5,
                value=20,
                max=50,
                step=1,
                tooltip="Size of moving average filter window (in frames)",
                **self.wgt_opts,
            )
            self.param_ma_wnd = 20
            w_wnd.observe(self.on_ma_wnd, names="value")
            display(widgets.VBox([w_txt, w_wnd]))
        else:
            self.param_ma_wnd = wnd

    def on_ma_wnd(self, change) -> None:
        self.param_ma_wnd = int(change["new"])

    def set_base_med_wnd(self, wnd: int = None) -> None:
        if wnd is None:
            w_txt = widgets.Label("Filter window size (baseline)")
            w_wnd = widgets.IntSlider(
                min=5,
                value=100,
                max=600,
                step=1,
                tooltip="Size of median filter window for baseline smoothing (in frames)",
                **self.wgt_opts,
            )
            self.param_base_med_wnd = 100
            w_wnd.observe(self.on_base_med_wnd, names="value")
            display(widgets.VBox([w_txt, w_wnd]))
        else:
            self.param_base_med_wnd = wnd

    def on_base_med_wnd(self, change) -> None:
        self.param_base_med_wnd = int(change["new"])

    def load_data(self) -> None:
        assert self.data is not None, "Please set data first!"
        assert self.param_roi_dict is not None, "Please set ROIs first!"
        assert self.param_discard_time is not None, "Please set time to discard first!"
        self.data = load_data(
            self.data, self.param_discard_time, self.param_led_dict, self.param_roi_dict
        )
        fig = plot_signals(
            self.data, list(self.param_roi_dict.values()), default_window=(0, 10)
        )
        # fig.write_html(os.path.join(self.fig_path, "raw_signals.html"))
        nroi = len(self.param_roi_dict)
        fig.update_layout(height=350 * nroi)
        display(fig)

    def photobleach_correction(self, **kwargs) -> None:
        assert self.data is not None, "Please set data first!"
        assert self.param_roi_dict is not None, "Please set ROIs first!"
        assert self.param_base_sig is not None, "Please set baseline signal first!"
        self.data_norm = photobleach_correction(
            self.data,
            self.param_base_sig,
            rois=list(self.param_roi_dict.values()),
            med_wnd=self.param_base_med_wnd,
            **kwargs,
        )
        fig = plot_signals(
            self.data_norm,
            list(self.param_roi_dict.values()),
            group_dict=lambda s: s.split("-")[0],
        )
        fig.write_html(os.path.join(self.fig_path, "photobleaching_correction.html"))
        nroi = len(self.param_roi_dict)
        fig.update_layout(height=350 * nroi)
        display(fig)

    def moving_filter(self, wnd=None, mode="same", apply_to=["470nm-norm"]) -> None:
        if wnd is None:
            wnd = self.param_ma_wnd
        for sig in apply_to:
            for roi in list(self.param_roi_dict.values()):
                self.data_norm.loc[self.data_norm["signal"] == sig, roi] = (
                    moving_average_filter(
                        self.data_norm.loc[self.data_norm["signal"] == sig, roi],
                        wnd=wnd,
                        mode=mode,
                    )
                )

    def find_peaks(self) -> None:
        self.data_norm = find_pks(
            self.data_norm,
            rois=list(self.param_roi_dict.values()),
            prominence=self.param_pk_prominence,
            sigs=["470nm-norm-zs"],
        )
        fig = plot_peaks(
            self.data_norm[self.data_norm["signal"] == "470nm-norm-zs"].copy(),
            rois=list(self.param_roi_dict.values()),
        )
        fig.write_html(os.path.join(self.fig_path, "peaks.html"))
        nroi = len(self.param_roi_dict)
        fig.update_layout(height=350 * nroi)
        display(fig)

    def export_data(self, sigs=["415nm", "470nm-norm", "470nm-norm-zs"]) -> None:
        assert self.data_norm is not None, "Please process data first!"
        d = self.data_norm
        for sig in sigs:
            if self.prefix is not None:
                fpath = os.path.join(
                    self.out_path, "{}_{}.csv".format(self.prefix, sig)
                )
            else:
                fpath = os.path.join(self.out_path, "{}.csv".format(sig))
            d[d["signal"] == sig].drop(columns=["signal"]).to_csv(fpath, index=False)
            print("data saved to {}".format(fpath))


class NPMAlign(NPMBase):
    def __init__(self, fig_path="./figs/process", out_path="./output/process") -> None:
        super().__init__(fig_path, out_path)
        self.ts_dict = dict()
        self.data_align = None
        print("Alignment initialized")

    def set_ts(self, ts_ls: list = None, source: str = "local") -> None:
        if ts_ls is None:
            if source == "local":
                fs = pn.widgets.FileSelector(
                    directory=".",
                    root_directory="/",
                    only_files=True,
                    name="Select Timestamp Files",
                )
                fs.param.watch(self.on_ts_local, ["value"], onlychanged=True)
                display(fs)
            elif source == "remote":
                w_ts = widgets.FileUpload(
                    accept=".csv",
                    multiple=True,
                    description="Upload Timestamp Files",
                    tooltip="Select timestamps to align",
                    **self.wgt_opts,
                )
                w_ts.observe(self.on_ts_remote, names="value")
                display(w_ts)
        else:
            for ts_path in ts_ls:
                ts_name, ts = self.load_ts(ts_path)
                self.ts_dict[ts_name] = ts

    def on_ts_remote(self, change) -> None:
        for dfile in change["new"]:
            dname = dfile["name"]
            dat = dfile["content"].tobytes()
            self.ts_dict[dname] = pd.read_csv(
                io.BytesIO(dat), encoding="utf8", header=None
            )

    def on_ts_local(self, event) -> None:
        for dpath in event.new:
            ts_name, ts = self.load_ts(dpath)
            self.ts_dict[ts_name] = ts

    def load_ts(self, ts_path: str) -> pd.DataFrame:
        ts_name = os.path.split(ts_path)[1]
        if ts_name.endswith(".csv"):
            return ts_name, pd.read_csv(ts_path, header=None)
        elif ts_path.endswith(".xlsx"):
            return ts_name, pd.read_excel(ts_path, header=None)
        else:
            raise NotImplementedError("Unable to read {}".format(ts_path))

    def align_data(self, **kwargs) -> None:
        # self.data = label_bout(self.data, "Stimulation") # depracated
        self.data_align, self.ts = align_ts(self.data, self.ts_dict, **kwargs)

    def export_data(self) -> None:
        assert self.data_align is not None, "Please align ts first!"
        if self.prefix is not None:
            fpath = os.path.join(
                self.out_path, "{}_alignedevents.csv".format(self.prefix)
            )
        else:
            fpath = os.path.join(self.out_path, "alignedevents.csv")
        self.data_align.to_csv(fpath, index=False)
        print("data saved to {}".format(fpath))


class NPMPolling(NPMBase):
    def __init__(self, fig_path="./figs/process", out_path="./output/process") -> None:
        super().__init__(fig_path, out_path)
        self.param_evt_range = None
        self.param_evt_sep = 1
        self.param_evt_duration = 1
        print("Pooling initialized")

    def set_evt_range(self, evt_range: tuple = None) -> None:
        assert self.data is not None, "Please set data first!"
        self.fps = compute_fps(self.data)
        print("Assuming Framerate of {:.2f}".format(self.fps))
        if evt_range is None:
            txt_evt_range = widgets.Label(
                "Time (seconds) to Include Before and After Event"
            )
            w_evt_range = widgets.FloatRangeSlider(
                value=(-10, 10),
                min=-100,
                max=100,
                step=0.01,
                tooltip="Use the markers to specify the time (seconds) before and after each event",
                **self.wgt_opts,
            )
            self.param_evt_range = tuple(
                np.around(np.array((-10, 10)) * self.fps).astype(int)
            )
            w_evt_range.observe(self.on_evt_range, names="value")
            display(widgets.VBox([txt_evt_range, w_evt_range]))
        else:
            self.param_evt_range = evt_range

    def on_evt_range(self, change) -> None:
        self.param_evt_range = tuple(
            np.around(np.array(change["new"]) * self.fps).astype(int)
        )

    def set_evt_sep(self, evt_sep: float = None) -> None:
        if evt_sep is None:
            w_txt = widgets.Label("Minimum seperation between events (seconds)")
            w_evt_sep = widgets.FloatSlider(
                min=0, value=0, max=10, step=0.01, **self.wgt_opts
            )
            self.param_evt_sep = 0
            w_evt_sep.observe(self.on_evt_sep, names="value")
            display(widgets.VBox([w_txt, w_evt_sep]))
        else:
            self.param_evt_sep = evt_sep

    def on_evt_sep(self, change) -> None:
        self.param_evt_sep = float(change["new"])

    def set_evt_duration(self, evt_duration: float = None) -> None:
        if evt_duration is None:
            w_txt = widgets.Label("Minimum duration of events (seconds)")
            w_evt_dur = widgets.FloatSlider(
                min=0, value=0, max=10, step=0.01, **self.wgt_opts
            )
            self.param_evt_duration = 0
            w_evt_dur.observe(self.on_evt_dur, names="value")
            display(widgets.VBox([w_txt, w_evt_dur]))
        else:
            self.param_evt_duration = evt_duration

    def on_evt_dur(self, change) -> None:
        self.param_evt_duration = float(change["new"])

    def set_roi(self, roi_dict: dict = None) -> None:
        assert self.data is not None, "Please set data first!"
        if roi_dict is None:
            w_txt = widgets.Label("ROIs to analyze (CTRL/CMD click to Select Multiple)")
            w_roi = widgets.SelectMultiple(
                options=self.data.columns,
                tooltip="Region1G Region2R etc",
                **self.wgt_opts,
            )
            w_roi.observe(self.on_roi, names="value")
            display(widgets.VBox([w_txt, w_roi]))
        else:
            self.param_roi_dict = roi_dict

    def on_roi(self, change) -> None:
        rois = change["new"]
        self.param_roi_dict = {r: r for r in rois}

    def poll_events(self, tabs=None, **kwargs) -> None:
        self.evtdf = poll_events(
            self.data,
            self.param_evt_range,
            list(self.param_roi_dict.values()),
            self.param_evt_sep,
            self.param_evt_duration,
            **kwargs,
        )
        cmap = construct_cmap(self.evtdf["evt_id"].unique(), qualitative.Plotly)
        fig = plot_events(
            self.data,
            self.evtdf,
            list(self.param_roi_dict.values()),
            ts_col="ts_fp",
            cmap=cmap,
        )
        display(fig)
        fig.write_html(os.path.join(self.fig_path, "events.html"))
        fig = plot_polled_signal(
            self.evtdf,
            list(self.param_roi_dict.values()),
            tabs=tabs,
            fps=self.fps,
            cmap=cmap,
        )
        if tabs is None:
            fig.write_html(os.path.join(self.fig_path, "pooled_signals.html"))
        display(fig)

    def agg_polled_events(self) -> None:
        self.evt_agg = agg_polled_events(self.evtdf, list(self.param_roi_dict.values()))
        fig, figs = plot_agg_polled(self.evt_agg)
        for met, cur_fig in figs.items():
            cur_fig.write_html(
                os.path.join(self.fig_path, "polled_signals-{}.html".format(met))
            )
        display(fig)

    def label_trials(
        self,
        t_thres=7,
        labs={
            (0.002, 0.002): "LL",
            (0.003, 0.006): "RR",
            (0.006, 0.006): "RR",
            (0.002, 0.003): "LRr",
            (0.002, 0.006): "Lrur",
            (0.003, 0.001): "R-pellet",
            (0.001, 0.002): "pellet-L",
            (0.001, 0.006): "pellet-R",
            (0.003, 0.002): "RL",
            (0.006, 0.002): "RL",
            (0.006, 0.01): "RR-pellet",
        },
        lab_key="pulsewidth",
    ):
        evts = (
            self.evtdf[self.evtdf["fm_evt"] == 0]
            .sort_values("ts_fp")
            .reset_index(drop=True)
        )
        evts["tdiff"] = evts["ts_fp"].diff()
        evtdf = self.evtdf.set_index("evt_id")
        trial_df = []
        for idx, evt_row in evts.iterrows():
            if evt_row["tdiff"] < t_thres:
                evt_seq = tuple(evts.loc[idx - 1 : idx, lab_key].to_list())
                try:
                    lab = labs[evt_seq]
                except KeyError:
                    warnings.warn(
                        "Event sequence within {:.2f} seconds but not labeled: {}".format(
                            evt_row["tdiff"], evt_seq
                        )
                    )
                    continue
                evt_id = evts.loc[idx - 1, "evt_id"]
                cur_df = evtdf.loc[evt_id].reset_index()
                cur_df["label"] = lab
                cur_df["evt_id-next"] = evts.loc[idx, "evt_id"]
                trial_df.append(cur_df)
        self.trial_df = pd.concat(trial_df, ignore_index=True)
        fig = px.bar(
            self.trial_df.groupby("label")["evt_id"]
            .nunique()
            .rename("count")
            .reset_index(),
            x="label",
            y="count",
        )
        display(fig)

    def export_data(self, pvt_use_norm=True) -> None:
        assert self.evtdf is not None, "Please poll events first!"
        assert self.evt_agg is not None, "Please aggregate polled events first!"
        if self.prefix is not None:
            fpath = os.path.join(
                self.out_path, "{}_polledevents.csv".format(self.prefix)
            )
        else:
            fpath = os.path.join(self.out_path, "polledevents.csv")
        self.evtdf.to_csv(fpath, index=False)
        print("data saved to {}".format(fpath))
        if pvt_use_norm:
            val_vars = ["{}-norm".format(r) for r in self.param_roi_dict.values()]
        else:
            val_vars = list(self.param_roi_dict.values())
        evt_pvt = self.evtdf.melt(
            id_vars=["evt_id", "fm_evt"],
            value_vars=val_vars,
            var_name="roi",
        ).pivot(columns=["roi", "evt_id"], index="fm_evt", values="value")
        if self.prefix is not None:
            fpath = os.path.join(
                self.out_path, "{}_polledevents_pivot.csv".format(self.prefix)
            )
        else:
            fpath = os.path.join(self.out_path, "polledevents_pivot.csv")
        evt_pvt.to_csv(fpath)
        print("data saved to {}".format(fpath))
        trial_pvt = (
            self.trial_df.melt(
                id_vars=["label", "evt_id", "evt_id-next", "fm_evt"],
                value_vars=val_vars,
                var_name="roi",
            )
            .sort_values(["roi", "label", "evt_id", "fm_evt"])
            .pivot(
                columns=["roi", "label", "evt_id", "evt_id-next"],
                index="fm_evt",
                values="value",
            )
        )
        if self.prefix is not None:
            fpath = os.path.join(
                self.out_path, "{}_trials_pivot.csv".format(self.prefix)
            )
        else:
            fpath = os.path.join(self.out_path, "trials_pivot.csv")
        trial_pvt.to_csv(fpath)
        print("data saved to {}".format(fpath))
        if self.prefix is not None:
            fpath = os.path.join(
                self.out_path, "{}_polledevents_agg.csv".format(self.prefix)
            )
        else:
            fpath = os.path.join(self.out_path, "polledevents_agg.csv")
        self.evt_agg.to_csv(fpath, index=False)
        print("data saved to {}".format(fpath))
