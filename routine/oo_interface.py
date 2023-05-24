import io
import os

import pandas as pd
from IPython.display import display
from ipywidgets import Layout, widgets

from routine.plotting import plot_signals
from routine.processing import photobleach_correction
from routine.utilities import load_data


class NPMBase:
    def __init__(self, fig_path="./figs/process", out_path="./output/process") -> None:
        self.wgt_opts = {
            "style": {"description_width": "initial"},
            "layout": Layout(width="80%"),
        }
        self.data = None
        self.fig_path = fig_path
        self.out_path = out_path
        os.makedirs(self.fig_path, exist_ok=True)
        os.makedirs(self.out_path, exist_ok=True)

    def set_data(self, dpath: str = None) -> None:
        if dpath is None:
            w_data = widgets.FileUpload(
                accept=".csv",
                multiple=False,
                description="Upload Data File",
                tooltip="Select data file to analyze",
                **self.wgt_opts,
            )
            w_data.observe(self.on_upload, names="value")
            display(w_data)
        else:
            self.data = pd.read_csv(dpath)

    def on_upload(self, change) -> None:
        dat = change["new"][0]["content"].tobytes()
        self.data = pd.read_csv(io.BytesIO(dat), encoding="utf8")


class NPMProcess(NPMBase):
    def __init__(self, fig_path="./figs/process", out_path="./output/process") -> None:
        super().__init__(fig_path, out_path)
        self.param_nfm_discard = None
        self.param_led_dict = {7: "initial", 1: "415nm", 2: "470nm", 4: "560nm"}
        self.param_roi_dict = None
        self.param_base_sig = None
        self.data_norm = None
        print("Process initialized")

    def set_nfm_discard(self, nfm: int = None) -> None:
        assert self.data is not None, "Please set data first!"
        if nfm is None:
            w_txt = widgets.Label(
                "Number of Frames to Discard from Beginning of Recording"
            )
            w_nfm = widgets.IntSlider(
                min=0,
                value=0,
                max=self.data["FrameCounter"].max(),
                step=1,
                tooltip="Cropping data points at the beginning of the recording can improve curve fitting. 100 frames is a good start",
                **self.wgt_opts,
            )
            self.param_nfm_discard = 0
            w_nfm.observe(self.on_nfm, names="value")
            display(widgets.VBox([w_txt, w_nfm]))
        else:
            self.param_nfm_discard = nfm

    def on_nfm(self, change) -> None:
        self.param_nfm_discard = int(change["new"])

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
            self.param_roi_dict = roi_dict

    def on_roi_name(self, change) -> None:
        k, v = change["owner"].placeholder, change["new"]
        self.param_roi_dict[k] = v

    def set_baseline(self, base_sig: str = None):
        assert self.data is not None, "Please set data first!"
        if base_sig is None:
            w_base = widgets.ToggleButtons(
                value="415nm",
                options=["415nm", "470nm", "560nm"],
                description="Channel to use as Reference Signal:",
                disabled=False,
                button_style="",
                tooltips=[
                    "Best for most recordings",
                    "Alternative for certain neurotransmitter sensors",
                    "Alternative for certain red-shifted sensors",
                ],
                **self.wgt_opts,
            )
            w_base.observe(self.on_baseline, names="value")
            self.param_base_sig = "415nm"
            display(w_base)

    def on_baseline(self, change) -> None:
        self.param_base_sig = change["new"]

    def load_data(self) -> None:
        assert self.data is not None, "Please set data first!"
        assert self.param_roi_dict is not None, "Please set ROIs first!"
        assert self.param_nfm_discard is not None, "Please set frames to discard first!"
        self.data = load_data(
            self.data, self.param_nfm_discard, self.param_led_dict, self.param_roi_dict
        )
        fig = plot_signals(self.data, list(self.param_roi_dict.values()))
        fig.write_html(os.path.join(self.fig_path, "raw_signals.html"))
        nroi = len(self.param_roi_dict)
        fig.update_layout(height=350 * nroi)
        display(fig)

    def photobleach_correction(self) -> None:
        assert self.data is not None, "Please set data first!"
        assert self.param_roi_dict is not None, "Please set ROIs first!"
        assert self.param_base_sig is not None, "Please set baseline signal first!"
        self.data_norm = photobleach_correction(
            self.data, list(self.param_roi_dict.values()), self.param_base_sig
        )
        fig = plot_signals(
            self.data_norm,
            list(self.param_roi_dict.values()),
            group_dict={
                "415nm": "415nm",
                "415nm-fit": "415nm",
                "470nm": "470nm",
                "470nm-norm": "470nm",
            },
        )
        fig.write_html(os.path.join(self.fig_path, "photobleaching_correction.html"))
        nroi = len(self.param_roi_dict)
        fig.update_layout(height=350 * nroi)
        display(fig)

    def export_data(self, sigs=["415nm", "470nm-norm"]) -> None:
        assert self.data_norm is not None, "Please process data first!"
        d = self.data_norm
        ds_path = os.path.join(self.out_path, "signals")
        os.makedirs(ds_path, exist_ok=True)
        for sig in sigs:
            fpath = os.path.join(ds_path, "{}.csv".format(sig))
            d[d["signal"] == sig].drop(columns=["signal"]).to_csv(fpath, index=False)
            print("data saved to {}".format(fpath))
