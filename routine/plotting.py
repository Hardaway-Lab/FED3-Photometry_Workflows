import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utilities import enumerated_product


def plot_signals(data, rois, fps=30, default_window=(0, 10)):
    dat_long = data[["Timestamp", "signal"] + rois].melt(
        id_vars=["Timestamp", "signal"], var_name="roi", value_name="raw"
    )
    t0 = dat_long["Timestamp"].min()
    dat_long["time (s)"] = (dat_long["Timestamp"] - t0) / fps
    return px.line(
        dat_long,
        x="time (s)",
        y="raw",
        facet_row="roi",
        color="signal",
        range_x=default_window,
    )


def plot_events(evt_df):
    return px.line(
        evt_df,
        x="evt_fm",
        y="Region0G",
        color="evt_id",
        facet_row="event",
        facet_col="signal",
    )


def facet_plotly(
    data: pd.DataFrame,
    facet_row: str,
    facet_col: str,
    title_dim: str = None,
    specs: dict = None,
    col_wrap: int = None,
    **kwargs,
):
    row_crd = data[facet_row].unique()
    col_crd = data[facet_col].unique()
    layout_ls = []
    iiter = 0
    for (ir, ic), (r, c) in enumerated_product(row_crd, col_crd):
        dat_sub = data[(data[facet_row] == r) & (data[facet_col] == c)]
        if not len(dat_sub) > 0:
            continue
        if title_dim is not None:
            title = dat_sub[title_dim].unique().item()
        else:
            if facet_row == "DUMMY_FACET_ROW":
                title = "{}={}".format(facet_col, c)
            elif facet_col == "DUMMY_FACET_COL":
                title = "{}={}".format(facet_row, r)
            else:
                title = "{}={}; {}={}".format(facet_row, r, facet_col, c)
        if col_wrap is not None:
            ir = iiter // col_wrap
            ic = iiter % col_wrap
            iiter += 1
        layout_ls.append(
            {"row": ir, "col": ic, "row_label": r, "col_label": c, "title": title}
        )
    layout = pd.DataFrame(layout_ls).set_index(["row_label", "col_label"])
    if col_wrap is not None:
        nrow, ncol = int(layout["row"].max() + 1), int(layout["col"].max() + 1)
    else:
        nrow, ncol = len(row_crd), len(col_crd)
    if specs is not None:
        specs = np.full((nrow, ncol), specs).tolist()
    fig = make_subplots(
        rows=nrow,
        cols=ncol,
        subplot_titles=layout["title"].values,
        specs=specs,
        **kwargs,
    )
    return fig, layout


def construct_layout(row_crd, col_crd, row_name="", col_name="", **kwargs):
    layout_ls = []
    for (ir, ic), (r, c) in enumerated_product(row_crd, col_crd):
        tt = ""
        if row_name:
            tt = tt + row_name + ": " + r
        else:
            tt = tt + r
        tt + " "
        if col_name:
            tt = tt + col_name + ": " + c
        else:
            tt = tt + c
        layout_ls.append(
            {"row": ir, "col": ic, "row_label": r, "col_label": c, "title": tt}
        )
    layout = pd.DataFrame(layout_ls)
    nrow, ncol = len(row_crd), len(col_crd)
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=layout["title"].values)
    return fig, layout


def plot_peaks(data, rois, fps=30, default_window=None):
    sigs = data["signal"].unique()
    t0 = data["Timestamp"].min()
    data["t"] = (data["Timestamp"] - t0) / fps
    fig, layout = construct_layout(rois, sigs, "roi", "signal", shared_xaxes=True)
    for (roi, sig), ly in layout.groupby(["row_label", "col_label"]):
        dat = data[data["signal"] == sig]
        pks = dat[dat[roi + "-pks"]]
        ly = ly.squeeze()
        if len(dat) > 0:
            fig.add_trace(
                go.Scatter(
                    x=dat["t"],
                    y=dat[roi],
                    mode="lines",
                    name="signal",
                    legendgroup="signal",
                    line={"color": "#636EFA"},
                    range_x=default_window,
                ),
                row=ly["row"] + 1,
                col=ly["col"] + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=dat["t"],
                    y=dat[roi + "-freq"],
                    mode="lines",
                    name="freq",
                    legendgroup="freq",
                    line={"color": "grey"},
                    range_x=default_window,
                ),
                row=ly["row"] + 1,
                col=ly["col"] + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=pks["t"],
                    y=pks[roi],
                    mode="markers",
                    marker={"size": 8, "color": "#EF553B", "symbol": "cross"},
                    name="peaks",
                    range_x=default_window,
                ),
                row=ly["row"] + 1,
                col=ly["col"] + 1,
            )
    return fig
