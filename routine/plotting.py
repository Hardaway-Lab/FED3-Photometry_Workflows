import plotly.express as px


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
