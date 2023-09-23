import os
from routine.oo_interface import NPMProcess
from routine.oo_interface import NPMAlign
from routine.oo_interface import NPMPooling

IN_DPATH = {"d5": "./data/example/d5/In0.csv", "d6": "./data/example/d6/In0.csv"}
IN_TS = {
    "d5": [
        "./data/example/d5/In1.csv",
        "./data/example/d5/In2.xlsx",
        "./data/example/d5/In4.xlsx",
    ],
    "d6": ["./data/example/d6/In5.xlsx"],
}
PARAM_ROIS = {"G0": "G0", "G1": "G1"}
OUT_PATH = "./output/test"
FIG_PATH = "./figs/test"

for ds_name, dpath in IN_DPATH.items():
    # process
    process = NPMProcess()
    process.set_data(dpath)
    process.set_paths(out_path=OUT_PATH, fig_path=FIG_PATH)
    process.set_roi(PARAM_ROIS)
    process.set_roi_names()
    process.set_nfm_discard()
    process.set_baseline()
    process.load_data()
    process.photobleach_correction()
    process.export_data()
    # align
    align = NPMAlign()
    align.set_data(os.path.join(OUT_PATH, "signals", "470nm-norm.csv"))
    align.set_paths(out_path=OUT_PATH, fig_path=FIG_PATH)
    align.set_ts(IN_TS[ds_name])
    align.align_data()
    align.export_data()
    # pooling
    pooling = NPMPooling()
    pooling.set_data(os.path.join(OUT_PATH, "aligned", "master.csv"))
    pooling.set_paths(out_path=OUT_PATH, fig_path=FIG_PATH)
    pooling.set_roi(PARAM_ROIS)
    pooling.set_evt_range()
    pooling.pool_events()
