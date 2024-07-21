# %% imports and definition
import os

import pandas as pd

from routine.ts_alignment import classify_opto

IN_TS_PATH = "./data/testing/opto_class/"
OUT_PATH = "./output/testing/opto_class"

os.makedirs(OUT_PATH, exist_ok=True)

# %% process timestamps
files = os.listdir(IN_TS_PATH)
csv_files = list(filter(lambda fn: fn.lower().endswith(".csv"), files))
for csv_f in csv_files:
    ts = pd.read_csv(os.path.join(IN_TS_PATH, csv_f))
    ts_out = classify_opto(ts)
    ts_out.to_csv(os.path.join(OUT_PATH, csv_f), index=False)
