#!/usr/bin/env python3

from transmorph.datasets import load_zhou_10x
from transmorph.utils.plotting import plot_result

datasets = load_zhou_10x()

# Plotting results

for ds_id in datasets:
    plot_result(
        datasets=datasets[ds_id],
        color_by="malignant",
        title=f"Dataset {ds_id} (n,d={datasets[ds_id].X.shape})",
        show=False,
        save=True,
        caller_path=f"{__file__}",
        suffix=ds_id + "_malignant",
    )
