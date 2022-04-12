#!/usr/bin/env python3

from transmorph.datasets import load_travaglini_10x
from transmorph.utils.plotting import plot_result


datasets = load_travaglini_10x()

# Plotting results

plot_result(
    datasets=list(datasets.values()),
    show=False,
    save=True,
    use_cache=True,
    caller_path=f"{__file__}",
)

plot_result(
    datasets=list(datasets.values()),
    color_by="compartment",
    show=False,
    save=True,
    use_cache=True,
    caller_path=f"{__file__}",
)
