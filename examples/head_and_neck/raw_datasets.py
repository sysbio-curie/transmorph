#!/usr/bin/env python3

from transmorph.datasets import load_chen_10x
from transmorph.utils.plotting import plot_result


datasets = load_chen_10x()

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
    color_by="cell_type",
    show=False,
    save=True,
    use_cache=True,
    caller_path=f"{__file__}",
)

plot_result(
    datasets=list(datasets.values()),
    color_by="ebv",
    show=False,
    save=True,
    use_cache=True,
    caller_path=f"{__file__}",
)
