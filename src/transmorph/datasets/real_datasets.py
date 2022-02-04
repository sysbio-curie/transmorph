#!/usr/bin/env python3

import numpy as np
from os.path import dirname
import scanpy as sc

DPATH = "%s/data/%s"


def load_dataset(module_path, filename):
    return np.loadtxt(DPATH % (module_path, filename), delimiter=",")


def load_spirals():
    xs = sc.AnnData(load_dataset(dirname(__file__), "spiralA.csv"))
    yt = sc.AnnData(load_dataset(dirname(__file__), "spiralB.csv"))
    return xs, yt


def load_spirals_labels():
    xs = load_dataset(dirname(__file__), "spiralA_labels.csv")
    yt = load_dataset(dirname(__file__), "spiralB_labels.csv")
    return xs, yt


def load_cell_cycle():
    xs = load_dataset(dirname(__file__), "pdx352.csv")
    yt = load_dataset(dirname(__file__), "chla9.csv")
    return xs, yt
