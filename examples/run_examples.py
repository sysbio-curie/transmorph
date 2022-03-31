#!/usr/bin/env python3

import os

EXDIR = "/".join(__file__.split("/")[:-1]) + "/"

for f in os.listdir(EXDIR):
    if "." in f:
        continue
    for fp in os.listdir(EXDIR + f + "/"):
        if ".py" not in fp:
            continue
        os.system(f"python {EXDIR + f + '/' + fp}")
