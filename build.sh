#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade build
python3 -m build
