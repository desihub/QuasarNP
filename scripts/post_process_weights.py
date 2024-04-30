#!/usr/bin/env python3

"""
Post process a weights file to add wavelength information to it.
"""

import h5py
import numpy as np

import sys
sys.path.insert(0, "../")
from quasarnp.utils import wave, linear_wave

import argparse

p = argparse.ArgumentParser()
p.add_argument("-i", "--in_file", type=str, required=True, help="original weights file to process")
p.add_argument("-o", "--out", type=str, help="Where to save output file.")
p.add_argument("--linear", required=False, action="store_true", help="whether to add linear or logarithmic information")
args = p.parse_args()

with h5py.File(args.in_file, "r") as f1:
    with h5py.File(args.out, "w") as f2:
        # Copies the extent data to the new file
        for k in f1.keys():
            f1[k].copy(f1[k], f2["/"])

        # Copies the attributes (which includes the model config)
        for k in f1.attrs.keys():
            f2.attrs[k] = f1.attrs[k]

        # Add grid information
        f2.attrs["grid_spacing"] = "linear" if args.linear else "logarithmic"
        f2["model_grid"] = linear_wave if args.linear else wave