"""Validate requested dimensions, RGB byte count, and a nonuniform render."""

import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=256)
size = parser.parse_args().size
raw = (Path(os.environ["DISPATCH_OUTPUT_DIR"]) / "mandelbrot.ppm").read_bytes()
magic, dimensions, maximum, pixels = raw.split(b"\n", 3)
if magic != b"P6" or dimensions != f"{size} {size}".encode() or maximum != b"255":
    raise ValueError("Image header or dimensions do not match the requested render")
if len(pixels) != size * size * 3 or len(set(pixels)) < 2:
    raise ValueError("Image is incomplete or uniform")
