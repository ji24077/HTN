"""Render a Mandelbrot image; use --size 16 for a bounded probe."""

import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=256)
size = parser.parse_args().size
pixels = bytearray()
for y in range(size):
    for x in range(size):
        c = complex(x / size * 3 - 2, y / size * 2 - 1)
        z = 0j
        count = 0
        while abs(z) <= 2 and count < 64:
            z = z * z + c
            count += 1
        pixels.extend((count * 4 % 256, count * 7 % 256, count * 11 % 256))
out = Path(os.environ["DISPATCH_OUTPUT_DIR"])
(out / "mandelbrot.ppm").write_bytes(f"P6\n{size} {size}\n255\n".encode() + pixels)
