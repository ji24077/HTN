"""Play one short completion note through the default Windows audio device."""

import argparse
import math
import struct
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / ".cache/completion-bell.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    rate, duration = 44100, 0.55
    samples = []
    for i in range(int(rate * duration)):
        t = i / rate
        envelope = min(1, t / 0.008) * min(1, (duration - t) / 0.02) * math.exp(-7 * t)
        samples.append(int(7000 * envelope * math.sin(2 * math.pi * 880 * t)))
    with wave.open(str(path), "wb") as sound:
        sound.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        sound.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    if not args.prepare_only:
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_NODEFAULT)


if __name__ == "__main__":
    main()
