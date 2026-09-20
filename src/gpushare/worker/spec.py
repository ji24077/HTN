"""Jack — N-3. GPU introspection. classify_chip lives in contracts, not here."""

from gpushare.contracts import ChipClass, classify_chip


def collect_spec() -> dict:
    import torch

    p = torch.cuda.get_device_properties(0)
    is_amd = torch.version.hip is not None
    cc = float(f"{p.major}.{p.minor}")
    vram_gb = round(p.total_memory / 1e9, 1)
    chip_class: ChipClass = classify_chip(
        gpu_name=p.name, cc=cc, vram_gb=vram_gb, is_amd=is_amd
    )
    return {
        "gpu": p.name,
        "vram_gb": vram_gb,
        "cc": cc,
        "vendor": "amd" if is_amd else "nvidia",
        "chip_class": chip_class,
    }
