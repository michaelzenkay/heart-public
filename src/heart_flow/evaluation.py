"""Segmentation evaluation with equal weight per anatomy/background group."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_manifest, load_sample
from .model import load_model, overlap
from .preprocessing import tensors


def group_mean(rows, key="dice"):
    """Average frames within series, then series within groups, then groups.

    Equal weight at each level prevents longer cycles or groups with more
    acquisition variants from dominating the reported score.
    """
    if not rows:
        raise ValueError("Cannot evaluate an empty split")
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], {}).setdefault(row["series"], []).append(
            row[key]
        )
    return float(
        np.mean(
            [
                np.mean([np.mean(values) for values in group.values()])
                for group in groups.values()
            ]
        )
    )


@torch.inference_mode()
def evaluate(model, cache, rows, device="cpu"):
    """Evaluate cached acquisitions and return per-frame metrics and grouped Dice.

    The model must already be on device. Exclude padding through the FOV mask;
    use mixed precision on CUDA and float32 on CPU, without augmentation.
    """
    model.eval()
    device = torch.device(device)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    metrics = []
    for row in rows:
        arrays = load_sample(str(Path(cache) / row["file"]), row["sha256"])
        x, y, fov = tensors(*arrays, device)
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
        ):
            logits = model(x)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite prediction")
        metrics.append(
            dict(
                series=row["series"],
                group=row["group"],
                frame=row["frame"],
                **overlap(logits, y, fov),
            )
        )
    return dict(group_dice=group_mean(metrics), frames=metrics)


def main():
    """Evaluate a selected split after checking cache/checkpoint preprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = load_manifest(args.cache)
    model, checkpoint = load_model(args.checkpoint, args.device)
    expected = {key: manifest[key] for key in ("spacing_mm", "velocity_scale_mps")}
    if checkpoint["preprocessing"] != expected:
        raise ValueError("Cache preprocessing differs from the checkpoint")
    rows = [r for r in manifest["samples"] if r["split"] == args.split]
    print(
        json.dumps(
            evaluate(model, args.cache, rows, args.device), indent=2, allow_nan=False
        )
    )
