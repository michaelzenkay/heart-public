"""Whole-volume magnitude/XYZ training with grouped sampling and validation."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .data import load_manifest, load_sample, sha256, write_json
from .evaluation import evaluate
from .model import CHANNELS, VERSION, loss_fn, model_for
from .preprocessing import augment, tensors


def sample_balanced(rows, rng):
    """Draw a group, then a series, then a frame, uniformly at each level.

    This keeps anatomy groups with many related acquisitions from dominating
    training simply because they contribute more cached frames.
    """
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], {}).setdefault(row["series"], []).append(row)
    group = groups[rng.choice(sorted(groups))]
    return rng.choice(group[rng.choice(sorted(group))])


def train(
    cache,
    output,
    epochs=80,
    steps_per_epoch=64,
    lr=1e-3,
    seed=42,
    device="cuda",
    width=8,
):
    """Train a four-channel 3D U-Net, one augmented whole volume per update.

    Use grouped sampling, BCE plus soft Dice, and AdamW. CUDA uses BF16 when
    available or scaled FP16; CPU uses float32. Save last/best model checkpoints
    and epoch metrics in a new directory, choosing best by grouped validation
    Dice. Return the metric history; the test split is not used for selection.
    """
    if epochs < 1 or steps_per_epoch < 1 or width < 2 or not np.isfinite(lr) or lr <= 0:
        raise ValueError(
            "Positive epochs, steps and learning rate; width >= 2 required"
        )
    cache, output = Path(cache), Path(output)
    manifest = load_manifest(cache)
    training = [r for r in manifest["samples"] if r["split"] == "train"]
    validation = [r for r in manifest["samples"] if r["split"] == "val"]
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; request --device cpu")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    config = dict(mode="mag-xyz", blocks=3, width=width)
    model = model_for(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    amp = device.type == "cuda"
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaled = amp and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaled, init_scale=1024.0)
    preprocessing = {key: manifest[key] for key in ("spacing_mm", "velocity_scale_mps")}
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "run.json",
        dict(
            config=config,
            preprocessing=preprocessing,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            lr=lr,
            seed=seed,
            cache_sha256=sha256(cache / "manifest.json"),
        ),
    )
    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            row = sample_balanced(training, rng)
            x, y, fov = load_sample(str(cache / row["file"]), row["sha256"])
            flips = [axis for axis in range(3) if rng.random() < 0.5]
            x, y, fov = augment(x, y, fov, flips, (0, 1, rng.randrange(4)))
            # Change magnitude contrast without rescaling physical velocity.
            x[0] = np.clip(x[0] * rng.uniform(0.85, 1.15), 0, 1)
            x, y, fov = tensors(x, y, fov, device)
            for _ in range(8):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                    loss = loss_fn(model(x), y, fov)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                finite = all(
                    bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                    if p.grad is not None
                )
                if finite:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 5.0, error_if_nonfinite=True
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(float(loss.detach()))
                    break
                if not scaled:
                    raise ValueError("Nonfinite gradients")
                # FP16 overflow can come from gradient scaling, not the data.
                # Retry the same batch at a lower scale before abandoning it.
                scaler.update(new_scale=scaler.get_scale() / 2)
            else:
                raise ValueError("Repeated FP16 gradient overflow")
        score = evaluate(model, cache, validation, device)["group_dice"]
        result = dict(
            epoch=epoch, train_loss=float(np.mean(losses)), validation_group_dice=score
        )
        history.append(result)
        write_json(output / "metrics.json", history)
        checkpoint = dict(
            version=VERSION,
            channels=CHANNELS,
            config=config,
            preprocessing=preprocessing,
            epoch=epoch,
            model_state=model.state_dict(),
            validation_group_dice=score,
        )
        torch.save(checkpoint, output / "last.pth")
        if score > best:
            best = score
            torch.save(checkpoint, output / "best.pth")
        print(result, flush=True)
    return history


def main():
    """Run training with cache/output paths and hyperparameters from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(**vars(args))
