"""Four-channel 3D U-Net, segmentation loss and overlap metrics."""

import torch
import torch.nn.functional as F
from unet import UNet3D

VERSION = 1
CHANNELS = ["magnitude", "velocity_x", "velocity_y", "velocity_z"]


def model_for(config):
    """Build a padded 3D U-Net with configured depth/width and batch normalisation.

    Input is (B, 4, X, Y, Z): magnitude and world XYZ velocity. Output is one
    segmentation logit per voxel, (B, 1, X, Y, Z); sigmoid gives probabilities.
    """
    if config["mode"] != "mag-xyz":
        raise ValueError("Expected magnitude plus XYZ velocity")
    return UNet3D(
        in_channels=4,
        out_classes=1,
        num_encoding_blocks=config["blocks"],
        out_channels_first_layer=config["width"],
        padding=True,
        normalization="batch",
    )


def load_model(path, device="cpu"):
    """Return the model in evaluation mode and its checkpoint metadata.

    Validate the format/channel order and load weights strictly on device.
    Only load trusted checkpoints: torch.load here permits Python objects.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["version"] != VERSION or checkpoint["channels"] != CHANNELS:
        raise ValueError("Expected the four-channel whole-volume checkpoint")
    model = model_for(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), checkpoint


def loss_fn(logits, y, fov):
    """Combine mean binary cross-entropy and soft Dice loss inside the image FOV.

    Inputs share shape (B, 1, X, Y, Z). Both terms exclude padded voxels;
    soft Dice sums over the supplied batch, which is one volume during training.
    """
    bce = (
        F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none") * fov
    ).sum() / fov.sum().clamp_min(1)
    prob = torch.sigmoid(logits.float()) * fov
    dice = 1 - (2 * (prob * y).sum() + 1e-5) / (prob.sum() + (y * fov).sum() + 1e-5)
    return bce + dice


def overlap(logits, y, fov):
    """Count TP/FP/FN and compute Dice, precision and recall inside the FOV.

    Positive logits mean probability > 0.5. Empty prediction and target give
    Dice 1; precision is 0 with no predictions and recall is 1 with no targets.
    """
    pred = logits > 0
    truth = y > 0.5
    valid = fov > 0.5
    tp = int((pred & truth & valid).sum())
    fp = int((pred & ~truth & valid).sum())
    fn = int((~pred & truth & valid).sum())
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        dice=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 1.0,
    )
