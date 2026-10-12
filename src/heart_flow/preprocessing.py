"""World-grid magnitude/velocity features and vector-aware augmentation."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .io import point_volume, read_pv_image
from .velocity import decode_velocity


def native_affine(metadata):
    """Return native dimensions, voxel-to-world linear map and origin in metres.

    The origin includes the stored grid offset, so world XYZ = affine @ IJK
    + origin, including for fractional voxel indices.
    """
    dims = np.asarray(metadata["dimensions"], int)
    spacing = np.asarray(metadata["spacing"], float)
    direction = np.asarray(metadata["direction_matrix"], float)
    if np.any(spacing <= 0) or not np.allclose(
        direction.T @ direction, np.eye(3), atol=1e-5
    ):
        raise ValueError("Expected positive spacing and orthonormal direction matrix")
    a = direction @ np.diag(spacing)
    origin = np.asarray(metadata["origin"], float) + a @ np.asarray(
        metadata.get("offset", [0, 0, 0]), float
    )
    return dims, a, origin


def world_grid(metadata, spacing_mm):
    """Define an isotropic world XYZ grid enclosing the native voxel centres.

    Return shape, origin in metres and spacing in millimetres. This common
    orientation lets input channels always mean magnitude, Vx, Vy and Vz.
    """
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("Spacing must be positive")
    dims, a, origin = native_affine(metadata)
    corners = (
        np.array(np.meshgrid(*[(0, n - 1) for n in dims], indexing="ij"))
        .reshape(3, -1)
        .T
    )
    xyz = corners @ a.T + origin
    low = xyz.min(0)
    step = spacing_mm / 1000
    shape = np.ceil((xyz.max(0) - low) / step - 1e-9).astype(int) + 1
    if np.prod(shape) > 8000000:
        raise ValueError(
            "Requested full-volume grid exceeds 8 million voxels; choose coarser spacing"
        )
    return dict(
        shape=shape.tolist(),
        origin_m=low.tolist(),
        spacing_mm=spacing_mm,
        direction="world XYZ identity; array axes X/Y/Z",
    )


def resample_world(volume, metadata, target, order):
    """Resample a native IJK scalar volume onto the canonical world XYZ grid.

    Use order=1 for continuous inputs and order=0 for labels/validity. Coordinates
    map from output to input, as required by scipy; outside values are zero.
    """
    _, a, origin = native_affine(metadata)
    transform = np.linalg.solve(a, np.eye(3) * target["spacing_mm"] / 1000)
    offset = np.linalg.solve(a, np.asarray(target["origin_m"]) - origin)
    return ndimage.affine_transform(
        np.asarray(volume, dtype=np.float32),
        transform,
        offset,
        output_shape=tuple(target["shape"]),
        order=order,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def preprocess(path, spacing_mm=3.0, velocity_scale_mps=2.0, with_target=True):
    """Build four model channels from one acquisition on the canonical grid.

    Scale magnitude by its positive 99th percentile and clip to [0, 1]; divide
    XYZ velocity by velocity_scale_mps. Invalid velocities become zero only for
    network input. Return (4, X, Y, Z) features, optional target, field of view,
    velocity-validity mask and preprocessing metadata. Labels never form inputs.
    """
    if not np.isfinite(velocity_scale_mps) or velocity_scale_mps <= 0:
        raise ValueError("Velocity scale must be positive")
    image = read_pv_image(Path(path))
    meta = image.metadata
    dims = tuple(meta["dimensions"])
    decoded = decode_velocity(
        image.point_data["real"],
        image.point_data["imag"],
        image.field_data["kvs"],
        image.field_data["enc_dirs"],
        image.field_data["vencs"],
    )
    mag = decoded["magnitude"].reshape(dims, order="F")
    valid = decoded["valid"].reshape(dims, order="F")
    velocity = decoded["velocity"].reshape((*dims, 3), order="F")
    target = world_grid(meta, spacing_mm)
    positive = mag[np.isfinite(mag) & (mag > 0)]
    scale = float(np.percentile(positive, 99)) if len(positive) else 0.0
    if scale <= 0:
        raise ValueError("Acquisition has no finite magnitude signal")
    mag = np.clip(np.nan_to_num(mag / scale, nan=0, posinf=0, neginf=0), 0, 1)
    valid = valid & np.all(np.isfinite(velocity), axis=-1)
    velocity = np.where(valid[..., None], velocity, 0)
    features = np.stack(
        [resample_world(mag, meta, target, 1)]
        + [
            resample_world(velocity[..., i] / velocity_scale_mps, meta, target, 1)
            for i in range(3)
        ]
    )
    fov = resample_world(np.ones(dims, np.float32), meta, target, 0) > 0.5
    valid_world = resample_world(valid, meta, target, 0) > 0.5
    features[1:, ~valid_world] = 0
    label = None
    if with_target:
        # Labels are read only here, after every input feature has been formed.
        if "mask" not in image.point_data:
            raise ValueError("Supplied native-grid aorta target absent")
        label = (
            resample_world(
                point_volume(image, image.point_data["mask"]) > 0, meta, target, 0
            )
            > 0.5
        )
    info = dict(
        native_grid=meta,
        world_grid=target,
        magnitude_p99=scale,
        velocity_scale_mps=velocity_scale_mps,
        invalid_velocity_fraction=float(1 - valid.mean()),
        high_venc_channels=decoded["channels"],
        reference_channel=decoded["reference_channel"],
        direction_condition=decoded["direction_condition"],
        validity_note="Invalid velocity filled with zero; validity saved for QC, not a fifth input. No signal/speed/residual threshold mask.",
        reconstruction="highest-VENC direction least squares, no unwrapping or background-phase correction",
    )
    return features.astype(np.float32), label, fov, valid_world, info


def augment(x, y, fov, flips=(), rotation=None):
    """Flip/rotate features, target and field of view together, returning copies.

    x is (4, X, Y, Z); y and fov are (X, Y, Z). flips lists spatial axes;
    rotation=(a, b, k) applies k quarter-turns in that plane. Velocity components
    must also change sign or exchange axes: they are vectors, not scalar images.
    """
    x = x.copy()
    y = y.copy()
    fov = fov.copy()
    for axis in flips:
        x = np.flip(x, axis=axis + 1).copy()
        y = np.flip(y, axis=axis).copy()
        fov = np.flip(fov, axis=axis).copy()
        x[axis + 1] *= -1
    if rotation is not None:
        a, b, k = rotation
        for _ in range(k % 4):
            x = np.rot90(x, axes=(a + 1, b + 1)).copy()
            y = np.rot90(y, axes=(a, b)).copy()
            fov = np.rot90(fov, axes=(a, b)).copy()
            va = x[a + 1].copy()
            vb = x[b + 1].copy()
            x[a + 1] = -vb
            x[b + 1] = va
    return x, y, fov


def tensors(x, y, fov, device):
    """Add a batch dimension and pad spatial axes to multiples of eight.

    Return features (1, 4, X, Y, Z), target and field of view (1, 1, X, Y, Z)
    on device, with padded sizes. Zero FOV padding excludes it from loss/metrics.
    """
    # No center cropping: pad only, and exclude padding from loss and metrics.
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    y = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    fov = torch.from_numpy(fov.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    pad = []
    for n in reversed(x.shape[-3:]):
        pad.extend((0, (-n) % 8))
    return F.pad(x, pad), F.pad(y, pad), F.pad(fov, pad)
