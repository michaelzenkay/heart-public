"""Decode XYZ velocity in m/s from relative complex phase."""

import numpy as np
from scipy import ndimage

from .io import xyz_to_voxel


def encoding_channels(kvs, directions, vencs):
    """Select the reference and highest-VENC channel for each encoding direction.

    Sensitivity k has units rad/(m/s), VENC is in m/s, and directions are unit
    world XYZ vectors. Require k * VENC = pi and enough independent directions
    to recover all three velocity components. Return channel indices.
    """
    k = np.asarray(kvs, float).reshape(-1)
    d = np.asarray(directions, float)
    venc = np.asarray(vencs, float).reshape(-1)
    if d.shape != (len(k), 3) or len(venc) != len(k):
        raise ValueError("Encoding metadata shapes disagree")
    if not (
        np.all(np.isfinite(k)) and np.all(np.isfinite(d)) and np.all(np.isfinite(venc))
    ):
        raise ValueError("Nonfinite encoding metadata")
    ref = np.flatnonzero((np.linalg.norm(d, axis=1) < 1e-8) & (np.abs(k) < 1e-8))
    if len(ref) != 1:
        raise ValueError("Expected one non-encoded reference")
    encoded = np.flatnonzero(k > 0)
    if not len(encoded) or np.any(venc[encoded] <= 0):
        raise ValueError("Missing positive encoding sensitivity")
    if not np.allclose(k[encoded] * venc[encoded], np.pi, rtol=1e-5):
        raise ValueError("k and VENC disagree")
    if not np.allclose(np.linalg.norm(d[encoded], axis=1), 1, atol=1e-6):
        raise ValueError("Directions must be unit vectors")
    groups = []
    for j in encoded:
        group = next((g for g in groups if np.allclose(d[g[0]], d[j], atol=1e-6)), None)
        if group is None:
            groups.append([int(j)])
        else:
            group.append(int(j))
    high = np.array([max(g, key=lambda j: venc[j]) for g in groups])
    if np.linalg.matrix_rank(d[high]) < 3:
        raise ValueError("Encodings cannot recover 3D velocity")
    return int(ref[0]), high


def decode_velocity(real, imag, kvs, directions, vencs):
    """Decode complex signals shaped (..., channels) to velocity (..., 3) in m/s.

    Remove reference phase, then solve direction projections by equal-weight
    least squares using the highest VENC per direction. No mask or phase
    unwrapping is used. Return velocity, reference magnitude and diagnostics;
    tiny/undefined signals give NaN velocity. Residuals are not calibrated SNR.
    """
    s = np.asarray(real, dtype=np.float64) + 1j * np.asarray(imag, dtype=np.float64)
    k = np.asarray(kvs, float).reshape(-1)
    d = np.asarray(directions, float)
    ref, high = encoding_channels(k, d, vencs)
    if s.shape[-1] != len(k):
        raise ValueError("Signal channels do not match metadata")
    magnitude = np.abs(s[..., ref])
    scale = float(np.max(magnitude))
    amp = np.minimum(magnitude, np.min(np.abs(s[..., high]), axis=-1))
    valid = np.all(np.isfinite(s), axis=-1) & (
        amp > max(scale * 1e-10, np.finfo(float).tiny)
    )
    # Multiplying by the conjugate subtracts the common reference phase.
    phase = np.angle(s[..., high] * np.conj(s[..., ref, None]))
    projected = phase / k[high]
    # Each measured projection is direction dot velocity; invert all at once.
    velocity = projected @ np.linalg.pinv(d[high]).T
    residual = np.sqrt(np.mean((velocity @ d[high].T - projected) ** 2, axis=-1))
    velocity[~valid] = np.nan
    residual[~valid] = np.nan
    return dict(
        velocity=velocity.astype(np.float32),
        magnitude=magnitude.astype(np.float32),
        valid=valid,
        residual_mps=residual.astype(np.float32),
        near_wrap=np.any(np.abs(phase) > 0.95 * np.pi, axis=-1),
        channels=high.tolist(),
        reference_channel=ref,
        direction_condition=float(np.linalg.cond(d[high])),
    )


def sample_decoded(image, decoded, xyz, normal):
    """Trilinearly sample through-plane velocity at world XYZ points in metres.

    xyz has shape (..., 3); normal is a single direction, normalised here.
    Return matching spatial arrays of velocity (m/s), magnitude and residual
    (m/s). Sampling is unmasked; undefined/out-of-grid values remain NaN.
    """
    shape = xyz.shape[:-1]
    coords = xyz_to_voxel(image, np.asarray(xyz).reshape(-1, 3)).T
    valid = np.all(
        (coords >= 0)
        & (coords <= (np.array(image.metadata["dimensions"]) - 1)[:, None]),
        axis=0,
    )
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    vn = np.zeros(coords.shape[1])
    for c in range(3):
        vn += normal[c] * ndimage.map_coordinates(
            decoded["velocity"][..., c],
            coords,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
    vn[~valid] = np.nan
    diagnostics = {}
    for key in ("magnitude", "residual_mps"):
        diagnostics[key] = ndimage.map_coordinates(
            decoded[key], coords, order=1, mode="constant", cval=np.nan, prefilter=False
        ).reshape(shape)
    return dict(normal_velocity=vn.reshape(shape), **diagnostics)


def load_acquisition_velocity(path):
    """Read one PV acquisition and decode velocity on its native IJK grid.

    Return image geometry/field metadata and decoded arrays. Velocity has shape
    (I, J, K, 3); scalar diagnostics have shape (I, J, K).
    """
    from .io import PvImage, read_pv_image

    image = read_pv_image(path)
    result = decode_velocity(
        image.point_data["real"],
        image.point_data["imag"],
        image.field_data["kvs"],
        image.field_data["enc_dirs"],
        image.field_data["vencs"],
    )
    dims = tuple(image.metadata["dimensions"])
    for key in ("velocity", "magnitude", "valid", "residual_mps", "near_wrap"):
        shape = (*dims, 3) if key == "velocity" else dims
        result[key] = result[key].reshape(shape, order="F")
    return PvImage(image.metadata, {}, image.field_data), result


def integrate_plane_velocity(normal_velocity, mask, pixel_mm):
    """Sum lumen velocity (m/s) times pixel area (mm^2) to obtain flow in mL/s.

    Return signed flow and finite-velocity coverage. An empty lumen or any
    nonfinite value inside it gives NaN flow rather than an underestimated sum.
    """
    values = np.asarray(normal_velocity)[np.asarray(mask, dtype=bool)]
    if pixel_mm <= 0:
        raise ValueError("Pixel size must be positive")
    count = int(values.size)
    valid = int(np.isfinite(values).sum())
    flow = (
        float(np.sum(values) * pixel_mm**2)
        if count and valid == count
        else float("nan")
    )
    return dict(
        flow_mlps=flow,
        valid_fraction=valid / count if count else 0.0,
        valid_pixels=valid,
        total_pixels=count,
    )
