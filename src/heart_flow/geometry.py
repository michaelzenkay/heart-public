"""Physical plane sampling, lumen selection and area/flow measurements."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .io import voxel_xyz, xyz_to_voxel


def orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two perpendicular unit axes spanning a plane with nonzero normal."""
    normal = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v / np.linalg.norm(v)


def validate_grid(image):
    """Require positive spacing and orthonormal directions for coordinate mapping."""
    direction = np.asarray(image.metadata["direction_matrix"], dtype=float)
    spacing = np.asarray(image.metadata["spacing"], dtype=float)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6):
        raise ValueError(
            "Expected orthonormal grid direction; cannot assume transpose inverse"
        )
    if np.any(spacing <= 0):
        raise ValueError("Grid spacing must be positive")


def resample_mask(source, mask, target):
    """Transfer a boolean mask between physical grids using nearest neighbours.

    Return the target's native IJK shape, processing one slice at a time to
    bound memory. Points outside the source image become background.
    """
    validate_grid(source)
    validate_grid(target)
    shape = tuple(target.metadata["dimensions"])
    result = np.empty(shape, dtype=bool)
    xy = np.indices(shape[:2]).reshape(2, -1).T
    for k in range(shape[2]):
        ijk = np.column_stack((xy, np.full(len(xy), k)))
        coords = xyz_to_voxel(source, voxel_xyz(target, ijk))
        # Stabilise exact half-voxel ties and grid edges after floating-point
        # rotation round-trips, preserving scipy's nearest-neighbour rule.
        half = np.rint(coords * 2.0) / 2.0
        coords = np.where(np.abs(coords - half) < 1e-10, half, coords)
        result[:, :, k] = (
            ndimage.map_coordinates(
                mask.astype(np.uint8, copy=False),
                coords.T,
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            ).reshape(shape[:2])
            > 0
        )
    return result


def plane_sample(image, mask, velocity, center, normal, radius_mm=18.0, pixel_mm=0.4):
    """Sample a square plane and select a connected lumen within its disk.

    mask and velocity use native IJK axes; velocity's last axis is world XYZ
    in m/s. center is world XYZ in metres; normal sets positive flow direction.
    Return sampled arrays, area in cm^2, flow in mL/s and clipping flags.
    Use inference.measure_plane for flow with undefined-velocity checks;
    this lower-level sampler fills out-of-grid velocity with zero.
    """
    validate_grid(image)
    if radius_mm <= 0 or pixel_mm <= 0:
        raise ValueError("Plane radius and pixel size must be positive")
    normal = np.asarray(normal, dtype=float)
    if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-12:
        raise ValueError("Plane normal must be finite and nonzero")
    normal = normal / np.linalg.norm(normal)
    u, v = orthonormal_basis(normal)
    count = int(np.ceil(radius_mm / pixel_mm))
    axis = np.arange(-count, count + 1) * pixel_mm
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    xyz = np.asarray(center) + (xx[..., None] * u + yy[..., None] * v) / 1000.0
    coords = xyz_to_voxel(image, xyz.reshape(-1, 3)).T
    valid = np.all(
        (coords >= 0) & (coords <= (np.asarray(mask.shape) - 1)[:, None]), axis=0
    ).reshape(xx.shape)
    sampled_mask = (
        ndimage.map_coordinates(
            mask.astype(np.uint8),
            coords,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).reshape(xx.shape)
        > 0
    )
    disk = xx**2 + yy**2 <= radius_mm**2
    labels, n = ndimage.label(sampled_mask & disk, structure=np.ones((3, 3)))
    selected = np.zeros_like(sampled_mask)
    if n:
        # Keep the component closest to the centre, avoiding another vessel limb
        # in the same plane. center_inside still flags a misplaced plane centre.
        candidates = np.where(labels > 0, xx**2 + yy**2, np.inf)
        nearest = np.unravel_index(np.argmin(candidates), candidates.shape)
        selected = labels == labels[nearest]
    vn = np.zeros(xx.shape, dtype=float)
    for c in range(3):
        vn += normal[c] * ndimage.map_coordinates(
            velocity[..., c], coords, order=1, mode="constant", cval=0, prefilter=False
        ).reshape(xx.shape)
    area_cm2 = float(selected.sum() * pixel_mm**2 / 100.0)
    # m/s * mm^2 -> mL/s has numerical factor 1.
    flow_mlps = float(vn[selected].sum() * pixel_mm**2)
    boundary = disk & ~ndimage.binary_erosion(disk)
    return dict(
        axis_mm=axis,
        normal_velocity=vn,
        mask=selected,
        xyz=xyz,
        valid_grid=valid,
        area_cm2=area_cm2,
        flow_mlps=flow_mlps,
        center_inside=bool(selected[count, count]),
        radius_clipped=bool(np.any(selected & boundary)),
        grid_clipped=bool(np.any(selected & ndimage.binary_dilation(~valid))),
    )
