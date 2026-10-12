"""Whole-volume segmentation and fixed-plane flow analysis."""

import numpy as np
import torch
from scipy import ndimage

from .centerline import cnn_centerline
from .geometry import plane_sample
from .model import load_model
from .preprocessing import native_affine, preprocess, tensors
from .velocity import integrate_plane_velocity, sample_decoded


def native_probability(probability, info):
    """Linearly resample world-grid probabilities back to the native IJK grid.

    Use the transforms saved by preprocess; values outside the model grid are 0.
    Resampling precedes thresholding to avoid interpolating a binary mask.
    """
    dims, affine, origin = native_affine(info["native_grid"])
    target = info["world_grid"]
    step = target["spacing_mm"] / 1000
    return ndimage.affine_transform(
        probability,
        affine / step,
        (origin - np.asarray(target["origin_m"])) / step,
        output_shape=tuple(dims),
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )


def segment_acquisition(path, checkpoint_path, device="cpu"):
    """Segment one acquisition using magnitude and XYZ velocity, without labels.

    Return native-grid probability, mask (probability > 0.5), and preprocessing
    metadata. CUDA uses BF16 where supported, otherwise FP16; CPU uses float32.
    Only load trusted checkpoint files.
    """
    model, checkpoint = load_model(checkpoint_path, device)
    x, _, fov, _, info = preprocess(
        path, **checkpoint["preprocessing"], with_target=False
    )
    device = torch.device(device)
    inputs, _, _ = tensors(x, np.zeros(fov.shape, bool), fov, device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
        ),
    ):
        logits = model(inputs)
    probability = logits.float().sigmoid()[0, 0].cpu().numpy()
    probability = probability[tuple(slice(0, n) for n in fov.shape)]
    probability[~fov] = 0
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite prediction")
    native = native_probability(probability, info)
    return native, native > 0.5, info


def orient(image, decoded, line):
    """Orient spline points and tangents along the anchor frame's median flow.

    Require finite velocity at >=90% of points and median projected speed
    >=0.05 m/s in magnitude. Reverse order and tangent signs if necessary;
    return points, normals and the positive median speed in m/s.
    """
    values = np.array(
        [
            sample_decoded(image, decoded, x[None, :], n)["normal_velocity"][0]
            for x, n in zip(line.spline_xyz_m, line.tangent_xyz)
        ]
    )

    finite = values[np.isfinite(values)]

    if len(finite) < 0.9 * len(values) or abs(np.median(finite)) < 0.05:
        raise ValueError("Insufficient velocity to orient centerline")

    if np.median(finite) < 0:
        points = line.spline_xyz_m[::-1]
        normals = -line.tangent_xyz[::-1]

    else:
        points = line.spline_xyz_m
        normals = line.tangent_xyz

    return points, normals, float(abs(np.median(finite)))


def fit_planes(image, mask, decoded):
    """Fit and orient 36 measurement planes from one anchor acquisition.

    Skip the spline's extreme ends. Return centres (36, 3) in metres, unit
    normals and geometry diagnostics. Reuse these planes across the cycle so
    a changing segmentation does not also move the measurement locations.
    """
    line, checks = cnn_centerline(image, mask)
    points, normals, speed = orient(image, decoded, line)
    indices = np.linspace(4, len(points) - 5, 36).round().astype(int)
    return points[indices], normals[indices], dict(checks, orientation_median_mps=speed)


def measure_plane(image, mask, decoded, center, normal, radius_mm=18.0, pixel_mm=0.4):
    """Measure lumen area (cm^2) and signed flow (mL/s) on a physical plane.

    center is world XYZ in metres; normal defines positive through-plane flow.
    Return the square sampled field, circular-aperture lumen, clipping flags
    and velocity coverage. Undefined lumen velocity gives NaN flow. Reusing
    the same plane/velocity with different masks isolates segmentation effects.
    """
    plane = plane_sample(
        image, mask, decoded["velocity"], center, normal, radius_mm, pixel_mm
    )
    field = sample_decoded(image, decoded, plane["xyz"], normal)
    flow = integrate_plane_velocity(field["normal_velocity"], plane["mask"], pixel_mm)
    return dict(plane, normal_velocity=field["normal_velocity"], **flow)
