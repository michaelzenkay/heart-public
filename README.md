# Cardiac MRI 4D flow analysis

Phase-derived velocity, 3D aorta segmentation, spline centerlines and plane-based
flow measurement. Developed for RACLETTE simulated MRI in `.pv` format; images,
trained weights and the viewer are not included.

## Method

[Velocity decoding](src/heart_flow/velocity.py) takes phase differences relative
to the unencoded complex signal, selects the highest VENC for each direction,
and solves for XYZ velocity in m/s. It does not use a segmentation mask.
Background-phase correction and phase unwrapping are not implemented.

[Preprocessing](src/heart_flow/preprocessing.py) resamples magnitude and XYZ
velocity to a 3 mm world grid. Magnitude is divided by its positive-value 99th
percentile and clipped to [0, 1]; velocity is divided by 2 m/s. Undefined velocity
is zero-filled for the network and tracked separately for measurement.

The [3D U-Net](src/heart_flow/model.py) has four input channels, three encoding
blocks, first-layer width 8, batch normalization and one output logit per voxel.
Probabilities are interpolated to the acquisition grid, then thresholded at >0.5.

A [ridge graph and smoothed spline](src/heart_flow/centerline.py) define 36 fixed
planes. Geometry is rejected if the vessel/path is fragmented, the spline leaves
the mask, or adjacent plane normals turn more than 30 degrees. Velocity at an
anchor frame with sufficient flow sets the direction of the normals.

[Plane measurements](src/heart_flow/geometry.py) use 0.4 mm pixels and an 18 mm
radius aperture, selecting the lumen component nearest the center. Area is
reported in cm^2; summed normal velocity times pixel area gives signed flow in
mL/s. Boundary-contact flags identify potentially clipped measurements. Missing
velocity inside the lumen makes flow unavailable.

## Install

Requires Python 3.11+. Install a suitable PyTorch build first if you need CUDA.

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install .
```

## Inference and measurement

```python
from heart_flow.inference import segment_acquisition, fit_planes, measure_plane
from heart_flow.velocity import load_acquisition_velocity

# Choose an anchor acquisition with sufficient flow for orientation.
path = "anchor.pv"
probability, mask, info = segment_acquisition(path, "checkpoint.pth", device="cuda")
image, velocity = load_acquisition_velocity(path)
centers, normals, checks = fit_planes(image, mask, velocity)
result = measure_plane(image, mask, velocity, centers[0], normals[0])
print(result["area_cm2"], result["flow_mlps"])
```

Reuse the anchor centers/normals across frames, decoding velocity and predicting
the mask for each acquisition. Comparison masks must be on the acquisition grid;
use `resample_mask` for a different physical grid. Only load trusted checkpoints.

## Train and evaluate

Edit the [series manifest](configs/cases.example.json) with paths relative to the
manifest, frames in acquisition order, and validation/test geometry IDs.
Preparation reads `source_geometry` and `source_background` from RACLETTE
`meta.json` (or `meta_high/mid/low.json`). Series sharing either identity remain
together, including indirect links. Validation and test groups cannot overlap.

```bash
heart-flow-prepare --manifest configs/cases.example.json --output data/cache
heart-flow-train --cache data/cache --output runs/reference --device cuda
heart-flow-evaluate --cache data/cache --checkpoint runs/reference/best.pth --split val
```

The [trainer](src/heart_flow/training.py) uses full volumes, batch size 1, BCE plus
soft Dice, AdamW and gradient clipping. Padding is excluded from loss and metrics.
Sampling weights groups equally, then series within groups, then frames. Spatial
flips and quarter-turns also transform velocity components; intensity augmentation
affects magnitude only. CUDA uses mixed precision; CPU uses float32.

Validation averages frame Dice within each series, series within each group, then
groups equally. The best validation checkpoint is saved alongside the last one,
run settings and epoch metrics. The test partition is evaluated only when
explicitly selected. This compact trainer starts new runs; it does not resume them.

## Tests and scope

```bash
python -B -m unittest discover -s src/tests -v
```

Three synthetic CPU checks cover phase-to-flow units, centerline geometry and
training/reload. They use the installed package; reinstall after source changes.
These verify software behavior, not model accuracy. Geometry checks do not prove
anatomical correctness; lumen area is not valve-orifice area.

U-Net layers use Fernando Perez-Garcia's [unet](https://github.com/fepegar/unet)
(MIT). This source is licensed under [Apache 2.0](LICENSE). Data and model
weights retain their separate terms.
