"""Contained ridge-graph spline and numerical geometry checks."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

from .io import PvImage, voxel_xyz, xyz_to_voxel


@dataclass(frozen=True)
class Centerline:
    """Ordered vessel path and its smoothed representation.

    Path and spline points are N-by-3 world coordinates in metres. Unit tangents
    define the measurement-plane normals. sample_t runs from 0 to 1 along the
    fitted path parameter; it is not cardiac time or exact spline arc length.
    """

    path_xyz_m: np.ndarray
    spline_xyz_m: np.ndarray
    tangent_xyz: np.ndarray
    sample_t: np.ndarray


def ridge_points(
    mask: np.ndarray, image: PvImage, min_radius_m: float, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find candidate vessel centres at local maxima of distance to background.

    min_radius_m excludes thin regions; window is the neighbourhood in voxels.
    Return integer IJK indices and their world XYZ coordinates in metres.
    """
    spacing = tuple(float(x) for x in image.metadata["spacing"])
    distance = ndimage.distance_transform_edt(mask, sampling=spacing)
    local_max = distance == ndimage.maximum_filter(distance, size=window)
    ridge = local_max & mask & (distance >= min_radius_m)
    ridge_ijk = np.argwhere(ridge)
    if len(ridge_ijk) < 8:
        raise ValueError(f"Too few ridge points found: {len(ridge_ijk)}")
    return ridge_ijk, voxel_xyz(image, ridge_ijk)


def ridge_graph(
    xyz_m: np.ndarray, connect_m: float, bridge_m: float | None = None
) -> nx.Graph:
    """Connect nearby ridge points with edges weighted by distance in metres.

    Nodes index xyz_m. Optional bridges join separate components; the caller
    must still check that each edge stays inside the vessel.
    """
    tree = cKDTree(xyz_m)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(xyz_m)))
    for i, j in tree.query_pairs(connect_m):
        weight = float(np.linalg.norm(xyz_m[i] - xyz_m[j]))
        graph.add_edge(i, j, weight=weight)
    if graph.number_of_edges() == 0:
        raise ValueError("Ridge graph has no edges; increase connect_m")
    if bridge_m is not None:
        bridge_ridge_components(graph, xyz_m, bridge_m)
    return graph


def bridge_ridge_components(
    graph: nx.Graph, xyz_m: np.ndarray, bridge_m: float
) -> None:
    """Modify the graph by joining nearest components up to bridge_m metres.

    Repeat until connected or no short enough gap remains. This checks distance,
    not vessel containment.
    """
    while True:
        components = list(nx.connected_components(graph))
        if len(components) <= 1:
            return
        comp_arrays = [np.array(sorted(c)) for c in components]
        best: tuple[float, int, int] | None = None
        for a in range(len(comp_arrays)):
            tree_a = cKDTree(xyz_m[comp_arrays[a]])
            for b in range(a + 1, len(comp_arrays)):
                dists, nearest_in_a = tree_a.query(xyz_m[comp_arrays[b]])
                k = int(np.argmin(dists))
                gap = float(dists[k])
                if best is None or gap < best[0]:
                    best = (
                        gap,
                        int(comp_arrays[a][nearest_in_a[k]]),
                        int(comp_arrays[b][k]),
                    )
        assert best is not None
        gap, i, j = best
        if gap > bridge_m:
            return
        graph.add_edge(i, j, weight=float(np.linalg.norm(xyz_m[i] - xyz_m[j])))


def longest_shortest_path(graph: nx.Graph) -> list[int]:
    """Return the most widely separated pair's shortest path as ordered node IDs.

    Use weighted distances within the largest connected component. This gives
    a candidate main vessel route rather than a path through every branch.
    """
    largest_nodes = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_nodes).copy()
    lengths = dict(nx.all_pairs_dijkstra_path_length(subgraph, weight="weight"))
    start, end, best_length = None, None, -1.0
    for source, target_lengths in lengths.items():
        target, length = max(target_lengths.items(), key=lambda item: item[1])
        if length > best_length:
            start, end, best_length = source, target, length
    if start is None or end is None:
        raise ValueError("Could not find a centerline path in ridge graph")
    return nx.shortest_path(subgraph, start, end, weight="weight")


def fit_centerline(
    path_xyz_m: np.ndarray, samples: int, smoothing: float
) -> Centerline:
    """Fit a cubic spline to an ordered path in metres and compute unit tangents.

    Parameterise by cumulative path distance, then sample uniformly in that
    parameter. smoothing bounds the summed squared fit residuals in m^2;
    zero interpolates the path points exactly.
    """
    if len(path_xyz_m) < 4:
        raise ValueError("Need at least 4 path points for a spline")
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path_xyz_m, axis=0), axis=1))]
    chord = chord / chord[-1]
    degree = min(3, len(path_xyz_m) - 1)
    tck, _u = splprep(path_xyz_m.T, u=chord, s=smoothing, k=degree)
    sample_t = np.linspace(0.0, 1.0, samples)
    spline_xyz_m = np.column_stack(splev(sample_t, tck, der=0))
    tangent_xyz = np.column_stack(splev(sample_t, tck, der=1))
    tangent_xyz = tangent_xyz / np.linalg.norm(tangent_xyz, axis=1, keepdims=True)
    return Centerline(
        path_xyz_m=path_xyz_m,
        spline_xyz_m=spline_xyz_m,
        tangent_xyz=tangent_xyz,
        sample_t=sample_t,
    )


def smooth_mask(image: PvImage, mask: np.ndarray, smooth_mm: float) -> np.ndarray:
    """Apply a Gaussian with standard deviation smooth_mm, then threshold at 0.5.

    This reduces voxel-scale irregularities before finding vessel ridges.
    Final containment is checked against the original, unsmoothed mask.
    """
    if smooth_mm <= 0:
        return mask
    spacing_mm = np.asarray(image.metadata["spacing"], dtype=float) * 1000.0
    sigma_vox = smooth_mm / spacing_mm
    blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma_vox)
    return blurred > 0.5


def inside(image, mask, xyz):
    """Test N-by-3 world points (metres) against the nearest mask voxels.

    Return one boolean per point; coordinates outside the image are false.
    """
    return (
        ndimage.map_coordinates(
            mask.astype(np.uint8),
            xyz_to_voxel(image, xyz).T,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        > 0
    )


def cnn_centerline(image, mask):
    """Return a contained CNN-derived centerline and geometry diagnostics.

    Follow distance-transform ridges through the dominant mask component, fit
    a spline, and reject fragmentation, sampled escapes or abrupt normal turns.
    No reference labels or inlet-normal metadata guide the fit. Passing these
    numerical checks does not establish anatomical segmentation accuracy.
    """
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if not count:
        raise ValueError("Empty CNN mask")
    main = int(np.argmax(sizes))
    main_fraction = float(sizes[main] / mask.sum())
    if main_fraction < 0.9:
        raise ValueError(
            f"No dominant predicted vessel component: largest fraction {main_fraction:.3f} < 0.9"
        )
    vessel = labels == main
    # Smoothing stabilises ridge locations, but must not expand accepted anatomy.
    _, xyz = ridge_points(smooth_mask(image, vessel, 3.0), image, 0.0015, 3)
    retained = inside(image, vessel, xyz)
    outside_ridges = int((~retained).sum())
    xyz = xyz[retained]
    if len(xyz) < 4:
        raise ValueError("Too few contained vessel ridges")
    graph = ridge_graph(xyz, 0.008, bridge_m=0.024)
    # A short edge can shortcut a curved vessel or bridge neighbouring limbs.
    # Check its interior at half-voxel intervals, not just its endpoints.
    step = float(np.min(image.metadata["spacing"])) / 2
    removed = 0
    for a, b in list(graph.edges):
        n = max(2, int(np.ceil(np.linalg.norm(xyz[a] - xyz[b]) / step)) + 1)
        if not inside(image, vessel, np.linspace(xyz[a], xyz[b], n)).all():
            graph.remove_edge(a, b)
            removed += 1
    groups = sorted(nx.connected_components(graph), key=lambda g: (-len(g), min(g)))
    group_paths = []
    for group in groups:
        path = (
            longest_shortest_path(graph.subgraph(group))
            if len(group) > 1
            else list(group)
        )
        length = float(np.linalg.norm(np.diff(xyz[path], axis=0), axis=1).sum())
        group_paths.append((length, path, group))
    group_paths.sort(key=lambda row: row[0], reverse=True)
    # Reject a substantial competing path rather than silently discarding it.
    length, nodes, selected = group_paths[0]
    retained_fraction = len(selected) / len(xyz)
    secondary_length = group_paths[1][0] if len(group_paths) > 1 else 0.0
    if length <= 0 or retained_fraction < 0.9 or secondary_length > 0.25 * length:
        raise ValueError(
            "Ambiguous or fragmented main-vessel ridge path; no geometry accepted"
        )
    selection = dict(
        mask_components=count,
        main_mask_fraction=main_fraction,
        excluded_mask_voxels=int(mask.sum() - sizes[main]),
        outside_ridge_nodes=outside_ridges,
        ridge_groups=[
            dict(nodes=len(g), path_length_mm=1000 * path_length)
            for path_length, _, g in group_paths
        ],
        selected_ridge_fraction=retained_fraction,
        excluded_ridge_nodes=len(xyz) - len(selected),
        selection="Dominant predicted component >=90%; contained path retains >=90% of its ridges; competing path <=25% length",
        graph_path_length_mm=1000 * length,
    )
    # Truncated acquisitions can put terminal ridges against the FOV edge.
    # Try less smoothing, then explicitly shorten both endpoints (at most 2
    # ridge nodes per end). Never repair an interior escape by trimming it away.
    for trim in (0, 1, 2):
        path = xyz[nodes] if not trim else xyz[nodes[trim:-trim]]
        if len(path) < 4:
            break
        end_only = True
        for smoothing in (0.00008, 0.00002, 0.000005, 0.0):
            cl = fit_centerline(path, 96, smoothing)
            # Dense sampling catches spline excursions between displayed points.
            dense = fit_centerline(path, 2048, smoothing)
            contained = inside(image, vessel, dense.spline_xyz_m)
            ids = np.linspace(4, 91, 36).round().astype(int)
            n = cl.tangent_xyz[ids]
            turns = np.degrees(
                np.arccos(np.clip(np.sum(n[1:] * n[:-1], axis=1), -1, 1))
            )
            # Predeclared review gate, not a clinical curvature threshold.
            stable = float(turns.max()) <= 30.0
            if contained.all() and stable:
                return cl, dict(
                    **selection,
                    normal_max_adjacent_turn_deg=float(turns.max()),
                    normal_median_adjacent_turn_deg=float(np.median(turns)),
                    ridge_points=len(xyz),
                    graph_edges=graph.number_of_edges(),
                    rejected_edges=removed,
                    path_points=len(path),
                    dense_spline_points=2048,
                    spline_inside_fraction=1.0,
                    spline_smoothing=smoothing,
                    trimmed_nodes_per_end=trim,
                )
            end_only &= bool(contained[128:-128].all())
        if not end_only:
            break
    raise ValueError(
        "No contained, stable CNN spline (36-plane maximum turn <=30 deg); inspect segmentation and field of view"
    )
