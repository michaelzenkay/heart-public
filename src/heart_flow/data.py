"""Prepare acquisition caches and keep related anatomy/backgrounds in one split."""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .model import CHANNELS, VERSION
from .preprocessing import preprocess


def sha256(path):
    """Hash file bytes to identify the exact cached input used by a run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    """Read UTF-8 JSON, accepting an optional byte-order mark."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """Write readable UTF-8 JSON, rejecting nonfinite numbers."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def grouped_series(entries, validation_geometry, test_geometry):
    """Group series sharing source anatomy or background and assign data splits.

    Links are transitive: A sharing anatomy with B and B sharing background
    with C puts all three together. Hold out whole groups containing the named
    validation/test geometries to prevent related acquisitions leaking across
    splits. Return each entry with its identities, group and split.
    """
    rows = []
    for entry in entries:
        directory = Path(entry["series_dir"])
        candidates = []
        if directory.name.lower() in ("high", "mid", "low"):
            candidates.append(
                directory.parent.parent / f"meta_{directory.name.lower()}.json"
            )
        candidates += [p / "meta.json" for p in (directory, *directory.parents)]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise ValueError(f"Source metadata missing for {entry['id']}")
        metadata = read_json(path)
        if not metadata.get("source_geometry") or not metadata.get("source_background"):
            raise ValueError(f"Source anatomy/background identity missing: {path}")
        rows.append(
            dict(
                entry=entry,
                geometry=metadata["source_geometry"],
                background=metadata["source_background"],
            )
        )

    remaining, result = rows.copy(), []
    while remaining:
        members = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            geometries = {r["geometry"] for r in members}
            backgrounds = {r["background"] for r in members}
            for row in remaining.copy():
                if row["geometry"] in geometries or row["background"] in backgrounds:
                    members.append(row)
                    remaining.remove(row)
                    changed = True
        geometries = {r["geometry"] for r in members}
        if validation_geometry in geometries and test_geometry in geometries:
            raise ValueError("Validation and test anatomy/background groups are linked")
        split = (
            "val"
            if validation_geometry in geometries
            else "test"
            if test_geometry in geometries
            else "train"
        )
        identity = "|".join(sorted(r["entry"]["id"] for r in members))
        group = hashlib.sha256(identity.encode()).hexdigest()[:12]
        result.extend(dict(row, group=group, split=split) for row in members)
    if {r["split"] for r in result} != {"train", "val", "test"}:
        raise ValueError("Expected nonempty train, validation and test partitions")
    return result


def validate_splits(rows):
    """Reject empty splits, shared source identities across splits, or repeats.

    Each (series, frame) must be unique, and each anatomy/background group must
    belong entirely to train, validation or test.
    """
    if {r["split"] for r in rows} != {"train", "val", "test"}:
        raise ValueError("Cache needs nonempty train, validation and test partitions")
    for key in ("group", "geometry", "background", "series"):
        seen = {}
        for row in rows:
            value = row[key]
            if not value:
                raise ValueError(f"Missing split identity: {key}")
            if value in seen and seen[value] != row["split"]:
                raise ValueError(f"Split leakage: {key} {value}")
            seen[value] = row["split"]
    identities = [(r["series"], r["frame"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate series/frame in cache")


def prepare(manifest_path, output, spacing_mm=3.0, velocity_scale_mps=2.0):
    """Write preprocessed frame arrays and a grouped manifest to a new directory.

    Resolve input paths relative to the source manifest. Each NPZ holds four
    input channels, target and field of view; manifest entries record their
    hashes and split identities. An existing output directory is not overwritten.
    """
    manifest_path, output = Path(manifest_path).resolve(), Path(output)
    manifest = read_json(manifest_path)
    entries = manifest["cases"]
    if len({e["id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate series ID")
    for entry in entries:
        entry["series_dir"] = str(
            (manifest_path.parent / entry["series_dir"]).resolve()
        )
        entry["acquisition_files"] = [
            (manifest_path.parent / p).resolve() for p in entry["acquisition_files"]
        ]
        if not entry["acquisition_files"]:
            raise ValueError(f"No acquisition frames: {entry['id']}")
    groups = grouped_series(
        entries, manifest["validation_geometry"], manifest["test_geometry"]
    )
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for row in groups:
        entry = row["entry"]
        series_key = hashlib.sha256(entry["id"].encode()).hexdigest()[:12]
        for frame, path in enumerate(entry["acquisition_files"]):
            x, y, fov, _, _ = preprocess(path, spacing_mm, velocity_scale_mps)
            target = output / f"{series_key}-{frame:03d}.npz"
            np.savez_compressed(target, x=x, y=y, fov=fov)
            samples.append(
                dict(
                    series=entry["id"],
                    frame=frame,
                    group=row["group"],
                    geometry=row["geometry"],
                    background=row["background"],
                    split=row["split"],
                    file=target.name,
                    sha256=sha256(target),
                )
            )
    validate_splits(samples)
    write_json(
        output / "manifest.json",
        dict(
            version=VERSION,
            channels=CHANNELS,
            spacing_mm=spacing_mm,
            velocity_scale_mps=velocity_scale_mps,
            source_manifest_sha256=sha256(manifest_path),
            samples=samples,
        ),
    )


def load_manifest(cache):
    """Read a prepared cache manifest and validate its format, channels and splits."""
    manifest = read_json(Path(cache) / "manifest.json")
    if manifest["version"] != VERSION or manifest["channels"] != CHANNELS:
        raise ValueError("Expected the magnitude/XYZ acquisition cache")
    validate_splits(manifest["samples"])
    return manifest


@lru_cache(maxsize=3)
def load_sample(path, digest):
    """Load input, target and FOV arrays, checking the file hash on cache misses.

    Keep up to three samples in memory. Treat returned arrays as read-only;
    augmentation makes its own copies before changing them.
    """
    if sha256(path) != digest:
        raise ValueError(f"Cache sample hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        return data["x"].copy(), data["y"].astype(bool), data["fov"].astype(bool)


def main():
    """Prepare an acquisition cache from command-line paths and grid spacing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing-mm", type=float, default=3.0)
    args = parser.parse_args()
    prepare(args.manifest, args.output, args.spacing_mm)
