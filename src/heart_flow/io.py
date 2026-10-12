"""RACLETTE PV arrays and physical coordinates (metres)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zstandard as zstd

ZSTD_MAGIC = bytes.fromhex("28b52ffd")


@dataclass(frozen=True)
class PvImage:
    """Native image geometry plus voxel arrays and acquisition-wide metadata.

    metadata describes grid dimensions/orientation, with spacing/origin in metres.
    point_data contains flattened voxel arrays; field_data holds values such
    as encoding directions and VENCs shared by the acquisition.
    """

    metadata: dict[str, Any]
    point_data: dict[str, np.ndarray]
    field_data: dict[str, np.ndarray]


def read_pv_image(path: Path) -> PvImage:
    """Read a PyVista-Zstandard .pv file without reorienting or scaling its data."""
    payloads = decompress_zstd_frames(Path(path))
    container_metadata = json.loads(payloads[-1].decode("utf-8"))
    frame_names = container_metadata["frame_names"]
    array_payloads = payloads[1:-1:2]

    if len(frame_names) != len(array_payloads):
        raise ValueError(
            f"{path}: expected {len(frame_names)} array payloads, found {len(array_payloads)}"
        )

    ds_metadata_name = next(
        name for name in frame_names if name.endswith("__ds_metadata")
    )
    ds_metadata = json.loads(
        array_payloads[frame_names.index(ds_metadata_name)].decode("utf-8")
    )
    uid = ds_metadata["uid"]

    point_data: dict[str, np.ndarray] = {}
    field_data: dict[str, np.ndarray] = {}
    for frame_name, payload in zip(frame_names, array_payloads, strict=True):
        if frame_name == f"{uid}__ds_metadata":
            continue
        if frame_name.endswith("__point_data"):
            name = frame_name.removeprefix(uid).removesuffix("__point_data")
            point_data[name] = decode_array(
                payload, ds_metadata["point_data_keys"][name]
            )
        elif frame_name.endswith("__field_data"):
            name = frame_name.removeprefix(uid).removesuffix("__field_data")
            field_data[name] = decode_array(
                payload, ds_metadata["field_data_keys"][name]
            )

    return PvImage(metadata=ds_metadata, point_data=point_data, field_data=field_data)


def decompress_zstd_frames(path: Path) -> list[bytes]:
    """Read indexed compression frames, not cardiac time frames.

    PV trailer: pairs of uint64 cumulative compressed-end offsets and decoded
    byte lengths, followed by uint64 frame count, all little-endian. Validate
    offsets and exact decoded lengths; magic bytes can also occur in payloads,
    so scanning for headers would be unreliable.
    """
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path}: truncated PV frame index")
    count = int.from_bytes(raw[-8:], "little")
    if count < 3 or count > (len(raw) - 8) // 16:
        raise ValueError(f"{path}: invalid PV frame count")
    index_start = len(raw) - 8 - 16 * count
    index = np.frombuffer(
        raw, dtype="<u8", count=2 * count, offset=index_start
    ).reshape(count, 2)
    ends = [int(x) for x in index[:, 0]]
    if ends[-1] != index_start or any(b <= a for a, b in zip([0] + ends[:-1], ends)):
        raise ValueError(f"{path}: invalid PV frame offsets")
    payloads = []
    start = 0
    for frame, (end, expected) in enumerate(zip(ends, index[:, 1])):
        encoded = raw[start:end]
        if not encoded.startswith(ZSTD_MAGIC):
            raise ValueError(f"{path}: frame {frame} missing Zstandard header")
        decoder = zstd.ZstdDecompressor().decompressobj()
        payload = decoder.decompress(encoded)
        if not decoder.eof or decoder.unused_data or len(payload) != int(expected):
            raise ValueError(f"{path}: frame {frame} does not match PV index")
        payloads.append(payload)
        start = end
    return payloads


def decode_array(payload: bytes, spec: dict[str, Any]) -> np.ndarray:
    """Interpret raw bytes using the stored dtype and shape, without copying."""
    return np.frombuffer(payload, dtype=np.dtype(spec["dtype"])).reshape(
        tuple(spec["shape"])
    )


def point_volume(
    image: PvImage, point_array: np.ndarray, channel: int | None = None
) -> np.ndarray:
    """Reshape one voxel scalar/channel to native IJK using PV's Fortran order."""
    values = point_array if channel is None else point_array[:, channel]
    return values.reshape(tuple(image.metadata["dimensions"]), order="F")


def voxel_xyz(image: PvImage, indices_ijk: np.ndarray) -> np.ndarray:
    """Map N-by-3 voxel indices to world XYZ in metres, including grid offset."""
    origin = np.asarray(image.metadata["origin"], dtype=float)
    spacing = np.asarray(image.metadata["spacing"], dtype=float)
    direction = np.asarray(image.metadata["direction_matrix"], dtype=float)
    offset = np.asarray(image.metadata.get("offset", [0, 0, 0]), dtype=float)
    return origin + ((indices_ijk + offset) * spacing) @ direction.T


def xyz_to_voxel(image: PvImage, xyz_m: np.ndarray) -> np.ndarray:
    """Inverse of voxel_xyz: world coordinates (m) -> fractional voxel indices.

    Assumes direction_matrix is orthonormal, as in the RACLETTE PV grids,
    so its inverse is its transpose. Grid spacing and origin are in metres.
    """
    origin = np.asarray(image.metadata["origin"], dtype=float)
    spacing = np.asarray(image.metadata["spacing"], dtype=float)
    direction = np.asarray(image.metadata["direction_matrix"], dtype=float)
    offset = np.asarray(image.metadata.get("offset", [0, 0, 0]), dtype=float)
    local = (np.asarray(xyz_m, dtype=float) - origin) @ direction
    return local / spacing - offset
