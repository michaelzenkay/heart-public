import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from heart_flow.centerline import cnn_centerline, inside
from heart_flow.data import sha256, write_json
from heart_flow.evaluation import evaluate
from heart_flow.inference import measure_plane
from heart_flow.io import PvImage
from heart_flow.model import CHANNELS, VERSION, load_model, model_for
from heart_flow.training import train
from heart_flow.velocity import decode_velocity


class FlowTests(unittest.TestCase):
    """Small checks of physical units, geometry rejection and the training path."""

    def test_phase_to_area_and_signed_flow(self):
        """Recover known velocity, integrate a cylinder and reject missing values."""
        shape = (41, 41, 41)
        x, y, _ = np.indices(shape)
        mask = (x - 20) ** 2 + (y - 20) ** 2 <= 8**2
        image = PvImage(
            dict(
                dimensions=shape,
                origin=[0, 0, 0],
                spacing=[0.001] * 3,
                direction_matrix=np.eye(3),
            ),
            {},
            {},
        )
        directions = np.vstack(([0, 0, 0], np.repeat(np.eye(3), 2, axis=0)))
        venc = np.array([0, 0.3, 3, 0.3, 3, 0.3, 3])
        k = np.divide(np.pi, venc, out=np.zeros_like(venc), where=venc > 0)
        velocity = np.broadcast_to([0, 0, 0.4], (*shape, 3))
        signal = np.exp(1j * (velocity @ directions.T * k + 0.7))
        decoded = decode_velocity(signal.real, signal.imag, k, directions, venc)
        np.testing.assert_allclose(decoded["velocity"], velocity, atol=1e-6)
        result = measure_plane(image, mask, decoded, [0.02] * 3, [0, 0, 1])
        self.assertLess(abs(result["area_cm2"] / (np.pi * 0.8**2) - 1), 0.03)
        self.assertAlmostEqual(result["flow_mlps"], result["area_cm2"] * 40, places=5)
        reverse = measure_plane(image, mask, decoded, [0.02] * 3, [0, 0, -1])
        self.assertAlmostEqual(reverse["flow_mlps"], -result["flow_mlps"])
        decoded["velocity"][20, 20, 20] = np.nan
        self.assertTrue(
            np.isnan(
                measure_plane(image, mask, decoded, [0.02] * 3, [0, 0, 1])["flow_mlps"]
            )
        )

    def test_contained_centerline_and_fragment_rejection(self):
        """Keep a straight-vessel spline inside its mask and reject fragmentation."""
        shape = (110, 40, 40)
        x, y, z = np.indices(shape)
        mask = (x >= 8) & (x <= 101) & ((y - 20) ** 2 + (z - 20) ** 2 <= 36)
        image = PvImage(
            dict(
                dimensions=shape,
                origin=[0, 0, 0],
                spacing=[0.001] * 3,
                direction_matrix=np.eye(3),
            ),
            {},
            {},
        )
        line, checks = cnn_centerline(image, mask)
        self.assertTrue(inside(image, mask, line.spline_xyz_m).all())
        self.assertLess(checks["normal_max_adjacent_turn_deg"], 1.0)
        mask[51:59] = False
        with self.assertRaisesRegex(ValueError, "No dominant"):
            cnn_centerline(image, mask)

    def test_training_and_checkpoint_reload(self):
        """Run a tiny training step and verify saved weights reproduce evaluation."""
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = np.random.default_rng(4).random((4, 16, 16, 16), dtype=np.float32)
            y = x[0] > 0.5
            np.savez_compressed(root / "sample.npz", x=x, y=y, fov=np.ones_like(y))
            rows = [
                dict(
                    series=split,
                    frame=0,
                    group=split,
                    geometry=split,
                    background=split,
                    split=split,
                    file="sample.npz",
                    sha256=sha256(root / "sample.npz"),
                )
                for split in ("train", "val", "test")
            ]
            # Shared synthetic arrays exercise the plumbing, not model generalization.
            write_json(
                root / "manifest.json",
                dict(
                    version=VERSION,
                    channels=CHANNELS,
                    spacing_mm=3.0,
                    velocity_scale_mps=2.0,
                    samples=rows,
                ),
            )
            torch.manual_seed(42)
            initial = (
                next(model_for(dict(mode="mag-xyz", blocks=3, width=2)).parameters())
                .detach()
                .clone()
            )
            history = train(
                root, root / "run", epochs=1, steps_per_epoch=2, device="cpu", width=2
            )
            model, checkpoint = load_model(root / "run" / "best.pth")
            self.assertGreater(
                float((next(model.parameters()).detach() - initial).abs().max()), 0
            )
            score = evaluate(model, root, [rows[1]])["group_dice"]
            self.assertEqual(score, checkpoint["validation_group_dice"])
            self.assertEqual(score, history[0]["validation_group_dice"])
