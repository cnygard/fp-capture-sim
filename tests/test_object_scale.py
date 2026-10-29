"""Physical image scaling, sampling, and acquisition provenance."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

import cv2
import numpy as np
import torch

from fp_capture_sim import CaptureSimulator, OpticalSetup, RoughSample
from fp_capture_sim.cli import main


class ObjectScaleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "object.png"

    def load(self, image, setup, pitch):
        self.assertTrue(cv2.imwrite(str(self.path), image))
        return RoughSample.from_image(self.path, setup, 0, seed=3,
                                      pixel_pitch_m=pitch, device="cpu", dtype=torch.float64)

    def test_equal_pitch_preserves_pixel_values_and_centers(self):
        for shape in ((6, 8), (5, 7)):
            with self.subTest(shape=shape):
                setup = OpticalSetup(sensor_shape=shape)
                image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
                sample = self.load(image, setup, setup.object_pixel_pitch_m)
                np.testing.assert_allclose(sample.amplitude.square().numpy(), image / 255, atol=1e-7)

    def test_smaller_object_has_zero_background(self):
        setup = OpticalSetup(sensor_shape=(8, 8))
        sample = self.load(np.full((4, 6), 255, np.uint8), setup, setup.object_pixel_pitch_m)
        expected = np.zeros((8, 8))
        expected[2:6, 1:7] = 1
        np.testing.assert_allclose(sample.amplitude.square().numpy(), expected, atol=1e-7)

    def test_larger_object_is_center_cropped(self):
        setup = OpticalSetup(sensor_shape=(4, 6))
        image = np.arange(120, dtype=np.uint8).reshape(10, 12)
        sample = self.load(image, setup, setup.object_pixel_pitch_m)
        np.testing.assert_allclose(sample.amplitude.square().numpy(), image[3:7, 3:9] / 255, atol=1e-7)

    def test_fractional_scales_match_pixel_overlap_and_conserve_integrated_intensity(self):
        image = np.arange(35, dtype=np.uint8).reshape(5, 7) * 7
        setup = OpticalSetup(sensor_shape=(12, 14))
        for ratio in (0.7, 1.3, 2.5):
            with self.subTest(ratio=ratio):
                pitch = setup.object_pixel_pitch_m / ratio
                sample = self.load(image, setup, pitch)
                weights = []
                for source_n, target_n in zip(image.shape, setup.object_shape):
                    source_centers = (np.arange(source_n) - source_n // 2) * pitch
                    target_centers = (np.arange(target_n) - target_n // 2) * setup.object_pixel_pitch_m
                    left = np.maximum(target_centers[:, None] - setup.object_pixel_pitch_m / 2,
                                      source_centers[None, :] - pitch / 2)
                    right = np.minimum(target_centers[:, None] + setup.object_pixel_pitch_m / 2,
                                       source_centers[None, :] + pitch / 2)
                    weights.append(np.maximum(right - left, 0) / setup.object_pixel_pitch_m)
                expected = weights[0] @ (image / 255) @ weights[1].T
                actual = sample.amplitude.square().numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-7)
                self.assertAlmostEqual(actual.sum() * ratio**2, (image / 255).sum(), places=5)

    def test_unspecified_pitch_keeps_resize_behavior(self):
        setup = OpticalSetup(sensor_shape=(8, 10))
        image = np.arange(35, dtype=np.uint8).reshape(5, 7)
        sample = self.load(image, setup, None)
        expected = cv2.resize(image.astype("float32") / 255, (10, 8), interpolation=cv2.INTER_AREA)
        np.testing.assert_allclose(sample.amplitude.square().numpy(), expected, atol=1e-7)
        self.assertEqual(sample.original_image_shape, (5, 7))
        self.assertIsNone(sample.original_pixel_pitch_m)

    def test_original_scale_survives_device_conversion_and_export(self):
        setup = OpticalSetup(sensor_shape=(24, 24))
        pitch = 0.5060728745e-6
        sample = self.load(np.full((10, 12), 255, np.uint8), setup, pitch).to("cpu", torch.float32)
        self.assertEqual(sample.original_pixel_pitch_m, pitch)
        sim = CaptureSimulator(setup, sample, device="cpu")
        directory = sim.save_scan(self.root / "scan", [(0, 0)], count=1, scan_step_m=0.001)
        metadata = json.loads((directory / "config.json").read_text())["object_image"]
        self.assertEqual(metadata["original_shape"], [10, 12])
        self.assertEqual(metadata["original_pixel_pitch_m"], pitch)
        np.testing.assert_allclose(metadata["original_size_m"], [10 * pitch, 12 * pitch])
        self.assertEqual(metadata["path"], str(self.path.resolve()))
        config = tomllib.loads((directory / "config.toml").read_text())
        self.assertAlmostEqual(config["pixel_size_um"], setup.sensor_pixel_pitch_m / setup.magnification * 1e6)
        self.assertNotAlmostEqual(config["pixel_size_um"], pitch * 1e6)

    def test_cli_reports_original_scale_and_rejects_invalid_pitch(self):
        self.assertTrue(cv2.imwrite(str(self.path), np.full((10, 12), 255, np.uint8)))
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--image", str(self.path), "--object-pixel-pitch-um", "0.5060728745",
                  "--headless", "--resolution", "24", "--samples", "1", "--device", "cpu"])
        self.assertIn("Original image: 12 x 10 pixels", output.getvalue())
        self.assertIn("Original object pixel pitch: 0.5060728745 um", output.getvalue())
        self.assertIn("Original object size: 0.006072874494 x 0.005060728745 mm", output.getvalue())
        for pitch in (0, -1, float("nan"), float("inf")):
            with self.subTest(pitch=pitch), self.assertRaisesRegex(ValueError, "pixel_pitch_m"):
                RoughSample.from_image(self.path, OpticalSetup(), pixel_pitch_m=pitch)


if __name__ == "__main__":
    unittest.main()
