"""Regression coverage for the reconstruction handoff and capture destinations."""
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fp_capture_sim import CaptureSimulator, OpticalSetup, RoughSample, Spectrum, spiral_positions
from fp_capture_sim.cli import main
from fp_capture_sim.simulation import CAPTURES_DIRECTORY, capture_directory


class ExportTests(unittest.TestCase):
    def simulator(self, spectrum=None, oversampling=1):
        setup = OpticalSetup(sensor_shape=(24, 32), oversampling=oversampling)
        sample = RoughSample.from_intensity(torch.ones(setup.object_shape), 0)
        return CaptureSimulator(setup, sample, spectrum=spectrum, device="cpu", seed=7)

    def test_acquisition_only_toml_and_units(self):
        sim = self.simulator(oversampling=2)
        with tempfile.TemporaryDirectory() as temporary:
            directory = sim.save_scan(Path(temporary) / "scan", spiral_positions(3, 0.001), count=2)
            config = tomllib.loads((directory / "config.toml").read_text())
            self.assertEqual(set(config), {
                "source", "file_extension", "frame_pattern", "samples_count", "resolution",
                "wavelength", "pixel_size_um", "aperture_diameter", "object_distance", "translation",
            })
            self.assertEqual(config["source"], "array_sim")
            self.assertEqual(config["file_extension"], ".npy")
            self.assertEqual(config["samples_count"], 9)
            self.assertEqual(config["resolution"], 24)
            self.assertAlmostEqual(config["wavelength"], 532)
            self.assertAlmostEqual(config["translation"], 1)
            self.assertAlmostEqual(config["aperture_diameter"], 75 / 22)
            self.assertEqual(config["object_distance"], 2)
            self.assertAlmostEqual(config["pixel_size_um"], 1.85 / sim.setup.magnification)
            for index in range(9):
                frame = directory / (config["frame_pattern"].format(index=index) + config["file_extension"])
                self.assertEqual(np.load(frame).shape, (24, 32))
            self.assertTrue(json.loads((directory / "config.json").read_text())["complete"])
            with self.assertRaises(FileExistsError):
                sim.save_scan(directory, [(0, 0)], count=1)

    def test_spectral_power_mean(self):
        for spectrum, expected in (
            (Spectrum.monochromatic(633e-9), 633),
            (Spectrum.uniform(500e-9, 600e-9), 550),
            (Spectrum((500e-9, 600e-9), (1, 3)), 575),
            (Spectrum((500e-9, 600e-9)), 550),
        ):
            with self.subTest(spectrum=spectrum):
                config = tomllib.loads(self.simulator(spectrum)._reconstruction_config(spiral_positions(3, 0.001)))
                self.assertAlmostEqual(config["wavelength"], expected)

    def test_unknown_geometry_does_not_invent_translation(self):
        sim = self.simulator()
        for positions in ([(0, 0)], [(0, 0), (0.001, 0)], spiral_positions(3, 0.001)[::-1]):
            config = tomllib.loads(sim._reconstruction_config(np.asarray(positions)))
            self.assertNotIn("translation", config)
        config = tomllib.loads(sim._reconstruction_config(np.array([[0, 0]]), 0.001))
        self.assertAlmostEqual(config["translation"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bad"
            with self.assertRaisesRegex(ValueError, "match"):
                sim.save_scan(directory, spiral_positions(3, 0.001), count=1, scan_step_m=0.002)
            self.assertFalse(directory.exists())

    def test_failed_capture_does_not_publish_reconstruction_config(self):
        sim = self.simulator()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "failed"
            with patch.object(sim, "scan", side_effect=ValueError("capture failed")):
                with self.assertRaisesRegex(ValueError, "capture failed"):
                    sim.save_scan(directory, [(0, 0)], count=1)
            self.assertFalse((directory / "config.toml").exists())
            self.assertFalse(json.loads((directory / "config.json").read_text())["complete"])

    def test_path_resolution_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            self.assertEqual(capture_directory("named"), CAPTURES_DIRECTORY / "named")
            self.assertEqual(capture_directory(Path("named")), CAPTURES_DIRECTORY / "named")
            self.assertEqual(capture_directory("./local"), Path("local"))
            self.assertEqual(capture_directory("../elsewhere/scan"), Path("../elsewhere/scan"))
            self.assertEqual(capture_directory("nested/scan"), Path("nested/scan"))
            self.assertEqual(capture_directory(temporary), Path(temporary))
            self.assertEqual(capture_directory("~/scan"), Path.home() / "scan")

    def test_cli_named_default_and_explicit_destinations(self):
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            root = Path(temporary)
            captures = root / "project" / "captures"
            sample = RoughSample.from_intensity(torch.ones(24, 24), 0)
            with patch("fp_capture_sim.simulation.CAPTURES_DIRECTORY", captures), patch(
                "fp_capture_sim.cli.RoughSample.from_image", return_value=sample
            ), redirect_stdout(io.StringIO()):
                for save_args, expected in (
                    (["--save", "named"], captures / "named"),
                    (["--save"], captures / "capture"),
                    (["--save", "./local"], root / "local"),
                    (["--save", str(root / "absolute")], root / "absolute"),
                ):
                    with self.subTest(save_args=save_args):
                        main(["--resolution", "24", "--scan-size", "1", "--samples", "1",
                              "--device", "cpu", "--bandwidth-nm", "0", "--emitter-radius-mm", "0",
                              *save_args])
                        config = tomllib.loads((expected / "config.toml").read_text())
                        self.assertEqual(config["samples_count"], 1)
                        self.assertAlmostEqual(config["translation"], (75 / 22) * 0.35)


if __name__ == "__main__":
    unittest.main()
