"""Physics mappings, stable physical realizations, replay and update routing."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fp_capture_sim import CaptureSimulator, OpticalSetup, SensorNoise
from fp_capture_sim.backend import make_generator
from fp_capture_sim.playground import (PlaygroundConfig, Realizations, PlaygroundEngine,
    LatestRequest, CaptureCancelled, geometry_specs, save_preset, load_preset, save_capture)


class PlaygroundTests(unittest.TestCase):
    def setUp(self):
        self.config = PlaygroundConfig(setup=OpticalSetup(sensor_shape=(24, 32)),
            object_width_m=.00152, preview_samples=3, final_samples=7,
            batch_size=2, device="cpu", dtype="float64")
        image = np.linspace(0, 1, 24*32, dtype=np.float32).reshape(24, 32)
        self.assets = Realizations.create(image, "/image/no-longer-present.png", self.config)

    def test_scalar_specifications_and_aspect_ratio(self):
        c = self.config
        spec = geometry_specs(c, self.assets.image.shape)
        self.assertAlmostEqual(spec["object_na"], c.setup.object_na)
        self.assertAlmostEqual(spec["f_number"], 22)
        self.assertAlmostEqual(spec["measurement_pitch_m"], 1.85e-6/c.setup.magnification)
        self.assertAlmostEqual(spec["object_size_m"][1], c.object_width_m*24/32)
        np.testing.assert_allclose(spec["wavelength_range_m"], [522e-9, 542e-9])
        finer = replace(c, setup=replace(c.setup, oversampling=2))
        spec2 = geometry_specs(finer, self.assets.image.shape)
        self.assertEqual(spec["field_of_view_m"], spec2["field_of_view_m"])
        self.assertEqual(spec["grid_pitch_m"]/2, spec2["grid_pitch_m"])

    def test_surface_preserves_physical_coordinates_and_scales_roughness(self):
        c = self.config
        original = self.assets.sample(c)
        rougher = self.assets.sample(replace(c, height_rms_m=c.height_rms_m*2))
        torch.testing.assert_close(rougher.height_m, original.height_m*2, atol=0, rtol=0)
        finer = self.assets.sample(replace(c, setup=replace(c.setup, oversampling=2)))
        torch.testing.assert_close(finer.height_m[::2, ::2], original.height_m, atol=0, rtol=0)
        larger = self.assets.sample(replace(c, setup=replace(c.setup, sensor_shape=(48, 64))))
        torch.testing.assert_close(larger.height_m[12:36, 16:48], original.height_m, atol=0, rtol=0)
        # Object size changes do not move or regenerate the physical surface.
        smaller_object = self.assets.sample(replace(c, object_width_m=c.object_width_m/2))
        torch.testing.assert_close(smaller_object.height_m, original.height_m, atol=0, rtol=0)
        self.assertLess(smaller_object.amplitude.count_nonzero(), original.amplitude.count_nonzero())

    def test_illumination_prefix_and_common_random_mapping(self):
        c = self.config
        samples = self.assets.illumination(c, 7)
        preview = self.assets.illumination(c, 3)
        torch.testing.assert_close(samples.wavelengths_m[:3], preview.wavelengths_m, atol=0, rtol=0)
        torch.testing.assert_close(samples.source_xy_m[:3], preview.source_xy_m, atol=0, rtol=0)
        changed = replace(c, bandwidth_m=c.bandwidth_m*2, emitter=replace(c.emitter, radius_m=c.emitter.radius_m*2))
        other = self.assets.illumination(changed, 7)
        torch.testing.assert_close(other.wavelengths_m-c.wavelength_m, (samples.wavelengths_m-c.wavelength_m)*2)
        torch.testing.assert_close(other.source_xy_m, samples.source_xy_m*2)
        extended = self.assets.updated(replace(c, final_samples=20))
        np.testing.assert_array_equal(extended.light_uniform[:7], self.assets.light_uniform)
        light = self.assets.updated(replace(c, light_seed=1))
        self.assertFalse(np.array_equal(light.light_uniform, self.assets.light_uniform))
        np.testing.assert_array_equal(light.surface, self.assets.surface)
        surface = self.assets.updated(replace(c, surface_seed=1))
        self.assertFalse(np.array_equal(surface.surface, self.assets.surface))
        np.testing.assert_array_equal(surface.light_uniform, self.assets.light_uniform)

    def test_direct_simulator_equivalence_and_sensor_only_reuse(self):
        c = self.config
        engine = PlaygroundEngine()
        capture = engine.run(c, self.assets)
        sim = CaptureSimulator(c.setup, self.assets.sample(c), c.spectrum, c.emitter,
                               device=c.device, dtype=torch.float64, batch_size=2)
        direct = sim.capture(c.camera_xy_m, samples=self.assets.illumination(c, c.final_samples))
        torch.testing.assert_close(capture.result.mean_intensity, direct.mean_intensity, atol=0, rtol=0)
        torch.testing.assert_close(capture.result.standard_error, direct.standard_error, atol=0, rtol=0)
        changed = replace(c, sensor=SensorNoise(exposure_scale=2, read_noise_std=.1, electrons_per_unit=1000))
        with patch.object(CaptureSimulator, "capture", side_effect=AssertionError("optics recomputed")):
            noisy = engine.run(changed, self.assets)
            repeated = engine.run(changed, self.assets)
            new_read = engine.run(replace(changed, sensor_seed=1), self.assets)
        expected = changed.sensor.apply(direct.mean_intensity, make_generator(0))
        torch.testing.assert_close(noisy.result.image, expected, atol=0, rtol=0)
        torch.testing.assert_close(repeated.result.image, noisy.result.image, atol=0, rtol=0)
        self.assertFalse(torch.equal(new_read.result.image, noisy.result.image))

    def test_geometry_rebuilds_and_matches_direct_simulator(self):
        engine = PlaygroundEngine()
        engine.run(self.config, self.assets)
        for setup in (replace(self.config.setup, sensor_shape=(28, 36)),
                      replace(self.config.setup, oversampling=2),
                      replace(self.config.setup, focal_length_m=.08),
                      replace(self.config.setup, sensor_pixel_pitch_m=1.6e-6)):
            with self.subTest(setup=setup):
                c = replace(self.config, setup=setup)
                result = engine.run(c, self.assets)
                sim = CaptureSimulator(c.setup, self.assets.sample(c), c.spectrum, c.emitter,
                                       device="cpu", dtype=torch.float64, batch_size=2)
                expected = sim.capture(samples=self.assets.illumination(c, c.final_samples))
                torch.testing.assert_close(result.result.mean_intensity, expected.mean_intensity, atol=0, rtol=0)
                self.assertEqual(result.sample.amplitude.shape, setup.object_shape)

    def test_cancellation_between_batches_and_latest_mailbox(self):
        progress = []
        with self.assertRaises(CaptureCancelled):
            PlaygroundEngine().run(self.config, self.assets, cancelled=lambda: bool(progress),
                                   progress=lambda n, total: progress.append(n))
        self.assertEqual(progress, [2])
        box = LatestRequest()
        a = box.invalidate()
        box.submit((a, "old"))
        b = box.invalidate()
        box.submit((a, "too late"))
        box.submit((b, "preview"))
        box.submit((b, "refined"))
        self.assertEqual(box.take(), (b, "refined"))
        self.assertFalse(box.accepts(a))
        box.close()
        self.assertIsNone(box.take())
        self.assertFalse(box.accepts(b))

    def test_preset_and_capture_round_trip_without_original_image(self):
        c = replace(self.config, sensor=SensorNoise(read_noise_std=.01, electrons_per_unit=500))
        capture = PlaygroundEngine().run(c, self.assets, quality="preview")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"preset.npz"
            save_preset(path, c, self.assets, {"auto_contrast": False, "low": 0, "high": .3})
            restored, assets, display = load_preset(path)
            self.assertEqual(restored, c)
            self.assertFalse(display["auto_contrast"])
            np.testing.assert_array_equal(assets.surface, self.assets.surface)
            np.testing.assert_array_equal(assets.light_uniform, self.assets.light_uniform)
            replay = PlaygroundEngine().run(restored, assets, quality="preview")
            torch.testing.assert_close(replay.result.image, capture.result.image, atol=0, rtol=0)
            directory = save_capture(Path(tmp)/"capture", capture, display)
            np.testing.assert_array_equal(np.load(directory/"0000.npy"), capture.result.image.numpy())
            with np.load(directory/"0000_optical.npz") as data:
                self.assertEqual(len(data["wavelengths_m"]), c.preview_samples)
            import json
            meta = json.loads((directory/"config.json").read_text())
            self.assertTrue(meta["complete"])
            self.assertEqual(meta["quality"], "preview")
            with self.assertRaises(FileExistsError):
                save_capture(directory, capture)

    def test_invalid_configurations_rejected(self):
        for change in (dict(bandwidth_m=-1), dict(wavelength_m=1e-9, bandwidth_m=20e-9),
                       dict(height_rms_m=-1), dict(object_width_m=0), dict(final_samples=0),
                       dict(preview_samples=100), dict(camera_xy_m=(float("nan"), 0))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change).validate()
        with self.assertRaisesRegex(ValueError, "Pupil"):
            PlaygroundEngine().run(replace(self.config, camera_xy_m=(1, 0)), self.assets)
        with self.assertRaisesRegex(ValueError, "Spherical"):
            PlaygroundEngine().run(replace(self.config, emitter=replace(self.config.emitter, distance_m=.001, wavefront="spherical")), self.assets)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_playground_matches_cpu_with_same_realizations(self):
        c = replace(self.config, dtype="float32")
        cpu = PlaygroundEngine().run(c, self.assets)
        gpu = PlaygroundEngine().run(replace(c, device="mps"), self.assets)
        torch.testing.assert_close(cpu.result.image, gpu.result.image, atol=2e-6, rtol=5e-4)
        self.assertEqual(gpu.device, "mps")
