import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from fp_capture_sim import (OpticalSetup, RoughSample, Spectrum, CircularEmitter,
                            SensorNoise, CaptureSimulator, IlluminationSamples, spiral_positions)
from fp_capture_sim.backend import as_numpy, make_generator, resolve_device


def numpy_mode(setup, sample, emitter, wavelength, source, camera):
    """Independent NumPy reference for the pre-PyTorch forward operator."""
    ny, nx = setup.object_shape
    pitch = setup.object_pixel_pitch_m
    x, y = np.meshgrid((np.arange(nx)-nx//2)*pitch, (np.arange(ny)-ny//2)*pitch)
    fx, fy = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(nx, pitch)),
                         np.fft.fftshift(np.fft.fftfreq(ny, pitch)))
    sx, sy = source
    cx, cy = np.asarray(camera)/(wavelength*setup.object_distance_m)
    radius = setup.aperture_diameter_m/(2*wavelength*setup.object_distance_m)
    field = as_numpy(sample.amplitude)*np.exp(4j*np.pi*as_numpy(sample.height_m)/wavelength)
    if emitter.wavefront == "plane":
        cx += sx/(wavelength*emitter.distance_m)
        cy += sy/(wavelength*emitter.distance_m)
    else:
        r0 = np.sqrt(emitter.distance_m**2+sx*sx+sy*sy)
        r = np.sqrt(emitter.distance_m**2+(x-sx)**2+(y-sy)**2)
        path = (x*x+y*y-2*x*sx-2*y*sy)/(r+r0)
        field *= (r0/r)*np.exp(2j*np.pi*path/wavelength)
    pupil = (fx-cx)**2+(fy-cy)**2 <= radius**2
    spectrum = np.fft.fftshift(np.fft.fft2(field, norm="ortho"))
    image = np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum*pupil), norm="ortho"))**2
    s = setup.oversampling
    image = image.reshape(*setup.sensor_shape[:1], s, setup.sensor_shape[1], s).mean(axis=(1, 3))
    return image, pupil, np.abs(spectrum)**2


class CaptureTests(unittest.TestCase):
    def test_model_imports_preserve_class_identity(self):
        """Package and domain-module imports resolve to the same model classes."""
        import fp_capture_sim.models as models
        from fp_capture_sim.models.emitter import CircularEmitter as Emitter
        from fp_capture_sim.models.emitter import Spectrum as Lines
        from fp_capture_sim.models.object import RoughSample as Sample
        from fp_capture_sim.models.optical import OpticalSetup as Optics
        from fp_capture_sim.models.sensor import SensorNoise as Noise

        for name, implementation in (
            ("OpticalSetup", Optics),
            ("RoughSample", Sample),
            ("Spectrum", Lines),
            ("CircularEmitter", Emitter),
            ("SensorNoise", Noise),
        ):
            with self.subTest(model=name):
                self.assertIs(getattr(models, name), implementation)
                self.assertIs(globals()[name], implementation)

    def setUp(self):
        self.setup = OpticalSetup(sensor_shape=(32, 40))
        self.sample = RoughSample.from_intensity(torch.ones(self.setup.object_shape, dtype=torch.float64), 0)

    def sim(self, **kwargs):
        options = dict(device="cpu", dtype=torch.float64, batch_size=4)
        options.update(kwargs)
        return CaptureSimulator(self.setup, self.sample, **options)

    def modes(self):
        return IlluminationSamples(
            torch.tensor([522e-9, 542e-9, 531e-9, 537e-9, 525e-9, 534e-9, 529e-9], dtype=torch.float64),
            torch.tensor([[0, 0], [0.4e-3, 0], [-0.3e-3, 0.2e-3], [0, -0.2e-3],
                          [0.1e-3, 0], [0.2e-3, -0.1e-3], [-0.1e-3, 0.3e-3]], dtype=torch.float64))

    def test_uniform_field_and_repeated_modes_preserve_brightness(self):
        sim = self.sim(seed=2)
        one, many = sim.capture(count=1), sim.capture(count=40)
        torch.testing.assert_close(one.mean_intensity, torch.ones_like(one.mean_intensity), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(many.mean_intensity, one.mean_intensity, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(many.standard_error, torch.zeros_like(many.standard_error), atol=1e-14, rtol=0)
        self.assertTrue(torch.isnan(one.standard_error).all())

    def test_intensity_average_and_standard_error(self):
        sample = RoughSample.from_intensity(torch.ones(self.setup.object_shape, dtype=torch.float64), 0.5e-6, 3)
        sim = CaptureSimulator(self.setup, sample, device="cpu", dtype=torch.float64)
        modes = self.modes()
        images = torch.stack([sim.coherent_capture(w, xy)[0] for w, xy in zip(modes.wavelengths_m, modes.source_xy_m)])
        result = sim.capture(samples=modes)
        torch.testing.assert_close(result.mean_intensity, images.mean(dim=0), atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(result.standard_error, images.std(dim=0)/len(images)**0.5, atol=1e-14, rtol=1e-13)

    def test_batches_match_numpy_for_both_wavefronts_and_partial_batch(self):
        for wavefront in ("plane", "spherical"):
            for shape, oversampling in (((32, 40), 1), ((17, 21), 2), ((33, 41), 1)):
                with self.subTest(wavefront=wavefront, shape=shape, oversampling=oversampling):
                    setup = OpticalSetup(sensor_shape=shape, oversampling=oversampling)
                    sample = RoughSample.from_intensity(torch.ones(setup.object_shape, dtype=torch.float64), 0.5e-6, 4)
                    emitter = CircularEmitter(wavefront=wavefront)
                    modes, camera = self.modes(), (0.0003, -0.0002)
                    reference = [numpy_mode(setup, sample, emitter, w, xy, camera)
                                 for w, xy in zip(as_numpy(modes.wavelengths_m), as_numpy(modes.source_xy_m))]
                    reference_images = np.stack([item[0] for item in reference])
                    for batch_size in (1, 4, 16):
                        sim = CaptureSimulator(setup, sample, emitter=emitter, device="cpu",
                                               dtype=torch.float64, batch_size=batch_size)
                        result = sim.capture(camera, samples=modes)
                        np.testing.assert_allclose(as_numpy(result.mean_intensity), reference_images.mean(axis=0), atol=1e-13, rtol=1e-11)
                        np.testing.assert_allclose(as_numpy(result.standard_error), reference_images.std(axis=0, ddof=1)/len(reference)**0.5,
                                                   atol=1e-13, rtol=1e-11)
                        np.testing.assert_allclose(as_numpy(result.mean_pupil), np.mean([item[1] for item in reference], axis=0), atol=1e-14)
                        np.testing.assert_allclose(as_numpy(result.mean_spectrum_power), np.mean([item[2] for item in reference], axis=0), atol=1e-12)

    def test_camera_translation_selects_expected_object_frequency(self):
        wavelength = 532e-9
        x, _ = self.setup.object_grid(dtype=torch.float64)
        frequency = 4/(self.setup.object_shape[1]*self.setup.object_pixel_pitch_m)
        sample = RoughSample(torch.ones_like(x), wavelength*frequency*x/2)
        sim = CaptureSimulator(self.setup, sample, device="cpu", dtype=torch.float64)
        centered = sim.coherent_capture(wavelength)[0]
        moved = sim.coherent_capture(wavelength, camera_xy_m=(frequency*wavelength*self.setup.object_distance_m, 0))[0]
        self.assertLess(centered.max(), 1e-25)
        torch.testing.assert_close(moved, torch.ones_like(moved), atol=1e-13, rtol=1e-13)

    def test_source_and_camera_shift_sign(self):
        sim = self.sim()
        a = sim.coherent_capture(532e-9, source_xy_m=(0.4e-3, -0.2e-3))
        b = sim.coherent_capture(532e-9, camera_xy_m=(0.4e-3, -0.2e-3))
        torch.testing.assert_close(a[1], b[1])

    def test_wavelength_scales_pupil(self):
        sim = self.sim()
        short, long = sim.coherent_capture(450e-9)[1], sim.coherent_capture(650e-9)[1]
        self.assertGreater(short.sum(), long.sum())
        self.assertTrue(short[long].all())

    def test_roughness_is_physical_and_copies_input(self):
        height = torch.full(self.setup.object_shape, 532e-9/8, dtype=torch.float64)
        sample = RoughSample(torch.ones_like(height), height)
        height.zero_()
        torch.testing.assert_close(sample.field(532e-9), torch.full_like(height, 1j, dtype=torch.complex128), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(sample.field(1064e-9), torch.full_like(height, np.exp(1j*np.pi/4), dtype=torch.complex128))
        self.assertFalse(sample.amplitude.requires_grad)

    def test_disk_uniform_area_and_spectrum_probabilities(self):
        points = CircularEmitter(radius_m=2).sample(100000, make_generator(5), dtype=torch.float64)
        r2 = points.square().sum(dim=1)
        self.assertLessEqual(r2.max(), 4)
        self.assertAlmostEqual(r2.mean().item(), 2, delta=0.02)
        torch.testing.assert_close(points.mean(dim=0), torch.zeros(2, dtype=torch.float64), atol=0.015, rtol=0)
        values = Spectrum((500e-9, 600e-9), (1, 3)).sample(100000, make_generator(5), dtype=torch.float64)
        self.assertAlmostEqual((values == 600e-9).double().mean().item(), 0.75, delta=0.01)
        values = Spectrum.uniform().sample(10000, make_generator(0), dtype=torch.float64)
        self.assertTrue(((values >= 522e-9) & (values <= 542e-9)).all())

    def test_spherical_field_and_far_distance_limit(self):
        x, y = self.setup.object_grid(dtype=torch.float64)
        field = CircularEmitter(wavefront="spherical").field(x, y, (0, 0), 532e-9)
        self.assertAlmostEqual(field[x.shape[0]//2, x.shape[1]//2].abs().item(), 1)
        self.assertFalse(torch.allclose(field, torch.ones_like(field)))
        sphere = CircularEmitter(distance_m=1e8, wavefront="spherical")
        plane = CircularEmitter(distance_m=1e8)
        torch.testing.assert_close(sphere.field(x, y, (1e3, 0), 532e-9),
                                   plane.field(x, y, (1e3, 0), 532e-9), atol=2e-6, rtol=0)

    def test_sensor_noise_once_independent_of_mode_count(self):
        sensor = SensorNoise(exposure_scale=3, read_noise_std=0.1, bias=0.2, electrons_per_unit=1000)
        one = self.sim(sensor=sensor, seed=8).capture(count=1)
        many = self.sim(sensor=sensor, seed=8).capture(count=20)
        torch.testing.assert_close(one.image, many.image, atol=1e-14, rtol=1e-14)
        self.assertGreater(one.image.std(), 0.08)
        self.assertAlmostEqual(one.image.mean().item(), 3.2, delta=0.03)

    def test_pixel_integration_and_noise_statistics(self):
        setup = OpticalSetup(sensor_shape=(2, 3), oversampling=2)
        array = torch.arange(24, dtype=torch.float64).reshape(4, 6)
        expected = torch.tensor([[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]], dtype=torch.float64)
        torch.testing.assert_close(setup.integrate_pixels(array), expected)
        torch.testing.assert_close(setup.integrate_pixels(torch.stack((array, array))), torch.stack((expected, expected)))
        output = SensorNoise(read_noise_std=2, bias=5, electrons_per_unit=10).apply(
            torch.full((400, 400), 3.0), make_generator(4))
        self.assertAlmostEqual(output.mean().item(), 8, delta=0.02)
        self.assertAlmostEqual(output.var().item(), 4.3, delta=0.07)
        clipped = SensorNoise(bias=-10, clip_min=0).apply(torch.ones(2))
        torch.testing.assert_close(clipped, torch.zeros(2))

    def test_reproducibility_independent_noise_and_no_global_rng_mutation(self):
        state = torch.random.get_rng_state()
        sensor = SensorNoise(read_noise_std=0.1)
        a, b = self.sim(seed=9, sensor=sensor), self.sim(seed=9, sensor=sensor)
        torch.testing.assert_close(a.capture(count=4).image, b.capture(count=4).image, atol=0, rtol=0)
        scan = list(a.scan([(0, 0), (0, 0)], count=3))
        torch.testing.assert_close(scan[0].mean_intensity, scan[1].mean_intensity)
        self.assertFalse(torch.equal(scan[0].image, scan[1].image))
        torch.testing.assert_close(torch.random.get_rng_state(), state)

    def test_parameters_devices_and_sampling_validation(self):
        with self.assertRaises(ValueError):
            self.sim().capture(count=0)
        with self.assertRaises(ValueError):
            self.sim(batch_size=0)
        with self.assertRaises(ValueError):
            self.sim(dtype=torch.float16)
        with self.assertRaises(ValueError):
            self.sim(device="meta")
        with self.assertRaises(ValueError):
            self.sim().coherent_capture(532e-9, camera_xy_m=(1, 0))
        with self.assertRaises(ValueError):
            self.sim().coherent_capture(0)
        with self.assertRaises(ValueError):
            self.sim().capture(camera_xy_m=(float("nan"), 0))
        with self.assertRaises(ValueError):
            self.sim(emitter=CircularEmitter(distance_m=0.001, wavefront="spherical")).capture(count=1)
        with self.assertRaises(ValueError):
            Spectrum((532e-9,), (-1,))
        with self.assertRaises(ValueError):
            IlluminationSamples([], [])
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "CUDA"):
                resolve_device("cuda")
            with self.assertRaisesRegex(ValueError, "MPS"):
                resolve_device("mps")
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(resolve_device("auto"), torch.device("mps"))
            self.assertEqual(resolve_device("auto", torch.float64), torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "float32"):
                resolve_device("mps", torch.float64)

    def test_saved_scan_raw_values_and_execution_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/"scan"
            self.sim(seed=3).save_scan(directory, [(0, 0), (0.001, 0)], count=3)
            metadata = json.loads((directory/"config.json").read_text())
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["backend"], "torch")
            self.assertEqual(metadata["device"], "cpu")
            self.assertEqual(metadata["dtype"], "torch.float64")
            np.testing.assert_allclose(np.load(directory/"0000.npy"), 1, atol=1e-14)
            with np.load(directory/"0000_optical.npz") as data:
                self.assertEqual(data["source_xy_m"].shape, (3, 2))
            with self.assertRaises(FileExistsError):
                self.sim().save_scan(directory, [(0, 0)], count=1)

    def test_spiral_complete_and_centered(self):
        for size in (1, 3, 5, 15):
            xy = spiral_positions(size, 0.001)
            self.assertEqual(len(np.unique(xy, axis=0)), size**2)
            np.testing.assert_allclose(xy.mean(axis=0), 0, atol=1e-17)
            if size > 1:
                np.testing.assert_allclose(np.abs(np.diff(xy, axis=0)).sum(axis=1), 0.001)

    def _exercise_gpu(self, device):
        modes = self.modes()
        sample = RoughSample.from_intensity(torch.ones(self.setup.object_shape, dtype=torch.float64), 0.5e-6, 3)
        for wavefront in ("plane", "spherical"):
            emitter = CircularEmitter(wavefront=wavefront)
            cpu = CaptureSimulator(self.setup, sample, emitter=emitter, device="cpu", dtype=torch.float32, batch_size=4)
            gpu = CaptureSimulator(self.setup, sample, emitter=emitter, device=device, dtype=torch.float32, batch_size=4)
            a, b = cpu.capture(samples=modes), gpu.capture(samples=modes)
            self.assertEqual(b.image.device.type, device)
            self.assertEqual(b.samples.wavelengths_m.device.type, device)
            self.assertEqual(gpu.sample.height_m.device.type, device)
            self.assertFalse(b.image.requires_grad)
            host = b.cpu()
            for name in ("mean_intensity", "standard_error", "mean_pupil", "mean_spectrum_power"):
                torch.testing.assert_close(getattr(host, name), getattr(a, name), atol=2e-6, rtol=5e-4)
            noise = SensorNoise(read_noise_std=0.001, electrons_per_unit=1000, clip_min=0)
            noisy = CaptureSimulator(self.setup, sample, spectrum=Spectrum.uniform(), emitter=emitter,
                                     sensor=noise, device=device, seed=7, batch_size=4)
            draws = noisy.draw_samples(7)
            self.assertEqual(draws.source_xy_m.device.type, device)
            result = noisy.capture(samples=draws)
            self.assertEqual(result.image.device.type, device)
            self.assertTrue(torch.isfinite(result.image).all())
            self.assertTrue((result.image >= 0).all())
            self.assertFalse(torch.equal(result.image, result.mean_intensity))
            # Also exercise discrete spectral sampling and GPU->NumPy export.
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)/wavefront
                gpu.save_scan(directory, [(0, 0)], count=3)
                self.assertEqual(np.load(directory/"0000.npy").shape, self.setup.sensor_shape)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS GPU unavailable")
    def test_mps_measurements_match_cpu_and_noise_export_stay_usable(self):
        self._exercise_gpu("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
    def test_cuda_measurements_match_cpu_and_noise_export_stay_usable(self):
        self._exercise_gpu("cuda")


if __name__ == "__main__":
    unittest.main()
