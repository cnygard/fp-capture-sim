"""Optional Qt integration coverage; controls stay live while capture runs."""
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6 import QtWidgets
    from fp_capture_sim.playground import PlaygroundConfig, Realizations
    from fp_capture_sim.playground_ui import PlaygroundWindow
    from fp_capture_sim import OpticalSetup


@unittest.skipUnless(HAS_QT, "optional PySide6 dependency not installed")
class PlaygroundUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        config = PlaygroundConfig(setup=OpticalSetup(sensor_shape=(24, 32)),
            preview_samples=2, final_samples=4, device="cpu", batch_size=2,
            object_width_m=.00152)
        assets = Realizations.create(np.ones((24, 32), dtype=np.float32), "synthetic.png", config)
        self.window = PlaygroundWindow(config, assets)
        self.window.show()

    def wait_for(self, predicate, timeout=12):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        self.fail(f"Timed out: {self.window.status.text()}")

    def refined(self):
        self.wait_for(lambda: self.window.displayed is not None
                      and self.window.displayed.revision == self.window.revision
                      and self.window.displayed.quality == "refined")

    def tearDown(self):
        self.window.close()
        self.wait_for(lambda: not self.window.worker.isRunning())
        self.app.processEvents()

    def edit(self, key, value):
        self.window.controls[key].spin.setValue(value)

    def test_refinement_units_linked_size_and_immediate_specifications(self):
        self.refined()
        w = self.window
        w._show_info()
        self.edit("focal_length_m", 80)
        self.assertEqual(w.config.setup.focal_length_m, .08)
        self.assertIn("PENDING", w.info_text.toPlainText())
        self.assertIn("STALE", w.figure.axes[0].get_title())
        self.edit("object_height_m", 1.5)
        self.assertAlmostEqual(w.config.object_width_m, .002)
        self.assertAlmostEqual(w.controls["object_width_m"].value(), 2)
        self.edit("wavelength_m", 600)
        self.edit("radius_m", .3)
        self.edit("height_rms_m", .8)
        self.assertAlmostEqual(w.config.wavelength_m, 600e-9)
        self.assertAlmostEqual(w.config.emitter.radius_m, .3e-3)
        self.assertAlmostEqual(w.config.height_rms_m, .8e-6)
        self.refined()
        self.assertIn("Realized surface RMS:", w.info_text.toPlainText())
        self.assertTrue(w.info.isVisible())

    def test_rapid_changes_invalid_recovery_and_outdated_results(self):
        self.refined()
        w = self.window
        old = w.displayed
        for value in (76, 77, 78, 79, 80):
            self.edit("focal_length_m", value)
        w._completed(old)
        self.assertIs(w.displayed, old)
        self.refined()
        self.assertEqual(w.displayed.config.setup.focal_length_m, .08)
        last = w.displayed
        self.edit("object_distance_m", .01)
        self.assertFalse(w.valid)
        self.assertIs(w.displayed, last)
        self.assertIn("stale", w.status.text())
        self.assertTrue(w.controls["object_distance_m"].styleSheet())
        self.edit("object_distance_m", 2)
        self.refined()
        self.edit("camera_x", 20)
        self.wait_for(lambda: not w.valid)
        self.assertIn("Pupil", w.status.text())
        self.edit("camera_x", 0)
        self.refined()

    def test_sensor_only_contrast_reference_and_grid_rebuild(self):
        self.refined()
        w = self.window
        original = w.displayed
        from fp_capture_sim import CaptureSimulator
        with patch.object(CaptureSimulator, "capture", side_effect=AssertionError("optics recomputed")):
            self.edit("exposure_scale", 2)
            self.refined()
            self.assertTrue(np.array_equal(w.displayed.result.mean_intensity, original.result.mean_intensity))
            self.assertTrue(np.array_equal(w.displayed.result.image, original.result.image*2))
            w.auto_contrast.setChecked(False)
            w.high.setValue(.5)
            self.assertEqual(w.displayed.revision, w.revision)
        w._pin()
        w.tabs.setCurrentIndex(7)
        self.assertIn("Difference", w.figure.axes[0].get_title())
        self.edit("oversampling", 2)
        self.refined()
        self.assertIn("physical grids differ", w.figure.axes[0].texts[0].get_text())
        self.assertEqual(tuple(w.displayed.sample.height_m.shape), (48, 64))
        for index in range(7):
            w.tabs.setCurrentIndex(index)
            self.app.processEvents()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"object.png"
            import cv2
            cv2.imwrite(str(path), np.full((40, 20), 128, dtype=np.uint8))
            with patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(path), "")):
                w._choose_image()
            self.refined()
            self.assertEqual(w.assets.image.shape, (40, 20))
            self.assertAlmostEqual(w.controls["object_height_m"].value(), w.controls["object_width_m"].value()*2)

    def test_preset_precision_and_sensor_edit_during_preview_still_refines(self):
        w = self.window
        self.refined()
        precise = replace(w.config, setup=replace(w.config.setup, focal_length_m=.07512345678912345),
                          object_width_m=.001234567891234567)
        w.config = precise
        w._populate()
        self.assertEqual(w._collect(), precise)
        w._edited()
        self.wait_for(lambda: w.displayed.revision == w.revision and w.displayed.quality == "preview")
        self.edit("exposure_scale", 3)
        self.refined()
        self.assertEqual(w.displayed.config.sensor.exposure_scale, 3)

    def test_every_parameter_maps_to_models(self):
        w = self.window
        for key, value, read in [
            ("aperture_diameter_m", 4, lambda c: c.setup.aperture_diameter_m*1e3),
            ("sensor_pixel_pitch_m", 2, lambda c: c.setup.sensor_pixel_pitch_m*1e6),
            ("sensor_width", 36, lambda c: c.setup.sensor_shape[1]),
            ("sensor_height", 28, lambda c: c.setup.sensor_shape[0]),
            ("camera_x", .1, lambda c: c.camera_xy_m[0]*1e3),
            ("camera_y", -.1, lambda c: c.camera_xy_m[1]*1e3),
            ("bandwidth_m", 30, lambda c: c.bandwidth_m*1e9),
            ("distance_m", 3, lambda c: c.emitter.distance_m),
            ("emitter_x", .1, lambda c: c.emitter.center_xy_m[0]*1e3),
            ("emitter_y", -.1, lambda c: c.emitter.center_xy_m[1]*1e3),
            ("read_noise_std", .01, lambda c: c.sensor.read_noise_std),
            ("preview_samples", 3, lambda c: c.preview_samples),
            ("final_samples", 5, lambda c: c.final_samples),
            ("batch_size", 3, lambda c: c.batch_size)]:
            with self.subTest(key=key):
                self.edit(key, value)
                self.assertAlmostEqual(read(w.config), value)
        w.shot.setChecked(True)
        self.edit("electrons_per_unit", 500)
        self.assertEqual(w.config.sensor.electrons_per_unit, 500)
        w.wavefront.setCurrentText("spherical")
        self.assertEqual(w.config.emitter.wavefront, "spherical")
        w.precision.setCurrentText("float64")
        self.assertEqual(w.config.dtype, "float64")
        self.refined()
        self.assertEqual(w.displayed.result.image.shape, (28, 36))

    def test_close_during_capture(self):
        w = self.window
        self.edit("final_samples", 512)
        w._submit("refined")
        self.app.processEvents()
        w.close()
        self.assertTrue(w.closing)
        self.wait_for(lambda: not w.worker.isRunning())

    def test_closing_last_window_exits_application(self):
        import subprocess
        import sys
        script = """
import numpy as np
from PySide6 import QtCore, QtWidgets
from fp_capture_sim import OpticalSetup
from fp_capture_sim.playground import PlaygroundConfig, Realizations
from fp_capture_sim.playground_ui import PlaygroundWindow
app = QtWidgets.QApplication([])
c = PlaygroundConfig(setup=OpticalSetup(sensor_shape=(24, 32)), device="cpu")
a = Realizations.create(np.ones((24,32), dtype=np.float32), "test.png", c)
w = PlaygroundWindow(c, a)
w.show()
QtCore.QTimer.singleShot(200, w.close)
app.exec()
assert not w.worker.isRunning()
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_units_and_optional_gui_import(self):
        from fp_capture_sim.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"image.png"
            import cv2
            cv2.imwrite(str(path), np.ones((20, 40), dtype=np.uint8))
            with patch("fp_capture_sim.playground_ui.launch_playground") as launch:
                main(["--playground", "--image", str(path), "--resolution", "32",
                      "--focal-length-mm", "80", "--object-pixel-pitch-um", "10",
                      "--camera-x-mm", ".2", "--emitter-y-mm", ".3", "--device", "cpu"])
                config, assets = launch.call_args.args
                self.assertEqual(config.setup.focal_length_m, .08)
                self.assertAlmostEqual(config.object_width_m, 400e-6)
                self.assertEqual(config.camera_xy_m, (.0002, 0))
                self.assertAlmostEqual(config.emitter.center_xy_m[1], .0003)
