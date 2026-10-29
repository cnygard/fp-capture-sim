"""Optional PySide6 desktop playground. All rendering belongs to the GUI thread."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import time

import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from .backend import as_numpy
from .models import OpticalSetup, CircularEmitter, SensorNoise
from .playground import (
    PlaygroundConfig,
    Realizations,
    PlaygroundEngine,
    LatestRequest,
    CaptureCancelled,
    geometry_specs,
    save_preset,
    load_preset,
    save_capture,
)


class NumberControl(QtWidgets.QWidget):
    """Linked slider and numeric entry that preserves exact model/preset values."""
    changed = QtCore.Signal()

    def __init__(self, value, low, high, *, integer=False):
        """Create synchronized slider and numeric controls in display units.

        Args:
            value: Initial numeric value, preserved at full supplied precision.
            low: Usual slider minimum; extended to include the initial value.
            high: Usual slider maximum; extended to include the initial value.
            integer: Whether the numeric control accepts integers instead of
                floating-point values.
        """
        super().__init__()
        self.low, self.high = min(low, value), max(high, value)
        self.integer = integer
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.spin = QtWidgets.QSpinBox() if integer else QtWidgets.QDoubleSpinBox()
        self.spin.setRange(self.low, self.high)
        if not integer:
            self.spin.setDecimals(6)
            self.spin.setSingleStep(max((high-low)/1000, 0.000001))
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(112)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.set_value(value)
        self.slider.valueChanged.connect(self._slide)
        self.spin.valueChanged.connect(self._number)

    def _slide(self, position):
        """Map a slider position to the numeric control in display units.

        Args:
            position: Integer slider position in [0, 1000].
        """
        value = self.low+(self.high-self.low)*position/1000
        self.spin.setValue(round(value) if self.integer else value)

    def _number(self, value):
        """Record a numeric edit, synchronize the slider, and emit one change signal.

        Args:
            value: New numeric value in the display units of this control.
        """
        self._exact_value = value
        self.si_value = None
        # Synchronizing the slider must not recursively emit another user edit.
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(round(1000*(value-self.low)/(self.high-self.low)))
        del blocker
        self.changed.emit()

    def value(self):
        """Return the exact stored value in display units."""
        return self._exact_value

    def set_value(self, value):
        """Set a value silently, extending the slider range when necessary.

        Args:
            value: Numeric value in display units; retain precision beyond the visible
                spin box.
        """
        self._exact_value = value
        self.si_value = None
        # Preserve all CLI/preset values, even beyond the usual slider ranges.
        self.low, self.high = min(self.low, value), max(self.high, value)
        self.spin.setRange(self.low, self.high)
        with QtCore.QSignalBlocker(self.spin), QtCore.QSignalBlocker(self.slider):
            self.spin.setValue(value)
            self.slider.setValue(round(1000*(value-self.low)/(self.high-self.low)))


class CaptureWorker(QtCore.QThread):
    """Single simulation worker communicating with the GUI through revision-tagged signals."""
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(int, str)
    advanced = QtCore.Signal(int, int, int, float)

    def __init__(self, mailbox, parent=None):
        """Create a background capture worker with its own simulation cache.

        Args:
            mailbox: LatestRequest shared with the GUI for requests and cancellation.
            parent: Optional Qt parent that owns this object.
        """
        super().__init__(parent)
        self.mailbox = mailbox
        self.engine = PlaygroundEngine()

    def run(self):
        """Consume latest requests until closure, emitting results only for current
        revisions.

        This thread owns simulation work; Qt signals hand all UI updates to the GUI
        thread.
        """
        while (request := self.mailbox.take()) is not None:
            revision, config, assets, quality = request
            start = time.monotonic()
            try:
                capture = self.engine.run(
                    config, assets, quality, revision,
                    cancelled=lambda: not self.mailbox.accepts(revision),
                    progress=lambda n, total: self.advanced.emit(revision, n, total, time.monotonic()-start))
                if self.mailbox.accepts(revision):
                    self.completed.emit(capture)
            except CaptureCancelled:
                pass
            except Exception as exc:
                self.failed.emit(revision, str(exc))


# key, label, display conversion from SI, minimum, maximum, integer
CONTROLS = {
    "Optics and sensor": [
        ("focal_length_m", "Focal length (mm)", 1e3, .1, 500, False),
        ("aperture_diameter_m", "Aperture diameter (mm)", 1e3, .01, 100, False),
        ("sensor_pixel_pitch_m", "Sensor pixel pitch (µm)", 1e6, .01, 20, False),
        ("sensor_width", "Sensor width (pixels)", 1, 2, 2048, True),
        ("sensor_height", "Sensor height (pixels)", 1, 2, 2048, True),
        ("camera_x", "Camera x (mm)", 1e3, -20, 20, False),
        ("camera_y", "Camera y (mm)", 1e3, -20, 20, False)],
    "Emitter": [
        ("wavelength_m", "Center wavelength (nm)", 1e9, 200, 2000, False),
        ("bandwidth_m", "Full spectral bandwidth (nm)", 1e9, 0, 500, False),
        ("radius_m", "Emitter radius (mm)", 1e3, 0, 20, False),
        ("distance_m", "Emitter–object distance (m)", 1, .001, 20, False),
        ("emitter_x", "Emitter x (mm)", 1e3, -20, 20, False),
        ("emitter_y", "Emitter y (mm)", 1e3, -20, 20, False)],
    "Object": [
        ("object_distance_m", "Object–lens distance (m)", 1, .001, 20, False),
        ("object_width_m", "Physical width (mm)", 1e3, .001, 200, False),
        ("object_height_m", "Physical height (mm)", 1e3, .001, 200, False),
        ("height_rms_m", "Surface height RMS (µm)", 1e6, 0, 10, False)],
    "Acquisition": [
        ("exposure_scale", "Exposure scale", 1, 0, 10, False),
        ("read_noise_std", "Read noise std (intensity)", 1, 0, 1, False),
        ("electrons_per_unit", "Electrons per intensity unit", 1, 1, 100000, False)],
    "Computation": [
        ("preview_samples", "Preview samples", 1, 1, 512, True),
        ("final_samples", "Final samples", 1, 1, 4096, True),
        ("oversampling", "Oversampling per axis", 1, 1, 16, True),
        ("batch_size", "FFT batch size", 1, 1, 128, True)]}


class PlaygroundWindow(QtWidgets.QMainWindow):
    """Live parameter editor and capture viewer; all rendering runs on the GUI thread."""
    def __init__(self, config, assets, display=None):
        """Build the playground, restore display settings, and schedule the first capture.

        Args:
            config: PlaygroundConfig containing physical settings, execution options,
                and seeds.
            assets: Realizations holding the original image and shared surface/light
                draws.
            display: Optional mapping of contrast limits, automatic contrast, and
                selected tab.
        """
        super().__init__()
        self.setWindowTitle("Fourier ptychography · Simulation playground")
        self.resize(1440, 940)
        self.config, self.assets = config, assets
        self.displayed = self.reference = None
        self.controls = {}
        self.loading = False
        self.closing = False
        self.worker_done = False
        self.contrast_limits = None
        self.mailbox = LatestRequest()
        self.revision = self.mailbox.invalidate()
        self.worker = CaptureWorker(self.mailbox, self)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.advanced.connect(self._progress)
        self.worker.finished.connect(self._worker_finished)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(lambda: self._submit("preview"))
        self.refine_timer = QtCore.QTimer(self)
        self.refine_timer.setSingleShot(True)
        self.refine_timer.timeout.connect(lambda: self._submit("refined"))
        self.info = QtWidgets.QDialog(self)
        self.info.setWindowTitle("Live system information")
        self.info.resize(650, 700)
        self.info_text = QtWidgets.QPlainTextEdit()
        self.info_text.setReadOnly(True)
        QtWidgets.QVBoxLayout(self.info).addWidget(self.info_text)
        self._build()
        self._populate()
        self._restore_display(display or {})
        self.worker.start()
        self._edited()

    def _build(self):
        """Build parameter controls, capture views, and file/display actions."""
        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(440)
        panel = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QLabel("Simulation parameters")
        title.setStyleSheet("font-size: 20px; font-weight: 600; padding: 8px 0")
        left.addWidget(title)
        for group, rows in CONTROLS.items():
            box = QtWidgets.QGroupBox(group)
            form = QtWidgets.QFormLayout(box)
            if group == "Object":
                self.image_label = QtWidgets.QLabel()
                self.image_label.setWordWrap(True)
                form.addRow(self.image_label)
                button = QtWidgets.QPushButton("Choose object image…")
                button.clicked.connect(self._choose_image)
                form.addRow(button)
            for key, label, factor, low, high, integer in rows:
                control = NumberControl(low, low, high, integer=integer)
                control.changed.connect(lambda k=key: self._edited(k))
                self.controls[key] = control
                form.addRow(label, control)
            if group == "Emitter":
                self.wavefront = QtWidgets.QComboBox()
                self.wavefront.addItems(["plane", "spherical"])
                self.wavefront.currentTextChanged.connect(lambda: self._edited())
                form.addRow("Wavefront", self.wavefront)
                note = QtWidgets.QLabel("Emitter size and bandwidth hold total source power fixed.")
                note.setWordWrap(True)
                form.addRow(note)
            if group == "Object":
                note = QtWidgets.QLabel("Width and height stay linked to image aspect ratio.\nObject distance maintains focus automatically.\nSurface: fixed periodic physical lattice, interpolated onto the grid.")
                note.setWordWrap(True)
                form.addRow(note)
            if group == "Acquisition":
                self.shot = QtWidgets.QCheckBox("Enable shot noise")
                self.shot.toggled.connect(lambda: self._edited("sensor"))
                form.addRow(self.shot)
            if group == "Computation":
                self.device = QtWidgets.QComboBox()
                self.device.setEditable(True)
                self.device.addItems(["auto", "cpu", "cuda", "mps"])
                self.device.currentTextChanged.connect(lambda: self._edited())
                self.precision = QtWidgets.QComboBox()
                self.precision.addItems(["float32", "float64"])
                self.precision.currentTextChanged.connect(lambda: self._edited())
                form.addRow("Device", self.device)
                form.addRow("Precision", self.precision)
            left.addWidget(box)
        left.addStretch()
        scroll.setWidget(panel)
        splitter.addWidget(scroll)
        right = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(right)
        toolbar = QtWidgets.QHBoxLayout()
        for text, slot in [("System information", self._show_info), ("Load preset…", self._load),
                           ("Save preset…", self._save), ("Export displayed capture…", self._export)]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            toolbar.addWidget(button)
        layout.addLayout(toolbar)
        actions = QtWidgets.QHBoxLayout()
        for text, field in [("Resample light", "light_seed"), ("Regenerate surface", "surface_seed"),
                            ("New sensor read", "sensor_seed")]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, f=field: self._resample(f))
            actions.addWidget(button)
        pin = QtWidgets.QPushButton("Pin displayed reference")
        pin.clicked.connect(self._pin)
        actions.addWidget(pin)
        layout.addLayout(actions)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.status = QtWidgets.QLabel("Waiting for first capture")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("padding: 8px; background: #eef2f6; color: #182a40")
        layout.addWidget(self.status)
        self.tabs = QtWidgets.QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setUsesScrollButtons(True)
        for title in ["Sensor capture", "Noiseless intensity", "Reflectance", "Surface height",
                      "Pupil", "Spectral power", "Reference", "Difference"]:
            self.tabs.addTab(title)
        self.tabs.currentChanged.connect(self._render)
        layout.addWidget(self.tabs)
        self.figure = Figure(figsize=(8, 7), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        contrast = QtWidgets.QHBoxLayout()
        self.auto_contrast = QtWidgets.QCheckBox("Automatic contrast")
        self.auto_contrast.setChecked(True)
        self.auto_contrast.toggled.connect(self._contrast_changed)
        contrast.addWidget(self.auto_contrast)
        self.low = QtWidgets.QDoubleSpinBox()
        self.high = QtWidgets.QDoubleSpinBox()
        for label, spin, value in [("Low", self.low, 0), ("High", self.high, 1)]:
            spin.setDecimals(9)
            spin.setRange(-1e12, 1e12)
            spin.setValue(value)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._contrast_changed)
            contrast.addWidget(QtWidgets.QLabel(label))
            contrast.addWidget(spin)
        layout.addLayout(contrast)
        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)
        splitter.addWidget(right)
        splitter.setSizes([460, 980])

    def _populate(self):
        """Populate display-unit controls from the model while suppressing edit scheduling."""
        self.loading = True
        c, s, e, n = self.config, self.config.setup, self.config.emitter, self.config.sensor
        values = {**vars(s), **vars(e), **vars(n), **vars(c),
                  "sensor_width": s.sensor_shape[1], "sensor_height": s.sensor_shape[0],
                  "camera_x": c.camera_xy_m[0], "camera_y": c.camera_xy_m[1],
                  "emitter_x": e.center_xy_m[0], "emitter_y": e.center_xy_m[1],
                  "object_height_m": c.object_width_m*self.assets.image.shape[0]/self.assets.image.shape[1],
                  "electrons_per_unit": n.electrons_per_unit or 1000}
        for rows in CONTROLS.values():
            for key, _, factor, *_ in rows:
                self.controls[key].set_value(values[key]*factor)
                self.controls[key].si_value = values[key]
        self.wavefront.setCurrentText(e.wavefront)
        self.device.setCurrentText(c.device)
        self.precision.setCurrentText(c.dtype)
        self.shot.setChecked(n.electrons_per_unit is not None)
        self.image_label.setText(f"{Path(self.assets.image_path).name} · {self.assets.image.shape[1]} × {self.assets.image.shape[0]} px")
        self.image_label.setToolTip(self.assets.image_path)
        self.loading = False

    def _collect(self):
        """Return a validated config by converting display units back to SI."""
        v = {}
        for rows in CONTROLS.values():
            for key, _, factor, *_ in rows:
                control = self.controls[key]
                # Preserve exact CLI/preset SI values until this control is edited.
                v[key] = control.si_value if control.si_value is not None else control.value()/factor
        setup = OpticalSetup(**{k: v[k] for k in ("focal_length_m", "aperture_diameter_m", "sensor_pixel_pitch_m", "object_distance_m")},
                             sensor_shape=(int(v["sensor_height"]), int(v["sensor_width"])), oversampling=int(v["oversampling"]))
        emitter = CircularEmitter(radius_m=v["radius_m"], distance_m=v["distance_m"],
                                  center_xy_m=(v["emitter_x"], v["emitter_y"]), wavefront=self.wavefront.currentText())
        sensor = SensorNoise(exposure_scale=v["exposure_scale"], read_noise_std=v["read_noise_std"],
                             electrons_per_unit=v["electrons_per_unit"] if self.shot.isChecked() else None)
        return replace(self.config, setup=setup, emitter=emitter, sensor=sensor,
                       wavelength_m=v["wavelength_m"], bandwidth_m=v["bandwidth_m"],
                       object_width_m=v["object_width_m"], height_rms_m=v["height_rms_m"],
                       camera_xy_m=(v["camera_x"], v["camera_y"]),
                       preview_samples=int(v["preview_samples"]), final_samples=int(v["final_samples"]),
                       batch_size=int(v["batch_size"]), device=self.device.currentText(), dtype=self.precision.currentText()).validate()

    def _edited(self, key=None):
        """Invalidate obsolete work, validate edits, and debounce preview/refinement
        requests.

        Args:
            key: Optional edited control key, used to link object dimensions at the
                image aspect ratio.
        """
        if self.loading or self.closing:
            return
        self.preview_timer.stop()
        self.refine_timer.stop()
        # Cancel in-flight work now; debounce only submission of replacement work.
        self.revision = self.mailbox.invalidate()
        for widget in [*self.controls.values(), self.device, self.precision]:
            widget.setStyleSheet("")
        if key in ("object_width_m", "object_height_m"):
            ratio = self.assets.image.shape[0]/self.assets.image.shape[1]
            if key == "object_width_m":
                self.controls["object_height_m"].set_value(self.controls[key].value()*ratio)
            else:
                self.controls["object_width_m"].set_value(self.controls[key].value()/ratio)
        try:
            self.config = self._collect()
            self.assets = self.assets.updated(self.config)
        except (ValueError, RuntimeError) as exc:
            self._failed(self.revision, str(exc))
            return
        self.valid = True
        self._mark_stale()
        self.status.setText("Pending · displayed image is stale while the new configuration is computed.")
        self.progress.setValue(0)
        self._update_info()
        # Sensor-only edits reuse the displayed optical quality and sample count.
        if self.displayed and replace(self.config, sensor=self.displayed.config.sensor,
                                       sensor_seed=self.displayed.config.sensor_seed) == self.displayed.config:
            quality = self.displayed.quality
            QtCore.QTimer.singleShot(0, lambda: self._submit(quality))
            if quality == "preview":
                self.refine_timer.start(750)
        else:
            self.preview_timer.start(150)
            self.refine_timer.start(750)

    def _submit(self, quality):
        """Enqueue the current valid controls and assets at the requested capture quality.

        Args:
            quality: Capture level: "preview" uses preview_samples, "refined" uses
                final_samples.
        """
        if not self.closing and getattr(self, "valid", False):
            self.mailbox.submit((self.revision, self.config, self.assets, quality))

    @QtCore.Slot(object)
    def _completed(self, capture):
        """Accept a current result, update specifications, and redraw the displayed image.

        Args:
            capture: Completed PlaygroundCapture carrying its originating mailbox
                revision.
        """
        if not self.mailbox.accepts(capture.revision):
            return
        self.displayed = capture
        self.progress.setValue(100)
        self.status.setText(f"{capture.quality.capitalize()} · {len(capture.result.samples.wavelengths_m)} samples · "
                            f"{capture.device} · {capture.elapsed_s:.2f} s")
        self._update_info()
        self._render()

    @QtCore.Slot(int, str)
    def _failed(self, revision, message):
        """Retain the last image as stale and highlight controls related to a current
        failure.

        Args:
            revision: Mailbox generation used to reject work superseded by newer edits.
            message: Validation or capture failure message to display beside the
                controls.
        """
        if not self.mailbox.accepts(revision):
            return
        self.valid = False
        self._mark_stale()
        self.preview_timer.stop()
        self.refine_timer.stop()
        self.status.setText(f"Invalid configuration · last valid image retained (stale). {message}")
        groups = {
            "Pupil": ["oversampling", "aperture_diameter_m", "camera_x", "camera_y", "radius_m", "sensor_pixel_pitch_m"],
            "Spherical": ["oversampling", "distance_m"],
            "object_distance": ["object_distance_m", "focal_length_m"],
            "wavelength": ["wavelength_m", "bandwidth_m"],
            "bandwidth": ["bandwidth_m"],
            "samples": ["preview_samples", "final_samples"],
        }
        for token, keys in groups.items():
            if token.lower() in message.lower():
                for key in keys:
                    self.controls[key].setStyleSheet("border: 1px solid #c65345")
        if any(token in message.lower() for token in ("device", "mps", "cuda", "float")):
            self.device.setStyleSheet("border: 1px solid #c65345")
            self.precision.setStyleSheet("border: 1px solid #c65345")
        self._update_info(error=message)

    @QtCore.Slot(int, int, int, float)
    def _progress(self, revision, n, total, elapsed):
        """Update progress only while the reported request revision remains current.

        Args:
            revision: Mailbox generation used to reject work superseded by newer edits.
            n: Number of illumination modes completed.
            total: Total illumination modes requested.
            elapsed: Elapsed wall-clock capture time in seconds.
        """
        if self.mailbox.accepts(revision):
            self.progress.setValue(round(100*n/total))
            self.status.setText(f"Computing · {n}/{total} samples · {elapsed:.2f} s · displayed image may be stale")

    def _update_info(self, error=None):
        """Refresh immediate geometry and capture-dependent values only for a matching
        result.

        Args:
            error: Optional validation failure; geometry then describes the last valid
                configuration.
        """
        c = self.config
        spec = geometry_specs(c, self.assets.image.shape)
        pair = lambda values, factor: " × ".join(f"{v*factor:.6g}" for v in values)
        self.summary.setText(f"NA {spec['object_na']:.5g} · f/{spec['f_number']:.3g} · |M| {spec['magnification']:.4g} · "
                             f"Object FOV {pair(spec['field_of_view_m'], 1e3)} mm")
        matching = self.displayed is not None and self.displayed.revision == self.revision and error is None
        lines = [f"Current controls · revision {self.revision}" + (" (geometry below is the last valid configuration)" if error else ""),
                 "Sampling validity: " + (f"INVALID: {error}" if error else "validated for displayed draws" if matching else "pending capture validation"),
                 f"Object-side NA: {spec['object_na']:.8g}", f"F-number: f/{spec['f_number']:.6g}",
                 f"Magnification: {spec['magnification']:.8g}",
                 f"Lens–sensor distance: {spec['image_distance_m']*1e3:.6g} mm",
                 f"Sensor dimensions: {pair(spec['sensor_size_m'], 1e3)} mm",
                 f"Object-plane FOV: {pair(spec['field_of_view_m'], 1e3)} mm",
                 f"Measurement pitch: {spec['measurement_pitch_m']*1e6:.6g} µm",
                 f"Simulation pitch: {spec['grid_pitch_m']*1e6:.6g} µm",
                 f"Grid: {spec['grid_shape'][1]} × {spec['grid_shape'][0]}",
                 f"Physical object: {pair(spec['object_size_m'], 1e3)} mm",
                 f"Requested surface RMS: {c.height_rms_m*1e6:.6g} µm",
                 f"Wavelength range: {pair(spec['wavelength_range_m'], 1e9)} nm",
                 f"Emitter angular radius: {spec['emitter_angular_radius_rad']*1e3:.6g} mrad",
                 f"Requested execution: {c.device}, {c.dtype}; batch {c.batch_size}",
                 f"Preview / final samples: {c.preview_samples} / {c.final_samples}",
                 f"Light / surface / sensor seeds: {c.light_seed} / {c.surface_seed} / {c.sensor_seed}", ""]
        if matching:
            d = self.displayed
            error_mean = d.result.standard_error.mean().item()
            lines += [f"Displayed capture: revision {d.revision}, {d.quality}",
                      f"Device: {d.device}; {len(d.result.samples.wavelengths_m)} samples; {d.elapsed_s:.3f} s",
                      f"Realized surface RMS: {d.sample.height_m.square().mean().sqrt().item()*1e6:.6g} µm",
                      f"Mean Monte Carlo standard error: {error_mean:.6g} optical units" if math.isfinite(error_mean)
                      else "Monte Carlo standard error: unavailable for a single sample"]
        else:
            lines += ["Capture-dependent specifications: PENDING", "Realized surface RMS / device / uncertainty: pending matching result"]
        lines += ["", "Object distance automatically maintains focus.",
                  "Emitter radius and bandwidth hold total source power fixed.",
                  "Surface lattice pitch stays fixed in physical coordinates.",
                  "Interpolation can change realized RMS as grids change.",
                  "Sampling checks do not establish rough-surface convergence.",
                  "Sensor noise uses an independent CPU stream for reproducibility."]
        self.info_text.setPlainText("\n".join(lines))

    def _show_info(self):
        """Show and raise the modeless live system-information dialog."""
        self.info.show()
        self.info.raise_()

    def _contrast_changed(self, *_):
        """Redraw after contrast edits without scheduling another simulation.

        Args:
            _: Unused GUI signal or callback arguments.
        """
        self._render()

    @staticmethod
    def _extent(capture, frequency=False):
        """Return imshow pixel-edge bounds in mm or cycles/mm, including odd grid sizes.

        Args:
            capture: PlaygroundCapture whose registered physical grid defines the axes.
            frequency: Whether to return frequency bounds rather than spatial bounds.
        """
        s = capture.config.setup
        ny, nx = s.object_shape
        pitch = s.object_pixel_pitch_m
        if frequency:
            fx = np.fft.fftshift(np.fft.fftfreq(nx, pitch))/1e3
            fy = np.fft.fftshift(np.fft.fftfreq(ny, pitch))/1e3
            dx, dy = 1/(nx*pitch*1e3), 1/(ny*pitch*1e3)
            return [fx[0]-dx/2, fx[-1]+dx/2, fy[0]-dy/2, fy[-1]+dy/2]
        return [(-(nx//2)-.5)*pitch*1e3,
                (nx-nx//2-.5)*pitch*1e3, (-(ny//2)-.5)*pitch*1e3,
                (ny-ny//2-.5)*pitch*1e3]

    def _mark_stale(self):
        # Reusing the existing axes avoids rebuilding a figure on every slider tick.
        """Mark the retained capture stale without rebuilding its Matplotlib axes."""
        if self.displayed and self.figure.axes:
            ax = self.figure.axes[0]
            if "STALE" not in ax.get_title():
                ax.set_title(ax.get_title()+" · STALE")
            self.canvas.draw_idle()

    def _render(self, *_):
        """Render the selected diagnostic, reference, or registered difference with
        contrast controls.

        Args:
            _: Unused GUI signal or callback arguments.
        """
        if self.displayed is None:
            return
        d = self.displayed
        tab = self.tabs.currentIndex()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if tab in (6, 7) and self.reference is None:
            ax.text(.5, .5, "Pin a displayed capture to compare", ha="center", transform=ax.transAxes)
            self.canvas.draw_idle()
            return
        # Pixelwise subtraction requires identical registration and sampling.
        if tab == 7 and d.grid_key != self.reference.grid_key:
            ax.text(.5, .5, "Difference unavailable: physical grids differ.\nMatch sensor shape, measurement pitch, and oversampling.",
                    ha="center", transform=ax.transAxes)
            self.canvas.draw_idle()
            return
        r = d.result
        arrays = [r.image, r.mean_intensity, d.sample.amplitude.square(), d.sample.height_m*1e6,
                  r.mean_pupil, r.mean_spectrum_power]
        if tab == 6:
            array = as_numpy(self.reference.result.image)
            extent = self._extent(self.reference)
        elif tab == 7:
            array = as_numpy(r.image-self.reference.result.image)
            extent = self._extent(d)
        else:
            array = as_numpy(arrays[tab])
            if tab == 5:
                array = np.log1p(array)
            extent = self._extent(d, tab in (4, 5))
        image = ax.imshow(array, origin="lower", extent=extent,
                          cmap="coolwarm" if tab in (3, 7) else "gray", interpolation="nearest")
        if tab in (0, 1, 6):
            if self.auto_contrast.isChecked():
                low, high = float(np.min(array)), float(np.max(array))
                high = max(high, low+1e-12)
                self.contrast_limits = low, high
                with QtCore.QSignalBlocker(self.low), QtCore.QSignalBlocker(self.high):
                    self.low.setValue(low)
                    self.high.setValue(high)
            else:
                low, high = self.low.value(), self.high.value()
                if low >= high:
                    ax.set_title("Contrast low must be less than high")
                    self.canvas.draw_idle()
                    return
            image.set_clim(low, high)
        if tab == 4:
            image.set_clim(0, 1)
        if tab == 7:
            limit = max(float(np.abs(array).max()), 1e-12)
            image.set_clim(-limit, limit)
        ax.set_xlabel("Object frequency x (cycles/mm)" if tab in (4, 5) else "Registered object x (mm)")
        ax.set_ylabel("Object frequency y (cycles/mm)" if tab in (4, 5) else "Registered object y (mm)")
        title = self.tabs.tabText(tab)
        if tab == 3:
            title += " (µm)"
        if tab == 5:
            title += " · log(1 + mean power)"
        if d.revision != self.revision or not getattr(self, "valid", False):
            title += " · STALE"
        ax.set_title(title)
        self.figure.colorbar(image, ax=ax, shrink=.8)
        self.canvas.draw_idle()

    def _resample(self, field):
        """Advance one independent realization seed and schedule an updated capture.

        Args:
            field: Config seed field: light_seed, surface_seed, or sensor_seed.
        """
        self.config = replace(self.config, **{field: (getattr(self.config, field)+1) % (2**63)})
        self._edited(field)

    def _pin(self):
        """Retain the displayed snapshot as a reference for later comparisons."""
        if self.displayed:
            self.reference = self.displayed
            self._render()

    def _choose_image(self):
        """Choose an image and replace reflectance while retaining surface/light
        realizations.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose reflectance image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)")
        if path:
            try:
                new = Realizations.from_image(path, self.config)
                self.assets = replace(self.assets, image=new.image, image_path=new.image_path)
                self._populate()
                self._edited()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Image could not be loaded", str(exc))

    def _display_settings(self):
        """Return serializable contrast and view settings for a preset."""
        return dict(auto_contrast=self.auto_contrast.isChecked(), low=self.low.value(), high=self.high.value(), tab=self.tabs.currentIndex())

    def _save(self):
        """Choose a file and save current controls, realizations, and display settings."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save current controls and realizations", "playground.npz", "Playground preset (*.npz)")
        if path:
            try:
                config = self._collect()
                save_preset(path, config, self.assets, self._display_settings())
            except (OSError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(self, "Preset could not be saved", str(exc))

    def _restore_display(self, display):
        """Restore contrast controls and the selected diagnostic tab from preset settings.

        Args:
            display: Optional mapping of contrast limits, automatic contrast, and
                selected tab.
        """
        self.auto_contrast.setChecked(bool(display.get("auto_contrast", True)))
        self.low.setValue(float(display.get("low", 0)))
        self.high.setValue(float(display.get("high", 1)))
        self.tabs.setCurrentIndex(int(display.get("tab", 0)))

    def _load(self):
        """Choose a preset and restore its self-contained controls, assets, and display."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load playground preset", "", "Playground preset (*.npz)")
        if path:
            try:
                config, assets, display = load_preset(path)
                self.config, self.assets = config, assets
                self._populate()
                self._restore_display(display)
                self._edited()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                QtWidgets.QMessageBox.warning(self, "Preset could not be loaded", str(exc))

    def _export(self):
        """Choose a new folder and export the exact displayed capture snapshot."""
        if self.displayed is None:
            QtWidgets.QMessageBox.information(self, "No capture yet", "Wait for the first valid capture.")
            return
        parent = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose parent directory for a new capture folder")
        if parent:
            name, ok = QtWidgets.QInputDialog.getText(self, "Capture folder", "New folder name:", text="playground_capture")
            if ok and name:
                try:
                    if Path(name).name != name or name in (".", ".."):
                        raise ValueError("Enter a folder name without path separators")
                    path = save_capture(Path(parent)/name, self.displayed, self._display_settings())
                    self.status.setText(f"Exported displayed {self.displayed.quality} capture to {path}")
                except (OSError, ValueError) as exc:
                    QtWidgets.QMessageBox.warning(self, "Capture could not be exported", str(exc))

    def closeEvent(self, event):
        """Cancel outstanding work and defer closing until the worker exits safely.

        Args:
            event: Qt close event to ignore while a batch finishes, or accept after
                worker shutdown.
        """
        # Keep a live QThread alive; finish closing after its finished signal.
        if self.worker.isRunning() and not self.worker_done:
            self.closing = True
            self.preview_timer.stop()
            self.refine_timer.stop()
            self.mailbox.close()
            self.info.close()
            self.status.setText("Closing after the current simulation batch finishes…")
            event.ignore()
        else:
            event.accept()

    @QtCore.Slot()
    def _worker_finished(self):
        """Complete a deferred window close after the simulation worker has stopped."""
        self.worker_done = True
        if self.closing:
            self.close()


def launch_playground(config, assets, display=None):
    """Show the playground, run the Qt event loop, and join the worker on exit.

    Returns the Qt application exit code.

    Args:
        config: PlaygroundConfig containing physical settings, execution options, and
            seeds.
        assets: Realizations holding the original image and shared surface/light draws.
        display: Optional mapping of contrast limits, automatic contrast, and selected
            tab.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = PlaygroundWindow(config, assets, display)
    window.show()
    try:
        return app.exec()
    finally:
        window.mailbox.close()
        window.worker.wait()
