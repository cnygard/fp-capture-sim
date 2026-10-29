"""Optional Matplotlib views; importing the simulator never opens a window."""
import numpy as np
from .backend import as_numpy, make_generator


def plot_setup(simulator):
    """Return a Matplotlib figure of side-view geometry and sampled source coordinates.

    Plotting uses a dedicated RNG so it does not change subsequent captures.

    Args:
        simulator: CaptureSimulator supplying the physical models and execution
            settings.
    """
    import matplotlib.pyplot as plt
    sim = simulator
    setup, emitter = sim.setup, sim.emitter
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    z = setup.object_distance_m * 1e3
    source_z = emitter.distance_m * 1e3
    source_x = emitter.center_xy_m[0] * 1e3
    radius = setup.aperture_diameter_m * 500
    object_half_width = setup.sensor_shape[1] * setup.sensor_pixel_pitch_m / setup.magnification * 500
    sensor_half_width = setup.sensor_shape[1] * setup.sensor_pixel_pitch_m * 500
    ax.plot([0, 0], [-object_half_width, object_half_width], lw=4, label="Sample (z=0)")
    ax.plot([z, z], [-radius, radius], lw=4, label="Lens")
    sensor_z = z + setup.image_distance_m * 1e3
    ax.plot([sensor_z, sensor_z], [-sensor_half_width, sensor_half_width], lw=4, label="Sensor")
    ax.scatter([source_z], [source_x], label="Emitter center")
    ax.annotate("", xy=(0, 0), xytext=(source_z, source_x),
                arrowprops=dict(arrowstyle="->", color="orange"))
    ax.annotate("", xy=(z, radius), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="blue"))
    ax.annotate("", xy=(sensor_z, 0), xytext=(z, radius),
                arrowprops=dict(arrowstyle="->", color="blue"))
    ax.set(xlabel="Axial distance z (mm)", ylabel="Lateral x (mm)",
           title=f"f/{setup.f_number:.1f}; object NA={setup.object_na:.5f}; |M|={setup.magnification:.4f}")
    ax.legend(fontsize=8)
    # Dedicated RNG: plotting must not change subsequent captures.
    rng = make_generator(0)
    xy = as_numpy(emitter.sample(500, rng))
    wavelengths = as_numpy(sim.spectrum.sample(500, rng))
    points = axes[1].scatter(xy[:, 0]*1e3, xy[:, 1]*1e3, c=wavelengths*1e9, s=8, cmap="viridis")
    axes[1].set(xlabel="Source x (mm)", ylabel="Source y (mm)",
                title=f"Uniform disk emitter: {emitter.wavefront} waves")
    axes[1].set_aspect("equal")
    fig.colorbar(points, ax=axes[1], label="Wavelength (nm)")
    fig.tight_layout()
    return fig


class CaptureViewer:
    """Camera translation sliders with a fixed set of optical Monte Carlo draws."""
    def __init__(self, simulator, count=64):
        """Build camera sliders and diagnostic panels using a fixed optical sample set.

        Args:
            simulator: CaptureSimulator supplying the physical models and execution
                settings.
            count: Positive number of independent illumination modes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
        self.simulator = simulator
        self.samples = simulator.draw_samples(count)
        self.count = count
        self.figure, self.axes = plt.subplots(2, 3, figsize=(13, 8))
        self.figure.subplots_adjust(bottom=0.28, hspace=0.4)
        setup = simulator.setup
        image_extent = np.array([-setup.sensor_shape[1]/2, setup.sensor_shape[1]/2,
                                 -setup.sensor_shape[0]/2, setup.sensor_shape[0]/2])
        image_extent *= setup.sensor_pixel_pitch_m / setup.magnification * 1e3
        fx, fy = as_numpy(simulator.fx), as_numpy(simulator.fy)
        freq_extent = [fx.min()/1e3, fx.max()/1e3, fy.min()/1e3, fy.max()/1e3]
        result = simulator.capture(samples=self.samples)
        display = result.cpu()
        arrays = [as_numpy(simulator.sample.amplitude).copy()**2,
                  as_numpy(simulator.sample.height_m).copy()*1e6,
                  as_numpy(display.mean_pupil), np.log1p(as_numpy(display.mean_spectrum_power)),
                  as_numpy(display.mean_intensity), as_numpy(display.image)]
        titles = ["Object reflectance", "Fixed surface height (µm)",
                  "Mean pupil transmission", "Log mean spectrum power",
                  "Mean optical intensity", "Sensor capture"]
        self.artists = []
        for i, (ax, array, title) in enumerate(zip(self.axes.flat, arrays, titles)):
            extent = freq_extent if i in (2, 3) else image_extent
            self.artists.append(ax.imshow(array, cmap="viridis" if i == 1 else "gray",
                                          extent=extent, origin="lower"))
            ax.set_title(title)
            ax.set_xlabel("Object frequency (cycles/mm)" if i in (2, 3) else "Object x (mm)")
        self.artists[2].set_clim(0, 1)
        low = min(0, display.image.min().item())
        high = max(display.mean_intensity.max().item(), display.image.max().item(), 1e-12)
        self.artists[4].set_clim(low, high)
        self.artists[5].set_clim(low, high)
        # Bound translation by the shortest-wavelength pupil and worst source tilt.
        wavelength = min(simulator.spectrum.wavelengths_m)
        limit = wavelength * setup.object_distance_m * min(fx.max(), fy.max()) - setup.aperture_diameter_m / 2
        if simulator.emitter.wavefront == "plane":
            extent = max(abs(v) for v in simulator.emitter.center_xy_m) + simulator.emitter.radius_m
            limit -= setup.object_distance_m / simulator.emitter.distance_m * extent
        limit = min(2 * setup.aperture_diameter_m, 0.95 * limit)
        if limit <= 0:
            raise ValueError("No camera translation fits the Fourier grid; increase oversampling")
        self.x_slider = Slider(self.figure.add_axes([0.18, 0.10, 0.55, 0.025]),
                               "Camera x (mm)", -limit*1e3, limit*1e3, valinit=0)
        self.y_slider = Slider(self.figure.add_axes([0.18, 0.05, 0.55, 0.025]),
                               "Camera y (mm)", -limit*1e3, limit*1e3, valinit=0)
        self.resample_button = Button(self.figure.add_axes([0.79, 0.07, 0.17, 0.055]), "Resample light")
        self.status = self.figure.text(0.12, 0.16, "")
        self.x_slider.on_changed(self.update)
        self.y_slider.on_changed(self.update)
        self.resample_button.on_clicked(self.resample)
        self.result = result
        self._status(result)

    def _status(self, result):
        """Display the mean optical Monte Carlo standard error for a capture.

        Args:
            result: CaptureResult whose device-local uncertainty is summarized.
        """
        error = result.standard_error.mean().item()
        self.status.set_text(f"{self.count} modes on {self.simulator.device}; mean Monte Carlo SE = {error:.3g} (optical units)")

    def update(self, _=None):
        """Capture at the current slider position and refresh diagnostic image panels.

        Args:
            _: Unused GUI signal or callback arguments.
        """
        try:
            result = self.simulator.capture((self.x_slider.val*1e-3, self.y_slider.val*1e-3), samples=self.samples)
        except ValueError as exc:
            self.status.set_text(str(exc))
            self.figure.canvas.draw_idle()
            return
        self.result = result
        display = result.cpu()
        arrays = [as_numpy(display.mean_pupil), np.log1p(as_numpy(display.mean_spectrum_power)),
                  as_numpy(display.mean_intensity), as_numpy(display.image)]
        for artist, array in zip(self.artists[2:], arrays):
            artist.set_data(array)
        # Match the optical and sensor panels without normalizing saved data.
        low = min(0, display.image.min().item())
        high = max(display.mean_intensity.max().item(), display.image.max().item(), 1e-12)
        self.artists[3].autoscale()
        self.artists[4].set_clim(low, high)
        self.artists[5].set_clim(low, high)
        self._status(result)
        self.figure.canvas.draw_idle()

    def resample(self, _=None):
        """Draw a new optical sample set and refresh the current camera view.

        Args:
            _: Unused GUI signal or callback arguments.
        """
        self.samples = self.simulator.draw_samples(self.count)
        self.update()

    def show(self):
        """Enter Matplotlib display handling for the viewer windows."""
        import matplotlib.pyplot as plt
        plt.show()
