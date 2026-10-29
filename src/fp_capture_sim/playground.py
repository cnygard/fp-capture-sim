"""GUI-independent playground configuration, stable realizations, and replay.

Lengths at this boundary are SI. A surface is a fixed periodic physical lattice;
interpolation onto a different grid can change realized RMS, but never its phase
or physical correlation scale. Illumination uses common uniform variates.
"""
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch

from .backend import (
    as_numpy,
    make_generator,
    positive,
    positive_count,
    resolve_device,
    resolve_dtype,
)
from .models import OpticalSetup, CircularEmitter, RoughSample, SensorNoise, Spectrum
from .models.object import _resample_object_image
from .specifications import system_specifications
from .simulation import (
    CaptureSimulator,
    CaptureResult,
    IlluminationSamples,
    capture_directory,
)


@dataclass(frozen=True)
class PlaygroundConfig:
    """Immutable playground controls in SI units; validate before capture.

    Args:
        setup: Lens geometry, sensor dimensions, and simulation oversampling.
        emitter: Source disk geometry and plane/spherical wavefront choice.
        sensor: Exposure and sensor noise settings.
        wavelength_m: Center illumination wavelength in meters.
        bandwidth_m: Full uniform spectral bandwidth in meters; zero is monochromatic.
        object_width_m: Physical image width in meters; height follows image aspect
            ratio.
        height_rms_m: Requested surface height RMS in meters.
        camera_xy_m: Camera pupil center (x, y) in meters.
        preview_samples: Positive mode count for interactive previews.
        final_samples: Refined mode count, at least preview_samples.
        batch_size: Positive maximum number of modes in one FFT batch.
        device: PyTorch execution device or "auto".
        dtype: "float32" or "float64" real precision.
        light_seed: Seed for shared uniform wavelength/source draws.
        surface_seed: Seed for the fixed periodic Gaussian surface lattice.
        sensor_seed: Independent CPU sensor-read seed.

    Seeds must be integers in [0, 2**63).
    """
    setup: OpticalSetup = field(default_factory=OpticalSetup)
    emitter: CircularEmitter = field(default_factory=CircularEmitter)
    sensor: SensorNoise = field(default_factory=SensorNoise)
    wavelength_m: float = 532e-9
    bandwidth_m: float = 20e-9
    object_width_m: float = 0.01
    height_rms_m: float = 0.5e-6
    camera_xy_m: tuple[float, float] = (0., 0.)
    preview_samples: int = 8
    final_samples: int = 64
    batch_size: int = 16
    device: str = "auto"
    dtype: str = "float32"
    light_seed: int = 0
    surface_seed: int = 0
    sensor_seed: int = 0

    @property
    def spectrum(self):
        """Return the uniform band or monochromatic spectrum defined by the controls."""
        positive("bandwidth_m", self.bandwidth_m, allow_zero=True)
        return (Spectrum.uniform(self.wavelength_m - self.bandwidth_m / 2,
                                 self.wavelength_m + self.bandwidth_m / 2)
                if self.bandwidth_m else Spectrum.monochromatic(self.wavelength_m))

    def validate(self):
        """Validate control values and execution support, returning self.

        Fourier-grid sampling validity is checked later against actual capture modes.
        """
        self.spectrum
        positive("object_width_m", self.object_width_m)
        positive("height_rms_m", self.height_rms_m, allow_zero=True)
        for name in ("preview_samples", "final_samples", "batch_size"):
            positive_count(getattr(self, name), name)
        if self.preview_samples > self.final_samples:
            raise ValueError("preview_samples must not exceed final_samples")
        if len(self.camera_xy_m) != 2 or not all(map(math.isfinite, self.camera_xy_m)):
            raise ValueError("camera_xy_m must contain two finite coordinates")
        for name in ("light_seed", "surface_seed", "sensor_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 <= value < 2**63:
                raise ValueError(f"{name} must be an integer in [0, 2**63)")
        resolve_device(self.device, resolve_dtype(self.dtype))
        return self

    @classmethod
    def from_dict(cls, values):
        """Restore nested model dataclasses and coordinate tuples from serialized values.

        Args:
            values: Mapping in the asdict(PlaygroundConfig) schema, such as decoded
                preset JSON.
        """
        values = dict(values)
        setup = dict(values.pop("setup"))
        setup["sensor_shape"] = tuple(setup["sensor_shape"])
        emitter = dict(values.pop("emitter"))
        emitter["center_xy_m"] = tuple(emitter["center_xy_m"])
        values["camera_xy_m"] = tuple(values["camera_xy_m"])
        return cls(setup=OpticalSetup(**setup), emitter=CircularEmitter(**emitter),
                   sensor=SensorNoise(**values.pop("sensor")), **values).validate()


def geometry_specs(config, image_shape):
    """Return immediate geometry and requested object/source specifications.

    This performs no grid allocation, device work, or acquisition.

    Args:
        config: PlaygroundConfig containing physical settings, execution options, and
            seeds.
        image_shape: Original image dimensions as (height, width) in pixels.
    """
    return dict(system_specifications(config.setup, config.emitter),
                object_size_m=(config.object_width_m, config.object_width_m*image_shape[0]/image_shape[1]),
                requested_height_rms_m=config.height_rms_m,
                wavelength_range_m=(config.wavelength_m-config.bandwidth_m/2,
                                    config.wavelength_m+config.bandwidth_m/2))


@dataclass(frozen=True)
class Realizations:
    """Cached image and shared random assets for repeatable control comparisons.

    Args:
        image: Original 2-D reflectance intensity array before physical resampling.
        image_path: Image provenance path; replay uses embedded pixels.
        surface: Fixed periodic 2-D unit-RMS height lattice.
        surface_pitch_m: Fixed physical lattice spacing in meters.
        light_uniform: (N, 3) uniform variates for wavelength, disk radius, and angle.
        light_seed: Seed that generated the cached illumination variates.
        surface_seed: Seed that generated the cached surface lattice.
    """
    image: np.ndarray
    image_path: str
    surface: np.ndarray
    surface_pitch_m: float
    light_uniform: np.ndarray
    light_seed: int
    surface_seed: int

    @classmethod
    def from_image(cls, path, config):
        """Decode and cache grayscale reflectance, then initialize stable random assets.

        Args:
            path: Image file decoded once as grayscale reflectance intensity.
            config: PlaygroundConfig containing physical settings, execution options,
                and seeds.
        """
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read sample image: {path}")
        return cls.create(image.astype(np.float32)/255, str(Path(path).resolve()), config)

    @classmethod
    def create(cls, image, path, config):
        """Initialize a fixed physical surface lattice and row-wise light variates.

        Args:
            image: Original 2-D nonnegative NumPy reflectance intensity array.
            path: Image provenance path retained in presets and capture metadata.
            config: PlaygroundConfig containing physical settings, execution options,
                and seeds.
        """
        surface = np.random.default_rng(config.surface_seed).standard_normal((512, 512))
        surface /= np.sqrt(np.mean(surface**2))
        return cls(np.asarray(image, dtype=np.float32), path, surface.astype(np.float32),
                   config.setup.object_pixel_pitch_m,
                   np.random.default_rng(config.light_seed).random((config.final_samples, 3)),
                   config.light_seed, config.surface_seed)

    def updated(self, config):
        """Return assets with changed seeds applied and enough shared light variates.

        Unchanged seeds preserve existing light prefixes and the physical surface pitch.

        Args:
            config: PlaygroundConfig containing physical settings, execution options,
                and seeds.
        """
        result = self
        if config.surface_seed != self.surface_seed:
            surface = np.random.default_rng(config.surface_seed).standard_normal(self.surface.shape)
            surface /= np.sqrt(np.mean(surface**2))
            result = replace(result, surface=surface.astype(np.float32), surface_seed=config.surface_seed)
        if config.light_seed != self.light_seed or len(self.light_uniform) < config.final_samples:
            # Row-wise generation makes every existing prefix independent of N.
            draws = np.random.default_rng(config.light_seed).random((config.final_samples, 3))
            if config.light_seed == self.light_seed:
                draws[:len(self.light_uniform)] = self.light_uniform
            result = replace(result, light_uniform=draws, light_seed=config.light_seed)
        return result

    def sample(self, config):
        """Place reflectance and interpolate the fixed physical surface onto the current
        grid.

        Periodic bilinear surface interpolation preserves physical coordinates;
        changing the grid or FOV can change the realized height RMS.

        Args:
            config: PlaygroundConfig containing physical settings, execution options,
                and seeds.
        """
        setup = config.setup
        image = _resample_object_image(self.image, setup.object_shape,
                                      setup.object_pixel_pitch_m/(config.object_width_m/self.image.shape[1]))
        ny, nx = setup.object_shape
        # Physical coordinates are independent of object size and the current FOV.
        x = (np.arange(nx, dtype=np.float64)-nx//2)*setup.object_pixel_pitch_m
        y = (np.arange(ny, dtype=np.float64)-ny//2)*setup.object_pixel_pitch_m
        map_x, map_y = np.meshgrid(np.remainder(x/self.surface_pitch_m, self.surface.shape[1]),
                                  np.remainder(y/self.surface_pitch_m, self.surface.shape[0]))
        height = cv2.remap(self.surface, map_x.astype(np.float32), map_y.astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)*config.height_rms_m
        dtype = resolve_dtype(config.dtype)
        return RoughSample(torch.as_tensor(image, dtype=dtype).sqrt(),
                           torch.as_tensor(height, dtype=dtype), self.image_path,
                           tuple(self.image.shape), config.object_width_m/self.image.shape[1])

    def illumination(self, config, count):
        """Map the shared uniform prefix to wavelengths and uniform-disk source points.

        Args:
            config: PlaygroundConfig containing physical settings, execution options,
                and seeds.
            count: Number of cached variates to use; assets must contain at least this
                many rows.
        """
        # Reuse rows across edits and preview/refinement for common random numbers.
        u = self.light_uniform[:count]
        wavelengths = config.wavelength_m + config.bandwidth_m*(u[:, 0]-0.5)
        radius = config.emitter.radius_m*np.sqrt(u[:, 1])
        angle = 2*np.pi*u[:, 2]
        xy = np.column_stack((radius*np.cos(angle), radius*np.sin(angle))) + config.emitter.center_xy_m
        dtype = resolve_dtype(config.dtype)
        return IlluminationSamples(torch.as_tensor(wavelengths, dtype=dtype), torch.as_tensor(xy, dtype=dtype))


@dataclass(frozen=True)
class PlaygroundCapture:
    """Snapshot tying a displayed image to its exact settings and realizations.

    Args:
        config: Configuration that produced this capture.
        realizations: Image, surface lattice, and shared light variates used.
        sample: Reflectance amplitude and realized surface on the simulation grid.
        result: CPU measurement and optical diagnostics.
        quality: "preview" or "refined" acquisition level.
        elapsed_s: Wall-clock capture time in seconds.
        device: Actual PyTorch optical execution device name.
        revision: Mailbox generation that originated this capture.
    """
    config: PlaygroundConfig
    realizations: Realizations
    sample: RoughSample
    result: CaptureResult
    quality: str
    elapsed_s: float
    device: str
    revision: int = 0

    @property
    def grid_key(self):
        """Return sensor shape, measurement pitch, and oversampling for image comparisons."""
        s = self.config.setup
        return s.sensor_shape, s.sensor_pixel_pitch_m/s.magnification, s.oversampling


class CaptureCancelled(Exception):
    """Cooperative cancellation, checked between FFT batches."""


class PlaygroundEngine:
    """Owned by one worker. Sensor edits reuse the latest optical result."""
    def __init__(self):
        """Initialize empty sample and optical-result caches for a single worker."""
        self.cached = None
        self.sample_key = None
        self.sample = None

    @staticmethod
    def optical_key(config, assets, count):
        """Return the cache key for optics and realization identities, excluding sensor
        edits.

        Args:
            config: PlaygroundConfig containing physical settings, execution options,
                and seeds.
            assets: Realizations holding the original image and shared surface/light
                draws.
            count: Positive number of independent illumination modes.
        """
        return (config.setup, config.emitter, config.wavelength_m, config.bandwidth_m,
                config.object_width_m, config.height_rms_m, config.camera_xy_m,
                config.light_seed, config.surface_seed, config.device, config.dtype,
                config.batch_size, id(assets.image), id(assets.surface), id(assets.light_uniform), count)

    def run(self, config, assets, quality="refined", revision=0, cancelled=lambda: False, progress=None):
        """Compute a capture or reuse cached optics, then apply a reproducible CPU sensor
        read.

        Returns a PlaygroundCapture snapshot. Cancellation is cooperative and raises
        CaptureCancelled before obsolete work can become a displayed result.

        Args:
            config: PlaygroundConfig containing physical settings, execution options,
                and seeds.
            assets: Realizations holding the original image and shared surface/light
                draws.
            quality: Capture level: "preview" uses preview_samples, "refined" uses
                final_samples.
            revision: Mailbox generation used to reject work superseded by newer edits.
            cancelled: Zero-argument predicate that returns True when this request is
                obsolete.
            progress: Optional callback receiving (completed_modes, total_modes) after
                each FFT batch.
        """
        start = time.monotonic()
        config.validate()
        assets = assets.updated(config)
        count = config.preview_samples if quality == "preview" else config.final_samples
        key = self.optical_key(config, assets, count)
        if cancelled():
            raise CaptureCancelled()
        # Sensor settings are excluded from the optical key, so read/exposure edits
        # reuse the mean and Monte Carlo diagnostics without another optical capture.
        if self.cached is not None and self.cached[0] == key:
            prior = self.cached[1]
            result, sample, device = prior.result, prior.sample, prior.device
        else:
            sample_key = (config.setup, config.object_width_m, config.height_rms_m,
                          config.dtype, id(assets.image), id(assets.surface))
            if sample_key != self.sample_key:
                sample = assets.sample(config)
                self.sample_key, self.sample = sample_key, sample
            else:
                sample = self.sample
            if cancelled():
                raise CaptureCancelled()
            sim = CaptureSimulator(config.setup, sample, config.spectrum, config.emitter,
                                   SensorNoise(), device=config.device, dtype=resolve_dtype(config.dtype),
                                   batch_size=config.batch_size, seed=0)
            def check():
                """Raise CaptureCancelled when the owning request has become obsolete."""
                if cancelled():
                    raise CaptureCancelled()
            result = sim.capture(config.camera_xy_m, samples=assets.illumination(config, count),
                                 check_cancel=check, progress=progress).cpu()
            device = str(sim.device)
        if cancelled():
            raise CaptureCancelled()
        # A stable, separate sensor stream also gives common Gaussian read noise.
        # Apply on CPU so Poisson/read-noise replay works across chosen devices.
        image = config.sensor.apply(result.mean_intensity, make_generator(config.sensor_seed))
        capture = PlaygroundCapture(config, assets, sample, replace(result, image=image), quality,
                                    time.monotonic()-start, device, revision)
        self.cached = key, capture
        return capture


class LatestRequest:
    """Thread-safe one-item mailbox; revision guards also reject late results."""
    def __init__(self):
        """Create an open, empty mailbox with its condition lock and revision counter."""
        self.condition = threading.Condition()
        self.revision = 0
        self.pending = None
        self.closed = False

    def invalidate(self):
        """Advance and return the revision, discarding pending requests and waking waiters."""
        with self.condition:
            self.revision += 1
            self.pending = None
            self.condition.notify_all()
            return self.revision

    def submit(self, request):
        """Replace pending work only if the mailbox is open and the revision matches.

        Args:
            request: Tuple (revision, config, assets, quality) belonging to the current
                generation.
        """
        with self.condition:
            if not self.closed and request[0] == self.revision:
                self.pending = request
                self.condition.notify_all()

    def take(self):
        """Block until work arrives or closure; consume the pending request or return None."""
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.pending is not None)
            if self.closed:
                return None
            request, self.pending = self.pending, None
            return request

    def accepts(self, revision):
        """Return whether a revision is current and the mailbox remains open.

        Args:
            revision: Mailbox generation used to reject work superseded by newer edits.
        """
        with self.condition:
            return not self.closed and revision == self.revision

    def close(self):
        """Close the mailbox, discard pending work, and wake the waiting worker."""
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()


def save_preset(path, config, assets, display=None):
    """Write a self-contained NPZ preset with image pixels and exact random assets.

    Args:
        path: Filesystem path to the input or output file.
        config: PlaygroundConfig containing physical settings, execution options, and
            seeds.
        assets: Realizations holding the original image and shared surface/light draws.
        display: Optional mapping of contrast limits, automatic contrast, and selected
            tab.
    """
    assets = assets.updated(config)
    metadata = dict(schema_version=1, config=asdict(config), image_path=assets.image_path,
                    surface_pitch_m=assets.surface_pitch_m, display=display or {})
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata), image=assets.image,
                            surface=assets.surface, light_uniform=assets.light_uniform)


def load_preset(path):
    """Validate and load a preset, returning (config, realizations, display settings).

    Args:
        path: Self-contained NPZ preset path; loading never requires the original image.
    """
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta["schema_version"] != 1:
            raise ValueError("Unsupported playground preset version")
        # Validate metadata before trusting the embedded realization arrays.
        config = PlaygroundConfig.from_dict(meta["config"])
        image, surface, light = (data[k].copy() for k in ("image", "surface", "light_uniform"))
        if (image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all()
                or (image < 0).any() or surface.ndim != 2 or min(surface.shape) < 2
                or not np.isfinite(surface).all() or light.ndim != 2 or light.shape[1] != 3
                or len(light) < config.final_samples or not np.isfinite(light).all()
                or ((light < 0) | (light >= 1)).any()):
            raise ValueError("Invalid realization arrays in preset")
        positive("surface_pitch_m", meta["surface_pitch_m"])
        assets = Realizations(image, meta["image_path"], surface, meta["surface_pitch_m"], light,
                              config.light_seed, config.surface_seed)
        return config, assets, meta.get("display", {})


def save_capture(directory, capture, display=None):
    """Export the displayed capture without another optical acquisition or sensor draw.

    Returns the new directory Path. Metadata is marked complete only after every
    result, preset, and reconstruction handoff has been written.

    Args:
        directory: New output directory; a bare name resolves under project captures/.
        capture: PlaygroundCapture snapshot whose matching settings and arrays will be
            saved.
        display: Optional mapping of contrast limits, automatic contrast, and selected
            tab.
    """
    directory = capture_directory(directory)
    directory.mkdir(parents=True, exist_ok=False)
    c, r = capture.config, capture.result
    metadata = dict(schema_version=1, complete=False, config=asdict(c), quality=capture.quality,
                    samples=len(r.samples.wavelengths_m), elapsed_s=capture.elapsed_s,
                    device=capture.device, torch_version=str(torch.__version__),
                    image_units="linear arbitrary units; no per-frame normalization")
    target = directory/"config.json"
    # A failed export retains complete=False until all output files exist.
    target.write_text(json.dumps(metadata, indent=2)+"\n")
    np.save(directory/"0000.npy", as_numpy(r.image))
    np.savez(directory/"sample.npz", amplitude=as_numpy(capture.sample.amplitude),
             height_m=as_numpy(capture.sample.height_m))
    np.savez(directory/"0000_optical.npz", mean_intensity=as_numpy(r.mean_intensity),
             standard_error=as_numpy(r.standard_error), mean_pupil=as_numpy(r.mean_pupil),
             mean_spectrum_power=as_numpy(r.mean_spectrum_power),
             wavelengths_m=as_numpy(r.samples.wavelengths_m), source_xy_m=as_numpy(r.samples.source_xy_m))
    save_preset(directory/"playground.npz", c, capture.realizations, display)
    sim = CaptureSimulator(c.setup, capture.sample, c.spectrum, c.emitter, c.sensor,
                           device="cpu", dtype=resolve_dtype(c.dtype), seed=0)
    positions = np.array([c.camera_xy_m])
    (directory/"config.toml").write_text(sim._reconstruction_config(positions))
    metadata["complete"] = True
    target.write_text(json.dumps(metadata, indent=2)+"\n")
    return directory
