"""Batched PyTorch Monte Carlo capture generation on CPU, CUDA, or MPS."""

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .backend import (
    as_numpy,
    make_generator,
    positive_count,
    real_tensor,
    resolve_device,
    resolve_dtype,
)
from .models import CircularEmitter, OpticalSetup, RoughSample, SensorNoise, Spectrum

CAPTURES_DIRECTORY = Path(__file__).resolve().parents[2] / "captures"


def capture_directory(directory):
    """Resolve a dataset name under captures/, preserving explicit paths.

    Use a string to retain an explicit "./" prefix, which Path would remove.

    Args:
        directory: New output directory; a bare name resolves under project captures/.
    """
    raw = str(directory)
    path = Path(directory).expanduser()
    if not raw or raw in (".", ".."):
        raise ValueError("directory must name a new capture directory")
    if "/" not in raw and not path.is_absolute():
        return CAPTURES_DIRECTORY / path
    return path


@dataclass(frozen=True)
class IlluminationSamples:
    """Paired coherent modes copied into real tensors on a shared device/precision.

    Args:
        wavelengths_m: Positive finite (N,) wavelengths in meters, with N at least one.
        source_xy_m: Finite (N, 2) source positions (x, y) in meters.
    """

    wavelengths_m: torch.Tensor
    source_xy_m: torch.Tensor

    def __post_init__(self):
        """Copy and validate N positive wavelengths paired with N finite source positions."""
        w = real_tensor(self.wavelengths_m, copy=True)
        xy = real_tensor(self.source_xy_m, device=w.device, dtype=w.dtype, copy=True)
        if w.ndim != 1 or not len(w) or xy.shape != (len(w), 2):
            raise ValueError(
                "samples require N wavelengths and an (N, 2) source array, N >= 1"
            )
        if not bool(
            (torch.isfinite(w) & (w > 0) & torch.isfinite(xy).all(dim=-1)).all()
        ):
            raise ValueError(
                "sample wavelengths must be positive and all coordinates finite"
            )
        object.__setattr__(self, "wavelengths_m", w)
        object.__setattr__(self, "source_xy_m", xy)

    def to(self, device, dtype=None):
        """Return samples on a device/precision, reusing self if both already match.

        Args:
            device: Target PyTorch device; None preserves the input device where
                supported.
            dtype: Real tensor precision (float32 or float64); None preserves input
                precision.
        """
        dtype = self.wavelengths_m.dtype if dtype is None else resolve_dtype(dtype)
        if (
            self.wavelengths_m.device == torch.device(device)
            and self.wavelengths_m.dtype == dtype
        ):
            return self
        return IlluminationSamples(
            self.wavelengths_m.to(device=device, dtype=dtype),
            self.source_xy_m.to(device=device, dtype=dtype),
        )


@dataclass
class CaptureResult:
    """Measurement and diagnostic tensors retained on the simulation device.

    Args:
        image: Sensor image with shape sensor_shape, after exposure and noise.
        mean_intensity: Noiseless optical mean with shape sensor_shape.
        standard_error: Optical Monte Carlo error per sensor pixel; NaN for one mode.
        mean_pupil: Mean Fourier-grid pupil transmission with shape object_shape.
        mean_spectrum_power: Mean pre-pupil spectral power with shape object_shape.
        camera_xy_m: Camera pupil center (x, y) in meters.
        samples: Exact paired wavelengths and source positions used for the capture.
    """

    image: torch.Tensor
    mean_intensity: torch.Tensor
    standard_error: torch.Tensor
    mean_pupil: torch.Tensor
    mean_spectrum_power: torch.Tensor
    camera_xy_m: tuple[float, float]
    samples: IlluminationSamples

    def cpu(self):
        """Return a result with CPU tensors for plotting or serialization.

        Tensors already on CPU may share storage with the original result.
        """
        return replace(
            self,
            **{
                name: getattr(self, name).cpu()
                for name in (
                    "image",
                    "mean_intensity",
                    "standard_error",
                    "mean_pupil",
                    "mean_spectrum_power",
                )
            },
            samples=self.samples.to("cpu"),
        )


class CaptureSimulator:
    """Batched incoherent Monte Carlo imaging with independent optical and sensor RNGs."""

    def __init__(
        self,
        setup: OpticalSetup,
        sample: RoughSample,
        spectrum: Spectrum | None = None,
        emitter: CircularEmitter | None = None,
        sensor: SensorNoise | None = None,
        seed=None,
        *,
        device="auto",
        dtype=torch.float32,
        batch_size=16,
    ):
        """Initialize the physical models, cached grids, and independent random streams.

        Args:
            setup: OpticalSetup defining lens geometry and the object/sensor grids.
            sample: RoughSample matching setup.object_shape; copied to the simulation
                device.
            spectrum: Optional spectral power distribution; defaults to a 532 nm line.
            emitter: Optional CircularEmitter; defaults to an on-axis point source.
            sensor: Optional SensorNoise; defaults to an ideal sensor.
            seed: Local random seed; None selects a fresh seed without changing global
                RNG state.
            device: Execution device or "auto" for CUDA/MPS/CPU selection.
            dtype: Real float32/float64 precision for grids, sampling, and accumulation.
            batch_size: Positive maximum number of coherent modes evaluated in one FFT
                batch.
        """
        if sample.amplitude.shape != setup.object_shape:
            raise ValueError(
                f"sample shape must be setup.object_shape = {setup.object_shape}"
            )
        positive_count(batch_size, "batch_size")
        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device, self.dtype)
        self.batch_size = batch_size
        self.setup, self.sample = setup, sample.to(self.device, self.dtype)
        self.spectrum = spectrum if spectrum is not None else Spectrum.monochromatic()
        self.emitter = emitter if emitter is not None else CircularEmitter(radius_m=0)
        self.sensor = sensor if sensor is not None else SensorNoise()
        if seed is not None and (not isinstance(seed, int) or not 0 <= seed < 2**64):
            raise ValueError("seed must be an integer in [0, 2**64)")
        # Derive independent device-local streams without touching global RNG state.
        master = make_generator(seed)
        self.seed = master.initial_seed()
        optical_seed, noise_seed = torch.randint(
            0, 2**63 - 1, (2,), generator=master
        ).tolist()
        self._optical_rng = make_generator(optical_seed, self.device)
        self._noise_rng = make_generator(noise_seed, self.device)
        self.x, self.y = setup.object_grid(device=self.device, dtype=self.dtype)
        self.fx, self.fy = setup.frequency_grid(device=self.device, dtype=self.dtype)
        # Only these scalar limits are cached on the host; grids stay on device.
        self._frequency_limits = (
            self.fx.min().item(),
            self.fx.max().item(),
            self.fy.min().item(),
            self.fy.max().item(),
        )

    def draw_samples(self, count=64):
        """Draw paired independent wavelengths and source points on the simulation device.

        Args:
            count: Positive number of independent illumination modes.
        """
        positive_count(count)
        kwargs = dict(device=self.device, dtype=self.dtype)
        return IlluminationSamples(
            self.spectrum.sample(count, self._optical_rng, **kwargs),
            self.emitter.sample(count, self._optical_rng, **kwargs),
        )

    def _camera_position(self, camera_xy_m):
        """Validate and return a device tensor for one camera position.

        Args:
            camera_xy_m: Camera pupil center (x, y), in meters relative to the optical
                axis.
        """
        camera = real_tensor(camera_xy_m, device=self.device, dtype=self.dtype)
        if camera.shape != (2,) or not bool(torch.isfinite(camera).all()):
            raise ValueError("camera position must be a finite (x, y) pair")
        return camera

    def _coherent_batch(self, wavelengths, source_xy, camera):
        """Evaluate modes and return intensity, pupil, and spectral power tensors.

        Intensity has shape (N, *sensor_shape); pupil and power use object_shape.
        Sampling checks reject clipped/empty pupils and aliased spherical illumination.

        Args:
            wavelengths: Positive (N,) wavelength tensor in meters on the simulation
                device.
            source_xy: Matching (N, 2) source coordinates in meters on the simulation
                device.
            camera: Validated (2,) camera position tensor in meters.
        """
        w = wavelengths[:, None, None]
        sx, sy = source_xy[:, 0, None, None], source_xy[:, 1, None, None]
        cx = camera[0] / (w * self.setup.object_distance_m)
        cy = camera[1] / (w * self.setup.object_distance_m)
        radius = self.setup.aperture_diameter_m / (2 * w * self.setup.object_distance_m)
        # A plane-wave tilt translates the pupil in the untilted object spectrum;
        # the omitted output phase ramp has unit magnitude, preserving intensity.
        if self.emitter.wavefront == "plane":
            cx = cx + sx / (w * self.emitter.distance_m)
            cy = cy + sy / (w * self.emitter.distance_m)
        xmin, xmax, ymin, ymax = self._frequency_limits
        valid = (
            (cx - radius >= xmin)
            & (cx + radius <= xmax)
            & (cy - radius >= ymin)
            & (cy + radius <= ymax)
        )
        if not bool(valid.all()):
            raise ValueError(
                "Pupil exceeds the sampled Fourier grid; increase oversampling or reduce scan/source extent"
            )
        pupil = (self.fx - cx).square() + (self.fy - cy).square() <= radius.square()
        if not bool(pupil.flatten(start_dim=1).any(dim=1).all()):
            raise ValueError(
                "Pupil contains no Fourier samples; increase field of view"
            )
        field = self.sample.field(wavelengths)
        if self.emitter.wavefront == "spherical":
            r = (
                self.emitter.distance_m**2
                + (self.x - sx).square()
                + (self.y - sy).square()
            ).sqrt()
            # Local path slope / wavelength is spatial frequency. Enforce Nyquist
            # for each spherical mode before evaluating its phase.
            max_slope = torch.maximum(
                ((self.x - sx).abs() / r).amax(dim=(-2, -1)),
                ((self.y - sy).abs() / r).amax(dim=(-2, -1)),
            )
            if bool(
                (max_slope / wavelengths >= 0.5 / self.setup.object_pixel_pitch_m).any()
            ):
                raise ValueError(
                    "Spherical illumination is undersampled; increase oversampling"
                )
            field = field * self.emitter.field(self.x, self.y, source_xy, wavelengths)

        # Shift only spatial FFT axes, leaving the leading mode axis untouched.
        spectrum = torch.fft.fftshift(torch.fft.fft2(field, norm="ortho"), dim=(-2, -1))
        image_field = torch.fft.ifft2(
            torch.fft.ifftshift(spectrum * pupil, dim=(-2, -1)), norm="ortho"
        )
        return (
            self.setup.integrate_pixels(image_field.abs().square()),
            pupil,
            spectrum.abs().square(),
        )

    @torch.no_grad()
    def coherent_capture(
        self, wavelength_m, source_xy_m=(0.0, 0.0), camera_xy_m=(0.0, 0.0)
    ):
        """Return intensity, pupil, and spectral power for one coherent mode.

        The returned tensors remain on the simulator device; sensor noise is not
        applied.

        Args:
            wavelength_m: One positive scalar wavelength in meters.
            source_xy_m: Single source (x, y) coordinate pair in meters.
            camera_xy_m: Camera pupil center (x, y), in meters relative to the optical
                axis.
        """
        wavelength = real_tensor(wavelength_m, device=self.device, dtype=self.dtype)
        xy = real_tensor(source_xy_m, device=self.device, dtype=self.dtype)
        if wavelength.ndim != 0 or xy.shape != (2,):
            raise ValueError(
                "one mode requires a scalar wavelength and a source (x, y) pair"
            )
        samples = IlluminationSamples(wavelength.reshape(1), xy.reshape(1, 2))
        result = self._coherent_batch(
            samples.wavelengths_m,
            samples.source_xy_m,
            self._camera_position(camera_xy_m),
        )
        return tuple(tensor[0] for tensor in result)

    @torch.no_grad()
    def capture(
        self,
        camera_xy_m=(0.0, 0.0),
        count=64,
        *,
        samples=None,
        check_cancel=None,
        progress=None,
    ):
        """Average incoherent mode intensities, estimate uncertainty, then read the sensor.

        Explicit samples determine the mode count and bypass optical random draws.
        One-mode standard error is NaN. Cancellation exceptions propagate to callers.

        Args:
            camera_xy_m: Camera pupil center (x, y), in meters relative to the optical
                axis.
            count: Positive number of independent illumination modes.
            samples: Optional IlluminationSamples to reuse instead of drawing count new
                modes.
            check_cancel: Optional zero-argument callback that raises to abort between
                FFT batches.
            progress: Optional callback receiving (completed_modes, total_modes) after
                each FFT batch.
        """
        samples = (
            self.draw_samples(count)
            if samples is None
            else samples.to(self.device, self.dtype)
        )
        camera = self._camera_position(camera_xy_m)
        mean = torch.zeros(
            self.setup.sensor_shape, device=self.device, dtype=self.dtype
        )
        m2 = torch.zeros_like(mean)
        mean_pupil = torch.zeros(
            self.setup.object_shape, device=self.device, dtype=self.dtype
        )
        mean_power = torch.zeros_like(mean_pupil)
        n = 0
        for start in range(0, len(samples.wavelengths_m), self.batch_size):
            if check_cancel is not None:
                check_cancel()
            stop = start + self.batch_size
            intensity, pupil, power = self._coherent_batch(
                samples.wavelengths_m[start:stop],
                samples.source_xy_m[start:stop],
                camera,
            )
            batch_n = intensity.shape[0]
            batch_mean = intensity.mean(dim=0)
            batch_m2 = (intensity - batch_mean).square().sum(dim=0)
            delta = batch_mean - mean
            total = n + batch_n
            # Parallel/Chan variance merge: stable without retaining every mode.
            m2.add_(batch_m2 + delta.square() * (n * batch_n / total))
            mean.add_(delta * (batch_n / total))
            mean_pupil.add_(
                (pupil.to(self.dtype).mean(dim=0) - mean_pupil) * (batch_n / total)
            )
            mean_power.add_((power.mean(dim=0) - mean_power) * (batch_n / total))
            n = total
            if progress is not None:
                progress(n, len(samples.wavelengths_m))
        if check_cancel is not None:
            check_cancel()
        # m2/(n-1) estimates mode variance; dividing by n gives variance of the mean.
        error = (
            (m2.clamp_min(0) / (n * (n - 1))).sqrt()
            if n > 1
            else torch.full_like(mean, math.nan)
        )
        # A sensor read is performed only after all batches have been accumulated.
        image = self.sensor.apply(mean, self._noise_rng)
        return CaptureResult(
            image,
            mean,
            error,
            mean_pupil,
            mean_power,
            tuple(camera.detach().cpu().tolist()),
            samples,
        )

    def scan(self, positions_m, count=64, *, reuse_samples=True):
        """Yield captures in acquisition order, with independent sensor reads.

        Args:
            positions_m: Iterable of camera (x, y) positions in meters, in acquisition
                order.
            count: Positive number of independent illumination modes.
            reuse_samples: Whether every scan position shares one set of optical draws.
        """
        samples = self.draw_samples(count) if reuse_samples else None
        for xy in positions_m:
            yield self.capture(xy, count, samples=samples)

    def _reconstruction_config(self, positions, scan_step_m=None):
        """Return acquisition-derived fp TOML settings in reconstruction units.

        Translation is supplied only for a complete centered odd square spiral.
        An explicitly supplied spacing that does not match the positions is rejected.

        Args:
            positions: Finite CPU (N, 2) NumPy camera coordinates in meters, in file
                order.
            scan_step_m: Optional spiral spacing in meters, needed for a single centered
                frame.
        """
        if scan_step_m is not None and (
            not math.isfinite(scan_step_m) or scan_step_m <= 0
        ):
            raise ValueError("scan_step_m must be finite and positive")
        spectrum = self.spectrum
        if spectrum.continuous:
            wavelength = sum(spectrum.wavelengths_m) / 2
        else:
            weights = spectrum.weights or (1.0,) * len(spectrum.wavelengths_m)
            wavelength = sum(
                w * (p / sum(weights)) for w, p in zip(spectrum.wavelengths_m, weights)
            )
        settings = dict(
            source="array_sim",
            file_extension=".npy",
            frame_pattern="{index:04d}",
            samples_count=len(positions),
            resolution=min(self.setup.sensor_shape),
            wavelength=wavelength * 1e9,
            pixel_size_um=self.setup.sensor_pixel_pitch_m
            / self.setup.magnification
            * 1e6,
            aperture_diameter=self.setup.aperture_diameter_m * 1e3,
            object_distance=self.setup.object_distance_m,
        )
        # fp reconstructs a centered square spiral, not arbitrary saved positions.
        # A single center frame cannot determine spacing without an explicit step.
        step = scan_step_m
        if step is None and len(positions) > 1:
            step = float(np.linalg.norm(positions[1] - positions[0]))
        size = math.isqrt(len(positions))
        if (
            step is not None
            and step > 0
            and size * size == len(positions)
            and size % 2 == 1
            and np.allclose(
                positions, spiral_positions(size, step), rtol=1e-6, atol=step * 1e-7
            )
        ):
            settings["translation"] = step * 1e3
        elif scan_step_m is not None:
            raise ValueError(
                "scan_step_m must match a centered square spiral in file order"
            )
        lines = [
            "# wavelength is the spectral power mean in nm (a monochromatic approximation).",
            "# pixel_size_um is measurement spacing in the object plane, before cropping/resizing.",
        ]
        if "translation" not in settings:
            lines.append(
                "# translation omitted: spacing is unknown or positions are not an fp square spiral."
            )
        return (
            "\n".join(
                lines
                + [f"{key} = {json.dumps(value)}" for key, value in settings.items()]
            )
            + "\n"
        )

    def save_scan(
        self, directory, positions_m, count=64, *, reuse_samples=True, scan_step_m=None
    ):
        """Export raw frames, sample tensors, optical draws, JSON metadata, and fp TOML.

        The destination must be new. JSON remains marked incomplete until every frame
        and the reconstruction handoff have been saved. Returns the output Path.

        Args:
            directory: New output directory; a bare name resolves under project
                captures/.
            positions_m: Iterable of camera (x, y) positions in meters, in acquisition
                order.
            count: Positive number of independent illumination modes.
            reuse_samples: Whether every scan position shares one set of optical draws.
            scan_step_m: Optional spiral spacing in meters, needed for a single centered
                frame.
        """
        if isinstance(positions_m, torch.Tensor):
            positions = as_numpy(positions_m)
        else:
            positions = np.asarray(list(positions_m), dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 2
            or not len(positions)
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("positions_m must be a nonempty finite (N, 2) array")
        positive_count(count)
        reconstruction_config = self._reconstruction_config(positions, scan_step_m)
        directory = capture_directory(directory)
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(
            schema_version=2,
            complete=False,
            setup=asdict(self.setup),
            spectrum=asdict(self.spectrum),
            emitter=asdict(self.emitter),
            sensor=asdict(self.sensor),
            seed=self.seed,
            backend="torch",
            torch_version=str(torch.__version__),
            device=str(self.device),
            dtype=str(self.dtype),
            batch_size=self.batch_size,
            monte_carlo_samples=count,
            reuse_samples=reuse_samples,
            camera_positions_m=positions.tolist(),
            image_units="linear arbitrary units; no per-frame normalization",
            coordinate_system="registered object view; axes y,x; positions x,y",
        )
        if self.sample.original_image_shape is not None:
            pitch = self.sample.original_pixel_pitch_m
            metadata["object_image"] = dict(
                path=self.sample.image_path,
                original_shape=list(self.sample.original_image_shape),
                original_pixel_pitch_m=pitch,
                original_size_m=None
                if pitch is None
                else [n * pitch for n in self.sample.original_image_shape],
                mapping="resize_to_field_of_view"
                if pitch is None
                else "physical_scale_centered_zero_background",
            )
        config = directory / "config.json"
        # An incomplete manifest makes interrupted exports recognizable.
        config.write_text(json.dumps(metadata, indent=2) + "\n")
        np.savez(
            directory / "sample.npz",
            amplitude=as_numpy(self.sample.amplitude),
            height_m=as_numpy(self.sample.height_m),
        )
        for index, result in enumerate(
            tqdm(
                self.scan(positions, count, reuse_samples=reuse_samples),
                total=len(positions),
            )
        ):
            np.save(directory / f"{index:04d}.npy", as_numpy(result.image))
            np.savez(
                directory / f"{index:04d}_optical.npz",
                mean_intensity=as_numpy(result.mean_intensity),
                standard_error=as_numpy(result.standard_error),
                wavelengths_m=as_numpy(result.samples.wavelengths_m),
                source_xy_m=as_numpy(result.samples.source_xy_m),
            )
        (directory / "config.toml").write_text(reconstruction_config)
        metadata["complete"] = True
        config.write_text(json.dumps(metadata, indent=2) + "\n")
        return directory


def spiral_positions(size, step_m):
    """Return a centered square spiral as a CPU (size**2, 2) NumPy array.

    Args:
        size: Positive odd number of scan positions along each side.
        step_m: Positive physical spacing between adjacent scan positions in meters.
    """
    positive_count(size, "size")
    if size % 2 == 0:
        raise ValueError("size must be odd")
    if not math.isfinite(step_m) or step_m <= 0:
        raise ValueError("step_m must be finite and positive")
    positions = [(0, 0)]
    x = y = 0
    run = 1
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    direction = 0
    # Use each run length for two right-angle legs before extending the square.
    while len(positions) < size**2:
        for _ in range(2):
            dx, dy = directions[direction % 4]
            for _ in range(run):
                x, y = x + dx, y + dy
                positions.append((x, y))
                if len(positions) == size**2:
                    return np.asarray(positions, dtype=float) * step_m
            direction += 1
        run += 1
    return np.asarray(positions, dtype=float) * step_m
