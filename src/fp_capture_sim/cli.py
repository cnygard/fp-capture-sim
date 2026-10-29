"""Command-line construction of a setup, preview, or saved scan."""

import argparse
import math
from pathlib import Path

from . import (
    CaptureSimulator,
    CircularEmitter,
    OpticalSetup,
    RoughSample,
    SensorNoise,
    Spectrum,
    spiral_positions,
)
from .backend import resolve_device, resolve_dtype
from .specifications import system_specifications


def default_image_path():
    """Find the default sample in the working directory or source checkout.

    Return the working-directory candidate if no existing image is found, so the
    image loader can report the missing path.
    """
    name = "usaf_perfect_square.jpg"
    candidates = (
        Path.cwd() / name,
        Path.cwd().parent / name,
        Path(__file__).resolve().parents[3] / name,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def format_setup_summary(sim, image_path, requested_height_rms_m, samples):
    """Return a human-readable acquisition summary including units and realized RMS.

    Args:
        sim: Configured CaptureSimulator whose sample and settings are summarized.
        image_path: Original image path to include in the acquisition summary.
        requested_height_rms_m: Requested surface height RMS in meters, before
            finite-sample variation.
        samples: Number of Monte Carlo modes requested per capture.
    """
    setup, sample = sim.setup, sim.sample
    spectrum, emitter, sensor = sim.spectrum, sim.emitter, sim.sensor
    ny, nx = setup.sensor_shape
    object_ny, object_nx = setup.object_shape
    specs = system_specifications(setup, emitter)
    measurement_pitch = specs["measurement_pitch_m"]
    height_rms = sample.height_m.square().mean().sqrt().item()
    amplitude_min, amplitude_max = (
        sample.amplitude.amin().item(),
        sample.amplitude.amax().item(),
    )
    if spectrum.continuous:
        low, high = (w * 1e9 for w in spectrum.wavelengths_m)
        spectral_description = (
            f"uniform power band {low:.6g}-{high:.6g} nm; "
            f"center {(low + high) / 2:.6g} nm; full bandwidth {high - low:.6g} nm"
        )
    elif len(spectrum.wavelengths_m) == 1:
        spectral_description = (
            f"monochromatic, {spectrum.wavelengths_m[0] * 1e9:.6g} nm"
        )
    else:
        weights = spectrum.weights or (1.0,) * len(spectrum.wavelengths_m)
        total = sum(weights)
        spectral_description = "discrete power lines: " + ", ".join(
            f"{w * 1e9:.6g} nm ({p / total:.3%})"
            for w, p in zip(spectrum.wavelengths_m, weights)
        )
    spatial_description = (
        "single spatially coherent source point"
        if emitter.radius_m == 0
        else "uniform disk of mutually incoherent source points"
    )
    shot_noise = (
        "disabled"
        if sensor.electrons_per_unit is None
        else f"enabled, {sensor.electrons_per_unit:.6g} electrons per intensity unit"
    )
    image_description = []
    if sample.original_image_shape is not None:
        original_ny, original_nx = sample.original_image_shape
        image_description.append(
            f"  Original image: {original_nx} x {original_ny} pixels (width x height)"
        )
        if sample.original_pixel_pitch_m is not None:
            pitch = sample.original_pixel_pitch_m
            image_description.extend(
                [
                    f"  Original object pixel pitch: {pitch * 1e6:.12g} um",
                    f"  Original object size: {original_nx * pitch * 1e3:.10g} x {original_ny * pitch * 1e3:.10g} mm (width x height)",
                    "  Image placement: physical scale preserved; centered, cropped to field of view, zero reflectance outside image",
                ]
            )
        else:
            image_description.append(
                "  Image placement: resized to fill field of view (native object pixel pitch unspecified)"
            )
    return "\n".join(
        [
            "Optical setup (in-focus thin lens in air)",
            f"  Object-to-lens distance: {setup.object_distance_m:.6g} m",
            f"  Lens-to-sensor distance: {setup.image_distance_m * 1e3:.6g} mm",
            f"  Focal length: {setup.focal_length_m * 1e3:.6g} mm",
            f"  Aperture diameter: {setup.aperture_diameter_m * 1e3:.6g} mm; f/{setup.f_number:.6g}",
            f"  Object-side NA: {setup.object_na:.6g}; magnification: {setup.magnification:.6g}",
            f"  Sensor: {nx} x {ny} pixels (width x height); pitch {setup.sensor_pixel_pitch_m * 1e6:.6g} um",
            f"  Sensor area: {nx * setup.sensor_pixel_pitch_m * 1e3:.6g} x {ny * setup.sensor_pixel_pitch_m * 1e3:.6g} mm",
            "Object",
            f"  Image: {Path(image_path).resolve()}",
            *image_description,
            "  Grayscale image interpreted as reflectance intensity; field amplitude is its square root",
            f"  Simulation grid: {object_nx} x {object_ny}; oversampling: {setup.oversampling} per axis",
            f"  Object-plane field of view: {nx * measurement_pitch * 1e3:.6g} x {ny * measurement_pitch * 1e3:.6g} mm (width x height)",
            f"  Object-plane measurement pitch: {measurement_pitch * 1e6:.6g} um",
            f"  Simulation grid pitch: {setup.object_pixel_pitch_m * 1e6:.6g} um; subpixel intensities averaged into sensor pixels",
            f"  Reflectance amplitude range: {amplitude_min:.6g}-{amplitude_max:.6g}",
            f"  Surface height RMS: requested {requested_height_rms_m * 1e6:.6g} um; realized {height_rms * 1e6:.6g} um",
            "  Reflective surface phase: 4*pi*height/wavelength",
            "Light source",
            f"  Spectrum: {spectral_description}",
            f"  Spatial model: {spatial_description}",
            f"  Emitter radius: {emitter.radius_m * 1e3:.6g} mm; distance to object: {emitter.distance_m:.6g} m",
            f"  Emitter center (x, y): ({emitter.center_xy_m[0] * 1e3:.6g}, {emitter.center_xy_m[1] * 1e3:.6g}) mm",
            f"  Angular radius about emitter center: {specs['emitter_angular_radius_rad'] * 1e3:.6g} mrad",
            f"  Wavefront: {emitter.wavefront}; each Monte Carlo mode is individually coherent",
            f"  Monte Carlo modes per capture: {samples}; mode intensities averaged before sensor noise",
            "Sensor and execution",
            f"  Exposure scale: {sensor.exposure_scale:.6g}; read-noise std: {sensor.read_noise_std:.6g} intensity units",
            f"  Shot noise: {shot_noise}",
            f"  PyTorch device: {sim.device}; dtype: {sim.dtype}; batch size: {sim.batch_size}; seed: {sim.seed}",
        ]
    )


def main(argv=None):
    """Parse settings and run the CLI capture, scan export, or optional playground.

    Args:
        argv: Optional argument sequence; None reads command-line arguments from
            sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=default_image_path())
    parser.add_argument(
        "--object-pixel-pitch-um",
        type=float,
        help="object size per original image pixel in um; preserve physical scale (default: resize image to fill field of view)",
    )
    parser.add_argument(
        "--playground",
        action="store_true",
        help="Open the optional live PySide6 playground",
    )
    parser.add_argument(
        "--preset",
        type=Path,
        help="Load a self-contained playground NPZ preset (requires --playground)",
    )
    parser.add_argument("--preview-samples", type=int, default=8)
    parser.add_argument(
        "--sensor-width", type=int, help="Sensor columns (default: --resolution)"
    )
    parser.add_argument(
        "--sensor-height", type=int, help="Sensor rows (default: --resolution)"
    )
    parser.add_argument("--camera-x-mm", type=float, default=0)
    parser.add_argument("--camera-y-mm", type=float, default=0)
    parser.add_argument("--emitter-x-mm", type=float, default=0)
    parser.add_argument("--emitter-y-mm", type=float, default=0)
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument("--oversampling", type=int, default=1)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda[:index], or mps"
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Monte Carlo modes per FFT batch; reduce to save GPU memory",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wavelength-nm", type=float, default=532)
    parser.add_argument(
        "--bandwidth-nm",
        type=float,
        default=20,
        help="Full width of a uniform spectral power band; zero is monochromatic",
    )
    parser.add_argument("--emitter-radius-mm", type=float, default=0.5)
    parser.add_argument("--emitter-distance-m", type=float, default=2)
    parser.add_argument("--wavefront", choices=["plane", "spherical"], default="plane")
    parser.add_argument(
        "--height-rms-um",
        type=float,
        default=0.5,
        help="Zero disables surface roughness",
    )
    parser.add_argument("--object-distance-m", type=float, default=2)
    parser.add_argument("--focal-length-mm", type=float, default=75)
    parser.add_argument("--aperture-mm", type=float, default=75 / 22)
    parser.add_argument("--pixel-pitch-um", type=float, default=1.85)
    parser.add_argument("--read-noise-std", type=float, default=0)
    parser.add_argument("--exposure-scale", type=float, default=1)
    parser.add_argument(
        "--electrons-per-unit",
        type=float,
        default=None,
        help="Enable Poisson shot noise",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="capture",
        metavar="NAME_OR_PATH",
        help="Save under project captures/NAME (default: capture); use ./PATH or an absolute path to override",
    )
    parser.add_argument(
        "--scan-size", type=int, default=15, help="Odd side length of square spiral"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.65, help="Linear pupil diameter overlap"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Print center capture statistics and exit",
    )
    parser.add_argument("--plot-setup", action="store_true")
    args = parser.parse_args(argv)
    if args.preset and not args.playground:
        parser.error("--preset requires --playground")
    if args.playground and (args.headless or args.save is not None or args.plot_setup):
        parser.error(
            "--playground cannot be combined with --headless, --save, or --plot-setup"
        )
    # Import Qt only for the optional desktop path.
    if args.playground:
        try:
            from .playground_ui import launch_playground
        except ImportError as exc:
            parser.error(
                f"Playground GUI dependency unavailable: {exc}. Install with uv run --extra gui fp-capture-sim --playground"
            )
        if args.preset:
            from .playground import load_preset

            try:
                config, assets, display = load_preset(args.preset)
                return launch_playground(config, assets, display)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                parser.error(str(exc))
    try:
        if not math.isfinite(args.bandwidth_nm) or args.bandwidth_nm < 0:
            raise ValueError("bandwidth-nm must be finite and nonnegative")
        # Convert CLI display units to SI at the model boundary.
        setup = OpticalSetup(
            object_distance_m=args.object_distance_m,
            focal_length_m=args.focal_length_mm * 1e-3,
            aperture_diameter_m=args.aperture_mm * 1e-3,
            sensor_shape=(
                args.sensor_height
                if args.sensor_height is not None
                else args.resolution,
                args.sensor_width if args.sensor_width is not None else args.resolution,
            ),
            sensor_pixel_pitch_m=args.pixel_pitch_um * 1e-6,
            oversampling=args.oversampling,
        )
        spectrum = (
            Spectrum.uniform(
                (args.wavelength_nm - args.bandwidth_nm / 2) * 1e-9,
                (args.wavelength_nm + args.bandwidth_nm / 2) * 1e-9,
            )
            if args.bandwidth_nm
            else Spectrum.monochromatic(args.wavelength_nm * 1e-9)
        )
        emitter = CircularEmitter(
            args.emitter_radius_mm * 1e-3,
            args.emitter_distance_m,
            center_xy_m=(args.emitter_x_mm * 1e-3, args.emitter_y_mm * 1e-3),
            wavefront=args.wavefront,
        )
        if args.playground:
            from .playground import PlaygroundConfig, Realizations

            config = PlaygroundConfig(
                setup=setup,
                emitter=emitter,
                sensor=SensorNoise(
                    exposure_scale=args.exposure_scale,
                    read_noise_std=args.read_noise_std,
                    electrons_per_unit=args.electrons_per_unit,
                ),
                wavelength_m=args.wavelength_nm * 1e-9,
                bandwidth_m=args.bandwidth_nm * 1e-9,
                height_rms_m=args.height_rms_um * 1e-6,
                camera_xy_m=(args.camera_x_mm * 1e-3, args.camera_y_mm * 1e-3),
                preview_samples=min(args.preview_samples, args.samples),
                final_samples=args.samples,
                batch_size=args.batch_size,
                device=args.device,
                dtype=args.dtype,
                light_seed=args.seed,
                surface_seed=args.seed,
                sensor_seed=args.seed,
            )
            assets = Realizations.from_image(args.image, config)
            from dataclasses import replace

            width = (
                assets.image.shape[1] * args.object_pixel_pitch_um * 1e-6
                if args.object_pixel_pitch_um is not None
                else setup.sensor_shape[1]
                * setup.sensor_pixel_pitch_m
                / setup.magnification
            )
            config = replace(config, object_width_m=width).validate()
            return launch_playground(config, assets)
        dtype = resolve_dtype(args.dtype)
        device = resolve_device(args.device, dtype)
        sample = RoughSample.from_image(
            args.image,
            setup,
            args.height_rms_um * 1e-6,
            args.seed,
            pixel_pitch_m=None
            if args.object_pixel_pitch_um is None
            else args.object_pixel_pitch_um * 1e-6,
            device=device,
            dtype=dtype,
        )
        sensor = SensorNoise(
            exposure_scale=args.exposure_scale,
            read_noise_std=args.read_noise_std,
            electrons_per_unit=args.electrons_per_unit,
        )
        sim = CaptureSimulator(
            setup,
            sample,
            spectrum,
            emitter,
            sensor,
            seed=args.seed,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
        )
        print(
            format_setup_summary(
                sim, args.image, args.height_rms_um * 1e-6, args.samples
            ),
            flush=True,
        )
        if args.save is not None:
            step = setup.aperture_step(args.overlap)
            positions = spiral_positions(args.scan_size, step)
            path = sim.save_scan(
                args.save,
                positions,
                args.samples,
                reuse_samples=False,
                scan_step_m=step,
            )
            print(f"Saved {len(positions)} captures to {path.resolve()}")
        elif args.headless:
            result = sim.capture(
                (args.camera_x_mm * 1e-3, args.camera_y_mm * 1e-3), count=args.samples
            )
            print(
                f"Capture {tuple(result.image.shape)}: min={result.image.min().item():.6g}, "
                f"max={result.image.max().item():.6g}, mean={result.image.mean().item():.6g}"
            )
        else:
            from .visualization import CaptureViewer, plot_setup

            if args.plot_setup:
                plot_setup(sim)
            viewer = CaptureViewer(sim, args.samples)
            viewer.show()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
