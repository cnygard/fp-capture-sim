# Fourier ptychography capture simulation

A scalar, paraxial, in-focus model for camera-scanning macroscopic Fourier
ptychography. Define the optical setup, sample, illumination distribution and
sensor independently. Importing the package does not read images or open plots.

Run this standalone project from this directory:

```bash
uv run fp-capture-sim --device auto --plot-setup
uv run fp-capture-sim --wavefront spherical --samples 128
uv run fp-capture-sim --device cuda --headless --samples 256 --batch-size 16
uv run fp-capture-sim --device mps --headless --samples 128
uv run fp-capture-sim --device cpu --dtype float64 --headless --samples 32
uv run fp-capture-sim --save my_scan --scan-size 5 --read-noise-std 0.001
uv run python -m unittest discover -s tests -v
```

Every run prints the lens geometry, sensor dimensions, object field of view and
sampling, requested and realized surface roughness, source spectrum and spatial
model, and sensor noise and execution settings before generating captures.

The default image is found in the working directory, its parent, or the
source checkout's parent directory. Use `--image PATH` for another sample and
`--help` for setup, spectrum, emitter, roughness and noise controls. The existing
`--save my_scan` writes to `fp_capture_sim/captures/my_scan`, independent of the
working directory. `--save` alone uses `captures/capture`. To choose another
location, pass an explicit path, e.g. `--save ./my_scan`, `--save ../data/my_scan`,
or `--save /data/my_scan`. Saving exits afterward and requires a new directory.
The project's `captures/` directory is ignored by Git. The Python `save_scan`
API uses the same rules; use a string for `"./my_scan"` because `Path` removes
that prefix.

The implementation lives in `src/fp_capture_sim/`; the original monolithic
script has been replaced. To use the existing parent environment without
installing this project, run `../.venv/bin/python run_capture.py --plot-setup`.

## Live simulation playground

Install the optional GUI dependency and launch the desktop playground:

```bash
uv run --extra gui fp-capture-sim --playground
uv run --extra gui fp-capture-sim --playground --device cpu --samples 128 --preview-samples 8
uv run --extra gui fp-capture-sim --playground --object-pixel-pitch-um 0.5060728745
uv run --extra gui fp-capture-sim --playground --preset saved_playground.npz
```

In the existing parent environment, install `PySide6>=6.10,<7`, then use
`../.venv/bin/python run_capture.py --playground`. The GUI was exercised with
Python 3.14.7, PySide6 6.12.0, and Matplotlib 3.11.2 on macOS. Normal CLI and
Python simulator use do not import Qt or require the GUI extra.

Existing optical, source, sensor, image-scale, execution and seed arguments
initialize the controls. Additional initial settings include `--sensor-width`,
`--sensor-height`, `--camera-x-mm`, `--camera-y-mm`, `--emitter-x-mm`, and
`--emitter-y-mm`. Sensor width and height each default to `--resolution`.
`--preset` restores its saved configuration and embedded image, taking precedence
over the other initialization arguments. Playground mode is exclusive with
`--save`, `--headless`, and `--plot-setup`.

| Group | Live controls |
| --- | --- |
| Optics and sensor | Focal length, aperture diameter, pixel pitch, sensor width/height, camera x/y |
| Emitter | Center wavelength, full bandwidth, radius, source distance, source x/y, plane/spherical wavefront |
| Object | Image selection, object distance, linked physical width/height, surface height RMS |
| Acquisition | Exposure scale, read-noise standard deviation, optional shot noise and electron calibration |
| Computation | Preview/final sample counts, oversampling, FFT batch size, device, precision |

Sliders and numeric entries use explicit display units. The model boundary uses
meters. Aperture diameter determines f-number; changing object distance keeps
the thin lens in focus. Emitter radius and bandwidth changes hold total source
power fixed. Object dimensions retain the image aspect ratio. Without an initial
native image pitch, the object's width starts at the initial object-plane FOV;
afterward it retains that physical size as optics change. New fields of view
crop or reveal the centered object, with zero reflectance outside it.

**System information** opens a modeless dialog that stays open during edits.
Geometry, NA, f-number, magnification, sensor size, FOV, sampling pitches,
physical object size, wavelength range, and emitter angular radius update
immediately. Realized surface RMS, actual device, sample count, sampling checks,
and Monte Carlo uncertainty describe only a matching capture. These values are
marked pending while the last image remains visible with a stale label. Invalid
combinations retain the last valid image and highlight relevant controls; physical
parameters are never adjusted automatically to make a capture succeed.

Capture runs on one background worker. Edits cancel obsolete work between FFT
batches, replace the single pending request, and reject results from older
revisions. A preview starts after 150 ms without another edit; refinement is
scheduled after 750 ms. Preview and refinement share the same spatial grid and
the same prefix of illumination samples. Sensor-only edits reuse the optical
capture; contrast changes only redraw. Sampling checks apply to the illumination
modes used in that capture, and do not establish rough-surface convergence.

The main sensor view has tabs for noiseless intensity, object reflectance,
surface height, mean pupil transmission, and log mean spectral power. Axes use
registered object coordinates in mm or object frequencies in cycles/mm. Turn
off **Automatic contrast** to lock the displayed intensity limits for brightness
comparisons. **Pin displayed reference** retains a before image. Reference and
difference tabs compare it to later captures; pixelwise differences are available
only when sensor shape, object-plane measurement pitch, and oversampling match.

Comparisons use stable underlying realizations:

- The original image is decoded once per selection, then cached.
- Surface heights come from a fixed 512 × 512 periodic physical lattice, whose
  pitch is the initial simulation pitch. Bilinear periodic interpolation samples
  that same surface at new physical coordinates. Roughness scales its heights;
  interpolation and cropping can change realized RMS, which the dialog reports.
  This adds a fixed physical surface scale for playground comparisons; the
  ordinary `RoughSample.from_image` API retains its original IID behavior.
- Row-wise uniform variates map to the wavelength band and emitter disk. Changing
  bandwidth, radius or mode count preserves shared random variates and prefixes.
- **Resample light**, **Regenerate surface**, and **New sensor read** change
  separate streams. Playground sensor noise is applied on CPU with its own
  reproducible stream, regardless of the optical execution device.

**Save preset** stores current controls, seeds, image provenance, embedded original
image pixels, the exact surface lattice and its physical pitch, illumination
variates, and contrast/view settings in a self-contained NPZ file. Loading does
not need the original image file. To extend the sample count, existing variates
remain intact and additional rows are generated from the light seed. Exact
numerical replay assumes the same backend, precision, and compatible library
versions; floating-point FFT results can differ across devices.

**Export displayed capture** saves the exact image currently held by the viewer,
even if the controls have since changed. It creates a new directory containing
`0000.npy`, `0000_optical.npz` (optical mean, uncertainty, pupil, spectral power,
actual wavelengths and source coordinates), `sample.npz` (realized amplitude and
heights), `config.json` (matching parameters, preview/refined quality, count,
execution metadata and completion flag), `config.toml` (existing reconstruction
handoff), and `playground.npz` (self-contained preset). Saving does not draw a new
sensor read or run another optical capture. Existing capture directories are
rejected. Presets save current controls; exports save the displayed result.

Local measurements on the development Mac, using the default 300 × 300 grid,
8 preview / 64 final samples, float32, batch size 16, and the default image:

| Device | Preview median | Refined median |
| --- | --- | --- |
| CPU | 44 ms | 109 ms |
| Apple MPS | 48 ms | 109 ms |

These are medians of three captures after backend warm-up, including grid/sample
preparation and CPU result transfer. They exclude initial image decoding,
debounce delays, and Matplotlib rendering. Larger grids, oversampling, spherical
illumination, and other hardware change the timings.

The regression suite covers parameter mappings and units, derived geometry,
physical-surface preservation across grids, common illumination variates,
sensor-only reuse, simulator equivalence, cancellation and stale-result rejection,
self-contained replay/export, linked controls, invalid-setting recovery, image
changes, refinement, and application shutdown. Qt tests skip when PySide6 is
absent. CPU and available MPS execution are exercised; CUDA tests require CUDA
hardware.

## Object image scale

Use `--object-pixel-pitch-um` to specify the physical size of a pixel in the
**original object image**. Dimensions are read from the image file. For example,
an 8002 x 8002 image at 0.5060728745 um per pixel spans
4.049595142 x 4.049595142 mm:

```bash
uv run fp-capture-sim --object-pixel-pitch-um 0.5060728745 --save my_scan
```

The object is centered on the simulation grid at its physical scale. Native
image pixels are area-averaged onto that grid; the field of view crops larger
objects and uses zero reflectance outside smaller objects. The startup summary
reports original dimensions, native pixel pitch, physical size, and simulated
sampling. Without this option, the image is resized to fill the field of view.
`--pixel-pitch-um` continues to specify the **sensor** pixel pitch; magnification
sets the object-plane measurement pitch, and `--oversampling` refines the
simulation grid. The original object pitch does not change those optics or
the reconstruction TOML's measurement spacing.

For Python, pass `pixel_pitch_m=0.5060728745e-6` to `RoughSample.from_image`.
Saved `config.json` includes `object_image` with the original shape (height,
width), native pitch and physical size in meters, image path, and placement
rule; `sample.npz` holds the realized sample on the simulation grid.

## Code layout

The physical models live in `src/fp_capture_sim/models/`:

- `optical.py`: thin-lens geometry, object/frequency grids, and pixel integration.
- `object.py`: reflective samples, physical image placement, and surface phase.
- `emitter.py`: spectral power distributions and disk-source illumination.
- `sensor.py`: exposure, shot/read noise, bias, and clipping.

Models remain available from both `fp_capture_sim` and `fp_capture_sim.models`.
`simulation.py` evaluates and exports captures, `playground.py` manages stable
realizations and background-request state, and `playground_ui.py` contains Qt
controls and rendering. `backend.py` handles tensor/device conversion and shared
validation. Function docstrings describe inputs, physical units, and array shapes;
comments explain numerical and concurrency choices.

## Python API

```python
import torch
from fp_capture_sim import (
    OpticalSetup, RoughSample, Spectrum, CircularEmitter,
    SensorNoise, CaptureSimulator, spiral_positions,
)
from fp_capture_sim.visualization import CaptureViewer, plot_setup

setup = OpticalSetup(
    object_distance_m=2,
    focal_length_m=75e-3,
    aperture_diameter_m=75e-3 / 22,
    sensor_shape=(300, 300),
    sensor_pixel_pitch_m=1.85e-6,
    oversampling=2,
)
sample = RoughSample.from_image("../usaf_perfect_square.jpg", setup,
                               height_rms_m=0.5e-6, seed=7)
sim = CaptureSimulator(
    setup, sample,
    spectrum=Spectrum.uniform(522e-9, 542e-9),
    emitter=CircularEmitter(radius_m=0.5e-3, distance_m=2, wavefront="plane"),
    sensor=SensorNoise(read_noise_std=0.001),
    seed=42, device="auto", dtype=torch.float32, batch_size=16,
)
result = sim.capture(camera_xy_m=(1e-3, 0), count=256)
print(result.image.shape, result.image.device, result.standard_error.mean().item())
image_numpy = result.image.cpu().numpy()  # Explicit transfer for other CPU tools
plot_setup(sim)
CaptureViewer(sim, count=64).show()

positions = spiral_positions(5, setup.aperture_step(overlap=0.65))
sim.save_scan("my_scan", positions, count=256)
```

Use `Spectrum.monochromatic(532e-9)` for one wavelength, or
`Spectrum((522e-9, 532e-9, 542e-9), weights=(1, 2, 1))` for weighted discrete
lines. Set emitter radius to zero for a single source point. Set sample height
RMS to zero for a flat sample. Change `wavefront` to `"spherical"` for a point
source's curvature and relative amplitude falloff. The simulation core uses PyTorch; NumPy, Matplotlib, and OpenCV handle
image I/O, plotting, and scan export.

## PyTorch devices, precision, and batching

The numerical pipeline uses PyTorch tensors: source sampling, rough-surface
phase, plane/spherical illumination, batched FFTs, pupil filtering, pixel
integration, intensity accumulation, uncertainty, and sensor noise run on the
selected device. Image decoding, plot rendering, scan coordinates, and NumPy
file serialization remain CPU tasks.

`device="auto"` (also the CLI default) selects CUDA when available, then Apple
MPS for float32, then CPU. Request `"cpu"`, `"cuda"`, `"cuda:1"`, or `"mps"`
explicitly when needed. Unavailable explicit devices raise an error. CUDA
requires a CUDA-enabled PyTorch installation; MPS requires a supported Mac.
No CPU fallback for unsupported GPU operators is enabled by this project.

Float32/complex64 is the default for generation. For higher precision use
`dtype=torch.float64` or `--dtype float64` on CPU or CUDA. MPS supports only
the float32 option here; `auto` with float64 selects CUDA or CPU.

`batch_size` / `--batch-size` controls the number of optical modes processed
by each batched FFT. The default is 16. Reduce it if a large image exhausts
GPU memory, or increase it to explore throughput on your GPU. A final partial
batch is included. Streaming mean/variance merging bounds intermediate storage
by batch size rather than the total Monte Carlo count. Different batch sizes
can produce small floating-point reduction differences.

`RoughSample`, `IlluminationSamples`, and all arrays in `CaptureResult` now
hold tensors. NumPy arrays and lists are accepted as model/sample inputs;
non-tensor inputs default to float32. Existing float32/float64 tensors retain
precision unless a dtype is requested. The simulator takes a copy of the sample
on its chosen device and dtype. Treat sample tensors as fixed during a scan.
Methods such as `object_grid`, `frequency_grid`, spectrum/emitter sampling,
and coherent capture also return tensors.

Results stay on device for subsequent PyTorch work. Use `result.cpu()` to copy
a complete result to CPU, or `result.image.cpu().numpy()` for one NumPy image.
Plotting and export perform these copies explicitly. Generation uses
`torch.no_grad()` and does not retain an autograd graph.

Each simulator has separate device-local PyTorch generators for optical draws
and sensor reads; sample roughness has its own generator. Global PyTorch RNG
state is left unchanged. A seed is repeatable for the same backend, dtype,
PyTorch version, and sequence of calls; equal seeds are not guaranteed to
produce identical random draws across CPU, CUDA, and MPS or across versions.
For numerical comparisons between devices, reuse the same fixed sample and
`IlluminationSamples` (the simulator transfers them as needed). Existing NumPy
RNG seeds do not reproduce the new PyTorch sequences.

## Forward model and conventions

All API lengths are in meters. Arrays use (y, x), positions use (x, y), and
spatial frequency is cycles/m. The sample is a thin rough reflective surface:

```text
object(lambda) = amplitude * exp(i * 4*pi*height / lambda)
field_j        = inverse_FFT[pupil_j * FFT(object(lambda_j) * illumination_j)]
mean_intensity = (1/N) * sum_j |field_j|^2
capture        = sensor(mean_intensity)
```

Each independent draw samples one wavelength from the normalized spectral
**power** distribution and one point uniformly in disk area (radius R*sqrt(U)).
Spatial and spectral distributions are assumed independent. Source points and
wavelengths are mutually incoherent over the exposure, so intensities are
averaged; complex fields are never added across draws. Increasing N improves
integration accuracy without increasing exposure. `exposure_scale` controls
brightness separately. Changing disk radius or bandwidth holds total source
power fixed; it does not hold radiance per area or per wavelength fixed.

For plane waves from (sx, sy) at source distance L, the incident phase is
-2*pi*(sx*x+sy*y)/(lambda*L). Translation of the camera pupil by (cx, cy) at
distance z gives a circular passband centered at (cx,cy)/(lambda*z), with radius
D/(2*lambda*z). We implement the plane-wave tilt by shifting this passband by
+(sx,sy)/(lambda*L) in the untilted object's spectrum. The omitted output phase
ramp has unit magnitude and does not affect intensity. This also avoids
wrapping a discretely sampled illumination ramp around the FFT grid.

Spherical illumination explicitly multiplies the object by
(R0/R)*exp(i*2*pi*(R-R0)/lambda), where R is source-to-pixel distance and R0
is source-to-object-center distance. Its center irradiance is normalized to
match the plane-wave model. The source emits toward the sample and the camera
collects reflected light on the same side.

The thin-lens equation gives image distance v=f*z/(z-f), magnification |M|=v/z,
and object pixel pitch = sensor pitch / (|M|*oversampling). The physical
aperture is authoritative. A 75 mm f/22 lens has a 3.41 mm aperture and
object-side NA about 0.000852 at 2 m. The old code's simultaneous NA=0.02
implied an approximately 80 mm aperture, inconsistent with f/22. To model
that larger aperture explicitly, set `aperture_diameter_m` and increase
oversampling until the requested pupil fits the grid.

Images are registered to the same object coordinates, with inversion removed.
Actual sensor crops, camera translation-induced field-of-view shifts, lens
aberrations, defocus, polarization, multiple scattering, shadowing, and
angle-dependent surface reflectance are not modeled. The fixed height phase
uses the near-normal reflection approximation (path 2*h). The propagation
operator uses the ideal Fourier pupil imaging approximation: it is not a full
finite-distance Fresnel propagation or a ray tracer. Spherical illumination
does not remove those limitations in the collection optics.

FFT sampling implies a periodic finite object. The pupil must fit fully inside
the sampled frequency grid; invalid scans raise a useful error. Spherical
illumination also checks phase-gradient sampling. These checks do not prove
convergence of an arbitrary rough surface: check finer object sampling and a
larger field of view for quantitative work. The generated height map is IID
at the supplied pixel spacing, with no physical correlation-length model.
To compare resolutions for the same surface, supply your own consistent
height maps rather than regenerating IID heights.

## Noise, uncertainty, and visualization

Subpixel irradiance is averaged into each sensor pixel before noise.
Gaussian additive read noise and bias are applied once per capture, after
all modes have accumulated. Optional `electrons_per_unit` adds Poisson shot
noise to the accumulated exposure before read noise. All values use linear
arbitrary intensity units; electrons_per_unit is a user-specified calibration,
not an inferred photon budget. Negative read-noise values are retained unless
`clip_min` is specified. `clip_max` can represent saturation.

`standard_error` is the per-pixel Monte Carlo standard error of the optical
mean, estimated by merging per-batch sample variances. It excludes sensor noise
and exposure scaling. It is NaN for one draw. It is an estimator, not a
guarantee that a small sample has captured all variation.

The viewer shows reflectance, fixed height, mean pupil transmission, log mean
spectral power, optical intensity, and the sensor image. The source diagnostic
uses an independent RNG. Sliders reuse optical draws, making comparisons
between camera positions stable; the resample button draws a new ensemble.
Sensor noise is a new read on every update. Display contrast can change, but
saved intensities are never normalized per frame.

## Saved scans and migration

The scan iterator streams results. By default it reuses the same optical
samples across positions to reduce differences due to Monte Carlo sampling,
while drawing independent sensor noise. Use `reuse_samples=False` for fresh
optical samples at every position.

Each saved dataset contains:

- `config.toml`: acquisition defaults automatically read by `fp DATASET_DIRECTORY`.
- `config.json`: versioned setup, source, noise, seed, positions, PyTorch
  version, device, dtype, batch size, and completion flag (schema version 2).
- `sample.npz`: exact amplitude and fixed surface heights.
- `0000.npy`, etc.: raw sensor captures, with no running-max or 255 scaling.
- `0000_optical.npz`, etc.: noiseless optical means, standard errors and actual
  wavelength/source draws for replay.

The TOML uses the argument names consumed by `fp/arguments.py`: `source`,
`file_extension`, `frame_pattern`, `samples_count`, `resolution`, `wavelength`
(numeric nm), `pixel_size_um`, `aperture_diameter` (mm), `object_distance` (m),
and `translation` (mm) for a compatible spiral. Pixel size is the sensor pitch
divided by magnification, in the object plane; simulation oversampling does not
change the saved measurement spacing. Rectangular frames use their smaller
side as `resolution`, matching the reconstruction loader's center crop.

Algorithm, epochs, learning rates, reconstruction `upsample`, device, precision,
and output options are omitted so reconstruction defaults and explicit CLI
choices remain in control. For example:

```bash
fp /path/to/fp_capture_sim/captures/my_scan --algorithm alternating_projection
```

For a uniform band, `wavelength` is its midpoint; for discrete lines it is the
power-weighted mean. This is a single-wavelength reconstruction approximation:
full bandwidth, source extent, roughness, and sensor noise remain in JSON and
optical NPZ metadata. The TOML does not make the reconstruction model account
for those effects.

Translation is inferred only for a complete centered odd square spiral in
file order. For a single center frame, the CLI supplies the scan step; Python
callers can pass `scan_step_m=setup.aperture_step(overlap)`. Without that information,
translation is omitted. Arbitrary Python scan positions are still saved in JSON,
but require a custom reconstruction geometry adapter; `fp` cannot represent them
with just a sample count and spacing.

TOML is written after all frames are saved. Interrupted scans retain
`complete: false`. Existing output directories are
never silently overwritten. Optical draws and the sample are saved exactly;
the seed initializes independent device-local PyTorch optical/noise streams. Replaying sensor
noise from the seed also requires replaying the same preceding capture calls.

`aperture_step(overlap)` uses **linear diameter overlap** 1-step/D, not the
intersection area of two disks.

The camera-scanning Fourier-domain model follows the acquisition geometry
reviewed in [Concept, implementations and applications of Fourier ptychography](https://authors.library.caltech.edu/records/xpcdw-5m989).
