"""Reflective objects, image placement, and fixed surface heights."""

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ..backend import make_generator, positive, real_tensor


def _resample_object_image(image, shape, pitch_ratio):
    """Area-average native pixels onto a centered physical grid.

    Integration is separable and assumes zero reflectance outside the image,
    without allocating a resized image larger than the destination grid.

    Args:
        image: Two-dimensional NumPy reflectance intensity array.
        shape: Destination grid dimensions (height, width).
        pitch_ratio: Simulation pixel pitch divided by native image pixel pitch.
    """
    import numpy as np

    for axis, count in enumerate(shape):
        values = np.moveaxis(image, axis, 0)
        length = len(values)
        result = np.zeros((count, *values.shape[1:]), dtype=np.float32)
        for index in range(count):
            # Both native and simulation grids put index size//2 at x/y = 0.
            left = (index - count // 2 - 0.5) * pitch_ratio + length // 2 + 0.5
            right = left + pitch_ratio
            left, right = max(0.0, left), min(float(length), right)
            if right <= left:
                continue
            first, stop = math.floor(left), math.ceil(right)
            # Sum covered pixels, then subtract the two fractional edge portions.
            total = values[first:stop].sum(axis=0, dtype=np.float64)
            total -= (left - first) * values[first]
            total -= (stop - right) * values[stop - 1]
            # Divide by the full footprint so out-of-image area contributes zero.
            result[index] = np.maximum(total / pitch_ratio, 0)
        image = np.moveaxis(result, 0, axis)
    return image


@dataclass(frozen=True)
class RoughSample:
    """Reflectance amplitude and fixed surface heights stored as copied tensors.

    Args:
        amplitude: Nonnegative real (H, W) field amplitude, at least 2 by 2.
        height_m: Matching finite surface heights in meters; reflection adds path 2*h.
        image_path: Optional original image provenance path.
        original_image_shape: Optional native image dimensions (height, width).
        original_pixel_pitch_m: Optional calibrated native image pixel pitch in meters.

    Treat stored tensors as fixed during acquisition. Reflected phase is 4*pi*h/lambda.
    """

    amplitude: torch.Tensor
    height_m: torch.Tensor
    image_path: str | None = None
    original_image_shape: tuple[int, int] | None = None
    original_pixel_pitch_m: float | None = None

    def __post_init__(self):
        """Copy and validate matching amplitude/height tensors and image provenance."""
        if self.original_pixel_pitch_m is not None:
            positive("original_pixel_pitch_m", self.original_pixel_pitch_m)
        if self.original_image_shape is not None and (
            len(self.original_image_shape) != 2
            or any(not isinstance(n, int) or n < 1 for n in self.original_image_shape)
        ):
            raise ValueError("original_image_shape must contain two positive integers")
        amp = real_tensor(self.amplitude, copy=True)
        height = real_tensor(
            self.height_m, device=amp.device, dtype=amp.dtype, copy=True
        )
        if amp.ndim != 2 or min(amp.shape) < 2 or height.shape != amp.shape:
            raise ValueError("amplitude and height_m must be matching 2-D arrays")
        if not bool((torch.isfinite(amp) & (amp >= 0) & torch.isfinite(height)).all()):
            raise ValueError("sample arrays must be finite, with nonnegative amplitude")
        object.__setattr__(self, "amplitude", amp)
        object.__setattr__(self, "height_m", height)

    @classmethod
    def from_intensity(
        cls, intensity, height_rms_m=0.5e-6, seed=None, *, device=None, dtype=None
    ):
        """Create amplitude and an IID Gaussian rough surface from reflectance intensity.

        height_rms_m is the distribution standard deviation; finite samples need not
        have exactly that realized RMS.

        Args:
            intensity: Finite nonnegative 2-D reflectance intensity; amplitude is its
                square root.
            height_rms_m: Nonnegative requested RMS surface height, in meters.
            seed: Local random seed; None selects a fresh seed without changing global
                RNG state.
            device: Target PyTorch device; None preserves the input device where
                supported.
            dtype: Real tensor precision (float32 or float64); None preserves input
                precision.
        """
        positive("height_rms_m", height_rms_m, allow_zero=True)
        intensity = real_tensor(intensity, device=device, dtype=dtype)
        if intensity.ndim != 2 or not bool(
            (torch.isfinite(intensity) & (intensity >= 0)).all()
        ):
            raise ValueError("intensity must be a finite, nonnegative 2-D array")
        generator = make_generator(seed, intensity.device)
        height = (
            torch.randn(
                intensity.shape,
                device=intensity.device,
                dtype=intensity.dtype,
                generator=generator,
            )
            * height_rms_m
        )
        return cls(intensity.sqrt(), height)

    @classmethod
    def from_image(
        cls,
        path,
        setup,
        height_rms_m=0.5e-6,
        seed=None,
        *,
        pixel_pitch_m=None,
        device=None,
        dtype=None,
    ):
        """Decode grayscale reflectance and generate a fixed rough surface.

        A calibrated native pitch preserves centered physical scale with zero
        background; otherwise the image is resized to fill the field of view.

        Args:
            path: Image file decoded as 8-bit grayscale reflectance intensity.
            setup: OpticalSetup defining lens geometry and the object/sensor grids.
            height_rms_m: Nonnegative requested RMS surface height, in meters.
            seed: Local random seed; None selects a fresh seed without changing global
                RNG state.
            pixel_pitch_m: Optional positive native image pixel pitch in meters; None
                fills the FOV.
            device: Target PyTorch device; None preserves the input device where
                supported.
            dtype: Real tensor precision (float32 or float64); None preserves input
                precision.
        """
        import cv2

        if pixel_pitch_m is not None:
            positive("pixel_pitch_m", pixel_pitch_m)
        image = cv2.imread(str(Path(path)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read sample image: {path}")
        original_shape = tuple(image.shape)
        image = image.astype("float32") / 255
        ny, nx = setup.object_shape
        if pixel_pitch_m is None:
            image = cv2.resize(image, (nx, ny), interpolation=cv2.INTER_AREA)
        else:
            image = _resample_object_image(
                image, (ny, nx), setup.object_pixel_pitch_m / pixel_pitch_m
            )
        sample = cls.from_intensity(
            image, height_rms_m, seed, device=device, dtype=dtype
        )
        return replace(
            sample,
            image_path=str(Path(path).resolve()),
            original_image_shape=original_shape,
            original_pixel_pitch_m=pixel_pitch_m,
        )

    def to(self, device, dtype=None):
        """Return an independent sample copy on the requested device and precision.

        Args:
            device: Target PyTorch device; None preserves the input device where
                supported.
            dtype: Real tensor precision (float32 or float64); None preserves input
                precision.
        """
        dtype = self.amplitude.dtype if dtype is None else dtype
        return replace(
            self,
            amplitude=self.amplitude.to(device=device, dtype=dtype),
            height_m=self.height_m.to(device=device, dtype=dtype),
        )

    def field(self, wavelength_m):
        """Return the complex reflected field with phase 4*pi*height/wavelength.

        Scalar wavelengths produce (H, W); wavelength arrays prepend their dimensions.

        Args:
            wavelength_m: Positive wavelength(s) in meters.
        """
        wavelength = real_tensor(
            wavelength_m, device=self.amplitude.device, dtype=self.amplitude.dtype
        )
        if not bool((torch.isfinite(wavelength) & (wavelength > 0)).all()):
            raise ValueError("wavelength_m must be finite and positive")
        phase = 4 * math.pi * self.height_m / wavelength[..., None, None]
        return self.amplitude * torch.exp(1j * phase)
