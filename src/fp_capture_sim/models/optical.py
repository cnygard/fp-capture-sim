"""Thin-lens geometry, spatial grids, and sensor pixel integration."""

import math
from dataclasses import dataclass

import torch

from ..backend import positive, positive_count, real_tensor


@dataclass(frozen=True)
class OpticalSetup:
    """In-focus thin lens in air with images registered in object coordinates.

    Args:
        object_distance_m: Positive object-to-lens distance exceeding focal length, in
            meters.
        focal_length_m: Positive lens focal length in meters.
        aperture_diameter_m: Physical entrance pupil diameter in meters.
        sensor_shape: Sensor (height, width) in pixels, each at least two.
        sensor_pixel_pitch_m: Positive sensor pixel pitch in meters.
        oversampling: Positive number of simulation subpixels per sensor axis.

    Arrays use (..., y, x); positions use (x, y).
    """

    object_distance_m: float = 2.0
    focal_length_m: float = 75e-3
    aperture_diameter_m: float = 75e-3 / 22
    sensor_shape: tuple[int, int] = (300, 300)
    sensor_pixel_pitch_m: float = 1.85e-6
    oversampling: int = 1

    def __post_init__(self):
        """Validate physical lengths, focus, sensor dimensions, and oversampling."""
        for name in (
            "object_distance_m",
            "focal_length_m",
            "aperture_diameter_m",
            "sensor_pixel_pitch_m",
        ):
            positive(name, getattr(self, name))
        if self.object_distance_m <= self.focal_length_m:
            raise ValueError("object_distance_m must exceed focal_length_m")
        if len(self.sensor_shape) != 2 or any(
            not isinstance(n, int) or n < 2 for n in self.sensor_shape
        ):
            raise ValueError("sensor_shape must contain two integers >= 2")
        positive_count(self.oversampling, "oversampling")

    @property
    def image_distance_m(self):
        """Return the in-focus lens-to-sensor distance in meters."""
        return 1 / (1 / self.focal_length_m - 1 / self.object_distance_m)

    @property
    def magnification(self):
        """Return the positive image-to-object lateral magnification."""
        return self.image_distance_m / self.object_distance_m

    @property
    def f_number(self):
        """Return focal length divided by physical aperture diameter."""
        return self.focal_length_m / self.aperture_diameter_m

    @property
    def object_na(self):
        """Return the object-side numerical aperture for a lens in air."""
        return math.sin(
            math.atan(self.aperture_diameter_m / (2 * self.object_distance_m))
        )

    @property
    def object_pixel_pitch_m(self):
        """Return the object-plane simulation pitch in meters per subpixel."""
        return self.sensor_pixel_pitch_m / self.magnification / self.oversampling

    @property
    def object_shape(self):
        """Return the oversampled simulation dimensions as (height, width)."""
        return tuple(n * self.oversampling for n in self.sensor_shape)

    def object_grid(self, *, device="cpu", dtype=torch.float32):
        """Return centered (x, y) coordinate tensors in meters, each of object_shape.

        Args:
            device: Device on which to allocate the coordinate grids.
            dtype: Real precision of the coordinate tensors.
        """
        ny, nx = self.object_shape
        pitch = self.object_pixel_pitch_m
        x = (torch.arange(nx, device=device, dtype=dtype) - nx // 2) * pitch
        y = (torch.arange(ny, device=device, dtype=dtype) - ny // 2) * pitch
        y, x = torch.meshgrid(y, x, indexing="ij")
        return x, y

    def frequency_grid(self, *, device="cpu", dtype=torch.float32):
        """Return FFT-centered (fx, fy) grids in cycles per meter.

        Args:
            device: Device on which to allocate the frequency grids.
            dtype: Real precision of the frequency tensors.
        """
        ny, nx = self.object_shape
        pitch = self.object_pixel_pitch_m
        x = torch.fft.fftshift(torch.fft.fftfreq(nx, pitch, device=device, dtype=dtype))
        y = torch.fft.fftshift(torch.fft.fftfreq(ny, pitch, device=device, dtype=dtype))
        y, x = torch.meshgrid(y, x, indexing="ij")
        return x, y

    def aperture_step(self, overlap=0.65):
        """Return the scan spacing in meters for a linear diameter overlap.

        Args:
            overlap: Fraction 1 - step/diameter in [0, 1); this is not disk intersection
                area.
        """
        if not math.isfinite(overlap) or not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        return self.aperture_diameter_m * (1 - overlap)

    def integrate_pixels(self, intensity):
        """Average subpixel irradiance into sensor_shape, preserving batch axes.

        Args:
            intensity: Real array-like irradiance with final dimensions equal to
                object_shape.
        """
        intensity = real_tensor(intensity)
        if intensity.shape[-2:] != self.object_shape:
            raise ValueError(f"intensity must end in object_shape {self.object_shape}")
        ny, nx = self.sensor_shape
        s = self.oversampling
        # Group contiguous subpixels by sensor row/column, then average only those axes.
        return intensity.reshape(*intensity.shape[:-2], ny, s, nx, s).mean(dim=(-1, -3))
