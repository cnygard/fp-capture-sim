"""Spectral power distributions and spatially incoherent disk emitters."""

import math
from dataclasses import dataclass

import torch

from ..backend import positive, positive_count, real_tensor


@dataclass(frozen=True)
class Spectrum:
    """Normalized spectral power distribution: a uniform band or discrete lines.

    Args:
        wavelengths_m: Positive discrete wavelengths, or two increasing band endpoints.
        weights: Nonnegative discrete power weights; empty means equal line
            probabilities.
        continuous: Whether wavelengths_m defines a continuous uniform band without
            weights.
    """

    wavelengths_m: tuple[float, ...]
    weights: tuple[float, ...] = ()
    continuous: bool = False

    def __post_init__(self):
        """Validate positive wavelengths and a normalizable spectral power distribution."""
        if not self.wavelengths_m or any(
            not math.isfinite(w) or w <= 0 for w in self.wavelengths_m
        ):
            raise ValueError("wavelengths must be finite and positive")
        if self.continuous:
            if (
                len(self.wavelengths_m) != 2
                or self.wavelengths_m[1] <= self.wavelengths_m[0]
                or len(self.weights)
            ):
                raise ValueError(
                    "a uniform band needs increasing endpoints and no weights"
                )
        elif len(self.weights):
            if (
                len(self.weights) != len(self.wavelengths_m)
                or any(not math.isfinite(p) or p < 0 for p in self.weights)
                or not 0 < sum(self.weights) < math.inf
            ):
                raise ValueError(
                    "line weights must be finite, nonnegative, and have a positive finite sum"
                )

    @classmethod
    def monochromatic(cls, wavelength_m=532e-9):
        """Construct a single spectral line with unit probability.

        Args:
            wavelength_m: Positive wavelength(s) in meters.
        """
        return cls((wavelength_m,))

    @classmethod
    def uniform(cls, low_m=522e-9, high_m=542e-9):
        """Construct a continuous uniform spectral power band.

        Args:
            low_m: Positive lower wavelength endpoint, in meters.
            high_m: Upper wavelength endpoint in meters, strictly greater than low_m.
        """
        return cls((low_m, high_m), continuous=True)

    def sample(self, count, generator=None, *, device=None, dtype=torch.float32):
        """Draw a (count,) wavelength tensor from the spectral power distribution.

        Args:
            count: Positive number of independent illumination modes.
            generator: Optional PyTorch random generator on the output tensor device.
            device: Output device; defaults to the generator device, or CPU without a
                generator.
            dtype: Real precision of the returned wavelength tensor.
        """
        positive_count(count)
        device = (
            generator.device
            if device is None and generator is not None
            else device or "cpu"
        )
        if self.continuous:
            low, high = self.wavelengths_m
            return low + (high - low) * torch.rand(
                count, device=device, dtype=dtype, generator=generator
            )
        wavelengths = torch.tensor(self.wavelengths_m, device=device, dtype=dtype)
        weights = torch.tensor(
            self.weights or (1.0,) * len(self.wavelengths_m), device=device, dtype=dtype
        )
        indices = torch.multinomial(
            weights, count, replacement=True, generator=generator
        )
        return wavelengths[indices]


@dataclass(frozen=True)
class CircularEmitter:
    """Uniform mutually incoherent disk; each sampled source mode is coherent.

    Args:
        radius_m: Nonnegative disk radius in meters; zero gives one source point.
        distance_m: Positive axial emitter-to-object distance in meters.
        center_xy_m: Finite emitter center (x, y) in meters.
        wavefront: "plane" for a paraxial tilt or "spherical" for curvature and R0/R
            falloff.
    """

    radius_m: float = 0.5e-3
    distance_m: float = 2.0
    center_xy_m: tuple[float, float] = (0.0, 0.0)
    wavefront: str = "plane"

    def __post_init__(self):
        """Validate disk geometry, source coordinates, and wavefront selection."""
        positive("radius_m", self.radius_m, allow_zero=True)
        positive("distance_m", self.distance_m)
        if len(self.center_xy_m) != 2 or not all(
            math.isfinite(v) for v in self.center_xy_m
        ):
            raise ValueError("center_xy_m must contain two finite coordinates")
        if self.wavefront not in ("plane", "spherical"):
            raise ValueError("wavefront must be 'plane' or 'spherical'")

    def sample(self, count, generator=None, *, device=None, dtype=torch.float32):
        """Draw a (count, 2) tensor of source (x, y) coordinates in meters.

        Args:
            count: Positive number of independent illumination modes.
            generator: Optional PyTorch random generator on the output tensor device.
            device: Output device; defaults to the generator device, or CPU without a
                generator.
            dtype: Real precision of the returned source coordinates.
        """
        positive_count(count)
        device = (
            generator.device
            if device is None and generator is not None
            else device or "cpu"
        )
        # Disk area grows as r**2, so sqrt(U) gives uniform power per unit area.
        radius = (
            self.radius_m
            * torch.rand(count, device=device, dtype=dtype, generator=generator).sqrt()
        )
        angle = (
            2
            * math.pi
            * torch.rand(count, device=device, dtype=dtype, generator=generator)
        )
        center = torch.tensor(self.center_xy_m, device=device, dtype=dtype)
        return (
            torch.stack((radius * angle.cos(), radius * angle.sin()), dim=-1) + center
        )

    def field(self, x, y, source_xy_m, wavelength_m):
        """Evaluate plane or spherical illumination on an object grid.

        Leading source/wavelength axes broadcast over the final two spatial axes.
        Spherical amplitude is normalized to unit irradiance at the object center.

        Args:
            x: Object x-coordinate grid in meters; determines the device and precision.
            y: Object y-coordinate grid in meters, matching x.
            source_xy_m: Source position(s) in meters, with a final (x, y) axis.
            wavelength_m: Positive wavelength(s) in meters.
        """
        x = real_tensor(x)
        y = real_tensor(y, device=x.device, dtype=x.dtype)
        source = real_tensor(source_xy_m, device=x.device, dtype=x.dtype)
        wavelength = real_tensor(wavelength_m, device=x.device, dtype=x.dtype)[
            ..., None, None
        ]
        sx, sy = source[..., 0, None, None], source[..., 1, None, None]
        if self.wavefront == "plane":
            path = -(sx * x + sy * y) / self.distance_m
            return torch.exp((2j * math.pi) * path / wavelength)
        r0 = (self.distance_m**2 + sx.square() + sy.square()).sqrt()
        r = (self.distance_m**2 + (x - sx).square() + (y - sy).square()).sqrt()
        # Stable R-R0; essential for float32 at macroscopic source distances.
        path = (x.square() + y.square() - 2 * x * sx - 2 * y * sy) / (r + r0)
        return (r0 / r) * torch.exp((2j * math.pi) * path / wavelength)
