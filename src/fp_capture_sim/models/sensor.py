"""Exposure scaling and sensor noise applied after optical integration."""

import math
from dataclasses import dataclass

import torch

from ..backend import positive, real_tensor


@dataclass(frozen=True)
class SensorNoise:
    """One sensor read after optical integration, on the input tensor device.

    Args:
        exposure_scale: Nonnegative intensity multiplier independent of mode count.
        read_noise_std: Nonnegative Gaussian standard deviation in output intensity
            units.
        bias: Finite additive offset in output intensity units.
        electrons_per_unit: Optional positive calibration enabling Poisson shot noise.
        clip_min: Optional finite lower bound on output intensity.
        clip_max: Optional finite upper bound, at least clip_min when both are present.

    Output uses linear arbitrary intensity units; negative reads remain unless clipped.
    """

    exposure_scale: float = 1.0
    read_noise_std: float = 0.0
    bias: float = 0.0
    electrons_per_unit: float | None = None
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self):
        """Validate exposure, noise calibration, bias, and optional clipping bounds."""
        positive("exposure_scale", self.exposure_scale, allow_zero=True)
        positive("read_noise_std", self.read_noise_std, allow_zero=True)
        if not math.isfinite(self.bias):
            raise ValueError("bias must be finite")
        if self.electrons_per_unit is not None:
            positive("electrons_per_unit", self.electrons_per_unit)
        for bound in (self.clip_min, self.clip_max):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("clipping bounds must be finite")
        if (
            self.clip_min is not None
            and self.clip_max is not None
            and self.clip_min > self.clip_max
        ):
            raise ValueError("clip_min must not exceed clip_max")

    def apply(self, mean_intensity, generator=None):
        """Apply one exposure and sensor read to the integrated optical intensity.

        Shot noise precedes bias and Gaussian read noise; clipping is applied last.
        The result retains the input tensor device and real precision.

        Args:
            mean_intensity: Finite nonnegative real array-like optical intensity before
                exposure scaling.
            generator: Optional PyTorch random generator on the output tensor device.
        """
        image = self.exposure_scale * real_tensor(mean_intensity)
        if not bool((torch.isfinite(image) & (image >= 0)).all()):
            raise ValueError("mean_intensity must be finite and nonnegative")
        if self.electrons_per_unit is not None:
            image = (
                torch.poisson(image * self.electrons_per_unit, generator=generator)
                / self.electrons_per_unit
            )
        image = image + self.bias
        if self.read_noise_std:
            image = image + self.read_noise_std * torch.randn(
                image.shape, device=image.device, dtype=image.dtype, generator=generator
            )
        if self.clip_min is not None or self.clip_max is not None:
            image = image.clamp(min=self.clip_min, max=self.clip_max)
        return image
