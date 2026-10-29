"""Macroscopic Fourier ptychography capture simulation (SI units)."""
from .models import OpticalSetup, RoughSample, Spectrum, CircularEmitter, SensorNoise
from .simulation import (
    CaptureResult,
    CaptureSimulator,
    IlluminationSamples,
    spiral_positions,
)

__all__ = [
    "OpticalSetup",
    "RoughSample",
    "Spectrum",
    "CircularEmitter",
    "SensorNoise",
    "CaptureSimulator",
    "CaptureResult",
    "IlluminationSamples",
    "spiral_positions",
]


def main():
    """Run the command-line application."""
    from .cli import main as cli_main
    cli_main()
