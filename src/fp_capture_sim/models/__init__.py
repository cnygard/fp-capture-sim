"""Independent physical models, re-exported here for stable public imports.

Implementations live in optical, object, emitter, and sensor modules. Existing
``from fp_capture_sim.models import ...`` imports continue to work.
"""

from ..backend import positive as positive
from .emitter import CircularEmitter, Spectrum
from .object import RoughSample, _resample_object_image as _resample_object_image
from .optical import OpticalSetup
from .sensor import SensorNoise

__all__ = ["OpticalSetup", "RoughSample", "Spectrum", "CircularEmitter", "SensorNoise"]
