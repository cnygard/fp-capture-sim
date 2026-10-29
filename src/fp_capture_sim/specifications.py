"""Shared scalar specifications for CLI summaries and live interfaces."""
import math


def system_specifications(setup, emitter):
    """Return scalar geometry and sampling specifications without allocating grids.

    Lengths are meters, angles are radians, shapes use (height, width), and
    physical sensor/FOV extents use (width, height).

    Args:
        setup: OpticalSetup defining lens geometry and the object/sensor grids.
        emitter: CircularEmitter defining the source geometry and wavefront.
    """
    ny, nx = setup.sensor_shape
    pitch = setup.sensor_pixel_pitch_m / setup.magnification
    return dict(
        object_na=setup.object_na,
        f_number=setup.f_number,
        magnification=setup.magnification,
        image_distance_m=setup.image_distance_m,
        sensor_size_m=(
            nx * setup.sensor_pixel_pitch_m,
            ny * setup.sensor_pixel_pitch_m,
        ),
        field_of_view_m=(nx * pitch, ny * pitch),
        measurement_pitch_m=pitch,
        grid_pitch_m=setup.object_pixel_pitch_m,
        grid_shape=setup.object_shape,
        emitter_angular_radius_rad=math.atan(emitter.radius_m / emitter.distance_m),
    )
