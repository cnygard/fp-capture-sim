"""Device selection and tensor conversion at API and I/O boundaries."""

import math

import torch


def resolve_dtype(dtype):
    """Return a supported real PyTorch dtype; reject other precisions.

    Args:
        dtype: torch.float32/float64 or the corresponding "float32"/"float64" string.
    """
    if isinstance(dtype, str):
        dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    return dtype


def resolve_device(device="auto", dtype=torch.float32):
    """Resolve a device and verify availability and precision support.

    Automatic selection prefers CUDA, then MPS for float32, then CPU.

    Args:
        device: "auto", "cpu", "cuda[:index]", "mps", or a torch.device.
        dtype: Requested real precision; MPS requires float32.
    """
    dtype = resolve_dtype(dtype)
    if str(device) == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available() and dtype == torch.float32:
            device = "mps"
        else:
            device = "cpu"
    try:
        device = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid device: {device}") from exc
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda[:index], or mps")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this PyTorch installation")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        device = torch.device("cuda", index)
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable in this PyTorch installation")
        if device.index not in (None, 0):
            raise ValueError("Only MPS device 0 is supported")
        if dtype != torch.float32:
            raise ValueError("MPS requires float32; use cpu or cuda for float64")
        device = torch.device("mps")
    if device.type == "cpu":
        device = torch.device("cpu")
    return device


def real_tensor(value, *, device=None, dtype=None, copy=False):
    """Convert real array-like input without an implicit transfer to CPU.

    Floating tensors retain their precision unless dtype is supplied; other inputs
    default to float32. A copied result is detached from autograd.

    Args:
        value: Real scalar, list, NumPy array, or PyTorch tensor.
        device: Target PyTorch device; None preserves the input device where supported.
        dtype: Real tensor precision (float32 or float64); None preserves input
            precision.
        copy: Whether to return an independent detached copy of the converted tensor.
    """
    tensor = torch.as_tensor(value)
    if tensor.is_complex():
        raise ValueError("Expected real-valued data")
    if dtype is None:
        dtype = (
            tensor.dtype
            if isinstance(value, torch.Tensor)
            and tensor.dtype in (torch.float32, torch.float64)
            else torch.float32
        )
    dtype = resolve_dtype(dtype)
    if not isinstance(value, torch.Tensor):
        # Cast on CPU before transfer (MPS cannot accept NumPy float64).
        # Re-read scalar/list inputs at the requested precision.
        tensor = torch.as_tensor(value, dtype=dtype)
    tensor = tensor.to(device=device, dtype=dtype)
    return tensor.detach().clone() if copy else tensor


def as_numpy(tensor):
    """Return a detached CPU NumPy view for plotting or serialization.

    CPU storage may be shared with the input tensor.

    Args:
        tensor: PyTorch tensor to transfer to the host.
    """
    return tensor.detach().cpu().numpy()


def positive_count(count, name="count"):
    """Validate a strictly positive integer, rejecting booleans.

    Args:
        count: Value to validate as a positive integer.
        name: Parameter name used in the validation error.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"{name} must be a positive integer")


def make_generator(seed=None, device="cpu"):
    """Create an independent PyTorch random stream.

    Args:
        seed: Local random seed; None selects a fresh seed without changing global RNG
            state.
        device: Device on which this generator will draw random numbers.
    """
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def positive(name, value, allow_zero=False):
    """Validate a finite positive or nonnegative scalar.

    Args:
        name: Parameter name used in the validation error.
        value: Numeric scalar to validate.
        allow_zero: Whether zero is accepted in addition to positive values.
    """
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )

