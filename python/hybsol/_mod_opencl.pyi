"""Stub for the OpenCL backend extension module ``_mod_opencl``."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hybsol import BlockSystem, Decomposition, Precision

def device_count() -> int:
    """Count the devices the OpenCL runtime offers.

    Zero means there is no OpenCL runtime, no platform offers a device, or
    hybsol was built without the backend.
    """
    ...

def device_info(index: int) -> dict[str, object]:
    """Report what the OpenCL runtime says about one device.

    Returns keys ``index``, ``name``, ``vendor``, ``uuid``, ``kind``,
    ``double``, ``memory`` and ``compute_units``.

    Raises
    ------
    hybsol.DeviceError
        No device has that index.
    ValueError
        ``index`` is negative.
    TypeError
        ``index`` is not an integer, or was left out.
    """
    ...

def decompose(
    system: BlockSystem,
    *,
    device: int = 0,
    precision: Precision | None = None,
    n_threads: int = 0,
) -> Decomposition:
    """Factorize a system with its factors on an OpenCL device.

    Everything that touches values runs as device kernels; the factors stay on
    the device. ``n_threads`` is accepted and ignored.

    Raises
    ------
    hybsol.DeviceError
        No device has that index, it cannot do double arithmetic, or the
        device runtime failed.
    ValueError
        The system does not satisfy the solver's structural assumptions,
        ``device`` or ``n_threads`` is negative, or ``precision`` is not a
        :class:`hybsol.Precision` member.
    TypeError
        ``system`` is not a :class:`hybsol.BlockSystem`, or an argument is of
        the wrong type or shape.
    RuntimeError
        ``system`` is a :class:`hybsol.BlockSystem` that was never initialized.
    MemoryError
        The device could not hold the factors, or the host could not allocate
        the frame.
    hybsol.SingularSystemError
        A diagonal block is singular; the message names the block.
    """
    ...
