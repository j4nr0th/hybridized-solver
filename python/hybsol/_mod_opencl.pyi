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

    Keys ``index``, ``name``, ``vendor``, ``kind``, ``double``, ``memory``
    and ``compute_units``.

    Raises
    ------
    hybsol.DeviceError
        No device has that index.
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

    Everything that touches values runs as device kernels; the factors stay
    on the device.

    Raises
    ------
    hybsol.DeviceError
        No device has that index, it cannot do double arithmetic, or the
        device runtime failed.
    ValueError
        The system does not satisfy the solver's structural assumptions,
        ``n_threads`` or ``device`` is negative, or ``precision`` is not a
        :class:`hybsol.Precision` member.
    hybsol.SingularSystemError
        A diagonal block is singular; the message names the block.
    """
    ...
