"""The OpenCL backend: factorize a system with its factors on a device.

Importing this is the whole opt-in. Nothing here runs until :func:`devices` or
:func:`decompose` is called, and ``import hybsol`` alone never touches OpenCL --
not the runtime, not the kernels.

The backend is a build-time thing too: an installation compiled without it has
no ``hybsol.opencl`` at all, and this import raises :exc:`ImportError`. One with
it but no device imports fine and says so -- :func:`devices` returns an empty
list and :func:`decompose` raises :exc:`~hybsol.DeviceError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hybsol import DeviceError
from hybsol._mod_opencl import decompose as _decompose
from hybsol._mod_opencl import device_count as _device_count
from hybsol._mod_opencl import device_info as _device_info

if TYPE_CHECKING:
    from hybsol import BlockSystem, Decomposition, Precision

__all__ = ["DeviceError", "decompose", "device_info", "devices"]


def devices() -> list[dict[str, object]]:
    """List the OpenCL devices available, GPUs first.

    Enumerated once per process, so this is cheap to ask before deciding where
    to factorize. Empty when there is no OpenCL runtime, no device, or no
    backend in this installation.
    """
    return [device_info(i) for i in range(_device_count())]


def device_info(index: int) -> dict[str, object]:
    """Report what the OpenCL runtime says about one device.

    Parameters
    ----------
    index : int
        A device index, below ``device_count()``. GPUs come first, so ``0`` is
        a GPU wherever the machine has one.

    Returns
    -------
    dict
        Keys ``index``, ``name``, ``vendor``, ``uuid``, ``kind`` (``"gpu"``,
        ``"cpu"``, ``"accelerator"`` or ``"unknown"``), ``double`` (whether it
        can do double arithmetic, which every decomposition needs), ``memory``
        and ``compute_units``.

    Raises
    ------
    hybsol.DeviceError
        No device has that index.
    ValueError
        ``index`` is negative.
    TypeError
        ``index`` is not an integer.
    """
    return _device_info(index)


def decompose(
    system: BlockSystem,
    *,
    device: int = 0,
    precision: Precision | None = None,
    n_threads: int = 0,
) -> Decomposition:
    """Factorize a system with its factors on an OpenCL device.

    The symbolic walk and the assembly stay on the CPU; the factorization, the
    solve and the replay run as device kernels, and the factors never come back
    to the host. The result is used exactly as any other decomposition; only its
    ``device`` differs.

    Parameters
    ----------
    system : BlockSystem
        The system to factorize. It is only read.
    device : int, default: 0
        Which device of :func:`devices` to factorize on.
    precision : hybsol.Precision, optional
        The precision of the factors, which need not match the system's: a
        double system can produce single factors, the fast path on a GPU with
        weak double arithmetic. Defaults to the system's own precision.
    n_threads : int, default: 0
        Accepted for symmetry with :meth:`hybsol.BlockSystem.decompose` and
        ignored: the device schedules its own work.

    Returns
    -------
    Decomposition
        The factorized system, with its factors on the device.

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
        A diagonal block is singular, so the LU factorization hit a zero
        pivot. The message names the block.

    Examples
    --------
    >>> import numpy as np
    >>> from hybsol import BlockSystem, opencl
    >>> system = BlockSystem(2)
    >>> system.add_block(0, 0, np.array([[4.0, 1.0], [1.0, 3.0]]))
    >>> for index in [d["index"] for d in opencl.devices() if d["double"]]:
    ...     _ = opencl.decompose(system, device=index).solve(np.ones(2))
    """
    return _decompose(system, device=device, precision=precision, n_threads=n_threads)
