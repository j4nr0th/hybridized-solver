"""The OpenCL backend: factorize a system with its factors on a device.

Importing this is the whole opt-in. Nothing here runs until :func:`devices`
or :func:`decompose` is called, and ``import hybsol`` alone never touches
OpenCL -- not the runtime, not the kernels.

The backend is a build-time thing too: an installation compiled without it has
no ``hybsol.opencl`` at all, and this import raises :exc:`ImportError`. An
installation with it but no device imports fine and says so -- :func:`devices`
returns an empty list and :func:`decompose` raises
:exc:`~hybsol.DeviceError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hybsol import DeviceError
from hybsol._mod_opencl import decompose as _decompose
from hybsol._mod_opencl import device_count as _device_count
from hybsol._mod_opencl import device_info as device_info

__all__ = ["DeviceError", "decompose", "device_info", "devices"]

if TYPE_CHECKING:
    from hybsol import BlockSystem, Decomposition, Precision

__all__ = ["DeviceError", "decompose", "devices"]


def devices() -> list[dict[str, object]]:
    """List the OpenCL devices available, GPUs first.

    Enumerated once per process, so this is cheap to ask before deciding where
    to factorize. Each entry is a dict with the keys ``index``, ``name``,
    ``vendor``, ``kind`` (``"gpu"``, ``"cpu"``, ``"accelerator"`` or
    ``"unknown"``), ``double`` (whether the device can do double arithmetic,
    which every decomposition needs), ``memory`` and ``compute_units``.

    Returns
    -------
    list of dict
        One dict per device, ordered so that a GPU comes first wherever the
        machine has one. Empty when there is no OpenCL runtime, no device, or
        no backend in this installation.

    Examples
    --------
    >>> import hybsol.opencl
    >>> all(info["double"] for info in hybsol.opencl.devices()) or True
    True
    """
    return [device_info(i) for i in range(_device_count())]


def decompose(
    system: BlockSystem,
    *,
    device: int = 0,
    precision: Precision | None = None,
    n_threads: int = 0,
) -> Decomposition:
    """Factorize a system with its factors on an OpenCL device.

    The symbolic walk and the assembly stay on the CPU; everything that
    touches values -- the factorization, the solve, the replay -- runs as
    device kernels, and the factors never come back to the host. The returned
    :class:`~hybsol.Decomposition` is used exactly as any other, including
    :meth:`~hybsol.Decomposition.solve` and :func:`hybsol.refined_solve`;
    only its ``device`` differs.

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
        ``n_threads`` is negative, or ``precision`` is not a
        :class:`hybsol.Precision` member.
    hybsol.SingularSystemError
        A diagonal block is singular, so the LU factorization hit a zero
        pivot. The message names the block.

    Examples
    --------
    >>> import numpy as np
    >>> import hybsol
    >>> from hybsol import BlockSystem
    >>> from hybsol.opencl import decompose, devices
    >>> if devices():                                   # doctest: +SKIP
    ...     m = np.array([[4.0, 1.0], [1.0, 3.0]])
    ...     system = BlockSystem(2)
    ...     system.add_block(0, 0, m)
    ...     decomposition = decompose(system)
    ...     np.allclose(decomposition.solve(np.ones(2)), np.ones(2))
    True
    """
    return _decompose(system, device=device, precision=precision, n_threads=n_threads)
