"""The HIP/ROCm backend: factorize a system with its factors on a device.

Importing this is the whole opt-in. Nothing here runs until :func:`devices` or
:func:`decompose` is called, and ``import hybsol`` alone never touches ROCm --
not libamdhip64, not the kernels, which are compiled into the library at build
time rather than built at run time.

The backend is a build-time thing too: an installation compiled without it has
no ``hybsol.hip`` at all, and this import raises :exc:`ImportError`. One with
it but no device imports fine and says so -- :func:`devices` returns an empty
list and :func:`decompose` raises :exc:`~hybsol.DeviceError`.

Everything a decomposition can do is in :func:`decompose`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hybsol import DeviceError
from hybsol._mod_hip import decompose as _decompose
from hybsol._mod_hip import device_count as _device_count
from hybsol._mod_hip import device_info as _device_info

if TYPE_CHECKING:
    from hybsol import BlockSystem, Decomposition, Precision

__all__ = ["DeviceError", "decompose", "device_info", "devices"]


def devices() -> list[dict[str, object]]:
    """List the HIP devices available, GPUs first.

    Enumerated once per process, so this is cheap to ask before deciding where
    to factorize. Empty when the runtime offers no device; this module is
    absent altogether from a build without the backend.

    Returns
    -------
    list of dict
        One dict per device, ordered so a discrete GPU comes first wherever
        the machine has one, with the keys :func:`device_info` documents.

    Examples
    --------
    >>> import hybsol.hip
    >>> found = hybsol.hip.devices()
    >>> all(isinstance(info["double"], bool) for info in found) or not found
    True
    """
    return [device_info(i) for i in range(_device_count())]


def device_info(index: int) -> dict[str, object]:
    """Report what the HIP runtime says about one device.

    Parameters
    ----------
    index : int
        A device index, below ``device_count()``. GPUs come first, so ``0`` is
        a GPU wherever the machine has one.

    Returns
    -------
    dict
        Keys ``index``, ``name``, ``vendor``, ``uuid``, ``kind`` (``"gpu"``,
        ``"integrated"`` or ``"unknown"``), ``double`` (whether it can do
        double arithmetic, which every decomposition needs), ``memory`` and
        ``compute_units``.

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
    arith: str = "auto",
) -> Decomposition:
    """Factorize a system with its factors on a HIP device.

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
    arith : str, default: "auto"
        How the factor arithmetic runs on the device: ``"auto"`` picks
        ``"native"`` for single factors, and for double factors the
        fastest mode the selection harness gates clean -- ``"native"``
        on a gfx12-family device, where the split modes live, and
        ``"split"`` on every other AMD gfx device. ``"native"`` runs the
        factors in their own type. ``"split"`` carries double factors as
        double-single pairs on the FP32 FMA unit; ``"ozaki"`` slices
        them for the WMMA matrix cores, which needs a gfx12-family
        device and tiles only where a block reaches 16 wide -- smaller
        blocks, and every pass the tiles do not serve, keep the
        double-single path. ``"ozaki2"`` is the asymmetric variant: one
        operand keeps its five slices, the other its single top-11-bit
        window. Its accuracy is the refinement partner's, not a direct
        solve's: use it under :func:`hybsol.refined_solve`.

        Single factors ignore the mode on every device: they are already
        the narrow arithmetic and the matrix cores hold no double, so
        ``"ozaki"`` there runs the same kernels ``"native"`` does rather
        than refusing. Blocks the tiles cannot serve -- narrower than 16,
        wider than 32, or carrying magnitudes outside the range the
        arithmetic of the moment can hold -- fall back per block, silently
        and at full accuracy.

    Returns
    -------
    Decomposition
        The factorized system, with its factors on the device.

    Raises
    ------
    hybsol.DeviceError
        No device has that index, it cannot do double arithmetic, ``arith``
        asks for a mode the device cannot serve, or the device runtime
        failed.
    ValueError
        The system does not satisfy the solver's structural assumptions,
        ``device`` or ``n_threads`` is negative, or ``precision`` or ``arith``
        is not a valid name.
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
    >>> from hybsol import BlockSystem, hip
    >>> system = BlockSystem(2)
    >>> system.add_block(0, 0, np.array([[4.0, 1.0], [1.0, 3.0]]))
    >>> for index in [d["index"] for d in hip.devices() if d["double"]]:
    ...     _ = hip.decompose(system, device=index).solve(np.ones(2))
    """
    return _decompose(
        system, device=device, precision=precision, n_threads=n_threads, arith=arith
    )
