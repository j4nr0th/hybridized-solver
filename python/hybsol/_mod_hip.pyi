"""Stub for the HIP backend extension module ``_mod_hip``."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hybsol import BlockSystem, Decomposition, Precision

def device_count() -> int:
    """Count the devices the HIP runtime offers.

    Zero means the runtime offered no device; this module is absent
    altogether from a build without the backend.

    Returns
    -------
    int
        The number of devices, GPUs first.
    """
    ...

def device_info(index: int) -> dict[str, object]:
    """Report what the HIP runtime says about one device.

    Parameters
    ----------
    index : int
        A device index, below ``device_count()``. GPUs come first, so ``0``
        is a GPU wherever the machine has one.

    Returns
    -------
    dict
        Keys ``index``, ``name``, ``vendor``, ``uuid``, ``kind``,
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
    arith: str = "auto",
) -> Decomposition:
    """Factorize a system with its factors on a HIP device.

    Everything that touches values runs as device kernels; the factors stay
    on the device. ``n_threads`` is accepted and ignored.

    Parameters
    ----------
    system : BlockSystem
        The system to factorize. It is only read.
    device : int, default: 0
        Which device of ``devices()`` to factorize on.
    precision : hybsol.Precision, optional
        The precision of the factors, which need not match the system's.
        Defaults to the system's own precision.
    n_threads : int, default: 0
        Accepted for symmetry and ignored: the device schedules its own work.
    arith : str, default: "auto"
        The factor arithmetic mode: ``"auto"`` resolves single factors
        to ``"native"`` and double factors to the fastest mode the
        selection harness gates clean -- ``"native"`` today, on a
        gfx12-family device as on every other AMD gfx device.
        ``"native"`` runs the factors in their own type; ``"ozaki"``
        takes the two block GEMMs onto the WMMA matrix cores instead,
        which needs a gfx12-family device and tiles blocks of at least
        16, staying native below that. ``"ozaki2"`` is the asymmetric
        Ozaki variant, a refinement partner: accurate to roughly 11
        bits on the unsliced side, so it is meant for the inner solve
        of ``hybsol.refined_solve``, not a direct solve.

    Returns
    -------
    Decomposition
        The factorized system, with its factors on the device.

    Raises
    ------
    hybsol.DeviceError
        No device has that index, it cannot do double arithmetic, ``arith``
        asks for a mode the backend does not offer yet, or the device runtime
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
        A diagonal block is singular; the message names the block.
    """
    ...
