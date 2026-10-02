"""The pipeline above the extension types, and a solve refined against the system.

:func:`factorize` is the two-step path in one call: it walks the elimination
graph and factorizes from it, returning both so a caller can still ask what the
factorization was going to cost.

:func:`refined_solve` improves a solve by measuring the residual against the
:*system* and correcting. The residual and every correction are formed in double
precision whatever the system stores, so what comes back is a solve whose
residual against the *stored* matrix is as small as ``tolerance`` asks for --
not of the matrix the caller imagined.

That is worth doing for a single-precision system, whose factorization error it
removes. It is not a way past the storage: how far that solution sits from the
real matrix is still bounded by how the system rounded its own blocks, roughly
``cond * 1e-7`` for single precision. And the floor of the residual itself is
``eps * cond``, so the default ``tolerance`` of ``1e-14`` is not reachable for a
system with a condition number much above ``1e2`` -- see :func:`refined_solve`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from hybsol._mod import BlockSystem, Decomposition, Elimination

__all__ = ["factorize", "refined_solve"]


def factorize(
    system: BlockSystem,
    *,
    n_threads: int = 0,
    workspace: NDArray[np.uint8] | None = None,
) -> tuple[Elimination, Decomposition]:
    """Walk the elimination graph and factorize from it.

    The two halves of :meth:`BlockSystem.decompose` in one call, returning both
    so the graph stays available: the decomposition holds a copy of the
    schedule, so the returned :class:`~hybsol.Elimination` is yours to inspect
    or throw away, and the system is unchanged either way.

    Parameters
    ----------
    system : BlockSystem
        The system to factorize. It is only read.
    n_threads : int, default: 0
        Number of OpenMP threads; ``0`` selects the OpenMP default and ``1``
        runs the factorization serially.
    workspace : array, optional
        A writable, contiguous 1-D ``uint8`` array of at least
        :meth:`BlockSystem.workspace_bytes` bytes, written in place so one buffer
        can be reused across systems. A read-only array is refused rather than
        copied, since a copy would defeat the point of handing one in.

    Returns
    -------
    tuple
        The :class:`~hybsol.Elimination` and the factorized
        :class:`~hybsol.Decomposition`.

    Raises
    ------
    ValueError
        The system does not satisfy the solver's structural assumptions, a
        diagonal block holds nothing but zeros, ``n_threads`` is negative,
        or ``workspace`` is too small or read-only.
    hybsol.SingularSystemError
        A diagonal block is singular, so the LU factorization hit a zero
        pivot.

    Examples
    --------
    >>> import numpy as np
    >>> from hybsol import BlockSystem, factorize
    >>> m = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> system = BlockSystem(2)
    >>> system.add_block(0, 0, m)
    >>> graph, decomposition = factorize(system)
    >>> graph.n_operations
    1
    >>> np.allclose(decomposition.solve(m @ np.ones(2)), np.ones(2))
    True
    """
    graph = system.elimination()
    if workspace is None:
        return graph, system.decompose(n_threads=n_threads)
    return graph, system.decompose(n_threads=n_threads, workspace=workspace)


def refined_solve(
    system: BlockSystem,
    decomposition: Decomposition,
    rhs: ArrayLike,
    *,
    out: NDArray[np.float64] | None = None,
    n_threads: int = 0,
    tolerance: float = 1e-14,
    max_iterations: int = 5,
) -> NDArray[np.float64]:
    """Solve, then correct against the system until the residual is small.

    A solve is only as good as the factors behind it. This measures
    ``rhs - A x`` against the *system* rather than against the factors, solves
    the residual with the same decomposition, and adds the correction -- the
    residual and the correction both in double precision, whatever the system
    stores. What comes back is the solve of the matrix the system *holds*, to
    the accuracy ``tolerance`` pins down.

    For a double-precision system the plain solve usually already meets the
    default tolerance, and this returns after one check. For a single-precision
    system it removes the factorization's share of the error -- which is where
    most of it is -- but it cannot undo how the system rounded its own blocks,
    so the answer is still only as far from the real matrix as that rounding
    is, roughly ``cond * 1e-7``. Once the residual floor is reached the extra
    corrections add nothing, which is why the ill-conditioned cases raise
    rather than returning a worse answer.

    The system and the decomposition must describe the same matrix. Passing a
    decomposition of a different system is not diagnosed here -- the correction
    simply will not converge, and :exc:`RuntimeError` says so.

    Parameters
    ----------
    system : BlockSystem
        The system the decomposition was built from, and the one the residual
        is measured against. It is only read.
    decomposition : Decomposition
        A factorized decomposition of ``system``.
    rhs : array_like
        Right-hand side of length ``sum(block_sizes)``.
    out : array, optional
        Array to write the solution into. It may be ``rhs`` itself.
    n_threads : int, default: 0
        Number of OpenMP threads for each solve; ``0`` selects the OpenMP
        default and ``1`` runs them serially.
    tolerance : float, default: 1e-14
        Target for the relative residual ``||rhs - A x|| / ||rhs||``, measured
        against the stored system. It has to be reachable: the residual is
        formed in double precision whatever the system stores, so its floor is
        roughly ``2e-16 * cond``, and the loop will not get below it however
        many corrections it is allowed. The default suits a well-conditioned
        system; raise it for an ill-conditioned one. That floor is on the
        *residual*, not on the answer: the storage precision still bounds how
        far the solution sits from the real matrix, as described above.
    max_iterations : int, default: 5
        How many corrections to allow before giving up. Each one is a full solve,
        so this bounds the work.

    Returns
    -------
    array
        The refined solution, in double precision.

    Raises
    ------
    RuntimeError
        The residual was still above ``tolerance`` after ``max_iterations``
        corrections. The system is very ill-conditioned, ``rhs`` does not belong
        to it, or the two do not match.
    ValueError
        ``rhs`` has the wrong length, ``tolerance`` is negative, or
        ``max_iterations`` is below one.

    Examples
    --------
    >>> import numpy as np
    >>> from hybsol import BlockSystem, factorize, refined_solve
    >>> m = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> system = BlockSystem(2)
    >>> system.add_block(0, 0, m)
    >>> _, decomposition = factorize(system)
    >>> x = refined_solve(system, decomposition, m @ np.ones(2))
    >>> np.allclose(x, np.ones(2))
    True
    """
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, but it was {tolerance}.")
    if max_iterations < 1:
        raise ValueError(
            f"max_iterations must be at least 1, but it was {max_iterations}."
        )

    b = np.ascontiguousarray(rhs, dtype=np.float64).reshape(-1)
    dim = int(np.sum(np.asarray(system.block_sizes, dtype=int)))
    if b.shape[0] != dim:
        raise ValueError(f"rhs must hold {dim} values, but it holds {b.shape[0]}.")

    result = decomposition.solve(b.copy(), n_threads=n_threads)
    if not np.any(b):
        return _finish(result, out)

    scale = float(np.linalg.norm(b))
    target = tolerance * scale
    residual = b - system.matvec(result, n_threads=n_threads)
    for correction in range(max_iterations + 1):
        if float(np.linalg.norm(residual)) <= target:
            return _finish(result, out)
        if correction == max_iterations:
            break
        result = result + decomposition.solve(residual, n_threads=n_threads)
        residual = b - system.matvec(result, n_threads=n_threads)

    reached = float(np.linalg.norm(residual)) / scale
    raise RuntimeError(
        f"iterative refinement did not reach a relative residual of {tolerance:g} "
        f"in {max_iterations} corrections; it stopped at {reached:g}. The system "
        "is very ill-conditioned, or the decomposition is not of this system."
    )


def _finish(
    result: NDArray[np.float64], out: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    """Copy the solution into ``out`` when given, and return it either way."""
    if out is not None:
        out[...] = result
        return out
    return result
