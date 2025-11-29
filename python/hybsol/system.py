"""Types to define system information."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from hybsol._mod import BlockSystem


@dataclass(frozen=True)
class ScaleOperation:
    """Type to define a scale operation.

    This operation scales a block by a given factor.

    Parameters
    ----------
    block_index : int
        The index of the block to scale.
    scale : array
        The scale matrix to apply to the block.
    """

    block_index: int
    scale: npt.NDArray[np.float64]

    def __init__(self, block_index: int, scale: npt.ArrayLike) -> None:
        object.__setattr__(self, "block_index", int(block_index))
        object.__setattr__(self, "scale", np.asarray(scale, np.float64, copy=None))


@dataclass(frozen=True)
class EliminationOperation:
    """Type to define an elimination operation.

    This operation eliminates a block by subtracting a multiple of another block.

    Parameters
    ----------
    target_block_index : int
        The index of the block to eliminate.
    source_block_index : int
        The index of the block to use for elimination.
    multiplier : array
        The multiplier matrix to apply during elimination.
    """

    target_block_index: int
    source_block_index: int

    def __init__(
        self,
        target_block_index: int,
        source_block_index: int,
    ) -> None:
        object.__setattr__(self, "target_block_index", int(target_block_index))
        object.__setattr__(self, "source_block_index", int(source_block_index))


_Operation = ScaleOperation | EliminationOperation


@dataclass(frozen=True)
class TargetRow:
    """Target row, which must be eliminated."""

    index: int
    first_entry: int


def decompose_block_system_c(
    system: BlockSystem,
) -> tuple[tuple[_Operation, ...], BlockSystem]:
    """Create an LU decomposition of a block system.

    Parameters
    ----------
    system : BlockSystem
        The block system to decompose.

    Returns
    -------
    tuple of _Operation
        Operations, which when applied to the original system yield the upper triangular
        block system. As such, it is equivalent to the lower triangular block system's
        inverse.

    BlockSystem
        The upper triangular block system.
    """
    if not system.is_valid():
        raise ValueError("Block system is not valid.")

    operations: list[_Operation] = list()

    # Divide rows between the source and target rows
    rows_src: list[int] = list()
    rows_tgt: list[TargetRow] = list()

    for i_row, is_free in enumerate(system.no_lower_connections()):
        if not is_free:
            rows_tgt.append(TargetRow(i_row, system.first_column(i_row)))
        else:
            inv = np.linalg.inv(system.get_block(i_row, i_row))
            operations.append(ScaleOperation(i_row, inv))
            if i_row + 1 != system.n_blocks:
                system.multiply_row(i_row, inv, start=i_row + 1)
            rows_src.append(i_row)

    while rows_tgt:
        possible_eliminations: list[EliminationOperation] = list()
        remaining_targets: list[TargetRow] = list()

        while rows_tgt:
            tgt_row = rows_tgt.pop()
            if tgt_row.first_entry in rows_src:
                possible_eliminations.append(
                    EliminationOperation(
                        tgt_row.index,
                        tgt_row.first_entry,
                    )
                )
            else:
                remaining_targets.append(tgt_row)

        rows_tgt = remaining_targets

        assert possible_eliminations, (
            "There have to be SOME elimination we can do, right?"
        )

        # Apply the elimination operations
        for elim in possible_eliminations:
            # Add the elimination under performed operations
            operations.append(elim)

            # Get source, target, and multiplier
            i_src, i_tgt = elim.source_block_index, elim.target_block_index

            # Subtract the source from target after scaling
            system.eliminate_row(i_src, i_tgt, system.get_block(i_tgt, i_src))

            # Check if it can now be used as a source, otherwise re-list
            # the target
            first_idx = system.get_next_column_index(i_tgt, i_src)

            if first_idx == i_tgt:
                # Scale by the first block
                inv = np.linalg.inv(system.get_block(i_tgt, i_tgt))
                if i_tgt + 1 != system.n_blocks:
                    system.multiply_row(i_tgt, inv, start=i_tgt + 1)

                operations.append(ScaleOperation(i_tgt, inv))
                rows_src.append(i_tgt)
            else:
                rows_tgt.append(TargetRow(i_tgt, first_idx))

    return tuple(operations), system


def solve_decomposed_c(
    decomposed: BlockSystem, operations: Sequence[_Operation], forcing: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Solve the system using its decomposition."""
    y = np.asarray(forcing, np.float64, copy=True)
    offsets = np.pad(np.cumsum(decomposed.block_sizes), (1, 0))

    # Split the RHS up into blocks as well
    vec: list[npt.NDArray[np.float64]] = list()
    for i in range(len(decomposed.block_sizes)):
        vec.append(y[offsets[i] : offsets[i + 1]])

    # First step is to apply operations to simulate L^{-1}
    for op in operations:
        match op:
            case ScaleOperation() as scal:
                vec[scal.block_index] = scal.scale @ vec[scal.block_index]

            case EliminationOperation() as elim:
                multiplier = decomposed.get_block(
                    elim.target_block_index, elim.source_block_index
                )
                vec[elim.target_block_index] -= multiplier @ vec[elim.source_block_index]

            case _:
                raise TypeError(f"Invalid operation type {type(op)}")

    # Next step is to do the back substitution
    for i_row in reversed(range(decomposed.n_blocks - 1)):
        v = vec[i_row]

        indices = decomposed.get_row_block_indices(i_row)
        for i_col in reversed(indices):
            if i_col == i_row:
                break
            u = decomposed.get_block(i_row, i_col)
            v[:] -= u @ vec[i_col]
        vec[i_row] = v

    return np.concatenate(vec)
