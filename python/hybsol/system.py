"""Types to define system information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


class BlockSystem:
    """Type to define a block system."""

    block_sizes: tuple[int, ...]
    rows: tuple[list[tuple[int, npt.NDArray[np.float64]]], ...]

    def __init__(self, *block_sizes: int) -> None:
        """Initialize a block system.

        Args:
            block_sizes: The sizes of each block in the system.
        """
        self.block_sizes = tuple(int(v) for v in block_sizes)
        self.rows = tuple([] for _ in block_sizes)

    def add_block(
        self,
        row: int,
        col: int,
        val: npt.ArrayLike,
    ) -> None:
        """Add a block to the system.

        Paramters
        ---------
        row : int
            The row index of the block.
        col : int
            The column index of the block.
        val : array_like
            The block to add.
        """
        block = np.asarray(val, np.float64, copy=None)
        expected_shape = (self.block_sizes[row], self.block_sizes[col])
        if block.shape != expected_shape:
            raise ValueError(
                f"Block shape {block.shape} does not match expected shape"
                f" {expected_shape}."
            )
        row_l = self.rows[row]
        v: npt.NDArray[np.float64]
        for c, v in row_l:
            if col == c:
                v[:] += block
        row_l.append((col, block))
        row_l.sort(key=lambda x: x[0])

    def is_valid(self) -> bool:
        """Check if the block system is valid.

        Returns
        -------
        bool
            True if the block system is valid, False otherwise.
        """
        # The system must be symmetric (a block at (i, j) means a block at (j, i))
        # and have all diagonal blocks.
        for i, row in enumerate(self.rows):
            has_diagonal = False
            for j, _ in row:
                if i == j:
                    has_diagonal = True
                    continue
                col_row = self.rows[j]
                if not any(c == i for c, _ in col_row):
                    return False

            if not has_diagonal:
                return False

        return True

    def copy(self) -> BlockSystem:
        """Create a copy of the system."""
        sys = BlockSystem(*self.block_sizes)
        for row_in, row_out in zip(self.rows, sys.rows):
            for i, v in row_in:
                row_out.append((i, np.array(v, copy=True)))

        return sys

    def as_array(self) -> npt.NDArray[np.float64]:
        """Convert the system to an array."""
        offsets = np.pad(np.cumsum(self.block_sizes), (1, 0))
        out = np.zeros((offsets[-1], offsets[-1]))
        for i, row in enumerate(self.rows):
            for j, block in row:
                out[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] = block

        return out


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
    multiplier: npt.NDArray[np.float64]

    def __init__(
        self,
        target_block_index: int,
        source_block_index: int,
        multiplier: npt.ArrayLike,
    ) -> None:
        object.__setattr__(self, "target_block_index", int(target_block_index))
        object.__setattr__(self, "source_block_index", int(source_block_index))
        object.__setattr__(
            self, "multiplier", np.asarray(multiplier, np.float64, copy=True)
        )


_Operation = ScaleOperation | EliminationOperation


@dataclass(frozen=True)
class TargetRow:
    """Target row, which must be eliminated."""

    index: int
    first_entry: int


def decompose_block_system(
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

    sys = system.copy()
    del system

    operations: list[_Operation] = list()

    # Divide rows between the source and target rows
    rows_src: list[int] = list()
    rows_tgt: list[TargetRow] = list()
    rows = list(sys.rows)

    for i, row in enumerate(rows):
        j, v = row[0]
        if i != j:
            rows_tgt.append(TargetRow(i, j))
        else:
            inv = np.linalg.inv(v)
            operations.append(ScaleOperation(i, inv))
            for _, u in row:
                u[:] = inv @ u
            rows_src.append(i)

    while rows_tgt:
        possible_eliminations: list[EliminationOperation] = list()
        remaining_targets: list[TargetRow] = list()

        while rows_tgt:
            tgt_row = rows_tgt.pop()
            if tgt_row.first_entry in rows_src:
                possible_eliminations.append(
                    EliminationOperation(
                        tgt_row.index, tgt_row.first_entry, rows[tgt_row.index][0][1]
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
            mul, i_src, i_tgt = (
                elim.multiplier,
                elim.source_block_index,
                elim.target_block_index,
            )

            # Get the rows (skip first element, since it is eliminated).
            src = rows[i_src][1:]
            tgt = rows[i_tgt][1:]

            # Subtract the source from target after scaling
            for i, v in src:
                found = False
                for idx, (j, u) in enumerate(tgt):
                    if i == j:
                        u[:] -= mul @ v
                        if np.allclose(u, 0):
                            if j == i:
                                raise ValueError(
                                    "Diagonal block was eliminated as a side-effect."
                                )
                            del tgt[idx]
                        found = True
                        break
                if found:
                    continue
                tgt.append((i, -mul @ v))

            # Re-sort the target
            tgt.sort(key=lambda key: key[0])

            # Update the row
            rows[i_tgt] = tgt

            # Check if it can now be used as a source, otherwise re-list
            # the target
            first_idx = tgt[0][0]
            if first_idx == i_tgt:
                # Scale by the first block
                inv = np.linalg.inv(tgt[0][1])
                for _, u in tgt:
                    u[:] = u @ inv
                operations.append(ScaleOperation(i_tgt, inv))
                rows_src.append(i_tgt)
            else:
                rows_tgt.append(TargetRow(i_tgt, first_idx))

    # Update the system rows
    sys.rows = tuple(rows)

    return tuple(operations), sys


def solve_decomposed(
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
                vec[elim.target_block_index] -= (
                    elim.multiplier @ vec[elim.source_block_index]
                )

            case _:
                raise TypeError(f"Invalid operation type {type(op)}")

    # Next step is to do the back substitution
    for i, row in reversed(list(enumerate(decomposed.rows))):
        v = vec[i]
        assert np.allclose(row[0][1], np.eye(len(v)))
        for j, u in row[1:]:
            v[:] -= u @ vec[j]
        vec[i] = v

    return np.concatenate(vec)
