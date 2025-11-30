"""Check that the block system can properly decompose and solve a system."""

import numpy as np
import pytest
from hybsol.system import (
    BlockSystem,
    # OperationEliminate,
    # OperationInvDiag,
    decompose_block_system_c,
    solve_decomposed_c,
)


@pytest.mark.parametrize("n", (2, 4, 10))
def test_dense_matrix(n: int) -> None:
    """Check the decomposition works on normal dense matrices."""
    rng = np.random.default_rng(15)
    mat = rng.random((n, n))
    # Check the matrix that was randomly generated is not anywhere near singular
    assert np.abs(np.linalg.det(mat)) > 1e-3, "RNG needs a better seed."

    # Create the system from the full matrix
    sys = BlockSystem(*np.ones(n, dtype=int))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            sys.add_block(i, j, ((mat[i, j],),))

    # Check the system actually represents the right thing.
    sys_mat = sys.as_array()
    assert np.all(mat == sys_mat)

    # Generate the lhs
    lhs = rng.random(n)

    # Get the rhs
    rhs = mat @ lhs

    # Decompose
    ops, decomp = decompose_block_system_c(sys)

    # Solve
    sol = solve_decomposed_c(decomp, ops, rhs)

    assert pytest.approx(sol) == lhs


@pytest.mark.parametrize("n_blocks", (2, 4, 10))
@pytest.mark.parametrize("block_size", (2, 3, 4))
def test_dense_matrix_blocks(n_blocks: int, block_size: int) -> None:
    """Check the decomposition works on normal dense matrices."""
    rng = np.random.default_rng(15)
    mat = rng.random((n_blocks * block_size, n_blocks * block_size))
    # Check the matrix that was randomly generated is not anywhere near singular
    assert np.abs(np.linalg.det(mat)) > 1e-3, "RNG needs a better seed."

    # Create the system from the full matrix
    sys = BlockSystem(*np.full(n_blocks, block_size, int))
    for i in range(n_blocks):
        for j in range(n_blocks):
            sys.add_block(
                i,
                j,
                mat[
                    block_size * i : block_size * (i + 1),
                    block_size * j : block_size * (j + 1),
                ],
            )

    # Check the system actually represents the right thing.
    sys_mat = sys.as_array()
    assert np.all(mat == sys_mat)

    # Generate the lhs
    lhs = rng.random(n_blocks * block_size)

    # Get the rhs
    rhs = mat @ lhs

    # Decompose
    ops, decomp = decompose_block_system_c(sys)

    # Solve
    sol = solve_decomposed_c(decomp, ops, rhs)

    assert pytest.approx(sol) == lhs


@pytest.mark.parametrize("n_blocks", (2, 4, 10))
@pytest.mark.parametrize("block_size", (2, 3, 4))
def test_dense_matrix_blocks_with_c(n_blocks: int, block_size: int) -> None:
    """Check the decomposition works on normal dense matrices with C code."""
    rng = np.random.default_rng(15)
    mat = rng.random((n_blocks * block_size, n_blocks * block_size))
    # Check the matrix that was randomly generated is not anywhere near singular
    assert np.abs(np.linalg.det(mat)) > 1e-3, "RNG needs a better seed."

    # Create the system from the full matrix
    sys = BlockSystem(*np.full(n_blocks, block_size, int))
    sys_2 = BlockSystem(*np.full(n_blocks, block_size, int))
    for i in range(n_blocks):
        for j in range(n_blocks):
            sys.add_block(
                i,
                j,
                mat[
                    block_size * i : block_size * (i + 1),
                    block_size * j : block_size * (j + 1),
                ],
            )
            sys_2.add_block(
                i,
                j,
                mat[
                    block_size * i : block_size * (i + 1),
                    block_size * j : block_size * (j + 1),
                ],
            )

    # Check the system actually represents the right thing.
    sys_mat = sys.as_array()
    assert np.all(mat == sys_mat)
    assert np.all(sys_2.as_array() == sys_mat)

    # Generate the lhs
    lhs = rng.random(n_blocks * block_size)

    # Get the rhs
    rhs = mat @ lhs

    # Decompose
    ops, decomp = decompose_block_system_c(sys)
    sys_2.decompose()

    for i_row in range(decomp.n_blocks):
        cols = decomp.get_row_block_indices(i_row)
        cols_2 = sys_2.get_row_block_indices(i_row)
        assert cols == cols_2

        for c in cols:
            v = decomp.get_block(i_row, c)
            v_2 = sys_2.get_block(i_row, c)
            assert pytest.approx(v) == v_2

    # Can't compare ops, since Python does breath-first, but C does depth-first
    # c_ops = c_decomp.operations()
    # for op, cop in zip(ops, c_ops, strict=True):
    #     if len(cop) == 2:
    #         op_2 = OperationEliminate(cop[0], cop[1])
    #     else:
    #         assert len(cop) == 1
    #         op_2 = OperationInvDiag(cop[0])

    #     assert op == op_2

    # Solve
    sol = solve_decomposed_c(decomp, ops, rhs)

    assert pytest.approx(sol) == lhs


if __name__ == "__main__":
    with np.printoptions(precision=2, suppress=True):
        test_dense_matrix_blocks_with_c(5, 4)
