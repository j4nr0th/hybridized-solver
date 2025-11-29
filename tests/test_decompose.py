"""Check that the block system can properly decompose and solve a system."""

import numpy as np
import pytest
from hybsol import BlockSystem, decompose_block_system, solve_decomposed
from hybsol.system import (
    EliminationOperation,
    ScaleOperation,
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
    sys = BlockSystem(*np.ones(n))
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
    ops, decomp = decompose_block_system(sys)

    # Solve
    sol = solve_decomposed(decomp, ops, rhs)

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
    ops, decomp = decompose_block_system(sys)

    # Solve
    sol = solve_decomposed(decomp, ops, rhs)

    assert pytest.approx(sol) == lhs


@pytest.mark.parametrize("n_blocks", (2, 4, 10))
@pytest.mark.parametrize("block_size", (2, 3, 4))
def test_dense_matrix_blocks_with_c(n_blocks: int, block_size: int) -> None:
    """Check the decomposition works on normal dense matrices but with C code."""
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
    # print(sys.as_array())
    ops, decomp = decompose_block_system_c(sys)
    ops_2, decomp_2 = decompose_block_system(sys)

    # print(decomp.as_array())
    # print(decomp_2.as_array())

    # Check that decompositions match above the diagonal
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            b1 = decomp.get_block(i, j)
            row = decomp_2.rows[i]
            b2 = None
            for ic, v in row:
                if ic == j:
                    b2 = v
                    break

            assert b2 is not None
            assert pytest.approx(b2) == b1

    # Check that operations match
    for op1, op2 in zip(ops, ops_2, strict=True):
        match op1:
            case ScaleOperation():
                assert type(op2) is ScaleOperation
                assert op1.block_index == op2.block_index
                assert pytest.approx(op1.scale) == op2.scale
            case EliminationOperation():
                assert type(op2) is EliminationOperation
                assert op1.source_block_index == op2.source_block_index
                assert op1.target_block_index == op2.target_block_index
                assert pytest.approx(op1.multiplier) == op2.multiplier

    # Solve
    sol = solve_decomposed_c(decomp, ops, rhs)
    sol_2 = solve_decomposed(decomp_2, ops_2, rhs)

    assert pytest.approx(sol) == sol_2
    assert pytest.approx(sol) == lhs


if __name__ == "__main__":
    with np.printoptions(precision=2, suppress=True):
        test_dense_matrix_blocks_with_c(5, 1)
