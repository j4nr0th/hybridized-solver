"""Check that the block system can properly decompose and solve a system."""

import numpy as np
import pytest
from hybsol import BlockSystem, decompose_block_system, solve_decomposed


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


if __name__ == "__main__":
    test_dense_matrix(5)
