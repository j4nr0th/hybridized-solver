Quickstart
==========

Assembling a system
-------------------

A :class:`~hybsol.BlockSystem` is created from the sizes of its diagonal
blocks and then filled block by block. Every block ``(i, j)`` has the shape
``(block_sizes[i], block_sizes[j])``, so the easiest way to fill a system is to
slice the full matrix it represents::

    >>> import numpy as np
    >>> from hybsol import BlockSystem
    >>>
    >>> sizes = (2, 3, 2)
    >>> offsets = np.pad(np.cumsum(sizes), (1, 0))
    >>> matrix = np.arange(49, dtype=float).reshape(7, 7)
    >>>
    >>> system = BlockSystem(*sizes)
    >>> for i in range(3):
    ...     for j in range(3):
    ...         system.add_block(
    ...             i, j, matrix[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]]
    ...         )
    >>> np.all(system.as_array() == matrix)
    True

Assembly one block at a time is convenient but not the fastest way to fill a
large system. :meth:`~hybsol.BlockSystem.from_blocks` takes the same blocks as
flat index arrays and packs them in a single pass::

    >>> rows = np.array([0, 1, 2])
    >>> cols = np.array([0, 1, 2])
    >>> data = np.concatenate(
    ...     [
    ...         matrix[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]].ravel()
    ...         for i in range(3)
    ...     ]
    ... )
    >>> diagonal = BlockSystem.from_blocks(sizes, rows, cols, data)
    >>> np.all(diagonal.as_array() == np.diag(np.diag(matrix)))
    True

``data`` holds the blocks in the order the ``(rows, cols)`` pairs give them,
each raveled **row-major**, with block ``k`` taking
``block_sizes[rows[k]] * block_sizes[cols[k]]`` entries.

.. warning::
   A block and its transpose share a row-major ravel *only* when one of them
   has a single column. An ``n x 1`` block and its ``1 x n`` transpose ravel
   alike, which is why assembling one column per block is accidentally correct.
   For ``m x n`` against ``n x m`` with both greater than one they are different
   permutations — ``[[1, 2, 3], [4, 5, 6]]`` ravels to ``[1, 2, 3, 4, 5, 6]`` and
   its ``2 x 3`` transpose to ``[1, 4, 2, 5, 3, 6]`` — so a transposed block needs
   its own data. Getting this wrong ships a silently non-symmetric matrix, and
   the symptom surfaces later as a decomposition failure rather than at
   assembly.

Assembly does not have to go through a temporary either. Code that already
computes a block where it belongs — element matrices, constraint blocks — can
ask for the storage directly: :meth:`~hybsol.BlockSystem.block_storage` adds
the block to the pattern on first use, zeroes it, and hands back a view to
write into::

    >>> filler = BlockSystem(2, 2)
    >>> slot = filler.block_storage(0, 1)
    >>> slot.shape
    (2, 2)
    >>> np.all(slot == 0.0)
    True
    >>> slot[:] = [[4.0, 5.0], [6.0, 7.0]]
    >>> np.array_equal(filler.get_block(0, 1), [[4.0, 5.0], [6.0, 7.0]])
    True

The view keeps the system alive and re-fetching the same block returns the
same storage unchanged. Since :meth:`~hybsol.BlockSystem.decompose` and
:meth:`~hybsol.BlockSystem.reorder_blocks` move blocks around, they refuse to
run while any array from :meth:`~hybsol.BlockSystem.block_storage` is still
alive.

Structural requirements
-----------------------

The decomposition assumes a *symmetric* sparsity pattern: whenever block
``(i, j)`` is stored, block ``(j, i)`` has to be stored too, and every row must
contain its own diagonal block. :meth:`~hybsol.BlockSystem.is_valid` reports
whether a system satisfies this::

    >>> diagonal.is_valid()
    True

Solving
-------

:meth:`~hybsol.BlockSystem.decompose` factorizes the system in place and
records the operations it performed. The system is frozen afterwards, but it
can be solved any number of times::

    >>> system.decompose()
    >>> lhs = np.arange(1, 8, dtype=float)
    >>> solution = system.solve(matrix @ lhs)
    >>> np.allclose(solution, lhs)
    True

The factorization does not pivot, so the caller must hand it blocks that are
safe to factor without that. The requirement is on the *leading principal
minors*, not on the block as a whole: every determinant of the top-left ``k x k``
corner, for ``k = 1`` up to the block size, has to be nonzero. A block can be
nonsingular and still fail this — the saddle-point-shaped matrix below has
``det = -4`` and a leading entry of 2, but its leading 4x4 minor is
``1.8e-15``, and factorizing it returns a wrong answer with no error::

    >>> import numpy as np
    >>> from hybsol import BlockSystem
    >>> m = np.array([[2., -1., -1., 0., 1.],
    ...               [-1., 2., 0., -1., 0.],
    ...               [-1., 0., 2., -1., 0.],
    ...               [0., -1., -1., 2., 0.],
    ...               [1., 0., 0., 0., 0.]])
    >>> system = BlockSystem(5)
    >>> system.add_block(0, 0, m)
    >>> system.decompose_diagonal(0)          # reports success
    >>> rhs = np.arange(1., 6.)
    >>> np.linalg.norm(m @ system.solve_diagonal(0, rhs) - rhs)   # not zero
    3.5...

An *exactly* zero pivot does raise, as ``zero pivot in LU decomposition``, and
names the block it came from. It is a leading principal minor that merely
vanishes *numerically* that slips through, and that is inherent to
factorizing without pivoting rather than something the solver can detect.

Even when the condition holds, an unpivoted factorization gives up a few digits
against a pivoted reference. They are bought back with a step of iterative
refinement, which reuses the factors as they stand. The recipe is in
:doc:`this worked example </auto_examples/iterative_refinement>`.

Precision
---------

The type every block is stored in — and everything the decomposition computes
in — is chosen when the system is created. :class:`~hybsol.Precision.SINGLE`
halves the memory of the blocks and the factors, and on hardware where FP32 is
faster it factorizes faster too, at the cost of a solution that is only good to
about ``cond * 1e-7``::

    >>> small = BlockSystem(2, 3, precision=Precision.SINGLE)
    >>> for i in range(2):
    ...     for j in range(2):
    ...         small.add_block(
    ...             i, j, matrix[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]].astype(np.float32)
    ...         )
    >>> small.as_array().dtype
    dtype('float32')

The arrays a system hands back follow its precision, while
:meth:`~hybsol.BlockSystem.solve` keeps taking and returning doubles for both
precisions and converts at the boundary. Values passed in are converted too, so
a double array handed to a single-precision system is narrowed on the way in —
build the values in ``float32`` if those last bits matter.

Reordering
----------

For loosely coupled systems, grouping the blocks by "color" first makes the
elimination cheaper. :meth:`~hybsol.BlockSystem.compute_reordering` returns
such an ordering, and the vector helpers move right-hand sides and solutions
along with the system::

    >>> system = BlockSystem(2, 3, 2)
    >>> for i in range(3):
    ...     for j in range(3):
    ...         system.add_block(
    ...             i, j, matrix[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]]
    ...         )
    >>> ordering = system.compute_reordering("greedy")
    >>> system.reorder_blocks(ordering)
    >>> system.decompose()
    >>> reordered_rhs = system.reorder_vector(ordering, matrix @ lhs)
    >>> reordered_lhs = system.solve(reordered_rhs)
    >>> np.allclose(system.unorder_vector(ordering, reordered_lhs), lhs)
    True

.. warning::
   A coloring is not a factorization ordering. Feeding the result of
   :meth:`~hybsol.BlockSystem.compute_reordering` straight into
   :meth:`~hybsol.BlockSystem.reorder_blocks` can leave the system
   unfactorizable, and does so routinely for augmented systems
   ``[[A, N], [Nᵀ, 0]]``. The multiplier region there is structurally zero, so
   to a coloring based on which blocks share an off-diagonal block the
   multipliers look like a perfectly independent set — and the coloring is free
   to schedule them ahead of the element blocks they depend on. A block whose
   diagonal is still zero when it reaches the front cannot be inverted, and the
   factorization has no interchange to fall back on.

   For those systems, choose the permutation yourself with the elements ahead of
   the multipliers. :meth:`~hybsol.BlockSystem.reorder_blocks` checks the order
   it is given and names any block that cannot be factorized, so the mistake
   surfaces at the reorder rather than at :meth:`~hybsol.BlockSystem.decompose`.
