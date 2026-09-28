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

The factorization is unpivoted, so a zero pivot on one of the diagonal blocks
is reported as an error rather than silently producing garbage.

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
