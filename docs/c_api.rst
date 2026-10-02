C API
=====

The solver lives in a small, dependency-free C library that knows nothing
about Python. It is built as the ``hybsol_core`` static library and can be
consumed directly with ``add_subdirectory`` or FetchContent_:

.. code-block:: cmake

    add_subdirectory(hybsol)
    target_link_libraries(my_target PRIVATE hybsol::hybsol)

Include ``hybsol/hybsol.h`` — or the individual headers under ``hybsol/`` — to
get the whole API. The core is thread-safe as long as no single system is
touched by two threads at once; the ``n_threads`` arguments select how many
OpenMP threads are used, with ``0`` meaning the OpenMP default and ``1``
running serially.

.. _FetchContent: https://cmake.org/cmake/help/latest/module/FetchContent.html

Conventions
-----------

Fallible functions return a result code instead of setting ``errno``;
:c:func:`hybsol_result_str` turns a code into a short, stable description.
Those codes cover outcomes of the data rather than mistakes — allocation
failure, a singular matrix, a system that does not decompose.

Conditions the caller can be asked to guarantee are *preconditions* instead:
index ranges, non-``NULL`` output pointers, matching shapes, the precision
spelling the system stores, a real permutation, a row that has its diagonal
block. These are checked with cutl's ``CUTL_ASSERT``, which aborts with a
diagnostic naming the violated condition, and each function documents them with
its parameters. They are compiled in for Debug builds and the test suite;
Release leaves them off, and setting ``CUTL_ASSERTS`` overrides either way.
Code that must not abort — the Python bindings, most obviously — checks the
same conditions itself and raises instead.

Memory is allocated through a `cutl <https://github.com/j4nr0th/cutl>`_
allocator, chosen per system rather than globally:
:c:func:`hybsol_system_create` takes the one to use and the system keeps it
until :c:func:`hybsol_system_destroy` returns, so systems can use different
allocators and a bare ``malloc`` pointer may be handed to the library. Pass
``&CUTL_STD_ALLOCATOR`` for plain ``malloc``/``realloc``/``free``; cutl also
provides arena, fixed-size-pool and validating allocators.

The allocator need only be safe for the phases that run one at a time.
:c:func:`hybsol_decomposition_factorize_with_workspace` is parallel and never
calls it from inside a parallel region, so a plain bump arena with no locking
at all can be supplied. That falls out of the split: the destination's whole
layout — every block, including the fill-in — is fixed by
:c:func:`hybsol_elimination_create` before any thread starts, and carved in one
allocation, so the factorization itself allocates nothing. The operation list is
read back off the same schedule on demand rather than stored.

The pipeline has five stages, each usable on its own:

#. :c:func:`hybsol_elimination_create` walks the elimination graph
   symbolically, computing no values, and reports the final pattern, the passes
   the factorization runs in, the exact memory it will need, and whether the
   block order admits a factorization at all.
#. :c:func:`hybsol_decomposition_create` lays out the destination in one
   allocation and copies the system's blocks into it.
#. :c:func:`hybsol_decomposition_factorize` walks the graph, factorizing in
   place.
#. :c:func:`hybsol_decomposition_solve` substitutes forward — one pass at a
   time, parallel across the rows of a pass — and back-substitutes serially.

The transient scratch for the factorization comes from a buffer the caller
supplies, sized by :c:func:`hybsol_workspace_bytes`;
:c:func:`hybsol_decomposition_factorize` allocates that buffer itself. A
decomposition owns a copy of the blocks it factorizes, so the system it came
from is unchanged and may be released or decomposed again.

Indices and counts are ``uint64_t`` throughout. The three opaque types are
:c:type:`hybsol_system_t` (built by :c:func:`hybsol_system_create`),
:c:type:`hybsol_elimination_t` (by :c:func:`hybsol_elimination_create`) and
:c:type:`hybsol_decomposition_t` (by :c:func:`hybsol_decomposition_create`),
each with its own destroy.

The headers
-----------

.. c:autodoc:: include/hybsol/types.h

.. c:autodoc:: include/hybsol/matrix.h

.. c:autodoc:: include/hybsol/block_system.h

.. c:autodoc:: include/hybsol/elimination.h

.. c:autodoc:: include/hybsol/decomposition.h

.. c:autodoc:: include/hybsol/ordering.h
