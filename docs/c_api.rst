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
:c:func:`hybsol_system_decompose_with_workspace` is parallel and never calls it
from inside a parallel region, so a plain bump arena with no locking at all can
be supplied. The fill-in comes from one pool the library allocates up front,
sized exactly by :c:func:`hybsol_fill_plan`, which walks the elimination graph
symbolically before any thread starts; the pool is owned by the system and
released with it. A pool found too small would mean the symbolic walk and the
factorization disagree, so that one allocation falls back to the allocator
under a lock rather than failing.

The transient scratch comes from a buffer the caller supplies, sized by
:c:func:`hybsol_workspace_bytes`; :c:func:`hybsol_system_decompose` allocates
that buffer itself. The recorded operation list is sized to
:c:func:`hybsol_system_operation_bound` before the parallel region starts, so
appending to it neither grows nor locks.

Indices and counts are ``uint64_t`` throughout, and a system is an opaque
:c:type:`hybsol_system_t` that :c:func:`hybsol_system_create` builds and
:c:func:`hybsol_system_destroy` releases.

The headers
-----------

.. c:autodoc:: include/hybsol/types.h

.. c:autodoc:: include/hybsol/matrix.h

.. c:autodoc:: include/hybsol/block_system.h

.. c:autodoc:: include/hybsol/decompose.h

.. c:autodoc:: include/hybsol/ordering.h
