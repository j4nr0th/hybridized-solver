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
Those codes cover conditions that are outcomes of the data rather than
mistakes — allocation failure, a singular matrix, a system that does not
decompose.

Conditions the caller can be asked to guarantee are *preconditions* instead:
index ranges, non-``NULL`` output pointers, matching shapes, using the
precision spelling the system stores, passing a real permutation, a row that
has its diagonal block. These are checked with ``HYBSOL_ASSERT``, which aborts
with a diagnostic naming the violated condition, and each function documents
them with its parameters.

The checks are compiled in for Debug builds and for the test suite. Release
builds leave them off so nothing is paid for at runtime; define
``HYBSOL_ENABLE_ASSERTS`` when configuring to override either way. Code that
must not abort — the Python bindings, most obviously — checks the same
conditions itself and raises instead.

Memory is allocated with ``malloc``/``realloc``/``free`` unless
:c:func:`hybsol_set_allocator` installs a replacement before the first system
is created. Indices and counts are ``uint64_t`` throughout, and a system is an
opaque :c:type:`hybsol_system_t` that :c:func:`hybsol_system_create` builds
and :c:func:`hybsol_system_destroy` releases.

The headers
-----------

.. c:autodoc:: include/hybsol/types.h

.. c:autodoc:: include/hybsol/matrix.h

.. c:autodoc:: include/hybsol/block_system.h

.. c:autodoc:: include/hybsol/decompose.h

.. c:autodoc:: include/hybsol/ordering.h
