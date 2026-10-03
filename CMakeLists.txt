cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "hybsol")
endif ()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

# The core is meant to be consumed through `add_subdirectory`/FetchContent, so
# everything it needs is self-contained; the Python bindings pull in NumPy,
# cpyutl and the CPython headers in addition.
option(HYBSOL_ENABLE_OPENMP "Build the core with OpenMP support." ON)
option(HYBSOL_BUILD_PYTHON "Build the Python bindings." ON)
option(HYBSOL_BUILD_TESTS "Build the native test suite." OFF)

# Preconditions are checked with cutl's CUTL_ASSERT, which turns into an
# assumption the optimizer may exploit unless this is on. Debug builds and the
# test suite turn it on so a violated precondition aborts with a message
# instead of running into undefined behaviour. For Release the choice is left
# to whoever consumes the core via add_subdirectory(): set CUTL_ASSERTS
# yourself to keep the checks.
if (NOT DEFINED CUTL_ASSERTS)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR HYBSOL_BUILD_TESTS)
        set(CUTL_ASSERTS ON)
    else ()
        set(CUTL_ASSERTS OFF)
    endif ()
endif ()
message(STATUS "hybsol precondition asserts: ${CUTL_ASSERTS}")

# cutl supplies the core's allocator and its assertion macro. Both of these
# have to be settled before add_subdirectory() so cutl's own defaults do not
# apply, and as cache entries so a consumer that set them keeps its choice.
#
# CUTL_BUILD_TESTS defaults to ON, which additionally forces
# CUTL_ALLOCATOR_GUARD_BYTES to 16 even in a Release build, and those guard
# bytes are compiled into the library rather than only into its tests.
#
# CUTL_ASSERTS stays the single knob: cutl turns it into CUTL_ENABLE_ASSERTS,
# which hybsol_core re-exports to its own consumers.
set(CUTL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CUTL_ASSERTS ${CUTL_ASSERTS} CACHE BOOL "" FORCE)

add_subdirectory(cutl EXCLUDE_FROM_ALL)

if (HYBSOL_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS C)
    if (NOT OpenMP_C_FOUND)
        message(WARNING "OpenMP requested but not found; building serially.")
        set(HYBSOL_ENABLE_OPENMP OFF)
    endif ()
endif ()

# The OpenCL backend is a plugin: its own library, its own Python module, and
# nothing in hybsol_core links OpenCL. A consumer that never asks for a device
# never touches it, so a machine without the loader builds exactly as before --
# with the warning the missing OpenMP package gets.
option(HYBSOL_ENABLE_OPENCL "Build the OpenCL backend." ON)
if (HYBSOL_ENABLE_OPENCL)
    find_package(OpenCL)
    if (NOT OpenCL_FOUND)
        message(WARNING "OpenCL requested but not found; building without the OpenCL backend.")
        set(HYBSOL_ENABLE_OPENCL OFF)
    endif ()
endif ()

if (HYBSOL_ENABLE_OPENCL)
    add_library(hybsol_opencl STATIC
            src/opencl/device.c
            src/opencl/decomposition_opencl.c)
    add_library(hybsol::opencl ALIAS hybsol_opencl)

    target_compile_features(hybsol_opencl PUBLIC c_std_17)

    # PUBLIC: a consumer of hybsol::opencl links the loader too, and reaches
    # the core's internal declarations (the frame helpers) through src.
    target_link_libraries(hybsol_opencl PUBLIC hybsol::hybsol OpenCL::OpenCL)
    target_include_directories(hybsol_opencl PRIVATE src)

    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(hybsol_opencl PRIVATE -Wall -Wextra -Werror)
    endif ()
endif ()

# ---------------------------------------------------------------------------
# Core: a plain, Python-free static library.
# ---------------------------------------------------------------------------
add_library(hybsol_core STATIC
        src/core/block_system.c
        src/core/decomposition.c
        src/core/decomposition_numeric.c
        src/core/elimination.c
        src/core/matrix.c
        src/core/ordering.c
        src/core/result.c
        src/core/solve.c)
add_library(hybsol::hybsol ALIAS hybsol_core)

target_compile_features(hybsol_core PUBLIC c_std_17)


# PUBLIC: hybsol/types.h re-exports cutl_allocator_t, so a consumer that
# includes the core headers needs cutl's headers too.
target_link_libraries(hybsol_core PUBLIC cutl)
target_include_directories(hybsol_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(hybsol_core PRIVATE -Wall -Wextra -Werror)
endif ()

if (HYBSOL_ENABLE_OPENMP)
    target_link_libraries(hybsol_core PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(hybsol_core PUBLIC HYBSOL_ENABLE_OPENMP)
endif ()

# ---------------------------------------------------------------------------
# Native tests.
# ---------------------------------------------------------------------------
if (HYBSOL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/c)
endif ()

# ---------------------------------------------------------------------------
# Python bindings.
# ---------------------------------------------------------------------------
if (NOT HYBSOL_BUILD_PYTHON)
    return()
endif ()

# An interpreter found on PATH usually has no NumPy, which the include
# directory lookup below needs. Prefer the project's own environment when
# there is one; an explicit -DPython_EXECUTABLE still wins.
if (NOT DEFINED Python_EXECUTABLE)
    if (WIN32)
        set(_hybsol_venv_python "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
    else ()
        set(_hybsol_venv_python "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    endif ()
    if (EXISTS "${_hybsol_venv_python}")
        set(Python_EXECUTABLE "${_hybsol_venv_python}")
        message(STATUS "Using the project environment: ${Python_EXECUTABLE}")
    endif ()
    unset(_hybsol_venv_python)
endif ()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR
            "Could not get NumPy's include directory from ${Python_EXECUTABLE}. "
            "The bindings need NumPy; either activate the environment that has it, "
            "or configure with -DPython_EXECUTABLE=/path/to/python. "
            "Building only the core? Pass -DHYBSOL_BUILD_PYTHON=OFF.")
endif ()

add_subdirectory(cpyutl)

Python_add_library(_mod MODULE
        src/python/module.c
        src/python/numpy_convert.c
        src/python/block_system_type.c
        src/python/decomposition_type.c
        src/python/elimination_type.c
        WITH_SOABI)

target_link_libraries(_mod PRIVATE hybsol_core cpyutl)
set_property(TARGET _mod PROPERTY C_STANDARD 17)
target_compile_definitions(_mod PRIVATE CPYUTL_ENABLE_ASSERTS)
target_include_directories(_mod PRIVATE ${NumPy_INCLUDE_DIRS})

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(_mod PRIVATE -Wall -Wextra -Werror)
endif ()

install(TARGETS _mod DESTINATION ${SKBUILD_PROJECT_NAME})

# The backend's Python module. It links its own copy of the core -- that is
# what makes it a backend -- and reaches the bindings' helpers through the
# capsule hybsol._mod publishes, so the two extensions share no code and stay
# independently loadable.
if (HYBSOL_ENABLE_OPENCL)
    Python_add_library(_mod_opencl MODULE src/python/opencl_module.c WITH_SOABI)
    target_link_libraries(_mod_opencl PRIVATE hybsol_core hybsol_opencl cpyutl)
    set_property(TARGET _mod_opencl PROPERTY C_STANDARD 17)
    target_compile_definitions(_mod_opencl PRIVATE CPYUTL_ENABLE_ASSERTS)
    target_include_directories(_mod_opencl PRIVATE ${NumPy_INCLUDE_DIRS})

    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(_mod_opencl PRIVATE -Wall -Wextra -Werror)
    endif ()

    install(TARGETS _mod_opencl DESTINATION ${SKBUILD_PROJECT_NAME})
endif ()
