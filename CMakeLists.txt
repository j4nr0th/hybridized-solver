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

# The HIP backend is the second plugin: its own library and its own kernels,
# compiled ahead of time for the architectures asked for, and nothing in
# hybsol_core links libamdhip64. The runtime lives under /opt/rocm when CMake
# is not told otherwise, and a machine without it builds exactly as before --
# with the warning the missing OpenMP package gets.
option(HYBSOL_ENABLE_HIP "Build the HIP/ROCm backend." ON)
if (HYBSOL_ENABLE_HIP)
    find_package(hip CONFIG QUIET HINTS /opt/rocm/lib/cmake /opt/rocm)
    if (NOT hip_FOUND)
        message(WARNING "HIP requested but not found; building without the HIP backend.")
        set(HYBSOL_ENABLE_HIP OFF)
    endif ()
endif ()

if (HYBSOL_ENABLE_HIP)
    enable_language(HIP)

    # The kernels are compiled for the GPU architectures the machine has, unless
    # the caller pins the list. rocminfo names the agents the runtime sees --
    # gfxNNNN, not the generic ISA families it also prints -- and where it
    # cannot answer, this box's pair is the fallback.
    set(HYBSOL_HIP_ARCHITECTURES "" CACHE STRING
            "GPU architectures the HIP kernels compile for; empty auto-detects from rocminfo.")
    if (HYBSOL_HIP_ARCHITECTURES STREQUAL "")
        find_program(HYBSOL_ROCMINFO rocminfo HINTS /opt/rocm/bin)
        set(_hybsol_hip_archs "")
        if (HYBSOL_ROCMINFO)
            execute_process(COMMAND "${HYBSOL_ROCMINFO}"
                    OUTPUT_VARIABLE _hybsol_rocminfo OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET TIMEOUT 30)
            string(REGEX MATCHALL "gfx[0-9][0-9][0-9]+[0-9a-zA-Z]*" _hybsol_hip_archs "${_hybsol_rocminfo}")
            list(REMOVE_DUPLICATES _hybsol_hip_archs)
        endif ()
        if (_hybsol_hip_archs STREQUAL "")
            set(_hybsol_hip_archs "gfx1201;gfx1036")
        endif ()
        set(CMAKE_HIP_ARCHITECTURES "${_hybsol_hip_archs}")
        unset(_hybsol_rocminfo)
    else ()
        set(CMAKE_HIP_ARCHITECTURES "${HYBSOL_HIP_ARCHITECTURES}")
    endif ()
    message(STATUS "HIP kernels compiled for: ${CMAKE_HIP_ARCHITECTURES}")
    unset(_hybsol_hip_archs)

    # One kernel source, two object libraries: the single-precision and the
    # double-precision instantiations, named apart by the define, since both
    # suffixes are archived together and chosen by address at launch.
    set_source_files_properties(src/hip/hip_kernels.cpp PROPERTIES LANGUAGE HIP)
    add_library(hybsol_hip_k32 OBJECT src/hip/hip_kernels.cpp)
    add_library(hybsol_hip_k64 OBJECT src/hip/hip_kernels.cpp)
    target_compile_definitions(hybsol_hip_k64 PRIVATE HYBSOL_HIP_DOUBLE)
    target_link_libraries(hybsol_hip_k32 PUBLIC hip::device)
    target_link_libraries(hybsol_hip_k64 PUBLIC hip::device)

    add_library(hybsol_hip STATIC
            src/hip/device.c
            src/hip/decomposition_hip.c
            $<TARGET_OBJECTS:hybsol_hip_k32>
            $<TARGET_OBJECTS:hybsol_hip_k64>)
    add_library(hybsol::hip ALIAS hybsol_hip)

    target_compile_features(hybsol_hip PUBLIC c_std_17)

    # PUBLIC: a consumer of hybsol::hip links the runtime too, and reaches
    # the core's internal declarations (the frame helpers) through src.
    target_link_libraries(hybsol_hip PUBLIC hybsol::hybsol hip::device)
    target_include_directories(hybsol_hip PRIVATE src)

    # The host side is C17 and gets the core's warnings; the HIP compiler is
    # not -Werror clean against its own headers, so the kernels do not.
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(hybsol_hip PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Werror>)
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
