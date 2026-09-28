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

if (HYBSOL_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS C)
    if (NOT OpenMP_C_FOUND)
        message(WARNING "OpenMP requested but not found; building serially.")
        set(HYBSOL_ENABLE_OPENMP OFF)
    endif ()
endif ()

# ---------------------------------------------------------------------------
# Core: a plain, Python-free static library.
# ---------------------------------------------------------------------------
add_library(hybsol_core STATIC
        src/core/alloc.c
        src/core/block_system.c
        src/core/decompose.c
        src/core/matrix.c
        src/core/ordering.c)
add_library(hybsol::hybsol ALIAS hybsol_core)

target_compile_features(hybsol_core PUBLIC c_std_17)
target_include_directories(hybsol_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(hybsol_core PRIVATE -Wall -Wextra -Werror)
endif ()

if (HYBSOL_ENABLE_OPENMP)
    target_link_libraries(hybsol_core PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(hybsol_core PUBLIC HYBSOL_ENABLE_OPENMP)
endif ()

# ---------------------------------------------------------------------------
# Python bindings.
# ---------------------------------------------------------------------------
if (NOT HYBSOL_BUILD_PYTHON)
    return()
endif ()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()

add_subdirectory(cpyutl)

Python_add_library(_mod MODULE
        src/python/module.c
        src/python/numpy_convert.c
        src/python/block_system_type.c
        WITH_SOABI)

target_link_libraries(_mod PRIVATE hybsol_core cpyutl)
set_property(TARGET _mod PROPERTY C_STANDARD 17)
target_compile_definitions(_mod PRIVATE CPYUTL_ENABLE_ASSERTS)
target_include_directories(_mod PRIVATE ${NumPy_INCLUDE_DIRS})

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(_mod PRIVATE -Wall -Wextra -Werror)
endif ()

install(TARGETS _mod DESTINATION ${SKBUILD_PROJECT_NAME})
