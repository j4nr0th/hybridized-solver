cmake_minimum_required(VERSION 3.29)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "hybsol")
endif ()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

Python_add_library(_mod MODULE
        src/module.c
        src/blocks.c
        src/block_info.c
        src/block_system.c
        src/block_operation.c
        WITH_SOABI)

add_subdirectory(cpyutl)

target_link_libraries(_mod PRIVATE cpyutl)
set_property(TARGET _mod PROPERTY C_STANDARD 17)
install(TARGETS _mod DESTINATION ${SKBUILD_PROJECT_NAME})
target_compile_definitions(_mod PRIVATE CPYUTL_ASSERTS)

if ("${NumPy_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "NumPy_INCLUDE_DIRS was empty.")
endif ()
target_include_directories(_mod PRIVATE ${NumPy_INCLUDE_DIRS} ${Python_INCLUDE_DIRS} cpyutl/src)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(_mod PRIVATE -Wall -Wextra -Werror -fwhole-program -flto=auto)
endif ()
