# Native tests for the Python-free core. Enabled with HYBSOL_BUILD_TESTS.

set(HYBSOL_TEST_EXECUTABLES
        test_matrix
        test_block_system
        test_assembly
        test_elimination
        test_decomposition
        test_ordering
        test_precision
        test_allocator)

foreach (test IN LISTS HYBSOL_TEST_EXECUTABLES)
    add_executable(${test} ${test}.c)
    target_link_libraries(${test} PRIVATE hybsol_core)
    target_compile_features(${test} PRIVATE c_std_17)
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${test} PRIVATE -Wall -Wextra -Werror)
    endif ()
    add_test(NAME ${test} COMMAND ${test})
endforeach ()

# A separate target that builds the core with sanitizers enabled; useful when
# hunting for memory errors, but far too slow to run on every build.
add_custom_target(hybsol_sanitize
        COMMAND ${CMAKE_COMMAND}
        -S ${CMAKE_CURRENT_SOURCE_DIR}/../..
        -B ${CMAKE_CURRENT_BINARY_DIR}/../sanitize
        -DHYBSOL_ENABLE_OPENMP=OFF
        -DHYBSOL_BUILD_TESTS=ON
        -DCMAKE_BUILD_TYPE=Debug
        -DCMAKE_C_FLAGS=-fsanitize=address,undefined
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/../sanitize
        COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_CURRENT_BINARY_DIR}/../sanitize --output-on-failure)
