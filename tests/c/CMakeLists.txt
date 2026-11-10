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

# The device tests need a device, so the target exists only where the backend
# does; without it the file is not compiled and nothing is skipped.
if (TARGET hybsol_opencl)
    add_executable(test_gpu test_gpu.c)
    target_link_libraries(test_gpu PRIVATE hybsol_opencl)
    target_compile_features(test_gpu PRIVATE c_std_17)
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(test_gpu PRIVATE -Wall -Wextra -Werror)
    endif ()
    add_test(NAME test_gpu COMMAND test_gpu)
endif ()

if (TARGET hybsol_hip)
    # Like test_gpu, both HIP tests need ASAN_OPTIONS=detect_leaks=0 under
    # hybsol_sanitize: LeakSanitizer blames the vendor runtime's own
    # exit-time leaks on libamdhip64, and they drown out anything of ours.
    add_executable(test_hip test_hip.c)
    target_link_libraries(test_hip PRIVATE hybsol_hip)
    target_compile_features(test_hip PRIVATE c_std_17)
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(test_hip PRIVATE -Wall -Wextra -Werror)
    endif ()
    add_test(NAME test_hip COMMAND test_hip)

    # The permanent selection harness: it prints the timing/accuracy table
    # that names the default arithmetic, and gates the modes that have a
    # gate. Same vacuous pass without a device.
    add_executable(test_hip_precision test_hip_precision.c)
    target_link_libraries(test_hip_precision PRIVATE hybsol_hip)
    target_compile_features(test_hip_precision PRIVATE c_std_17)
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(test_hip_precision PRIVATE -Wall -Wextra -Werror)
    endif ()
    add_test(NAME test_hip_precision COMMAND test_hip_precision)
endif ()

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
