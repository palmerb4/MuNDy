# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2023 Flatiron Institute
# Author: Bryce Palmer
#
# Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Mundy. If not, see
# <https://www.gnu.org/licenses/>.
#
# **********************************************************************************************************************
# @HEADER

set(TEST_LIST
    unit_test_mundy_app
)

# Set up the test executables
foreach (CUR_TEST ${TEST_LIST})
    # Add the executable
    add_executable(${CUR_TEST} EXCLUDE_FROM_ALL ${CUR_TEST}.cpp)

    target_compile_definitions(${CUR_TEST} PRIVATE TEST_FILES_DIR="${TEST_FILES_DIR}")

    add_dependencies(test_all ${CUR_TEST})

    target_link_libraries(${CUR_TEST} PRIVATE
        mundy
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        GTest::gmock_main)
    target_include_directories(${CUR_TEST} PRIVATE
        "${CMAKE_SOURCE_DIR}/tests/unit_tests"
        "${googletest_SOURCE_DIR}/googlemock/include"
        "${googletest_SOURCE_DIR}/googletest/include")
endforeach (CUR_TEST)

# Add the non-mpi tests first (1 processor)
# NOTE: This has OMP_PROC_BIND=false as per Kokkos' suggestion on unit testing
foreach (CUR_TEST ${TEST_LIST})
    add_test(NAME ${CUR_TEST} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_POSTFLAGS} $<TARGET_FILE:${CUR_TEST}>)
    set_tests_properties(${CUR_TEST} PROPERTIES ENVIRONMENT "OMP_PROC_BIND=false")
endforeach (CUR_TEST)

add_subdirectory(mundy)
add_subdirectory(mundy_agent)
add_subdirectory(mundy_math)
add_subdirectory(mundy_mesh)
add_subdirectory(mundy_meta)
add_subdirectory(mundy_shape)
