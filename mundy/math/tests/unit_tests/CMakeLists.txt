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

# Here, we attempt to organize unit tests logically by dependencies (not by spelling). Simple classes are tested first 
# and more complex classes are tested later. This is not a hard and fast rule, but it is a good guideline to follow, 
# as it produces more logical error messages when a test fails.
tribits_add_executable_and_test(
  UnitTestViews
  SOURCES
    UnitTestViews.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1-4
)
  
tribits_add_executable_and_test(
  UnitTestVector
  SOURCES
    UnitTestVector.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1-4
)
  
tribits_add_executable_and_test(
  UnitTestVector3
  SOURCES
    UnitTestVector3.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1-4
)

tribits_add_executable_and_test(
  UnitTestMatrix3
  SOURCES
    UnitTestMatrix3.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
    NUM_MPI_PROCS 1-4
)

tribits_add_executable_and_test(
  UnitTestQuaternion
  SOURCES
    UnitTestQuaternion.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1-4
)

tribits_add_executable_and_test(
  UnitTestMinimize
  SOURCES
    UnitTestMinimize.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1-4
)

tribits_add_executable_and_test(
  UnitTestHilbert
  SOURCES
    UnitTestHilbert.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1-4
)
  
add_subdirectory(distance)