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

tribits_add_executable_and_test(
  UnitTestTest
  SOURCES
    UnitTestTest.cpp
  NUM_MPI_PROCS 1-4
)

tribits_add_executable_and_test(
  UnitTestTestNoMain
  SOURCES
    UnitTestTestNoMain.cpp
    ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
  NUM_MPI_PROCS 1-4
)
