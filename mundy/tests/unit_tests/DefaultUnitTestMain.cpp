// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
//                                                 Author: Bryce Palmer
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

//! \file DefaultUnitTestMain.cpp
/// \brief Main function for unit tests within MundyCore and other Mundy packages
/// This file is meant to reduce code duplication in unit tests. It is meant to be used in combination with TriBITS.
/// For example, a unit test file might look like this:
/// \code
/// tribits_add_executable_and_test(
///     TestName
///     SOURCES
///       TestName.cpp
///       ${MUNDY_DEFAULT_UNIT_TEST_MAIN}
///     COMM serial mpi
///     )
///  \endcode
///
///  Here, MUNDY_DEFAULT_UNIT_TEST_MAIN is set to this file, DefaultUnitTestMain.cpp. TestName.cpp contains a collection
///  of GTEST-based unit tests. The above code will compile TestName.cpp and DefaultUnitTestMain.cpp into an executable
///  called TestName. When the executable is run, it will run all of the unit tests in TestName.cpp in serial and MPI with one process.
///  If the test was only meant to run in parallel, then COMM should be set to mpi. Conversely, if the test was only meant to run in serial,
///  then COMM should be set to serial.

// External libs
#include <gmock/gmock.h>  // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

#ifdef MUNDY_HAVE_MPI
#include <mpi.h>
#endif

#ifdef MUNDY_HAVE_KOKKOS
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize
#endif

int main(int argc, char** argv) {
  #ifdef MUNDY_HAVE_MPI
  MPI_Init(&argc,&argv);
  #endif

  #ifdef MUNDY_HAVE_KOKKOS
  Kokkos::initialize(argc, argv);
  #endif

  testing::InitGoogleMock(&argc, argv);
  int return_val = RUN_ALL_TESTS();

  #ifdef MUNDY_HAVE_KOKKOS
  Kokkos::finalize();
  #endif

  #ifdef MUNDY_HAVE_MPI
  MPI_Finalize();
  #endif

  return return_val;
}
