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

#include "gtest/gtest.h"  // for InitGoogleTest, etc
#include "mpi.h"          // for MPI_Init, MPI_Finalize, etc

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  testing::InitGoogleTest(&argc, argv);

  int return_val = RUN_ALL_TESTS();

  MPI_Finalize();

  return return_val;
}