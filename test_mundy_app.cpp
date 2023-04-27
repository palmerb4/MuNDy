#include <iostream>
#include <stk_unit_test_utils/CommandLineArgs.hpp>
#include <stk_util/command_line/CommandLineParserParallel.hpp>
#include <stk_util/parallel/Parallel.hpp>

int main(int argc, char** argv) {
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "MPI_Init failed." << std::endl;
    return -1;
  }

  const bool proc0 = (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0);

  if (proc0) {
    std::cout << "Test-STK-App" << std::endl;
  }

  MPI_Finalize();

  if (proc0) {
    std::cout << "... exiting." << std::endl;
  }
  return 0;
}