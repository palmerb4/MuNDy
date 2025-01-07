#include <chrono>

#include "mock.hpp"
#include "mock_api.hpp"

int main() {
  const size_t field_size = 100;
  const size_t num_iterations = 1000000;
  mock::GlobalState state;
  mock::Field &field1 = state.declare_field("field1", field_size);
  mock::Field &field2 = state.declare_field("field2", field_size);
  mock::Field &result = state.declare_field("result", field_size);

  auto start = std::chrono::high_resolution_clock::now();
  mock::randomize_field(field1, 0.0, 1.0);
  mock::randomize_field(field2, 0.0, 1.0);
  for (size_t i = 0; i < num_iterations; ++i) {
    mock::add_fields(field1, field2, result);
    // mock::print_field(result);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

  return 0;
}