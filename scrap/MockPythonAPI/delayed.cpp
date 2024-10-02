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

  {
    auto start_time_trace = std::chrono::high_resolution_clock::now();
    auto trace = mock_api::create_trace();
    trace->start();

    mock_api::randomize_field(field1, 0.0, 1.0);
    mock_api::randomize_field(field2, 0.0, 1.0);
    for (size_t i = 0; i < num_iterations; ++i) {
      mock_api::add_fields(field1, field2, result);
      // mock_api::print_field(result);
    }
    trace->stop();
    auto end_time_trace = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time trace: " << std::chrono::duration<double>(end_time_trace - start_time_trace).count()
              << "s\n";

    auto start_time_run = std::chrono::high_resolution_clock::now();
    trace->run();
    auto end_time_run = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time run: " << std::chrono::duration<double>(end_time_run - start_time_run).count() << "s\n";
  }

  {
    auto start_time_trace = std::chrono::high_resolution_clock::now();
    auto trace = mock_api::create_trace();
    trace->start();
    mock_api::randomize_field(field1, 0.0, 1.0);
    mock_api::randomize_field(field2, 0.0, 1.0);
    auto block = [&]() {
      mock_api::add_fields(field1, field2, result);
      // mock_api::print_field(result);
    };
    mock_api::for_loop(0, num_iterations, block);
    trace->stop();
    auto end_time_trace = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time trace (for_loop): "
              << std::chrono::duration<double>(end_time_trace - start_time_trace).count() << "s\n";

    auto start_time_run = std::chrono::high_resolution_clock::now();
    trace->run();
    auto end_time_run = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time run (for_loop): " << std::chrono::duration<double>(end_time_run - start_time_run).count()
              << "s\n";
  }

  return 0;
}