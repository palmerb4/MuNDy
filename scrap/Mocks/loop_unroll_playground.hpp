#include <iostream>
#include <cmath>
#include <chrono>
#include <any>
#include <memory>

int main(int argc, char *argv[]) {
    using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::microseconds;

auto t1 = high_resolution_clock::now();
//   std::any a = static_cast<double>(rand());
  double b = rand();
  double a = rand();
  double sum = 0;
  for (int i = 0; i < 10000; i++) {
    sum += std::sqrt(std::abs(std::sin(
         a   
        )));
  }

auto t2 = high_resolution_clock::now();

/* Getting number of milliseconds as an integer. */
auto ms_int = duration_cast<microseconds>(t2 - t1);
  std::cout << sum << " in " << ms_int.count() << std::endl;
  return 0;
}