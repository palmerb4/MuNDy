#include <boost/math/tools/roots.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// Function to compute the next x value using Boost's Brent's method
double find_next_x(const double &x_prev, const double &segment_length, const double &amplitude,
                   const double &angular_frequency, const double &phase_shift) {
  auto f = [&x_prev, &segment_length, &amplitude, &angular_frequency, &phase_shift](const double &x) {
    const double delta_x = x - x_prev;
    const double delta_y = amplitude * std::sin(angular_frequency * x + phase_shift) -
                           amplitude * std::sin(angular_frequency * x_prev + phase_shift);
    const double current_ell = std::sqrt(delta_x * delta_x + delta_y * delta_y);
    return current_ell - segment_length;
  };

  std::uintmax_t max_iter = 1000;
  const boost::math::tools::eps_tolerance<double> double_tol(boost::math::tools::digits<double>());
  auto [x_new, y_new] = boost::math::tools::toms748_solve(f, x_prev, x_prev + segment_length, double_tol, max_iter);
  return x_new;
}

/// \brief Function to descretize a sin wave into a series of equall length segments
///
/// For us y = A * sin(w * x + phi) where A is the amplitude, w is the angular frequency, and phi is the phase shift.
///
/// \param x_start Starting x value
/// \param num_segments Number of segments to discretize the sin wave into
/// \param segment_length Length of each segment.
/// \param amplitude Amplitude of the sin wave
/// \param angular_frequency Angular frequency of the sin wave
/// \param phase_shift Phase shift of the sin wave
struct ComputeSinWaveSegmentsFunctor {
  ComputeSinWaveSegmentsFunctor &set_x_start(const double &x_start) {
    x_start_ = x_start;
    return *this;
  }

  ComputeSinWaveSegmentsFunctor &set_num_segments(const size_t &num_segments) {
    num_segments_ = num_segments;
    return *this;
  }

  ComputeSinWaveSegmentsFunctor &set_segment_length(const double &segment_length) {
    segment_length_ = segment_length;
    return *this;
  }

  ComputeSinWaveSegmentsFunctor &set_amplitude(const double &amplitude) {
    amplitude_ = amplitude;
    return *this;
  }

  ComputeSinWaveSegmentsFunctor &set_angular_frequency(const double &angular_frequency) {
    angular_frequency_ = angular_frequency;
    return *this;
  }

  ComputeSinWaveSegmentsFunctor &set_phase_shift(const double &phase_shift) {
    phase_shift_ = phase_shift;
    return *this;
  }

  std::vector<double> operator()() const {
    std::vector<double> x_values = {x_start_};
    for (size_t i = 0; i < num_segments_; ++i) {
      double x_next = find_next_x(x_values.back(), segment_length_, amplitude_, angular_frequency_, phase_shift_);
      x_values.push_back(x_next);
    }
    return x_values;
  }

 private:
  double x_start_;
  size_t num_segments_;
  double segment_length_;
  double amplitude_;
  double angular_frequency_;
  double phase_shift_;
};  // ComputeSinWaveSegmentsFunctor

std::vector<double> find_backbone(const double &x_start, const size_t &num_segments, const double &segment_length,
                                  const double &amplitude, const double &angular_frequency, const double &phase_shift) {
  std::vector<double> x_values = {x_start};
  for (size_t i = 0; i < num_segments; ++i) {
    double x_next = find_next_x(x_values.back(), segment_length, amplitude, angular_frequency, phase_shift);
    x_values.push_back(x_next);
  }
  return x_values;
}

int main() {
  // Define the parameters
  const double segment_length = 0.1;
  const int N = 3000;
  std::vector<double> x_values = ComputeSinWaveSegmentsFunctor()
                                     .set_x_start(0.0)
                                     .set_num_segments(N)
                                     .set_segment_length(segment_length)
                                     .set_amplitude(1.0)
                                     .set_angular_frequency(1.0)
                                     .set_phase_shift(0.0)();

  return 0;
}
