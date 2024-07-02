// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
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

// External
#include <gmock/gmock.h>      // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>      // for TEST, ASSERT_NO_THROW, etc
#include <openrand/philox.h>  // for openrand::Philox

// C++ core
#include <numeric>  // for std::accumulate
#include <vector>   // for std::vector

// Kokkos and Kokkos-Kernels
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
// #include <KokkosBlas_gemm.hpp>

// Mundy
#include <mundy_alens/periphery/Periphery.hpp>  // for gen_sphere_quadrature

namespace mundy {

namespace alens {

namespace periphery {

namespace {

//! \name Helper functions
//@{

/// \brief Compute the slope of a log-log plot of y(x) vs x
double compute_log_log_slope(const std::vector<double> &x, const std::vector<double> &y) {
  const size_t data_size = x.size();
  std::vector<double> log_x(data_size);
  std::vector<double> log_y(data_size);

  for (size_t i = 0; i < data_size; ++i) {
    assert(x[i] > 0.0 && "x values must be positive");
    log_x[i] = std::log(x[i]);
    log_y[i] = std::log(std::fabs(y[i]));  // Use fabs to avoid log of negative numbers
  }

  // Compute the slope of log-log plot (log_y vs log_x)
  double sum_log_x = std::accumulate(log_x.begin(), log_x.end(), 0.0);
  double sum_log_y = std::accumulate(log_y.begin(), log_y.end(), 0.0);
  double sum_log_x_log_y = 0.0;
  double sum_log_x_squared = 0.0;

  for (size_t i = 0; i < data_size; ++i) {
    sum_log_x_log_y += log_x[i] * log_y[i];
    sum_log_x_squared += log_x[i] * log_x[i];
  }

  double slope = (static_cast<double>(data_size) * sum_log_x_log_y - sum_log_x * sum_log_y) /
                 (static_cast<double>(data_size) * sum_log_x_squared - sum_log_x * sum_log_x);
  return slope;
}

double fd_l2_norm_difference(const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &a,
                             const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &b) {
  const size_t num_elements = a.extent(0);
  double l2_norm = 0.0;
  for (size_t i = 0; i < num_elements; ++i) {
    const double diff = a(i) - b(i);
    l2_norm += diff * diff;
  }
  return std::sqrt(l2_norm / static_cast<double>(num_elements));
}

/// \brief A function for generating a quadrature rule
using QuadGenerationFunc =
    std::function<std::array<Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>, 3>(const int order)>;

/// \brief A function for generating a Kokkos vector based on a quadrature rule
using QuadVectorFunc = std::function<Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>(
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights)>;

/// \brief A function that accepts a quadrature rule, an in field, and an out field.
using QuadInOutFunc = std::function<void(
    const double viscosity, const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &in_field,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &out_field)>;

/// \brief A struct for storing the results of a convergence study
struct ConvergenceResults {
  std::vector<double> num_quadrature_points;
  std::vector<double> error;
  double slope;
};

/// \brief Perform a convergence study against an expected result
ConvergenceResults perform_convergence_study(const double &viscosity, const QuadGenerationFunc &quad_gen,
                                             const QuadInOutFunc &func, const QuadVectorFunc &input_field_gen,
                                             const QuadVectorFunc &expected_results_gen) {
  std::vector<double> stashed_num_quadrature_points;
  std::vector<double> stashed_error;
  for (int order = 2; order <= 64; order *= 2) {
    // Generate the quadrature rule
    auto [points, weights, normals] = quad_gen(order);
    const size_t num_quadrature_points = weights.extent(0);

    // Fetch the input field and the expected results
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field =
        input_field_gen(points, normals, weights);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results =
        expected_results_gen(points, normals, weights);

    // Apply the function and compute the finite difference error
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> result_vector("result_vector",
                                                                                3 * num_quadrature_points);
    func(viscosity, points, normals, weights, input_field, result_vector);
    const double fd_l2_error = fd_l2_norm_difference(result_vector, expected_results);

    // Stash the error
    stashed_error.push_back(fd_l2_error);
    stashed_num_quadrature_points.push_back(static_cast<double>(num_quadrature_points));
  }

  // Check that the error converges to zero at the expected rate
  const double slope = compute_log_log_slope(stashed_num_quadrature_points, stashed_error);
  return {stashed_num_quadrature_points, stashed_error, slope};
}

/// \brief Apply the stokes double layer matrix to surface forces to get surface velocities
void apply_stokes_double_layer_matrix_wrapper(
    const double viscosity, const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_forces,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_velocities) {
  // Fill the stokes_double_layer_matrix
  const size_t num_quadrature_points = weights.extent(0);
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> stokes_double_layer_matrix(
      "stokes_double_layer_matrix", 3 * num_quadrature_points, 3 * num_quadrature_points);
  fill_stokes_double_layer_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                  num_quadrature_points, points, points, normals, weights, stokes_double_layer_matrix);
  // Apply the stokes_double_layer_matrix to a the surface forces
  KokkosBlas::gemv(Kokkos::DefaultHostExecutionSpace(), "N", 1.0, stokes_double_layer_matrix, surface_forces, 0.0,
                   surface_velocities);
}

/// Apply the stokes double layer matrix with singularity subtraction to surface forces to get surface velocities
void apply_stokes_double_layer_matrix_ss_wrapper(
    const double viscosity, const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_forces,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_velocities) {
  // Fill the stokes_double_layer_matrix
  const size_t num_quadrature_points = weights.extent(0);
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> stokes_double_layer_matrix(
      "stokes_double_layer_matrix", 3 * num_quadrature_points, 3 * num_quadrature_points);
  fill_stokes_double_layer_matrix_ss(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                     num_quadrature_points, points, points, normals, weights,
                                     stokes_double_layer_matrix);
  // Apply the stokes_double_layer_matrix to a the surface forces
  KokkosBlas::gemv(Kokkos::DefaultHostExecutionSpace(), "N", 1.0, stokes_double_layer_matrix, surface_forces, 0.0,
                   surface_velocities);
}

/// \brief Apply the stokes double layer operator to surface forces to get surface velocities
void apply_stokes_double_layer_kernel_wrapper(
    const double viscosity, const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_forces,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_velocities) {
  const size_t num_quadrature_points = weights.extent(0);
  apply_stokes_double_layer_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                   num_quadrature_points, points, points, normals, weights, surface_forces,
                                   surface_velocities);
}

/// \brief Apply the stokes double layer operator (with singularity subtraction) to surface forces to get surface
/// velocities
void apply_stokes_double_layer_kernel_ss_wrapper(
    const double viscosity, const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_forces,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_velocities) {
  const size_t num_quadrature_points = weights.extent(0);
  apply_stokes_double_layer_kernel_ss(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                      num_quadrature_points, points, points, normals, weights, surface_forces,
                                      surface_velocities);
}

/// \brief A functor for generate a quadrature rule on the sphere
class SphereQuadFunctor {
 public:
  explicit SphereQuadFunctor(const double &sphere_radius) : sphere_radius_(sphere_radius) {
  }

  std::array<Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>, 3> operator()(const int &order) {
    std::vector<double> weights_vec;
    std::vector<double> points_vec;
    std::vector<double> normals_vec;
    gen_sphere_quadrature(order, sphere_radius_, &points_vec, &weights_vec, &normals_vec);
    const size_t num_quadrature_points = weights_vec.size();

    // Convert the points, weights, and normals to Kokkos views
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> points("points", num_quadrature_points * 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> weights("weights", num_quadrature_points);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> normals("normals", num_quadrature_points * 3);
    for (size_t i = 0; i < num_quadrature_points; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        points(3 * i + j) = points_vec[3 * i + j];
        normals(3 * i + j) = normals_vec[3 * i + j];
      }
      weights(i) = weights_vec[i];
    }

    return {points, weights, normals};
  }

 private:
  double sphere_radius_;
};  // class SphereQuadFunctor
//@}

//! \name Quadrature tests
//@{

TEST(PeripheryTest, SphereQuadBasicChecks) {
  // Test that the sum of the quadrature weights is equal to the surface area of the sphere
  // There's no need to check convergence, as this result should converge to the exact value
  // almost immediately.

  // Define the sphere radius
  const double sphere_radius = 12.34;
  const double sphere_surface_area = 4.0 * M_PI * sphere_radius * sphere_radius;
  for (int order = 1; order <= 32; order *= 2) {
    std::vector<double> weights;
    std::vector<double> points;
    std::vector<double> normals;
    gen_sphere_quadrature(order, sphere_radius, &points, &weights, &normals);

    // Check that the number of quadrature points is correct
    EXPECT_EQ(weights.size() * 3, points.size());
    EXPECT_EQ(weights.size() * 3, normals.size());

    for (size_t i = 0; i < points.size(); i += 3) {
      // Check that the points have the right magnitude
      const double x = points[i];
      const double y = points[i + 1];
      const double z = points[i + 2];
      const double magnitude = std::sqrt(x * x + y * y + z * z);
      EXPECT_NEAR(magnitude, sphere_radius, 1.0e-10);

      // Check that the normals are equal to the normalized points
      const double nx = normals[i];
      const double ny = normals[i + 1];
      const double nz = normals[i + 2];
      const double normal_magnitude = std::sqrt(nx * nx + ny * ny + nz * nz);
      EXPECT_NEAR(normal_magnitude, 1.0, 1.0e-10);
      EXPECT_NEAR(nx, x / sphere_radius, 1.0e-10);
      EXPECT_NEAR(ny, y / sphere_radius, 1.0e-10);
      EXPECT_NEAR(nz, z / sphere_radius, 1.0e-10);
    }

    // Check that the sum of the weights is equal to the surface area of the sphere
    const double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    EXPECT_NEAR(sum_weights, sphere_surface_area, 1.0e-10) << "failed for order = " << order << std::endl;
  }
}

TEST(PeripheryTest, SphereQuadAnalyticallyIntegrableFunctions) {
  // Test that we converge to the correct values for analytically integrable functions
  // If you know of any other functions that can be integrated over the sphere analytically without adding additional
  // dependencies like special functions (e.g., spherical harmonics), let us know and we can add them to this
  // test!

  auto run_test_for_function = [](const std::string &function_name, const double &known_integral,
                                  const std::function<double(double, double, double)> &func_to_integrate) {
    const double sphere_radius = 1.0;

    // Weird behavior can occur for order = 1, so we start at order = 2
    std::vector<double> num_quadrature_points;
    std::vector<double> errors;
    for (int order = 2; order <= 64; order *= 2) {
      std::vector<double> weights;
      std::vector<double> points;
      std::vector<double> normals;
      gen_sphere_quadrature(order, sphere_radius, &points, &weights, &normals);

      // Compute the integral
      double integral = 0.0;
      for (size_t i = 0; i < weights.size(); ++i) {
        const double x = points[3 * i];
        const double y = points[3 * i + 1];
        const double z = points[3 * i + 2];
        integral += weights[i] * func_to_integrate(x, y, z);
      }

      // Stash the error
      const double error = std::fabs(integral - known_integral);
      num_quadrature_points.push_back(static_cast<double>(weights.size()));
      errors.push_back(error);

      std::cout << "function_name = " << function_name << ", order = " << order << ", integral = " << integral
                << ", known_integral = " << known_integral << std::endl;
    }

    // Check that the error converges to zero at the expected rate
    const double slope = compute_log_log_slope(num_quadrature_points, errors);
    std::cout << "function_name = " << function_name << ", slope = " << slope << std::endl;
  };

  // Test the function f(x, y, z) = 1, which integrates to 4 * pi
  run_test_for_function("f(x, y, z) = 1", 4.0 * M_PI, [](double, double, double) { return 1.0; });

  // Test the function f(x, y, z) = x, which integrates to 0
  run_test_for_function("f(x, y, z) = x", 0.0, [](double x, double, double) { return x; });

  // Test the function f(x, y, z) = exp(x-y), which integrates to 2^(3/2) * pi * sinh(sqrt(2))
  // https://math.stackexchange.com/questions/717202/surface-integral-on-unit-sphere
  run_test_for_function("f(x, y, z) = exp(x-y)", 2.0 * std::sqrt(2.0) * M_PI * std::sinh(std::sqrt(2.0)),
                        [](double x, double y, double) { return std::exp(x - y); });

  // Test the function f(x, y, z) = max(0, x, x * cos(t) + y * sin(t)), which integrates
  // to pi + pi / 2 * sqrt( (1 - cos t)^2 + (sin t)^2 )
  // https://math.stackexchange.com/questions/3246357/a-tricky-integration-over-the-unit-sphere
  const double t = M_PI / 4.0;  // Arbitrary value of t simply needs to be in [0, 2pi]
  run_test_for_function(
      "f(x, y, z) = max(0, x, x * cos(t) + y * sin(t))",
      M_PI + M_PI / 2.0 * std::sqrt((1.0 - std::cos(t)) * (1.0 - std::cos(t)) + std::sin(t) * std::sin(t)),
      [&t](double x, double y, double) { return std::max({0.0, x, x * std::cos(t) + y * std::sin(t)}); });
}
//@}

//! \name Periphery auxilary function tests
//@{

TEST(PeripheryTest, KokkosInvertMatrix) {
  // This test tests our understanding of how to use Kokkos to invert a matrix.
  // It explicitly tests that the matrix_inverse function works as expected.

  // Statistically speaking, nearly all randomly generated matrices are invertible, so we'll just generate a random
  // matrix and invert it!

  // Generate a random matrix with random normally distributed entities with mean 0 and variance 1
  const int matrix_size = 10;
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> matrix_scratch("matrix_scratch", matrix_size,
                                                                                matrix_size);
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> matrix_original("matrix_original", matrix_size,
                                                                                 matrix_size);

  size_t seed = 1234;
  size_t counter = 0;
  openrand::Philox rng(seed, counter);
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      matrix_scratch(i, j) = rng.randn<double>();
      matrix_original(i, j) = matrix_scratch(i, j);
    }
  }

  // Invert the matrix
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> matrix_inverse("matrix_inverse", matrix_size,
                                                                                matrix_size);
  invert_matrix(Kokkos::DefaultHostExecutionSpace(), matrix_scratch, matrix_inverse);

  // Multiply the original matrix by the inverse matrix and store the result in matrix_scratch
  KokkosBlas::gemm(Kokkos::DefaultHostExecutionSpace(), "N", "N", 1.0, matrix_original, matrix_inverse, 0.0,
                   matrix_scratch);

  // Check that the result (stored in matrix_scratch) is the identity matrix
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      if (i == j) {
        EXPECT_NEAR(matrix_scratch(i, j), 1.0, 1.0e-10);
      } else {
        EXPECT_NEAR(matrix_scratch(i, j), 0.0, 1.0e-10);
      }
    }
  }
}

TEST(PeripheryTest, ReadWriteKokkosMatrixToFromFile) {
  // Test that we can write a Kokkos matrix to a file and read it back in

  // Generate a random matrix with random normally distributed entities with mean 0 and variance 1
  const int matrix_size = 10;
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> matrix("matrix", matrix_size, matrix_size);

  size_t seed = 1234;
  size_t counter = 0;
  openrand::Philox rng(seed, counter);
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      matrix(i, j) = rng.randn<double>();
    }
  }

  // Write the matrix to a file
  const std::string file_name = "ReadWriteKokkosMatrixToFromFile_matrix.dat";
  write_matrix_to_file(file_name, matrix);

  // Read the matrix back in
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> matrix_read("matrix_read", matrix_size, matrix_size);
  read_matrix_from_file(file_name, matrix_size, matrix_size, matrix_read);

  // Check that the matrix read in is the same as the original matrix, to machine precision
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      EXPECT_DOUBLE_EQ(matrix(i, j), matrix_read(i, j));
    }
  }
}

TEST(PeripheryTest, ReadWriteKokkosVectorToFromFile) {
  // Test that we can write a Kokkos vector to a file and read it back in

  // Generate a random vector with random normally distributed entities with mean 0 and variance 1
  const int vector_size = 10;
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> vector("vector", vector_size);

  size_t seed = 1234;
  size_t counter = 0;
  openrand::Philox rng(seed, counter);
  for (int i = 0; i < vector_size; ++i) {
    vector(i) = rng.randn<double>();
  }

  // Write the vector to a file
  const std::string file_name = "ReadWriteKokkosVectorToFromFile_vector.dat";
  write_vector_to_file(file_name, vector);

  // Read the vector back in
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> vector_read("vector_read", vector_size);
  read_vector_from_file(file_name, vector_size, vector_read);

  // Check that the vector read in is the same as the original vector, to machine precision
  for (int i = 0; i < vector_size; ++i) {
    EXPECT_DOUBLE_EQ(vector(i), vector_read(i));
  }
}

TEST(PeripheryTest, StokesDoubleLayerConstantForce) {
  // We have the following identity for flow exterior to a surface D:
  //   int_{partial D} n(y) dot T(x - y) dS(y) = I / 2
  // This is a commonly used to check the convergence of the double layer integral.

  // Setup the convergence study
  const double sphere_radius = 12.34;
  const double viscosity = 1.0;

  const QuadGenerationFunc quad_gen = SphereQuadFunctor(sphere_radius);
  const QuadVectorFunc in_field_gen =
      []([[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights) {
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field("input_field", 3 * weights.extent(0));
        Kokkos::deep_copy(input_field, 1.0);
        return input_field;
      };
  const QuadVectorFunc expected_results_gen =
      []([[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights) {
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results("expected_results",
                                                                                       3 * weights.extent(0));
        Kokkos::deep_copy(expected_results, 0.5);
        return expected_results;
      };

  // Run the convergence study for the stokes double layer matrix
  const ConvergenceResults matrix_results = perform_convergence_study(
      viscosity, quad_gen, apply_stokes_double_layer_matrix_wrapper, in_field_gen, expected_results_gen);

  // Run the same convergence study for the stokes double layer kernel
  const ConvergenceResults kernel_results = perform_convergence_study(
      viscosity, quad_gen, apply_stokes_double_layer_kernel_wrapper, in_field_gen, expected_results_gen);

  // Run the convergence study for the stokes double layer matrix with singularity subtraction
  const QuadVectorFunc new_expected_results_gen =
      []([[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights) {
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results("new_expected_results",
                                                                                       3 * weights.extent(0));
        Kokkos::deep_copy(expected_results, 0.0);
        return expected_results;
      };
  const ConvergenceResults matrix_ss_results = perform_convergence_study(
      viscosity, quad_gen, apply_stokes_double_layer_matrix_ss_wrapper, in_field_gen, new_expected_results_gen);

  // Run the convergence study for the stokes double layer kernel with singularity subtraction
  const ConvergenceResults kernel_ss_results = perform_convergence_study(
      viscosity, quad_gen, apply_stokes_double_layer_kernel_ss_wrapper, in_field_gen, new_expected_results_gen);

  // For now, just print the results
  std::cout << "matrix_results.slope = " << matrix_results.slope << std::endl;
  for (size_t i = 0; i < matrix_results.num_quadrature_points.size(); ++i) {
    std::cout << "matrix_results.num_quadrature_points[" << i << "] =" << matrix_results.num_quadrature_points[i]
              << ", matrix_results.error[" << i << "] = " << matrix_results.error[i] << std::endl;
  }
  std::cout << "kernel_results.slope = " << kernel_results.slope << std::endl;
  for (size_t i = 0; i < kernel_results.num_quadrature_points.size(); ++i) {
    std::cout << "kernel_results.num_quadrature_points[" << i << "] =" << kernel_results.num_quadrature_points[i]
              << ", kernel_results.error[" << i << "] = " << kernel_results.error[i] << std::endl;
  }
  std::cout << "matrix_ss_results.slope = " << matrix_ss_results.slope << std::endl;
  for (size_t i = 0; i < matrix_ss_results.num_quadrature_points.size(); ++i) {
    std::cout << "matrix_ss_results.num_quadrature_points[" << i << "] =" << matrix_ss_results.num_quadrature_points[i]
              << ", matrix_ss_results.error[" << i << "] = " << matrix_ss_results.error[i] << std::endl;
  }
  std::cout << "kernel_ss_results.slope = " << kernel_ss_results.slope << std::endl;
  for (size_t i = 0; i < kernel_ss_results.num_quadrature_points.size(); ++i) {
    std::cout << "kernel_ss_results.num_quadrature_points[" << i << "] =" << kernel_ss_results.num_quadrature_points[i]
              << ", kernel_ss_results.error[" << i << "] = " << kernel_ss_results.error[i] << std::endl;
  }
}
//@}

//! \name Periphery tests
//@{

//@}

}  // namespace

}  // namespace periphery

}  // namespace alens

}  // namespace mundy
