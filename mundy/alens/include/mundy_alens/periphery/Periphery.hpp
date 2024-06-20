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

#ifndef MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_PERIPHERY_HPP_
#define MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_PERIPHERY_HPP_

/* This class evaluates the fluid flow on some points within the domain induced by the no-slip confition on the
periphery. We do so via storing the precomputed inverse of the self-interaction matrix. Ths class is in charge
of writing out that inverse if it doesn't already exist, reading it into memory if it does, evaluating the surface
forces induced by external flow on the surface (by evaluating f = M^{-1}u), and then evaluating the fluid flow at points
within the domain induced by these surface forces.

The periphery is described by a collection of nodes with inward-pointing surface normals and predefined quadrature
weights. We are not in charge of computing these quantities.

We will write this entire class with raw Kokkos.

Because the Periphery itself is only designed for smallish-scale problems (N<5000), we won't worry about distributing
the periphery across multiple devices. After all, a 5000x5000 matrix is only 200MB, which is small enough to fit on
a single GPU.


*/

// C++ core
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// Boost (because it's cleaner than zlib)
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filtering_stream.hpp>

// Kokkos and Kokkos-Kernels
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace periphery {

/// \brief Invert and LU decompose a dense square matrix of size n x n
///
/// \param space The execution space
/// \param[in & out] matrix The matrix to invert. On exit, the matrix is replaced with its LU decomposition
/// \param[out] M_inv The inverse of the matrix.
template <class ExecutionSpace, class MemorySpace, class Layout>
void matrix_inverse(const ExecutionSpace &space, const Kokkos::View<double **, Layout, MemorySpace> &matrix,
                    const Kokkos::View<double **, Layout, MemorySpace> &matrix_inv) {
  // Check the input sizes
  const size_t matrix_size = matrix.extent(0);
  MUNDY_THROW_ASSERT(matrix.extent(1) == matrix_size, std::invalid_argument, "matrix_inverse: matrix must be square.");
  MUNDY_THROW_ASSERT((matrix_inv.extent(0) == matrix_size) && (matrix_inv.extent(1) == matrix_size),
                     std::invalid_argument,
                     "matrix_inverse: matrix_inv must be the same size as the matrix to invert.");

  // Create a view to store the pivots
  Kokkos::View<int *, Layout, MemorySpace> pivots("pivots", matrix_size);

  // Fill matrix_inv with the identity matrix
  Kokkos::deep_copy(matrix_inv, 0.0);
  Kokkos::parallel_for(
      "FillIdentity", Kokkos::RangePolicy<>(0, matrix_size), KOKKOS_LAMBDA(const size_t i) { matrix_inv(i, i) = 1.0; });

  // Solve the dense linear equation system M*X = I, which results in X = M^{-1}
  // On exist, M is replaced with its LU decomposition
  //           M_inv is replaced with the solution X = M^{-1}
  KokkosBlas::gesv(matrix, matrix_inv, pivots);
}

/// \brief Write a matrix to a file (uncompressed)
///
/// \param[in] filename The filename
/// \param[in] matrix_host The matrix to write (host)
template <typename MatrixDataType, class Layout>
void write_matrix_to_file_uncompressed(const std::string &filename,
                                       const Kokkos::View<MatrixDataType **, Layout, Kokkos::HostSpace> &matrix_host) {
  // Write the matrix to a file
  std::ofstream outfile(filename);
  if (!outfile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }
  outfile << matrix_host.extent(0) << " " << matrix_host.extent(1) << std::endl;
  for (size_t i = 0; i < matrix_host.extent(0); ++i) {
    for (size_t j = 0; j < matrix_host.extent(1); ++j) {
      outfile << matrix_host(i, j) << " ";
    }
    outfile << std::endl;
  }
  outfile.close();
}

/// \brief Read a matrix from a file (uncompressed)
///
/// \param[in] filename The filename
/// \param[out] matrix_host The matrix to read (host)
template <typename MatrixDataType, class Layout>
void read_matrix_from_file_uncompressed(const std::string &filename, const size_t expected_num_rows,
                                        const size_t expected_num_columns,
                                        Kokkos::View<MatrixDataType **, Layout, Kokkos::HostSpace> &matrix_host) {
  // Read the matrix from a file
  std::ifstream infile(filename);
  if (!infile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Parse the input
  size_t num_rows;
  size_t num_columns;
  infile >> num_rows >> num_columns;
  if ((num_rows != expected_num_rows) || (num_columns != expected_num_columns)) {
    std::cerr << "Matrix size mismatch: expected (" << expected_num_rows << ", " << expected_num_columns << "), got ("
              << num_rows << ", " << num_columns << ")" << std::endl;
    return;
  }
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      infile >> matrix_host(i, j);
    }
  }
}

/// \brief Write a matrix to a file
///
/// \param[in] filename The filename
/// \param[in] matrix_host The matrix to write (host)
template <typename MatrixDataType, class Layout>
void write_matrix_to_file_compressed(const std::string &filename,
                                     const Kokkos::View<MatrixDataType **, Layout, Kokkos::HostSpace> &matrix_host) {
  // Write the matrix to a file using zlib compression mediated by Boost
  // Use a buffer to avoid writing to the file one character at a time

  // Buffer the output
  std::ostringstream buffer;
  buffer << matrix_host.extent(0) << " " << matrix_host.extent(1) << std::endl;
  for (size_t i = 0; i < matrix_host.extent(0); ++i) {
    for (size_t j = 0; j < matrix_host.extent(1); ++j) {
      buffer << matrix_host(i, j) << " ";
    }
    buffer << std::endl;
  }

  // Perform the write
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }
  boost::iostreams::filtering_ostream out;
  out.push(boost::iostreams::zlib_compressor());
  out.push(outfile);

  std::istringstream in(buffer.str());
  boost::iostreams::copy(in, out);

  outfile.close();
}

/// \brief Read a matrix from a file
///
/// \param[in] filename The filename
/// \param[out] expected_num_rows The expected number of rows
/// \param[out] expected_num_columns The expected number of columns
/// \param[out] matrix_host The matrix to read (host)
template <typename MatrixDataType, class Layout>
void read_matrix_from_file_compressed(const std::string &filename, const size_t expected_num_rows,
                                      const size_t expected_num_columns,
                                      Kokkos::View<MatrixDataType **, Layout, Kokkos::HostSpace> &matrix_host) {
  // Read the matrix from a file using zlib decompression mediated by Boost
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  // Parse the input
  boost::iostreams::filtering_istream in;
  in.push(boost::iostreams::zlib_decompressor());
  in.push(infile);

  std::ostringstream out;
  boost::iostreams::copy(in, out);

  // Parse the buffer
  std::istringstream iss(out.str());
  size_t num_rows;
  size_t num_columns;
  iss >> num_rows >> num_columns;
  if ((num_rows != matrix_host.extent(0)) || (num_columns != matrix_host.extent(1))) {
    std::cerr << "Matrix size mismatch: expected (" << matrix_host.extent(0) << ", " << matrix_host.extent(1)
              << "), got (" << num_rows << ", " << num_columns << ")" << std::endl;
    return;
  }
  for (size_t i = 0; i < matrix_host.extent(0); ++i) {
    for (size_t j = 0; j < matrix_host.extent(1); ++j) {
      iss >> matrix_host(i, j);
    }
  }
}

/// \brief Apply the RPY kernel to map source forces to target velocitiess: u_target = M f_source (summed into target)
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_values The source values (size num_source_points x 4 containing the force and radius)
/// \param[out] target_values The target values (size num_target_points x 6 containing the velocity and laplacian)
template <class ExecutionSpace, class MemorySpace, class Layout>
void apply_rpy_kernel(const ExecutionSpace &space, const double viscosity, const size_t num_source_points,
                      const size_t num_target_points,
                      const Kokkos::View<double **, Layout, MemorySpace> &source_positions,
                      const Kokkos::View<double **, Layout, MemorySpace> &target_positions,
                      const Kokkos::View<double **, Layout, MemorySpace> &source_values,
                      const Kokkos::View<double **, Layout, MemorySpace> &target_values) {
  // Zero out the target values
  Kokkos::deep_copy(target_values, 0.0);

  // Launch the parallel kernel
  Kokkos::parallel_for(
      "RPY", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_target_points, num_source_points}),
      KOKKOS_LAMBDA(const size_t t, const size_t s) {
        // Skip self-interaction
        if (t == s) {
          return;
        }

        // Compute the distance vector
        const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
        const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
        const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

        const double fx = source_values(4 * s + 0);
        const double fy = source_values(4 * s + 1);
        const double fz = source_values(4 * s + 2);
        const double a = source_values(4 * s + 3);

        const double a2_over_three = (1.0 / 3.0) * a * a;
        const double r2 = dx * dx + dy * dy + dz * dz;
        const double rinv = 1.0 / std::sqrt(r2);
        const double rinv3 = rinv * rinv * rinv;
        const double rinv5 = rinv * rinv * rinv3;
        const double fdotr = fx * dx + fy * dy + fz * dz;

        const double three_fdotr_rinv5 = 3 * fdotr * rinv5;
        const double cx = fx * rinv3 - three_fdotr_rinv5 * dx;
        const double cy = fy * rinv3 - three_fdotr_rinv5 * dy;
        const double cz = fz * rinv3 - three_fdotr_rinv5 * dz;

        const double fdotr_rinv3 = fdotr * rinv3;

        // Velocity
        Kokkos::atomic_add(&target_values(6 * t + 0), fx * rinv + dx * fdotr_rinv3 + a2_over_three * cx);
        Kokkos::atomic_add(&target_values(6 * t + 1), fy * rinv + dy * fdotr_rinv3 + a2_over_three * cy);
        Kokkos::atomic_add(&target_values(6 * t + 2), fz * rinv + dz * fdotr_rinv3 + a2_over_three * cz);

        // Laplacian
        Kokkos::atomic_add(&target_values(6 * t + 3), 2 * cx);
        Kokkos::atomic_add(&target_values(6 * t + 4), 2 * cy);
        Kokkos::atomic_add(&target_values(6 * t + 5), 2 * cz);
      });

  // Apply the scale factor
  const double scale_factor = 1.0 / (8.0 * M_PI);
  KokkosBlas::scal(u, scale_factor);
}

/// \brief Apply the stokes double layer kernel to map source forces to target velocitiess: u_target = M f_source
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_normals The normals of the source points (size num_source_points x 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
/// \param[in] source_forces The vector to apply the self-interaction matrix to (size num_nodes x 3)
/// \param[out] target_velocities The result of applying the self-interaction matrix to f (size num_nodes x 3)
template <class ExecutionSpace, class MemorySpace, class Layout>
void apply_stokes_double_layer_kernel(const ExecutionSpace &space, const double viscosity,
                                      const size_t num_source_points, const size_t num_target_points,
                                      const Kokkos::View<double **, Layout, MemorySpace> &source_positions,
                                      const Kokkos::View<double **, Layout, MemorySpace> &target_positions,
                                      const Kokkos::View<double **, Layout, MemorySpace> &source_normals,
                                      const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights,
                                      const Kokkos::View<double **, Layout, MemorySpace> &source_forces,
                                      const Kokkos::View<double **, Layout, MemorySpace> &target_velocities) {
  // Zero out the target velocities
  Kokkos::deep_copy(target_velocities, 0.0);

  // Launch the parallel kernel
  // For us the double layer potential is (following SkellySim blindly):
  // f_dl(i * 3 + j, node) = 2.0 * viscosity * source_normals(i, node) * surface_forces(j, node) *
  // quadrature_weights(i);
  Kokkos::parallel_for(
      "StokesDoubleLayerKernel", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_target_points, num_source_points}),
      KOKKOS_LAMBDA(const size_t t, const size_t s) {
        // Skip self-interaction
        if (t == s) {
          return;
        }

        // Compute the distance vector
        const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
        const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
        const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

        // Compute the distance squared and its inverse powers
        const double dr2 = dx * dx + dy * dy + dz * dz;
        const double rinv = 1.0 / sqrt(dr2);
        const double rinv2 = rinv * rinv;
        const double rinv5 = dr2 ? rinv * rinv2 * rinv2 : 0.0;

        // Compute the double layer potential
        const double sxx =
            2.0 * viscosity * source_normals(3 * s + 0) * source_forces(3 * s + 0) * quadrature_weights(s);
        const double sxy =
            2.0 * viscosity * source_normals(3 * s + 0) * source_forces(3 * s + 1) * quadrature_weights(s);
        const double sxz =
            2.0 * viscosity * source_normals(3 * s + 0) * source_forces(3 * s + 2) * quadrature_weights(s);
        const double syx =
            2.0 * viscosity * source_normals(3 * s + 1) * source_forces(3 * s + 0) * quadrature_weights(s);
        const double syy =
            2.0 * viscosity * source_normals(3 * s + 1) * source_forces(3 * s + 1) * quadrature_weights(s);
        const double syz =
            2.0 * viscosity * source_normals(3 * s + 1) * source_forces(3 * s + 2) * quadrature_weights(s);
        const double szx =
            2.0 * viscosity * source_normals(3 * s + 2) * source_forces(3 * s + 0) * quadrature_weights(s);
        const double szy =
            2.0 * viscosity * source_normals(3 * s + 2) * source_forces(3 * s + 1) * quadrature_weights(s);
        const double szz =
            2.0 * viscosity * source_normals(3 * s + 2) * source_forces(3 * s + 2) * quadrature_weights(s);

        double coeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
        coeff += (sxy + syx) * dx * dy;
        coeff += (sxz + szx) * dx * dz;
        coeff += (syz + szy) * dy * dz;
        coeff *= -3.0 * rinv5;

        Kokkos::atomic_add(&target_velocities(3 * t + 0), dx * coeff);
        Kokkos::atomic_add(&target_velocities(3 * t + 1), dy * coeff);
        Kokkos::atomic_add(&target_velocities(3 * t + 2), dz * coeff);
      });

  // Apply the scale factor
  const double scale_factor = 1.0 / (8.0 * M_PI);
  KokkosBlas::scal(u, scale_factor);
}

/// \brief Fill the stokes_double_layer matrix
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_normals The normals of the source points (size num_source_points x 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
template <class ExecutionSpace, class MemorySpace, class Layout>
void fill_stokes_double_layer_matrix(const ExecutionSpace &space, const double viscosity,
                                     const size_t num_source_points, const size_t num_target_points,
                                     const Kokkos::View<double **, Layout, MemorySpace> &source_positions,
                                     const Kokkos::View<double **, Layout, MemorySpace> &target_positions,
                                     const Kokkos::View<double **, Layout, MemorySpace> &source_normals,
                                     const Kokkos::View<double *, Layout, MemorySpace> &quadrature_weights,
                                     const Kokkos::View<double **, Layout, MemorySpace> &M) {
  // Compute the scale factor
  const double scale_factor = 1.0 / (8.0 * M_PI);

  // Launch the parallel kernel
  // For us the double layer potential is (following SkellySim blindly):
  // f_dl(i * 3 + j, node) = 2.0 * viscosity * source_normals(i, node) * surface_forces(j, node) *
  // quadrature_weights(i);
  Kokkos::parallel_for(
      "StokesDoubleLayerMatrix", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_target_points, num_source_points}),
      KOKKOS_LAMBDA(const size_t t, const size_t s) {
        // Compute the distance vector
        const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
        const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
        const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

        // Compute the distance squared and its inverse powers
        const double dr2 = dx * dx + dy * dy + dz * dz;
        const double rinv = 1.0 / sqrt(dr2);
        const double rinv2 = rinv * rinv;
        const double rinv5 = dr2 ? rinv * rinv2 * rinv2 : 0.0;

        // Compute the double layer potential
        const double sxx = 2.0 * viscosity * source_normals(3 * s + 0) * quadrature_weights(s);
        const double sxy = 2.0 * viscosity * source_normals(3 * s + 0) * quadrature_weights(s);
        const double sxz = 2.0 * viscosity * source_normals(3 * s + 0) * quadrature_weights(s);
        const double syx = 2.0 * viscosity * source_normals(3 * s + 1) * quadrature_weights(s);
        const double syy = 2.0 * viscosity * source_normals(3 * s + 1) * quadrature_weights(s);
        const double syz = 2.0 * viscosity * source_normals(3 * s + 1) * quadrature_weights(s);
        const double szx = 2.0 * viscosity * source_normals(3 * s + 2) * quadrature_weights(s);
        const double szy = 2.0 * viscosity * source_normals(3 * s + 2) * quadrature_weights(s);
        const double szz = 2.0 * viscosity * source_normals(3 * s + 2) * quadrature_weights(s);

        // coeff is just a vector dotted with f(j, :).
        // Lets compute that vector
        double coeff_vec0 = sxx * dx * dx + syx * dx * dy + szx * dx * dz;
        double coeff_vec1 = syy * dy * dy + sxy * dx * dy + szy * dy * dz;
        double coeff_vec2 = szz * dz * dz + sxz * dx * dz + syz * dy * dz;

        // We can then express u as a 3 x 3 matrix times f(j, :)
        // That matrix is just the outer product of coeff_vec with -3 dr / r^5
        // M (local) =
        //    [[dx * coeff_vec[0], dx * coeff_vec[1], dx * coeff_vec[2]],
        //     [dy * coeff_vec[0], dy * coeff_vec[1], dy * coeff_vec[2]],
        //     [dz * coeff_vec[0], dz * coeff_vec[1], dz * coeff_vec[2]]]
        coeff_vec0 *= -scale_factor * 3.0 * rinv5;
        coeff_vec1 *= -scale_factor * 3.0 * rinv5;
        coeff_vec2 *= -scale_factor * 3.0 * rinv5;
        M(t * 3 + 0, s * 3 + 0) = dx * coeff_vec0;
        M(t * 3 + 0, s * 3 + 1) = dx * coeff_vec1;
        M(t * 3 + 0, s * 3 + 2) = dx * coeff_vec2;
        M(t * 3 + 1, s * 3 + 0) = dy * coeff_vec0;
        M(t * 3 + 1, s * 3 + 1) = dy * coeff_vec1;
        M(t * 3 + 1, s * 3 + 2) = dy * coeff_vec2;
        M(t * 3 + 2, s * 3 + 0) = dz * coeff_vec0;
        M(t * 3 + 2, s * 3 + 1) = dz * coeff_vec1;
        M(t * 3 + 2, s * 3 + 2) = dz * coeff_vec2;
      });
}

class Periphery {
 public:
  //! \name Types
  //@{

  using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
  using DeviceMemorySpace = typename DeviceExecutionSpace::memory_space;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  Periphery() = delete;

  /// \brief No copy constructor
  Periphery(const Periphery &) = delete;

  /// \brief No copy assignment
  Periphery &operator=(const Periphery &) = delete;

  /// \brief Default move constructor
  Periphery(Periphery &&) = default;

  /// \brief Default move assignment
  Periphery &operator=(Periphery &&) = default;

  /// \brief Destructor
  ~Periphery() = default;

  /// \brief Full constructor from Kokkos views
  Periphery(const size_t num_surface_nodes, const double viscosity,
            const Kokkos::View<double *, DeviceMemorySpace> &surface_positions,
            const Kokkos::View<double *, DeviceMemorySpace> &surface_normals,
            const Kokkos::View<double *, DeviceMemorySpace> &quadrature_weights)
      : is_valid_(false),
        num_surface_nodes_(num_surface_nodes),
        viscosity_(viscosity),
        surface_positions_(surface_positions),
        surface_normals_(surface_normals),
        quadrature_weights_(quadrature_weights),
        M_inv_("M_inv", 3 * num_surface_nodes_, 3 * num_surface_nodes_) {
    // Initialize host Kokkos mirrors
    host_surface_positions_ = Kokkos::create_mirror_view(surface_positions_);
    host_surface_normals_ = Kokkos::create_mirror_view(surface_normals_);
    host_quadrature_weights_ = Kokkos::create_mirror_view(quadrature_weights_);
    host_M_inv_ = Kokkos::create_mirror_view(M_inv_);
  }

  /// \brief Full constructor from flattened arrays
  Periphery(const size_t num_surface_nodes, const double viscosity, const double *surface_positions,
            const double *surface_normals, const double *quadrature_weights)
      : is_valid_(false),
        num_surface_nodes_(num_surface_nodes),
        viscosity_(viscosity),
        surface_positions_("surface_positions", num_surface_nodes_, 3),
        surface_normals_("surface_normals", num_surface_nodes_, 3),
        quadrature_weights_("quadrature_weights", num_surface_nodes_),
        M_inv_("M_inv", 3 * num_surface_nodes_, 3 * num_surface_nodes_) {
    // Initialize host Kokkos mirrors
    surface_positions_host_ = Kokkos::create_mirror_view(surface_positions_);
    surface_normals_host_ = Kokkos::create_mirror_view(surface_normals_);
    quadrature_weights_host_ = Kokkos::create_mirror_view(quadrature_weights_);
    M_inv_host_ = Kokkos::create_mirror_view(M_inv_);

    // Copy data from the arrays to the Kokkos views
    for (size_t i = 0; i < num_surface_nodes_; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const size_t idx = 3 * i + j;
        surface_positions_host_(idx) = node_positions[idx];
        surface_normals_host_(idx) = surface_normals[idx];
      }
      quadrature_weights_host_(i) = quadrature_weights[i];
    }
  }
  //@}

  //! \name Public member functions
  //@{

  void precompute(const std::string &precomputed_matrix_filename, const bool &write_to_file = true) {
    // Check if the precomputed matrix exists
    std::ifstream infile(precomputed_matrix_filename);
    if (infile.good()) {
      // Read the precomputed inverse matrix from the file
      read_matrix_from_file(precomputed_matrix_filename, 3 * num_surface_nodes_, 3 * num_surface_nodes_, M_inv_host_);
      Kokkos::deep_copy(M_inv_, M_inv_host_);
    } else {
      // Fill the self-interaction matrix at temporary storage
      Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> M("M", 3 * num_surface_nodes_,
                                                                       3 * num_surface_nodes_);
      fill_self_interaction_matrix(DeviceExecutionSpace(), num_surface_nodes_, viscosity_, surface_positions_,
                                   surface_normals_, quadrature_weights_, M);

      // Now invert the matrix and store the result
      matrix_inverse(DeviceExecutionSpace(), num_surface_nodes_, M, M_inv_);

      if (write_to_file) {
        // Write the precomputed matrix to a file
        Kokkos::deep_copy(M_inv_host_, M_inv_);
        write_matrix_to_file_compressed(precomputed_matrix_filename, M_inv_host_);
      }
    }
    is_valid_ = true;
  }

  /// \brief Compute the surface forces induced by external flow on the surface
  ///
  /// \param[in] external_flow_velocity The external flow velocity (size num_nodes x 3)
  /// \param[out] surface_forces The surface forces induced by enforcing no-slip on the surface (size num_nodes x 3)
  void compute_surface_forces(
      const Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> &external_flow_velocity,
      Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> &surface_forces) {
    // Check if the periphery is in a valid state
    if (!is_valid_) {
      std::cerr << "Periphery is not in a valid state" << std::endl;
      return;
    }

    // Apply the inverse of the self-interaction matrix to the external flow velocity
    KokkosBlas::gemv(DeviceExecutionSpace(), 1.0, M_inv_, external_flow_velocity, 0.0, surface_forces);
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of nodes
  ///
  /// \return The number of nodes
  size_t get_num_nodes() const {
    return num_surface_nodes_;
  }

  /// \brief Get the viscosity
  ///
  /// \return The viscosity
  double get_viscosity() const {
    return viscosity_;
  }

  /// \brief Get if the current periphery is in a valid state or not
  ///
  /// \return Whether the periphery is valid or not
  bool is_valid() const {
    return is_valid_;
  }

  /// \brief Get the node positions
  ///
  /// \return The node positions
  const Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> &get_node_positions() const {
    return surface_positions_;
  }

  /// \brief Get the surface normals
  ///
  /// \return The surface normals
  const Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> &get_surface_normals() const {
    return surface_normals_;
  }

  /// \brief Get the surface forces
  ///
  /// \return The surface forces
  const Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> &get_surface_forces() const {
    return surface_forces_;
  }

  /// \brief Get the quadrature weights
  ///
  /// \return The quadrature weights
  const Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &get_quadrature_weights() const {
    return quadrature_weights_;
  }

  /// \brief Get the inverse of the self-interaction matrix
  ///
  /// \return The inverse of the self-interaction matrix
  const Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> &get_M_inv() const {
    return M_inv_;
  }
  //@}

 private:
  //! \name Private member variables
  //@{

  bool is_valid_;             //!< Whether the inverse of the self-interaction matrix is valid or not
  size_t num_surface_nodes_;  //!< The number of nodes
  double viscosity_;          //!< The viscosity

  // Host Kokkos views
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> node_positions_host_;   //!< The node positions (host)
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_normals_host_;  //!< The surface normals (host)
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>
      surface_forces_host_;  //!< The unknown surface forces (host)
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>
      quadrature_weights_host_;  //!< The quadrature weights (host)
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>
      M_inv_host_;  //!< The inverse of the self-interaction matrix (host)

  // Device Kokkos views
  Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> surface_positions_;  //!< The node positions (device)
  Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> surface_normals_;    //!< The surface normals (device)
  Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace>
      surface_forces_;  //!< The unknown surface forces (device)
  Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace>
      quadrature_weights_;  //!< The quadrature weights (device)
  Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace>
      &M_inv_;  //!< The inverse of the self-interaction matrix (device)
  //@}
};  // class Periphery

}  // namespace periphery

#endif  // MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_PERIPHERY_HPP_
