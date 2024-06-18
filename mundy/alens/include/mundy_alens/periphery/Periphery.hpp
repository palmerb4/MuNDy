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
  Periphery(const Kokkos::View<double *, DeviceMemorySpace> &node_positions,
            const Kokkos::View<double *, DeviceMemorySpace> &surface_normals,
            const Kokkos::View<double *, DeviceMemorySpace> &quadrature_weights, const size_t num_nodes,
            const double viscosity)
      : is_valid_(false),
        num_nodes_(num_nodes),
        viscosity_(viscosity),
        node_positions_(node_positions),
        surface_normals_(surface_normals),
        quadrature_weights_(quadrature_weights),
        M_inv_("M_inv", 3 * num_nodes, 3 * num_nodes) {
    // Initialize host Kokkos mirrors
    host_node_positions_ = Kokkos::create_mirror_view(node_positions_);
    host_surface_normals_ = Kokkos::create_mirror_view(surface_normals_);
    host_quadrature_weights_ = Kokkos::create_mirror_view(quadrature_weights_);
    host_M_inv_ = Kokkos::create_mirror_view(M_inv_);
  }

  /// \brief Full constructor from flattened arrays
  Periphery(const double *node_positions, const double *surface_normals, const double *quadrature_weights,
            const size_t num_nodes, const double viscosity)
      : is_valid_(false),
        num_nodes_(num_nodes),
        viscosity_(viscosity),
        node_positions_("node_positions", num_nodes, 3),
        surface_normals_("surface_normals", num_nodes, 3),
        quadrature_weights_("quadrature_weights", num_nodes),
        M_inv_("M_inv", 3 * num_nodes, 3 * num_nodes) {
    // Initialize host Kokkos mirrors
    host_node_positions_ = Kokkos::create_mirror_view(node_positions_);
    host_surface_normals_ = Kokkos::create_mirror_view(surface_normals_);
    host_quadrature_weights_ = Kokkos::create_mirror_view(quadrature_weights_);
    host_M_inv_ = Kokkos::create_mirror_view(M_inv_);

    // Copy data from the arrays to the Kokkos views
    for (size_t i = 0; i < num_nodes_; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const size_t idx = 3 * i + j;
        host_node_positions_(idx) = node_positions[idx];
        host_surface_normals_(idx) = surface_normals[idx];
      }
      host_quadrature_weights_(i) = quadrature_weights[i];
    }
  }
  //@}

  //! \name Public member functions
  //@{

  /// \brief Apply self-interaction matrix to a vector u = Mf
  ///
  /// \param[in] f The vector to apply the self-interaction matrix to (size num_nodes x 3)
  /// \param[out] u The result of applying the self-interaction matrix to f (size num_nodes x 3)
  void apply_self_interaction_matrix(const Kokkos::View<double *, DeviceMemorySpace, Kokkos::LayoutLeft> &f,
                                     Kokkos::View<double *, DeviceMemorySpace, Kokkos::LayoutLeft> &u) {
    // Capture the member variables by value. Use references to avoid copying the arrays.
    auto &node_positions = node_positions_;
    auto &surface_normals = surface_normals_;
    auto &quadrature_weights = quadrature_weights_;
    auto &num_nodes = num_nodes_;
    auto &viscosity = viscosity_;

    // Initialize the result vector
    Kokkos::deep_copy(u, 0.0);

    // Launch the parallel kernel
    // For us the double layer potential is (following SkellySim blindly):
    // f_dl(i * 3 + j, node) = 2.0 * viscosity * surface_normals(i, node) * surface_forces(j, node) *
    // quadrature_weights(i);
    Kokkos::parallel_for(
        "StokesDoubleLayer", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_nodes, num_nodes}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          // Compute the distance vector
          const double dr0 = node_positions(3 * i + 0) - node_positions(3 * j + 0);
          const double dr1 = node_positions(3 * i + 1) - node_positions(3 * j + 1);
          const double dr2 = node_positions(3 * i + 2) - node_positions(3 * j + 2);

          // Compute the distance squared and its inverse powers
          const double dr2 = dr0 * dr0 + dr1 * dr1 + dr2 * dr2;
          const double rinv = 1.0 / sqrt(dr2);
          const double rinv2 = rinv * rinv;
          const double rinv5 = dr2 ? rinv * rinv2 * rinv2 : 0.0;

          // Compute the double layer potential
          const double sxx = 2.0 * viscosity * surface_normals(3 * j + 0) * f(3 * j + 0) * quadrature_weights(j);
          const double sxy = 2.0 * viscosity * surface_normals(3 * j + 0) * f(3 * j + 1) * quadrature_weights(j);
          const double sxz = 2.0 * viscosity * surface_normals(3 * j + 0) * f(3 * j + 2) * quadrature_weights(j);
          const double syx = 2.0 * viscosity * surface_normals(3 * j + 1) * f(3 * j + 0) * quadrature_weights(j);
          const double syy = 2.0 * viscosity * surface_normals(3 * j + 1) * f(3 * j + 1) * quadrature_weights(j);
          const double syz = 2.0 * viscosity * surface_normals(3 * j + 1) * f(3 * j + 2) * quadrature_weights(j);
          const double szx = 2.0 * viscosity * surface_normals(3 * j + 2) * f(3 * j + 0) * quadrature_weights(j);
          const double szy = 2.0 * viscosity * surface_normals(3 * j + 2) * f(3 * j + 1) * quadrature_weights(j);
          const double szz = 2.0 * viscosity * surface_normals(3 * j + 2) * f(3 * j + 2) * quadrature_weights(j);

          double coeff = sxx * dr0 * dr0 + syy * dr1 * dr1 + szz * dr2 * dr2;
          coeff += (sxy + syx) * dr0 * dr1;
          coeff += (sxz + szx) * dr0 * dr2;
          coeff += (syz + szy) * dr1 * dr2;
          coeff *= -3.0 * rinv5;

          Kokkos::atomic_add(&u(3 * i + 0), dr0 * coeff);
          Kokkos::atomic_add(&u(3 * i + 1), dr1 * coeff);
          Kokkos::atomic_add(&u(3 * i + 2), dr2 * coeff);
        });

    // Apply the scale factor
    const double scale_factor = 1.0 / (8.0 * M_PI);
    KokkosBlas::scal(u, scale_factor);
  }

  /// \brief Fill the self-interaction matrix
  ///
  /// \param[in] M The self-interaction matrix (size 3 * num_nodes x 3 * num_nodes)
  void fill_self_interaction_matrix(const Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> &M) {
    // Capture the member variables by value. Use references to avoid copying the arrays.
    auto &node_positions = node_positions_;
    auto &surface_normals = surface_normals_;
    auto &quadrature_weights = quadrature_weights_;
    auto &num_nodes = num_nodes_;
    auto &viscosity = viscosity_;
    const double scale_factor = 1.0 / (8.0 * M_PI);

    // Initialize the result vector
    Kokkos::deep_copy(u, 0.0);

    // Launch the parallel kernel
    // For us the double layer potential is (following SkellySim blindly):
    // f_dl(i * 3 + j, node) = 2.0 * viscosity * surface_normals(i, node) * surface_forces(j, node) *
    // quadrature_weights(i);
    Kokkos::parallel_for(
        "SelfInteractionMatrix", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_nodes, num_nodes}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          // Compute the distance vector

          Index access needs to be based on left layout


          const double dr0 = node_positions(3 * i + 0) - node_positions(3 * j + 0);
          const double dr1 = node_positions(3 * i + 1) - node_positions(3 * j + 1);
          const double dr2 = node_positions(3 * i + 2) - node_positions(3 * j + 2);

          // Compute the distance squared and its inverse powers
          const double dr2 = dr0 * dr0 + dr1 * dr1 + dr2 * dr2;
          const double rinv = 1.0 / sqrt(dr2);
          const double rinv2 = rinv * rinv;
          const double rinv5 = dr2 ? rinv * rinv2 * rinv2 : 0.0;

          // Compute the double layer potential
          const double sxx = 2.0 * viscosity * surface_normals(3 * j + 0) * quadrature_weights(j);
          const double sxy = 2.0 * viscosity * surface_normals(3 * j + 0) * quadrature_weights(j);
          const double sxz = 2.0 * viscosity * surface_normals(3 * j + 0) * quadrature_weights(j);
          const double syx = 2.0 * viscosity * surface_normals(3 * j + 1) * quadrature_weights(j);
          const double syy = 2.0 * viscosity * surface_normals(3 * j + 1) * quadrature_weights(j);
          const double syz = 2.0 * viscosity * surface_normals(3 * j + 1) * quadrature_weights(j);
          const double szx = 2.0 * viscosity * surface_normals(3 * j + 2) * quadrature_weights(j);
          const double szy = 2.0 * viscosity * surface_normals(3 * j + 2) * quadrature_weights(j);
          const double szz = 2.0 * viscosity * surface_normals(3 * j + 2) * quadrature_weights(j);

          // coeff is just a vector dotted with f(j, :).
          // Lets compute that vector
          double coeff_vec0 = sxx * dr0 * dr0 + syx * dr0 * dr1 + szx * dr0 * dr2;
          double coeff_vec1 = syy * dr1 * dr1 + sxy * dr0 * dr1 + szy * dr1 * dr2;
          double coeff_vec2 = szz * dr2 * dr2 + sxz * dr0 * dr2 + syz * dr1 * dr2;

          // We can then express u as a 3 x 3 matrix times f(j, :)
          // That matrix is just the outer product of coeff_vec with -3 dr / r^5
          // M (local) =
          //    [[dr0 * coeff_vec[0], dr0 * coeff_vec[1], dr0 * coeff_vec[2]],
          //     [dr1 * coeff_vec[0], dr1 * coeff_vec[1], dr1 * coeff_vec[2]],
          //     [dr2 * coeff_vec[0], dr2 * coeff_vec[1], dr2 * coeff_vec[2]]]
          coeff_vec0 *= -scale_factor * 3.0 * rinv5;
          coeff_vec1 *= -scale_factor * 3.0 * rinv5;
          coeff_vec2 *= -scale_factor * 3.0 * rinv5;
          M(i * 3 + 0, j * 3 + 0) = dr0 * coeff_vec0;
          M(i * 3 + 0, j * 3 + 1) = dr0 * coeff_vec1;
          M(i * 3 + 0, j * 3 + 2) = dr0 * coeff_vec2;
          M(i * 3 + 1, j * 3 + 0) = dr1 * coeff_vec0;
          M(i * 3 + 1, j * 3 + 1) = dr1 * coeff_vec1;
          M(i * 3 + 1, j * 3 + 2) = dr1 * coeff_vec2;
          M(i * 3 + 2, j * 3 + 0) = dr2 * coeff_vec0;
          M(i * 3 + 2, j * 3 + 1) = dr2 * coeff_vec1;
          M(i * 3 + 2, j * 3 + 2) = dr2 * coeff_vec2;
        });
  }

  /// \brief Fill the inverse of the self-interaction matrix (M^{-1} of size 3 * num_nodes x 3 * num_nodes)
  void fill_inverse_self_interaction_matrix(
      const Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> &M_inv) {
    // Capture the member variables by value. Use references to avoid copying the arrays.
    auto &node_positions = node_positions_;
    auto &surface_normals = surface_normals_;
    auto &quadrature_weights = quadrature_weights_;
    auto &num_nodes = num_nodes_;

    // First, build the self-interaction matrix
    Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> M("M", 3 * num_nodes, 3 * num_nodes);
    build_self_interaction_matrix(M);

    // Next, compute the inverse of the self-interaction matrix
    Kokkos::View<int *, DeviceMemorySpace, Kokkos::LayoutLeft> pivots("pivots", 3 * num_nodes);

    // Fill M_inv with the identity matrix
    Kokkos::parallel_for(
        "FillIdentity", Kokkos::RangePolicy<>(0, 3 * num_nodes), KOKKOS_LAMBDA(const size_t i) { M_inv(i, i) = i; });

    // Solve the dense linear equation system M*X = I, which results in X = M^{-1}
    // On exist, M is replaced with its LU decomposition
    //           M_inv is replaced with the solution X = M^{-1}
    KokkosBlas::gesv(M, M_inv, pivots);
  }

  /// \brief Write a matrix to a file
  ///
  /// \param[in] filename The filename
  /// \param[in] matrix_host The matrix to write (host)
  void write_matrix_to_file(const std::string &filename,
                            const Kokkos::View<double **, Kokkos::HostSpace, Kokkos::LayoutLeft> &matrix_host) {
    // Write the matrix to a file using zlib compression mediated by Boost
    // Use a buffer to avoid writing to the file one character at a time

    // Buffer the output
    std::ostringstream buffer;
    buffer << num_nodes << std::endl;
    for (size_t i = 0; i < num_nodes_; ++i) {
      for (size_t j = 0; j < num_nodes_; ++j) {
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
  /// \param[out] matrix_host The matrix to read (host)
  void read_matrix_from_file(const std::string &filename,
                             Kokkos::View<double **, Kokkos::HostSpace, Kokkos::LayoutLeft> &matrix_host) {
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
    size_t num_nodes;
    iss >> num_nodes;
    for (size_t i = 0; i < num_nodes; ++i) {
      for (size_t j = 0; j < num_nodes; ++j) {
        iss >> matrix_host(i, j);
      }
    }
  }

  void precompute(const std::string &precomputed_matrix_filename, const bool &write_to_file = true) {
    // Check if the precomputed matrix exists
    std::ifstream infile(precomputed_matrix_filename);
    if (infile.good()) {
      // Read the precomputed inverse matrix from the file
      read_matrix_from_file(precomputed_matrix_filename, M_inv_host_);
      Kokkos::deep_copy(M_inv_, M_inv_host_);
    } else {
      // Fill the inverse of the self-interaction matrix
      fill_inverse_self_interaction_matrix(M_inv_);

      if (write_to_file) {
        // Write the precomputed matrix to a file
        Kokkos::deep_copy(M_inv_host_, M_inv_);
        write_matrix_to_file(precomputed_matrix_filename, M_inv_host_);
      }
    }
    is_valid_ = true;
  }

  /// \brief Compute the surface forces induced by external flow on the surface
  ///
  /// \param[in] u The external flow velocity (size num_nodes x 3)
  /// \param[out] f The surface forces induced by the external flow (size num_nodes x 3)
  void compute_surface_forces(const Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> &u,
                              Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> &f) {
    // Check if the periphery is in a valid state
    if (!is_valid_) {
      std::cerr << "Periphery is not in a valid state" << std::endl;
      return;
    }

    // Apply the inverse of the self-interaction matrix to the external flow velocity
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of nodes
  ///
  /// \return The number of nodes
  size_t get_num_nodes() const {
    return num_nodes_;
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
  const Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> &get_node_positions() const {
    return node_positions_;
  }

  /// \brief Get the surface normals
  ///
  /// \return The surface normals
  const Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> &get_surface_normals() const {
    return surface_normals_;
  }

  /// \brief Get the surface forces
  ///
  /// \return The surface forces
  const Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> &get_surface_forces() const {
    return surface_forces_;
  }

  /// \brief Get the quadrature weights
  ///
  /// \return The quadrature weights
  const Kokkos::View<double *, DeviceMemorySpace, Kokkos::LayoutLeft> &get_quadrature_weights() const {
    return quadrature_weights_;
  }

  /// \brief Get the inverse of the self-interaction matrix
  ///
  /// \return The inverse of the self-interaction matrix
  const Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> &get_M_inv() const {
    return M_inv_;
  }
  //@}

 private:
  //! \name Private member variables
  //@{

  bool is_valid_;     //!< Whether the inverse of the self-interaction matrix is valid or not
  size_t num_nodes_;  //!< The number of nodes
  double viscosity_;  //!< The viscosity

  // Host Kokkos views
  Kokkos::View<double **, Kokkos::HostSpace, Kokkos::LayoutLeft> node_positions_host_;   //!< The node positions (host)
  Kokkos::View<double **, Kokkos::HostSpace, Kokkos::LayoutLeft> surface_normals_host_;  //!< The surface normals (host)
  Kokkos::View<double **, Kokkos::HostSpace, Kokkos::LayoutLeft>
      surface_forces_host_;  //!< The unknown surface forces (host)
  Kokkos::View<double *, Kokkos::HostSpace, Kokkos::LayoutLeft>
      quadrature_weights_host_;  //!< The quadrature weights (host)
  Kokkos::View<double **, Kokkos::HostSpace, Kokkos::LayoutLeft>
      M_inv_host_;  //!< The inverse of the self-interaction matrix (host)

  // Device Kokkos views
  Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> node_positions_;   //!< The node positions (device)
  Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft> surface_normals_;  //!< The surface normals (device)
  Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft>
      surface_forces_;  //!< The unknown surface forces (device)
  Kokkos::View<double *, DeviceMemorySpace, Kokkos::LayoutLeft>
      quadrature_weights_;  //!< The quadrature weights (device)
  Kokkos::View<double **, DeviceMemorySpace, Kokkos::LayoutLeft>
      &M_inv_;  //!< The inverse of the self-interaction matrix (device)
  //@}
};  // class Periphery

#endif  // MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_PERIPHERY_HPP_
