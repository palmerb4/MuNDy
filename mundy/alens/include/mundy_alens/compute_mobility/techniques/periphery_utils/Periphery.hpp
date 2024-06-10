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

#ifndef MUNDY_ALENS_COMPUTE_MOBILITY_PERIPHERY_UTILS_PERIPHERY_HPP_
#define MUNDY_ALENS_COMPUTE_MOBILITY_PERIPHERY_UTILS_PERIPHERY_HPP_

// C++ core
#include <iostream>
#include <msgpack.hpp>
#include <toml.hpp>
#include <vector>

// External libs

// Order matters here. These must come before including Eigen
#ifndef EIGEN_MATRIX_PLUGIN
#define EIGEN_MATRIX_PLUGIN <mundy_alens/compute_mobility/tequniues/periphery_utils/eigen_matrix_plugin.h>
#endif

#ifndef EIGEN_QUATERNION_PLUGIN
#define EIGEN_QUATERNION_PLUGIN <mundy_alens/compute_mobility/tequniues/periphery_utils/eigen_quaternion_plugin.h>
#endif

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Mundy
#include <mundy_alens/compute_mobility/tequniues/periphery_utils/kernels.hpp>

namespace mundy {

namespace alens {

namespace compute_mobility {

namespace periphery_utils {

/// Class to represent the containing boundary of the simulated system
///
/// There should be only periphery per system. The periphery, which is composed of smaller
/// discretized nodes, is distributed across all MPI ranks.
class Periphery {
 public:
  Periphery() = default;
  Periphery(const std::string &precompute_file, const std::string &pair_evaluator,
            const int periphery_stresslet_multipole_order, const int periphery_stresslet_max_points);

  Eigen::MatrixXd flow(const Eigen::Ref<const Eigen::MatrixXd> &trg, const Eigen::Ref<const Eigen::MatrixXd> &density, double eta) const;

  /// \brief Get the number of nodes local to the MPI rank
  int get_local_node_count() const {
    return M_inv_.rows() / 3;
  };

  /// \brief Get the rank local size of shell's contribution to the matrix problem solution
  int get_local_solution_size() const {
    return M_inv_.rows();
  };

  /// \brief Get the global size of the shell's contribution to the matrix problem solution
  int get_global_solution_size() const {
    return M_inv_.cols();
  };

  Eigen::MatrixXd get_local_node_positions() const {
    return node_pos_;
  };

  void update_RHS(const Eigen::Ref<const Eigen::MatrixXd> &v_on_shell);

  Eigen::VectorXd get_RHS() const {
    return RHS_;
  };

  Eigen::VectorXd apply_preconditioner(const Eigen::Ref<const Eigen::VectorXd> &x) const;
  Eigen::VectorXd matvec(const Eigen::Ref<const Eigen::VectorXd> &x_local, const Eigen::Ref<const Eigen::MatrixXd> &v_local) const;

  /// pointer to FMM object (pointer to avoid constructing object with empty Periphery)
  kernels::Evaluator stresslet_kernel_;
  Eigen::MatrixXd M_inv_;                             ///< Process local elements of inverse matrix
  Eigen::MatrixXd stresslet_plus_complementary_;      ///< Process local elements of stresslet tensor
  Eigen::MatrixXd node_pos_ = Eigen::MatrixXd(3, 0);  ///< [3xn_nodes_local] matrix representing node positions
  Eigen::MatrixXd node_normal_;         ///< [3xn_nodes_local] matrix representing node normal vectors (inward facing)
  Eigen::VectorXd quadrature_weights_;  ///< [n_nodes] array of 'far-field' quadrature weights
  Eigen::VectorXd RHS_;                 ///< [ 3 * n_nodes ] Current 'right-hand-side' for matrix formulation of solver
  Eigen::VectorXd solution_vec_;        ///< [ 3 * n_nodes ] Current 'solution' for matrix formulation of solver

  /// MPI_WORLD_SIZE array that specifies node_counts_[i] = number_of_nodes_on_rank_i*3
  Eigen::VectorXi node_counts_;

  /// MPI_WORLD_SIZE+1 array that specifies node displacements. Is essentially the CDF of node_counts_
  Eigen::VectorXi node_displs_;

  /// MPI_WORLD_SIZE array that specifies quad_counts_[i] = number_of_nodes_on_rank_i
  Eigen::VectorXi quad_counts_;

  /// MPI_WORLD_SIZE+1 array that specifies quadrature displacements. Is essentially the CDF of quad_counts_
  Eigen::VectorXi quad_displs_;

  /// MPI_WORLD_SIZE array that specifies row_counts_[i] = 3 * n_nodes_global_ * number_of_nodes_on_rank_i
  Eigen::VectorXi row_counts_;

  /// MPI_WORLD_SIZE+1 array that specifies row displacements. Is essentially the CDF of row_counts_
  Eigen::VectorXi row_displs_;

  void step(const Eigen::Ref<const Eigen::VectorXd> &solution) {
    solution_vec_ = solution;
  }
  void set_evaluator(const std::string &evaluator);

  bool is_active() const {
    return n_nodes_global_;
  }

  virtual std::tuple<double, double, double> get_dimensions() {
    if (!n_nodes_global_) return {0.0, 0.0, 0.0};
    throw std::runtime_error("Point cloud interaction undefined on base Periphery class\n");
  }

  int n_nodes_global_ = 0;  ///< Number of nodes across ALL MPI ranks
#ifdef CELLSIMULATOR_DEBUG
  MSGPACK_DEFINE_MAP(solution_vec_, RHS_);
#else
  MSGPACK_DEFINE_MAP(solution_vec_);
#endif

 protected:
  int periphery_stresslet_multipole_order_;
  int periphery_stresslet_max_points_;
  int world_size_;
  int world_rank_ = -1;
};  // Periphery

class SphericalPeriphery : public Periphery {
 public:
  double radius_;
  SphericalPeriphery(const toml::value &periphery_table, const std::string &pair_evaluator,
                     const int periphery_stresslet_multipole_order, const int periphery_stresslet_max_points)
      : Periphery(periphery_table, pair_evaluator, periphery_stresslet_multipole_order,
                  periphery_stresslet_max_points) {
    radius_ = toml::find_or<double>(periphery_table, "radius", 0.0);
  };

  virtual std::tuple<double, double, double> get_dimensions() {
    return {radius_, radius_, radius_};
  };
};  // SphericalPeriphery

class EllipsoidalPeriphery : public Periphery {
 public:
  double a_;
  double b_;
  double c_;
  EllipsoidalPeriphery(const toml::value &periphery_table, const std::string &pair_evaluator,
                       const int periphery_stresslet_multipole_order, const int periphery_stresslet_max_points)
      : Periphery(periphery_table, pair_evaluator, periphery_stresslet_multipole_order,
                  periphery_stresslet_max_points) {
    a_ = toml::find_or<double>(periphery_table, "a", 0.0);
    b_ = toml::find_or<double>(periphery_table, "b", 0.0);
    c_ = toml::find_or<double>(periphery_table, "c", 0.0);
  };

  virtual std::tuple<double, double, double> get_dimensions() {
    return {a_, b_, c_};
  };
};  // EllipsoidalPeriphery

class GenericPeriphery : public Periphery {
 public:
  double a_;
  double b_;
  double c_;
  GenericPeriphery(const toml::value &periphery_table, const std::string &pair_evaluator,
                   const int periphery_stresslet_multipole_order, const int periphery_stresslet_max_points)
      : Periphery(periphery_table, pair_evaluator, periphery_stresslet_multipole_order,
                  periphery_stresslet_max_points) {
    double a = node_pos_.row(0).array().abs().maxCoeff();
    double b = node_pos_.row(1).array().abs().maxCoeff();
    double c = node_pos_.row(2).array().abs().maxCoeff();
    MPI_Allreduce(&a, &a_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&b, &b_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&c, &c_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  };

  virtual std::tuple<double, double, double> get_dimensions() {
    return {a_, b_, c_};
  };
};  // GenericPeriphery

}  // namespace periphery_utils

}  // namespace compute_mobility

}  // namespace alens

}  // namespace mundy

#endif  // MUNDY_ALENS_COMPUTE_MOBILITY_PERIPHERY_UTILS_PERIPHERY_HPP_
