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

// External libs
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Mundy
#include <mundy_alens/compute_mobility/tequniues/periphery_utils/Periphery.hpp>
#include <mundy_alens/compute_mobility/tequniues/periphery_utils/cnpy.hpp>
#include <mundy_alens/compute_mobility/tequniues/periphery_utils/kernels.hpp>
#include <mundy_alens/compute_mobility/tequniues/periphery_utils/utils.hpp>

/// \brief Apply preconditioner for Periphery component of 'x'.  While local input is supplied,
/// the preconditioner result requires the 'global' set of 'x' across all ranks, so an
/// Allgatherv is required
///
/// \param[in] x_local [3 * n_nodes_local] vector of 'x' local to this rank
/// \return [3 * n_nodes_local] vector of P * x_local
Eigen::VectorXd Periphery::apply_preconditioner(const Eigen::Ref<const Eigen::VectorXd> &x_local) const {
  if (!n_nodes_global_) {
    return Eigen::VectorXd();
  }

  assert(x_local.size() == get_local_solution_size());
  Eigen::VectorXd x_shell(3 * n_nodes_global_);
  MPI_Allgatherv(x_local.data(), node_counts_[world_rank_], MPI_DOUBLE, x_shell.data(), node_counts_.data(),
                 node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
  return M_inv_ * x_shell;
}

/// \brief Apply matvec for Periphery component of 'x'.  While local input is supplied,
/// the matvec result requires the 'global' set of 'x' across all ranks, so an
/// Allgatherv is required (though not on v)
///
/// \param[in] x_local [3 * n_nodes_local] vector of 'x' local to this rank
/// \param[in] v_local [3 * n_nodes_local] vector of velocities 'v' local to this rank
/// \return [3 * n_nodes_local] vector of A * x_local
Eigen::VectorXd Periphery::matvec(const Eigen::Ref<const Eigen::VectorXd> &x_local,
                                  const Eigen::Ref<const Eigen::MatrixXd> &v_local) const {
  if (!n_nodes_global_) {
    return Eigen::VectorXd();
  }

  assert(x_local.size() == get_local_solution_size());
  assert(v_local.size() == get_local_solution_size());
  Eigen::VectorXd x_shell(3 * n_nodes_global_);
  MPI_Allgatherv(x_local.data(), node_counts_[world_rank_], MPI_DOUBLE, x_shell.data(), node_counts_.data(),
                 node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
  return stresslet_plus_complementary_ * x_shell + Eigen::Map<const Eigen::VectorXd>(v_local.data(), v_local.size());
}

/// \brief Calculate velocity at target coordinates due to the periphery
/// Input:
/// \param[in] r_trg [3 x n_trg_local] matrix of target coordinates to evaluate the velocity at
/// \param[in] density [3 x n_nodes_local] matrix of node source strengths
/// \param[in] eta fluid viscosity
/// \return [3 x n_trg_local] matrix of velocity at target coordinates
Eigen::MatrixXd Periphery::flow(const Eigen::Ref<const Eigen::MatrixXd> &r_trg,
                                const Eigen::Ref<const Eigen::MatrixXd> &density, double eta) const {
  std::cout << "Started shell flow" << std::endl;
  if (!n_nodes_global_) {
    return Eigen::MatrixXd::Zero(3, r_trg.cols());
  }

  const int n_dl = density.size() / 3;
  Eigen::MatrixXd f_dl(9, n_dl);
  Eigen::Map<const Eigen::MatrixXd> density_reshaped(density.data(), 3, n_dl);

  // double layer density is 2 * outer product of normals with density
  // scales with viscosity since the stresslet_kernel_ routine divides by the viscosity, and the double-layer
  // stresslet is independent of viscosity
  for (int node = 0; node < n_dl; ++node) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        f_dl(i * 3 + j, node) = 2.0 * eta * node_normal_(i, node) * density_reshaped(j, node);
      }
    }
  }

  Eigen::MatrixXd r_sl, f_sl;  // dummy SL positions/values
  Eigen::MatrixXd vel = stresslet_kernel_(r_sl, node_pos_, r_trg, f_sl, f_dl, eta);

  std::cout << "Finished shell flow" << std::endl;
  return vel;
}

/// \brief Update the internal right-hand-side state given the velocity at the shell nodes
/// No prerequisite calculations, beyond initialization, are needed
///
/// \param[in] v_on_shell [3 x n_nodes_local] matrix of velocity at shell nodes on local to this MPI rank
/// \return true if collision, false otherwise
void Periphery::update_RHS(const Eigen::Ref<const Eigen::MatrixXd> &v_on_shell) {
  RHS_ = -Eigen::Map<const Eigen::VectorXd>(v_on_shell.data(), v_on_shell.size());
}

void Periphery::set_evaluator(const std::string &evaluator) {
  if (evaluator == "FMM") {
    const int mult_order = periphery_stresslet_multipole_order_;
    const int max_pts = periphery_stresslet_max_points_;
    stresslet_kernel_ =
        FMM<stkfmm::Stk3DFMM>(mult_order, max_pts, stkfmm::PAXIS::NONE, stkfmm::KERNEL::PVel, stokes_pvel_fmm);
  } else if (evaluator == "DIRECT_CPU") {
    stresslet_kernel_ = kernels::stresslet_direct_cpu;
  } else {
    throw std::runtime_error("Invalid evaluator specified for periphery");
  }
}

/// \brief Construct Periphery base class object
///
/// \param[in] precompute_file '.npz' file generated by precompute script
Periphery::Periphery(const std::string &precompute_file, const std::string &pair_evaluator,
                     const int periphery_stresslet_multipole_order, const int periphery_stresslet_max_points) {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  set_evaluator(pair_evaluator);

  cnpy::npz_t precomp;
  if (!precompute_file.length()) {
    throw std::runtime_error(
        "Periphery specified, but no precompute file. In your config file under [periphery], "
        "set precompute_file and run 'skelly_precompute' on the config."
        "If using the config generator, it should automatically generate this variable, though "
        "you still need to run the precompute script after generating the config.");
  }

  std::cout << "Loading raw precomputation data from file " << precompute_file " for periphery into rank 0"
            << std::endl;
  int n_rows;
  int n_nodes;
  if (world_rank_ == 0) {
    precomp = cnpy::npz_load(precompute_file);
    n_rows = precomp.at("M_inv").shape[0];
    n_nodes = precomp.at("nodes").shape[0];
  }

  MPI_Bcast((void *)&n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void *)&n_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int n_cols = n_rows;
  const int node_size_big = 3 * (n_nodes / world_size_ + 1);
  const int node_size_small = 3 * (n_nodes / world_size_);
  const int node_size_local = (n_nodes % world_size_ > world_rank_) ? node_size_big : node_size_small;
  const int n_nodes_big = n_nodes % world_size_;
  const int nrows_local = node_size_local;

  solution_vec_ = Eigen::VectorXd::Zero(nrows_local);

  // TODO: prevent overflow for large matrices in periphery import
  node_counts_.resize(world_size_);
  node_displs_ = Eigen::VectorXi::Zero(world_size_ + 1);
  for (int i = 0; i < world_size_; ++i) {
    node_counts_[i] = ((i < n_nodes_big) ? node_size_big : node_size_small);
    node_displs_[i + 1] = node_displs_[i] + node_counts_[i];
  }
  row_counts_ = node_counts_;
  row_displs_ = node_displs_;
  quad_counts_ = node_counts_ / 3;
  quad_displs_ = node_displs_ / 3;

  const double *M_inv_raw = (world_rank_ == 0) ? precomp["M_inv"].data<double>() : NULL;
  const double *stresslet_plus_complementary_raw =
      (world_rank_ == 0) ? precomp["stresslet_plus_complementary"].data<double>() : NULL;
  const double *normals_raw = (world_rank_ == 0) ? precomp["normals"].data<double>() : NULL;
  const double *nodes_raw = (world_rank_ == 0) ? precomp["nodes"].data<double>() : NULL;
  const double *quadrature_weights_raw = (world_rank_ == 0) ? precomp["quadrature_weights"].data<double>() : NULL;

  MPI_Datatype mpi_matrix_row_t;
  MPI_Type_contiguous(n_cols, MPI_DOUBLE, &mpi_matrix_row_t);
  MPI_Type_commit(&mpi_matrix_row_t);

  // Numpy data is row-major, while eigen is column-major. Easiest way to rectify this is to
  // load in matrix as its transpose, then transpose back
  M_inv_.resize(n_cols, nrows_local);
  MPI_Scatterv(M_inv_raw, row_counts_.data(), row_displs_.data(), mpi_matrix_row_t, M_inv_.data(),
               row_counts_[world_rank_], mpi_matrix_row_t, 0, MPI_COMM_WORLD);

  stresslet_plus_complementary_.resize(n_cols, nrows_local);
  MPI_Scatterv(stresslet_plus_complementary_raw, row_counts_.data(), row_displs_.data(), mpi_matrix_row_t,
               stresslet_plus_complementary_.data(), row_counts_[world_rank_], mpi_matrix_row_t, 0, MPI_COMM_WORLD);

  M_inv_.transposeInPlace();
  stresslet_plus_complementary_.transposeInPlace();

  node_normal_.resize(3, node_size_local / 3);
  MPI_Scatterv(normals_raw, node_counts_.data(), node_displs_.data(), MPI_DOUBLE, node_normal_.data(),
               node_counts_[world_rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  node_pos_.resize(3, node_size_local / 3);
  MPI_Scatterv(nodes_raw, node_counts_.data(), node_displs_.data(), MPI_DOUBLE, node_pos_.data(),
               node_counts_[world_rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  quadrature_weights_.resize(node_size_local / 3);
  MPI_Scatterv(quadrature_weights_raw, quad_counts_.data(), quad_displs_.data(), MPI_DOUBLE, quadrature_weights_.data(),
               quad_counts_[world_rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  n_nodes_global_ = n_nodes;

  MPI_Type_free(&mpi_matrix_row_t);

  std::cout << "Done initializing base periphery" << std::endl;
}