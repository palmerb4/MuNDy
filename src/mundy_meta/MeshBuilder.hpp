// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
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

#ifndef MUNDY_META_MESHBUILDER_HPP_
#define MUNDY_META_MESHBUILDER_HPP_

/// \file MeshBuilder.hpp
/// \brief Declaration of the MeshBuilder class

// clang-format off
#include <gtest/gtest.h>                             // for AssertHelper, etc
#include <mpi.h>                                     // for MPI_COMM_WORLD, etc
#include <stddef.h>                                  // for size_t
#include <vector>                                    // for vector, etc
#include <random>                                    // for rand
#include <memory>                                    // for shared_ptr
#include <string>                                    // for string
#include <type_traits>                               // for static_assert
#include <stk_math/StkVector.hpp>                    // for Vec
#include <stk_mesh/base/BulkData.hpp>                // for BulkData
#include <stk_mesh/base/MetaData.hpp>                // for MetaData
#include <stk_mesh/base/GetEntities.hpp>             // for count_selected_entities
#include <stk_mesh/base/Types.hpp>                   // for EntityVector, etc
#include <stk_mesh/base/Ghosting.hpp>                // for create_ghosting
#include <stk_io/WriteMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_search/SearchMethod.hpp>               // for KDTREE
#include <stk_search/CoarseSearch.hpp>               // for coarse_search
#include <stk_search/BoundingBox.hpp>                // for Sphere, Box, tec.
#include <stk_balance/balance.hpp>                   // for balanceStkMesh
#include <stk_util/parallel/Parallel.hpp>            // for ParallelMachine
#include <stk_util/environment/WallTime.hpp>         // for wall_time
#include <stk_util/environment/perf_util.hpp>
#include <stk_unit_test_utils/BuildMesh.hpp>
#include "stk_util/util/ReportHandler.hpp"           // for ThrowAssert, etc
// clang-format on

namespace mundy {

namespace meta {

/// \class MeshBuilder
/// \brief A helper class for building an STK BulkData entity.
///
/// This class is merely a duplicate of STK's \c MeshBuilder. Although duplicative code is discouraged, we chose to copy
/// all of \c MeshBuilder's functionality to improve its readability and documentation.
class MeshBuilder {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor
  MeshBuilder()
      : builder_(MPI_COMM_NULL),
        comm_(MPI_COMM_NULL),
        has_comm_(false),
        aura_option_(stk::mesh::BulkData::AUTO_AURA),
        field_data_manager_ptr_(nullptr),
        bucket_capacity_(stk::impl::BucketRepository::default_bucket_capacity),
        spatial_dimension_(0),
        entity_rank_names(),
        upward_connectivity_flag_(true) {
  }

  /// \brief Constructor with given given communicator.
  explicit MeshBuilder(stk::ParallelMachine comm)
      : builder_(comm),
        comm_(comm),
        has_comm_(true),
        aura_option_(stk::mesh::BulkData::AUTO_AURA),
        field_data_manager_ptr_(nullptr),
        bucket_capacity_(stk::impl::BucketRepository::default_bucket_capacity),
        spatial_dimension_(0),
        entity_rank_names(),
        upward_connectivity_flag_(true) {
  }

  //@}

  //! @name Setters
  //@{

  /// \brief Set the spatial dimension of the mash.
  /// \param spatial_dimension [in] The dimension of the space within which the parts and entities reside.
  MeshBuilder &set_spatial_dimension(const unsigned spatial_dimension) {
    spatial_dimension_ = spatial_dimension;
    builder_.set_spatial_dimension(spatial_dimension_);
    return *this;
  }

  /// \brief Set the names assigned to each rank.
  /// \param entity_rank_names [in] The snames assigned to each rank.
  MeshBuilder &set_entity_rank_names(const std::vector<std::string> &entity_rank_names) {
    entity_rank_names_ = entity_rank_names;
    builder_.set_entity_rank_names(entity_rank_names_);
    return *this;
  }

  /// \brief Set the MPI communicator to be used by STK.
  /// \param comm [in] The MPI communicator.
  MeshBuilder &set_communicator(const stk::ParallelMachine &comm) {
    comm_ = comm;
    has_comm_ = true;
    builder_.set_communicator(comm_);
    return *this;
  }

  /// \brief Set the chosen Aura option. For example, stk::mesh::BulkData::AUTO_AURA.
  /// \param aura_option [in] The chosen Aura option.
  MeshBuilder &set_aura_option(const stk::mesh::BulkData::AutomaticAuraOption &aura_option) {
    aura_option_ = aura_option;
    builder_.set_aura_option(aura_option_);
    return *this;
  }

  /// \brief Set the field data manager.
  /// \param field_data_manager_ptr [in] Pointer to an existing field data manager.
  MeshBuilder &set_field_data_manager(stk::mesh::FieldDataManager *field_data_manager_ptr) {
    field_data_manager_ptr_ = field_data_manager_ptr;
    builder_.set_field_data_manager(field_data_manager_ptr_);
    return *this;
  }

  /// \brief Set the upper bound on the number of mesh entities that may be associated with a single bucket.
  ///
  /// Although subject to change, the maximum bucket capacity is currently 1024 and the default capacity is 512.
  ///
  /// \param bucket_capacity [in] The bucket capacity.
  MeshBuilder &set_bucket_capacity(const unsigned bucket_capacity) {
    bucket_capacity_ = bucket_capacity;
    builder_.set_bucket_capacity(bucket_capacity_);
    return *this;
  }

  /// \brief Set the flag specifying if upward connectivity will be enabled or not.
  /// \param enable_upward_connectivity [in] A flag specifying if upward connectivity will be enabled or not.
  MeshBuilder &set_upward_connectivity(const bool enable_upward_connectivity) {
    enable_upward_connectivity_ = enable_upward_connectivity;
    builder_.set_upward_connectivity(enable_upward_connectivity_);
    return *this;
  }
  //@}

  //! @name Actions
  //@{

  /// \brief Create a new MetaData instance.
  std::shared_ptr<MetaData> MeshBuilder::create_meta_data() {
    return builder_.create_meta_data();
  }

  /// \brief Create a new BulkData instance.
  std::unique_ptr<BulkData> create_bulk_data() {
    return this.create_bulk_data(this.create_meta_data());
  }

  /// \brief Create a new BulkData instance using an existing MetaData instance.
  std::unique_ptr<BulkData> create_bulk_data(std::shared_ptr<MetaData> metaData) {
    TEUCHOS_TEST_FOR_EXCEPTION(has_comm_, std::logic_error,
                               "mundy::meta::MeshBuilder must be given an MPI communicator before creating BulkData.");

    return builder_.create();
  }
  //@}

 private:
  //! \name Mesh settings
  //@{

  /// \brief An instance of STK's MeshBuilder class
  stk::mesh::MeshBuilder builder_;

  /// \brief MPI communicator to be used by STK.
  /// This must be set before BulkData can be created.
  stk::ParallelMachine comm_;

  /// \brief Flag specifying if comm has been set or not.
  bool has_comm_;

  /// \brief Chosen Aura option. For example, stk::mesh::BulkData::AUTO_AURA.
  stk::mesh::BulkData::AutomaticAuraOption aura_option_;

  /// \brief Pointer to an existing field data manager.
  stk::mesh::FieldDataManager *field_data_manager_ptr_;

  /// \brief Upper bound on the number of mesh entities that may be associated with a single bucket.
  unsigned bucket_capacity_;

  /// \brief Spatial dimension of the mash.
  unsigned spatial_dimension_;

  /// \brief Names assigned to each rank.
  std::vector<std::string> entity_rank_names_;

  /// \brief Flag specifying if upward connectivity will be enabled or not.
  bool upward_connectivity_flag_;
  //@}
}

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_MESHBUILDER_HPP_
