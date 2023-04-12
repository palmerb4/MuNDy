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

#ifndef MUNDY_META_PARTHIERARCHYBUILDER_HPP_
#define MUNDY_META_PARTHIERARCHYBUILDER_HPP_

/// \file PartHierarchyBuilder.hpp
/// \brief Declaration of the PartHierarchyBuilder class

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

/// \class PartHierarchyBuilder
/// \brief A helper class for building the hierarchy of parts and fields.
///
/// This class is merely a wrapper for STK's \c MeshBuilder with the added functionality of automatically constructing
/// the part/field hierarchy via a set of \c PartParams, a parameter list, or a YAML file.
///
/// Although duplicative code is discouraged, we chose to copy all of \c MeshBuilder's functionality to improve its
/// readability and documentation.
class PartHierarchyBuilder {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor
  PartHierarchyBuilder()
      : comm_(MPI_COMM_NULL),
        has_comm_(false),
        aura_option_(stk::mesh::BulkData::AUTO_AURA),
        field_data_manager_ptr_(nullptr),
        bucket_capacity_(stk::impl::BucketRepository::default_bucket_capacity),
        spatial_dimension_(0),
        entity_rank_names(),
        upward_connectivity_flag_(true) {
  }

  /// \brief Constructor with given given communicator.
  explicit PartHierarchyBuilder(stk::ParallelMachine comm)
      : comm_(comm),
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
  MeshBuilder &set_spatial_dimension(const unsigned spatial_dimension);

  /// \brief Set the names assigned to each rank.
  /// \param entity_rank_names [in] The spacial dimension within which the parts and entities reside.
  MeshBuilder &set_entity_rank_names(const std::vector<std::string> &entity_rank_names);

  /// \brief Set the spatial dimension of the mash.
  /// \param comm [in] The spacial dimension within which the parts and entities reside.
  MeshBuilder &set_communicator(const stk::ParallelMachine &comm);

  /// \brief Set the spatial dimension of the mash.
  /// \param aura_option [in] The spacial dimension within which the parts and entities reside.
  MeshBuilder &set_aura_option(const stk::mesh::BulkData::AutomaticAuraOption &aura_option);

  /// \brief Set the spatial dimension of the mash.
  /// \param field_data_manager_ptr [in] The spacial dimension within which the parts and entities reside.
  MeshBuilder &set_field_data_manager(stk::mesh::FieldDataManager *field_data_manager_ptr);

  /// \brief Set the capacity 
  /// \param bucket_capacity [in] The spacial dimension within which the parts and entities reside.
  MeshBuilder &set_bucket_capacity(const unsigned bucket_capacity);

  /// \brief Set the flag specifying of upward connectivity will be enabled or not.
  /// \param enable_upward_connectivity [in] A flag specifying if upward connectivity will be enabled or not.
  MeshBuilder &set_upward_connectivity(const bool enable_upward_connectivity);

  //@}

  //! @name Actions
  //@{

  std::unique_ptr<BulkData> create();
  std::unique_ptr<BulkData> create(std::shared_ptr<MetaData> metaData);
  //@}

 private:
  //! \name Mesh settings
  //@{

  stk::ParallelMachine comm_;
  bool has_comm_;
  stk::mesh::BulkData::AutomaticAuraOption aura_option_;
  stk::mesh::FieldDataManager *field_data_manager_ptr_;
  unsigned bucket_capacity_;
  unsigned spatial_dimension_;
  std::vector<std::string> entity_rank_names;
  bool upward_connectivity_flag_;
  //@}
}

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_PARTHIERARCHYBUILDER_HPP_
