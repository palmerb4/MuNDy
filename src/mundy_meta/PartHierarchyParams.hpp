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

/// \file PartHierarchyParams.hpp
/// \brief Declaration of the PartHierarchyParams class

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

/// \class PartHierarchyParams
/// \brief A helper class for building the parameters necessary to create a hierarchy of parts and fields.
class PartHierarchyParams {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor
  PartHierarchyParams();
  //@}

  //! @name Actions
  //@{

  /// \brief
  void MeshBuilder::add_part_hierarchy_from_params(const Teuchos::ParameterList &parameter_list) {
  }

  /// \brief
  void MeshBuilder::add_part(const Teuchos::ParameterList &parameter_list) {
    /// \brief Create a new MetaData instance.
    std::shared_ptr<MetaData> MeshBuilder::create_meta_data() {
      if (spatial_dimension_ > 0 || !entity_rank_names_.empty()) {
        return std::make_shared<MetaData>(spatial_dimension_, entity_rank_names_);
      } else {
        return std::make_shared<MetaData>();
      }
    }

    /// \brief Create a new BulkData instance.
    std::unique_ptr<BulkData> create_bulk_data() {
      return this.create_bulk_data(this.create_meta_data());
    }

    /// \brief Create a new BulkData instance using an existing MetaData instance.
    std::unique_ptr<BulkData> create_bulk_data(std::shared_ptr<MetaData> metaData) {
      TEUCHOS_TEST_FOR_EXCEPTION(has_comm_, std::logic_error,
                                 "PartHierarchyParams must be given an MPI communicator before creating BulkData.");

      return std::make_unique<BulkData>(metaData, comm_, aura_option_, field_data_manager_ptr_, bucket_capacity_,
                                        this.create_aura_ghosting(), upward_connectivity_flag_);
    }
    //@}

   private:
    //! \name Mesh settings
    //@{

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
#endif  // MUNDY_META_PARTHIERARCHYBUILDER_HPP_
