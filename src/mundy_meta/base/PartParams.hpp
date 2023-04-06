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

#ifndef MUNDY_META_PARTPARAMS_HPP_
#define MUNDY_META_PARTPARAMS_HPP_

/// \file PartParams.hpp
/// \brief Declaration of the PartParams class

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

/// \class PartParams
/// \brief A set of necessary parameters for declaring a new part with a some fields and enabled methods.
///
/// If we write out the final class stucture like a YAML file, then we get something the following.
/// Notice how easily this can be replaced with a GUI.
///
/// Naming convention became confusing quickly. This convention helped the most:
///   Variables are named variable_name.
///   Strings are named STRING_NAME
///
/// %YAML 1.1
/// ---
/// part_hierarchy:
///   - name: PARTICLES
///     topology: ENTITY
///     fields:
///       - name: FORCE
///         rank: ENTITY
///         dimension: 3
///         number_of_states: 1
///         type: DOUBLE
///       - name: COORDINATES
///         rank: Node
///         dimension: 3
///         number_of_states: 1
///         type: DOUBLE
///     methods:
///       - name: RESOLVE_CONSTRAINTS
///         resolve_constraints_technique:
///           name: NONSMOOTH_LCP
///           max_number_of_iterations: 10000
///           residual_tolerance: 1e-5
///     sub_parts:
///       - name: SPHERES
///         topology: ENTITY
///         fields:
///           - name: RADIUS
///             rank: ENTITY
///             dimension: 1
///             number_of_states: 1
///             type: DOUBLE
///         methods:
///           - name: TIME_INTEGRATION
///             variant:
///               - name: SPHERE
///                 time_integration_technique:
///                   name: NODE_EULER
///           - name: COMPUTE_MOBILITY
///               variant:
///                 name: SPHERE
///                 compute_mobility_technique:
///                   name: DRY_INERTIAL
///         sub_parts:
///           - name: COLORED_SPHERES
///             topology: ENTITY
///             fields:
///             - name: COLOR
///               rank: ENTITY
///               dimension: 1
///               number_of_states: 1
///               type: INT
///       - name: ELLIPSOIDS
///         topology: ENTITY
///         fields:
///         - AxisLength
///             rank: ENTITY
///             dimension: 3
///             number_of_states: 1
///             type: DOUBLE
///         methods:
///           - name: TIME_INTEGRATION
///             variant:
///               name: ELLIPSOID
///               time_integration_technique:
///                 name: NODE_EULER
///           - name: COMPUTE_MOBILITY
///               variant:
///                 name: ELLIPSOID
///                 compute_mobility_technique:
///                   name: DRY_INERTIAL
///
class PartParams {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is not allowed
  PartParams() = delete;

  /// \brief Constructor with known part topology.
  ///
  /// \param part_name [in] Name of the part. If the name already exists for the given topology, the two parameter
  /// sets must be valid.
  ///
  /// \param part_topology [in] Topology of entities within the part.
  ///
  /// \param part_fields [in] Vector of field parameters for the fields defined on this part.
  PartParams(const std::string &part_name, const stk::topology &part_topology)
      : part_name_(part_name), part_topology_(part_topology), part_rank_(part_topology_.rank()) {
  }

  /// \brief Constructor with known part rank.
  ///
  /// \param part_name [in] Name of the part. If the name already exists for the given topology, the two parameter
  /// sets must be valid.
  ///
  /// \param part_rank [in] Maximum rank of entities within the part. Can contain any element of lower rank, regardless
  /// of topology.
  ///
  /// \param part_fields [in] Vector of field parameters for the fields defined on this part.
  PartHierarchyParams(const std::string &part_name, const stk::topology::rank_t &part_rank)
      : part_name_(part_name), part_topology_(stk::topology::INVALID_TOPOLOGY), part_rank_(part_rank) {
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Return the part name.
  std::string get_part_name() {
    return part_name_;
  }

  /// \brief Return the part topology.
  stk::topology get_part_topology() {
    return part_topology_;
  }

  /// \brief Return the part rank.
  stk::topology::rank_t get_part_rank() {
    return part_rank_;
  }

  /// \brief Return the part field map.
  std::map<std::string, std::shared_ptr<const FieldParamsBase>> get_part_field_map() {
    return part_field_map_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means:
  ///   1. the rank of the fields does not exceed the rank of the part's topology.
  void check_if_valid() {
    ThrowRequireMsg(false, "not implemented yet");
  }

  /// \brief Add the provided field to the part, given that it is valid and does not conflict with existing fields.
  ///
  /// \param field_params [in] Field parameters to add to the part.
  void add_field_params(const std::shared_ptr<const FieldParamsBase> &field_params) {
    // Check if the provided parameters are valid.
    field_params.check_if_valid();

    // If a field with the same name exists, attempt to merge them if they are the same rank.
    // Otherwise, create a new field entity.
    const std::string field_name = field_params.get_field_name();
    const bool name_already_exists = (part_field_map_.count(field_name) != 0);
    if (name_already_exists && (part_field_map_[field_name].get_field_rank() == field_params.get_field_rank())) {
      part_field_map_[field_name]->merge(field_params);
    } else {
      part_field_map_[field_name] = field_params;
    }
  }

  /// \brief Merge the current parameters with any number of other \c PartParams.
  ///
  /// Here, merging two a \c PartParams object with this object amounts to merging their fields. For this process to be
  /// valid, the given \c PartParams must have the same topology and rank. The name of the other part does not need to
  /// match the current name of this part.
  ///
  /// \param list_of_part_params [in] A list of other \c PartParams objects to merge with the current object.
  template <class... ArgTypes,
            typename std::enable_if<std::conjunction<std::is_convertible<Ts, PartParams>...>::value>::type>
  void merge(const ArgTypes &...list_of_part_params) {
    for (const auto &part_params : list_of_part_params) {
      // Check if the provided parameters are valid.
      part_params.check_if_valid();

      // Check if the provided topology and rank are the same.
      TEUCHOS_TEST_FOR_EXCEPTION(
          get_part_rank() == part_params.get_part_rank(), std::invalid_argument,
          "mundy::PartParams: Part " << part_params.get_part_name() << " has incompatible rank.");
      TEUCHOS_TEST_FOR_EXCEPTION(
          get_part_topology() == part_params.get_part_topology(), std::invalid_argument,
          "mundy::PartParams: Part " << part_params.get_part_name() << " has incompatible topology.");

      // Loop over each field and attempt to merge it.
      for ([[maybe_unused]] auto const &[field_name, field_params_ptr] : part_params.get_part_field_map()) {
        this.add_field_params(field_params_ptr);
      }
    }
  }
  //@}

 private:
  /// \brief Name of the part.
  std::string part_name_;

  /// \brief Topology of entities in the part.
  stk::topology part_topology_;

  /// \brief Rank of the part.
  stk::topology::rank_t part_rank_;

  /// \brief A set of maps from field name to field params for each rank.
  std::vector<std::map<std::string, std::shared_ptr<FieldParamsBase>>>
      part_ranked_field_maps_[stk::topology::NUM_RANKS];
}

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_PARTPARAMS_HPP_
