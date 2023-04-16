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

/// \file PartRequirements.hpp
/// \brief Declaration of the PartRequirements class

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

/// \class PartRequirements
/// \brief A set requirements imposed upon a Part and its fields.
class PartRequirements {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements.
  PartRequirements() = default;

  /// \brief Constructor with partial requirements. Version 1.
  ///
  /// \param part_name [in] Name of the part.
  ///
  /// \param part_topology [in] Topology of entities within the part.
  PartRequirements(const std::string &part_name, const stk::topology &part_topology) {
    set_part_name(part_name);
    set_part_topology(part_topology);
  }

  /// \brief Constructor with partial requirements. Version 2.
  ///
  /// \param part_name [in] Name of the part. If the name already exists for the given topology, the two parameter
  /// sets must be valid.
  ///
  /// \param part_rank [in] Maximum rank of entities within the part. Can contain any element of lower rank, regardless
  /// of topology.
  ///
  /// \param part_fields [in] Vector of field parameters for the fields defined on this part.
  PartRequirements(const std::string &part_name, const stk::topology::rank_t &part_rank) {
    set_part_name(part_name);
    set_part_rank(part_rank);
  }
  //@}

  //! \name Setters and Getters
  //@{

  /// \brief Set the required part name.
  /// \brief part_name [in] Required name of the part.
  void set_part_name(const std::string &part_name) {
    part_name_ = part_name;
    part_name_is_set_ = true;
    check_if_valid();
  }

  /// \brief Set the required part topology.
  /// \brief part_topology [in] Required topology of the part.
  void set_part_topology(const stk::topology &part_topology) {
    part_topology_ = part_topology_;
    part_topology_is_set_ = true;
    check_if_valid();
  }

  /// \brief Set the required part rank.
  /// \brief part_rank [in] Required rank of the part.
  void set_part_rank(const stk::topology::rank_t &part_rank) {
    part_rank_ = part_rank;
    part_rank_is_set_ = true;
    check_if_valid();
  }

  /// \brief Get if the part name is constrained or not.
  bool constrains_part_name() {
    return part_name_is_set_;
  }

  /// \brief Get if the part topology is constrained or not.
  bool constrains_part_topology() {
    return part_name_is_set_;
  }

  /// \brief Get if the part rank is constrained or not.
  bool constrains_part_rank() {
    return part_name_is_set_;
  }

  /// \brief Return the part name.
  /// Will throw an error if the part name is not constrained.
  std::string get_part_name() {
    TEUCHOS_TEST_FOR_EXCEPTION(
        !this.constrains_part_name(), std::logic_error,
        "Attempting to access the part name requirement even though part name is unconstrained.");

    return part_name_;
  }

  /// \brief Return the part topology.
  /// Will throw an error if the part topology is not constrained.
  stk::topology get_part_topology() {
    TEUCHOS_TEST_FOR_EXCEPTION(
        !this.constrains_part_topology(), std::logic_error,
        "Attempting to access the part topology requirement even though part topology is unconstrained.");

    return part_topology_;
  }

  /// \brief Return the part rank.
  /// Will throw an error if the part rank is not constrained.
  stk::topology::rank_t get_part_rank() {
    TEUCHOS_TEST_FOR_EXCEPTION(
        !this.constrains_part_rank(), std::logic_error,
        "Attempting to access the part rank requirement even though part rank is unconstrained.");

    return part_rank_;
  }

  /// \brief Return the part field map.
  /// \brief field_rank [in] Rank associated with the retrieved fields.
  std::vector<std::map<std::string, std::shared_ptr<const FieldRequirementsBase>>> get_part_field_map(
      const stk::topology::rank_t &field_rank) {
    return part_ranked_field_maps_[field_rank];
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Delete the part name constraint (if it exists).
  void delete_part_name_constraint() {
    part_name_is_set_ = false;
  }

  /// \brief Delete the part topology constraint (if it exists).
  void delete_part_topology_constraint() {
    part_topology_is_set_ = false;
  }

  /// \brief Delete the part rank constraint (if it exists).
  void delete_part_rank_constraint() {
    part_rank_is_set_ = false;
  }

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
  void add_field_params(const std::shared_ptr<const FieldRequirementsBase> &field_params) {
    // Check if the provided parameters are valid.
    field_params.check_if_valid();

    // If a field with the same name and rank exists, attempt to merge them.
    // Otherwise, create a new field entity.
    const std::string field_name = field_params.get_field_name();
    const unsigned field_rank = field_params.get_field_rank();

    auto part_field_map_ptr = part_ranked_field_maps_.data() + field_rank;
    const bool name_already_exists = (part_field_map_ptr->count(field_name) != 0);
    if (name_already_exists) {
      *part_field_map_ptr[field_name]->merge(field_params);
    } else {
      *part_field_map_ptr[field_name] = field_params;
    }
  }

  /// \brief Add the provided part as a subpart of this part, given that it is valid.
  ///
  /// TODO: Are there any restrictions on what can and cannot be a subpart? If so, encode them here.
  ///
  /// \param field_params [in] Field parameters to add to the part.
  void add_subpart_reqs(const std::shared_ptr<const PartRequirements> &part_reqs) {
    // Check if the provided parameters are valid.
    part_reqs.check_if_valid();

    // Check for conflicts?

    // Store the params.
    part_subpart_map_[part_reqs.get_part_name(), part_reqs];
  }

  /// \brief Merge the current requirements with any number of other \c PartRequirements.
  ///
  /// Here, merging two a \c PartRequirements object with this object amounts to merging their fields. For this process
  /// to be valid, the given \c PartRequirements must have the same topology and rank. The name of the other part does
  /// not need to match the current name of this part.
  ///
  /// \param list_of_part_reqs [in] A list of other \c PartRequirements objects to merge with the current object.
  template <class... ArgTypes,
            typename std::enable_if<std::conjunction<std::is_convertible<Ts, PartRequirements>...>::value>::type>
  void merge(const ArgTypes &...list_of_part_reqs) {
    for (const auto &part_reqs : list_of_part_reqs) {
      // Check if the provided parameters are valid.
      part_reqs.check_if_valid();

      // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
      if (part_reqs.constrains_part_name()) {
        if (this.constrains_part_name() &&) {
          TEUCHOS_TEST_FOR_EXCEPTION(this.get_part_name() == part_reqs.get_part_name(), std::invalid_argument,
                                     "mundy::meta::PartRequirements: One of the inputs has incompatible name ("
                                         << part_reqs.get_part_name() << ").");
        } else {
          this.set_part_name(part_reqs.get_part_name());
        }
      }

      if (part_reqs.constrains_part_rank()) {
        if (this.constrains_part_rank() &&) {
          TEUCHOS_TEST_FOR_EXCEPTION(this.get_part_rank() == part_reqs.get_part_rank(), std::invalid_argument,
                                     "mundy::meta::PartRequirements: One of the inputs has incompatible rank ("
                                         << part_reqs.get_part_rank() << ").");
        } else {
          this.set_part_rank(part_reqs.get_part_rank());
        }
      }

      if (part_reqs.constrains_part_topology()) {
        if (this.constrains_part_topology() &&) {
          TEUCHOS_TEST_FOR_EXCEPTION(this.get_part_topology() == part_reqs.get_part_topology(), std::invalid_argument,
                                     "mundy::meta::PartRequirements: One of the inputs has incompatible topology ("
                                         << part_reqs.get_part_topology() << ").");
        } else {
          this.set_part_topology(part_reqs.get_part_topology());
        }
      }

      // Loop over each rank's map.
      for (auto const &part_field_map : part_reqs.get_part_field_map()) {
        // Loop over each field and attempt to merge it.
        for ([[maybe_unused]] auto const &[field_name, field_params_ptr] : part_field_map) {
          this.add_field_params(field_params_ptr);
        }
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

  /// \brief If the name of the part is set or not.
  bool part_name_is_set_ = false;

  /// \brief If the topology of entities in the part is set or not.
  bool part_topology_is_set_ = false;

  /// \brief If the rank of the part is set or not.
  bool part_rank_is_set_ = false;

  /// \brief A set of maps from field name to field params for each rank.
  std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>>
      part_ranked_field_maps_[stk::topology::NUM_RANKS];

  /// \brief A map from subpart name to the part params of each sub-part.
  std::map<std::string, std::shared_ptr<PartRequirements>> part_subpart_map_;
}

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_PARTPARAMS_HPP_
