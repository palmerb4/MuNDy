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

#ifndef UNIT_TESTS_MUNDY_AGENT_UTILS_EXAMPLEAGENT_HPP_
#define UNIT_TESTS_MUNDY_AGENT_UTILS_EXAMPLEAGENT_HPP_

/// \file ExampleMetaMethod.hpp
/// \brief Declaration of the ExampleMetaMethod class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::GlobalMetaMethodRegistry

namespace mundy {

namespace agent {

namespace utils {

/// \class ExampleAgent
/// \brief The static interface for all of Mundy's ExampleAgent objects.
template <int I>
class ExampleAgent {
 public:
  //! \name Getters
  //@{

  /// \brief Get the name of our part.
  static constexpr inline std::string_view get_name() {
    return name_;
  }

  /// @brief Get the name of our parent part.
  static constexpr inline std::string_view get_parent_name() {
    return parents_name_;
  }

  /// \brief Get the topology of our part.
  static constexpr inline stk::topology::topology_t get_topology() {
    return topology_;
  }

  /// \brief Get the rank of our part.
  static constexpr inline stk::topology::rank_t get_rank() {
    return rank_;
  }

  /// \brief Get if our part has a topology or not.
  static constexpr inline bool has_topology() {
    return has_topology_;
  }

  /// \brief Get if our part has a rank or not.
  static constexpr inline bool has_rank() {
    return has_rank_;
  }

  /// \brief Add new part requirements to ALL members of this agent part.
  /// These modifications are reflected in our mesh requirements.
  static inline void add_part_reqs(std::shared_ptr<mundy::meta::PartRequirements> part_reqs_ptr) {
    part_reqs_ptr_->merge(part_reqs_ptr);
  }

  /// \brief Add sub-part requirements.
  /// These modifications are reflected in our mesh requirements.
  static inline void add_subpart_reqs(std::shared_ptr<mundy::meta::PartRequirements> subpart_reqs_ptr) {
    part_reqs_ptr_->add_subpart_reqs(subpart_reqs_ptr);
  }

  /// \brief Get our mesh requirements.
  static inline std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements() {
    // By default, we assume that the ExampleAgents part is an agent with an INVALID_TOPOLOGY.

    // Declare our part as a subpart of our parent part.
    mundy::agent::AgentHierarchy::add_subpart_reqs(part_reqs_ptr_, parents_name_, grandparents_name_);

    // Fetch our parent's requirements.
    // If done correctly, this call will result in a upward tree traversal. Our part is declared as a subpart of our
    // parent, which is declared as a subpart of its parent. This process repeated until we reach a root node. The
    // combined requirements for all parts touched in this traversal are then returned here.
    return mundy::agent::AgentHierarchy::get_mesh_requirements(parents_name_, grandparents_name_);
  }

 private:
  //! \name Member variable definitions
  //@{

  /// \brief The name of the our part.
  static constexpr std::string_view name_ = []() {
    if constexpr (I == 0) {
      return "FAKE_NAME_0";
    } else if constexpr (I == 1) {
      return "FAKE_NAME_1";
    } else if constexpr (I == 2) {
      return "FAKE_NAME_2";
    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid ExampleAgent index.");
      return "FAKE_NAME_N";
    }
  }();

  /// \brief The name of the our parent part.
  static constexpr inline std::string_view parents_name_ = "AGENTS";

  /// \brief The name of the our grandparent part.
  static constexpr inline std::string_view grandparents_name_ = "";

  /// \brief Our topology
  static constexpr stk::topology::topology_t topology_ = stk::topology::INVALID_TOPOLOGY;

  /// \brief Our rank (we have a rank, so this is never used).
  static constexpr inline stk::topology::rank_t rank_ = stk::topology::INVALID_RANK;

  /// \brief If our part has a topology or not.
  static constexpr inline bool has_topology_ = true;

  /// \brief If our part has a rank or not.
  static constexpr inline bool has_rank_ = false;

  /// \brief Our part requirements.
  static inline std::shared_ptr<mundy::meta::PartRequirements> part_reqs_ptr_ = []() {
    auto part_reqs_ptr = std::make_shared<mundy::meta::PartRequirements>();
    part_reqs_ptr->set_part_name(std::string(name_));
    part_reqs_ptr->set_part_topology(topology_);
    return part_reqs_ptr;
  }();
  //@}
};  // ExampleAgent

}  // namespace utils

}  // namespace agent

}  // namespace mundy

#endif  // UNIT_TESTS_MUNDY_AGENT_UTILS_EXAMPLEAGENT_HPP_
