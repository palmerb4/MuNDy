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

#ifndef MUNDY_AGENT_AGENTS_HPP_
#define MUNDY_AGENT_AGENTS_HPP_

/// \file Agents.hpp
/// \brief Declaration of the Agents class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaFactory.hpp>           // for mundy::meta::GlobalMetaMethodFactory
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for MUNDY_REGISTER_METACLASS

namespace mundy {

namespace agent {

class Agents {
 public:
  //! \name Getters
  //@{

  /// \brief Get the Agents's name.
  static constexpr inline std::string_view get_name() {
    return our_name_;
  }

  static constexpr inline std::string_view get_parent_name() {
    return our_parents_name_;
  }

  /// \brief Get the Agents's topology.
  static constexpr inline stk::topology::topology_t get_topology() {
    return our_topology_;
  }

  /// \brief Get the Agents's rank.
  static constexpr inline stk::topology::rank_t get_rank() {
    return our_rank_;
  }

  /// \brief Get if the Agents has a topology or not.
  static constexpr inline bool has_topology() {
    return our_has_topology_;
  }

  /// \brief Get if the Agents has a rank or not.
  static constexpr inline bool has_rank() {
    return our_has_rank_;
  }

  /// \brief Add new part requirements to ALL members of this agent part.
  /// These modifications are reflected in our mesh requirements.
  static inline void add_part_reqs(std::shared_ptr<mundy::meta::PartRequirements> part_reqs_ptr) {
    our_part_reqs_ptr_->merge(part_reqs_ptr);
  }

  /// \brief Add sub-part requirements.
  /// These modifications are reflected in our mesh requirements.
  static inline void add_subpart_reqs(std::shared_ptr<mundy::meta::PartRequirements> subpart_reqs_ptr) {
    our_part_reqs_ptr_->add_subpart_reqs(subpart_reqs_ptr);
  }

  /// \brief Get the mesh requirements for the Agents.
  static inline std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements() {
    // Agents is an assembly part containing all agents.
     static auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_reqs(our_part_reqs_ptr_);
    return mesh_reqs;
  }

 private:
  //! \name Member variable definitions
  //@{

  /// \brief The name of the Agents part.
  static constexpr inline std::string_view our_name_ = "AGENTS";

  /// \brief The name of the Agents' parent part.
  static constexpr inline std::string_view our_parents_name_ = "";

  /// \brief The topology of the Agents (we don't have a topology, so this will never be used).
  static constexpr inline stk::topology::topology_t our_topology_ = stk::topology::INVALID_TOPOLOGY;

  /// \brief The rank of the Agents (INVALID_RANK is used to indicate assembly parts).
  static constexpr inline stk::topology::rank_t our_rank_ = stk::topology::INVALID_RANK;

  /// \brief If the Shape has a topology or not.
  static constexpr inline bool our_has_topology_ = false;

  /// brief If the Shape has a rank or not.
  static constexpr inline bool our_has_rank_ = true;

  static inline std::shared_ptr<mundy::meta::PartRequirements> our_part_reqs_ptr_ =
      std::make_shared<mundy::meta::PartRequirements>(our_name_, our_rank_);
};
//@}
};  // Agents

}  // namespace agent

}  // namespace mundy

#endif  // MUNDY_AGENT_AGENTS_HPP_
