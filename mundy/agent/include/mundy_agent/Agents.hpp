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
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_agent/AgentRegistry.hpp>    // for MUNDY_REGISTER_AGENT
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements

namespace mundy {

namespace agent {

class Agents {
 public:
  //! \name Getters
  //@{

  /// \brief Get the Agents's name.
  static inline std::string get_name() {
    return std::string(name_);
  }

  static inline std::string get_parent_name() {
    return std::string(parents_name_);
  }

  /// \brief Get the Agents's topology.
  static constexpr inline stk::topology::topology_t get_topology() {
    return topology_;
  }

  /// \brief Get the Agents's rank.
  static constexpr inline stk::topology::rank_t get_rank() {
    return rank_;
  }

  /// \brief Get if the Agents has a topology or not.
  static constexpr inline bool has_topology() {
    return has_topology_;
  }

  /// \brief Get if the Agents has a rank or not.
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

  /// \brief Get the mesh requirements for the Agents.
  static inline std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements() {
    // Agents is an assembly part containing all agents.
    static auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_reqs(part_reqs_ptr_);
    return mesh_reqs;
  }

 private:
  //! \name Member variable definitions
  //@{

  /// \brief The name of the Agents part.
  static constexpr inline std::string_view name_ = "AGENTS";

  /// \brief The name of the Agents' parent part.
  static constexpr inline std::string_view parents_name_ = "";

  /// \brief The topology of the Agents (we don't have a topology, so this will never be used).
  static constexpr inline stk::topology::topology_t topology_ = stk::topology::INVALID_TOPOLOGY;

  /// \brief The rank of the Agents (we don't have a rank, so this will never be used).
  static constexpr inline stk::topology::rank_t rank_ = stk::topology::INVALID_RANK;

  /// \brief If the Shape has a topology or not.
  static constexpr inline bool has_topology_ = false;

  /// brief If the Shape has a rank or not.
  static constexpr inline bool has_rank_ = false;

  static inline std::shared_ptr<mundy::meta::PartRequirements> part_reqs_ptr_ =
      std::make_shared<mundy::meta::PartRequirements>(std::string(name_));
  //@}
};  // Agents

}  // namespace agent

}  // namespace mundy

MUNDY_REGISTER_AGENT(mundy::agent::Agents)

#endif  // MUNDY_AGENT_AGENTS_HPP_
