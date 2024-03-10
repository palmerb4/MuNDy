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
#include <mundy_agent/AgentHierarchy.hpp>   // for mundy::agent::AgentHierarchy
#include <mundy_agent/Assembly.hpp>         // for mundy::agent::Assembly
#include <mundy_core/StringLiteral.hpp>     // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements

namespace mundy {

namespace agent {

/// \class Agents
/// \brief The static interface for all of Mundy's Agents.
///
/// The design of this class is in accordance with the static interface requirements of mundy::agent::AgentFactory.
///
/// \note This class is a assembly part containing all agents. It does not have a parent or a grandparent within the
/// agent hierarchy.
class Agents : public Assembly<mundy::core::make_string_literal("AGENTS")> {};  // Agents

}  // namespace agent

}  // namespace mundy

#endif  // MUNDY_AGENT_AGENTS_HPP_
