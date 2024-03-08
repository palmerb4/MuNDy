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

#ifndef MUNDY_LINKER_LINKERS_NEIGHBOR_LINKERS_SPHERESPHERES_HPP_
#define MUNDY_LINKER_LINKERS_NEIGHBOR_LINKERS_SPHERESPHERES_HPP_

/// \file NeighborLinkers.hpp
/// \brief Declaration of the NeighborLinkers part class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy includes
#include <mundy_agent/AgentHierarchy.hpp>    // for mundy::agent::AgentHierarchy
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements
#include <mundy_meta/MeshRequirements.hpp>   // for mundy::meta::MeshRequirements
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements

namespace mundy {

namespace linker {

namespace linkers {

namespace neighbor_linkers {

/// \class SphereSpheres
/// \brief The static interface for all of Mundy's SphereSpheres neighbor linkers.
///
/// The design of this class is in accordance with the static interface requirements of mundy::agent::AgentFactory.
///
/// \note This class is a constraint rank assembly part containing neighbor linkers between spheres. It is a subset of
/// the NEIGHBOR_LINKERS part.
class SphereSpheres
    : public mundy::agent::RankedAssembly<
          mundy::core::make_string_literal("SPHERE_SPHERE_LINKERS"), stk::topology::CONSTRAINT_RANK,
          mundy::core::make_string_literal("NEIGHBOR_LINKERS"), mundy::core::make_string_literal("LINKERS")> {
};  // SphereSpheres

}  // namespace neighbor_linkers

}  // namespace linkers

}  // namespace linker

}  // namespace mundy

#endif  // MUNDY_LINKER_LINKERS_NEIGHBOR_LINKERS_SPHERESPHERES_HPP_
