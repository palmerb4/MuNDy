// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
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

#ifndef MUNDY_LINKERS_NEIGHBOR_LINKERS_SPHEROCYLINDERSEGMENTSPHEROCYLINDERSEGMENTLINKERS_HPP_
#define MUNDY_LINKERS_NEIGHBOR_LINKERS_SPHEROCYLINDERSEGMENTSPHEROCYLINDERSEGMENTLINKERS_HPP_

/// \file SpherocylinderSegmentSpherocylinderSegmentLinkers.hpp
/// \brief Declaration of the SpherocylinderSegmentSpherocylinderSegmentLinkers part class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy includes
#include <mundy_linkers/NeighborLinkers.hpp>  // for mundy::linkers::NeighborLinkers
#include <mundy_meta/FieldReqs.hpp>           // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>            // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>            // for mundy::meta::PartReqs

namespace mundy {

namespace linkers {

namespace neighbor_linkers {

/// \class SpherocylinderSegmentSpherocylinderSegmentLinkers
/// \brief The static interface for all of Mundy's SpherocylinderSegmentSpherocylinderSegmentLinkers neighbor linkers.
///
/// The design of this class is in accordance with the static interface requirements of mundy::agents::AgentFactory.
///
/// \note This class is a constraint rank assembly part containing neighbor linkers between spherocylinder segments. It
/// is a subset of the NeighborLinkers agent.
class SpherocylinderSegmentSpherocylinderSegmentLinkers
    : public mundy::agents::RankedAssembly<mundy::core::make_string_literal(
                                               "SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKERS"),
                                           stk::topology::CONSTRAINT_RANK, mundy::linkers::NeighborLinkers> {
};  // SpherocylinderSegmentSpherocylinderSegmentLinkers

}  // namespace neighbor_linkers

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_NEIGHBOR_LINKERS_SPHEROCYLINDERSEGMENTSPHEROCYLINDERSEGMENTLINKERS_HPP_
