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

#ifndef MUNDY_SHAPE_SHAPE_HPP_
#define MUNDY_SHAPE_SHAPE_HPP_

/// \file ShapeBase.hpp
/// \brief Declaration of the ShapeBase class

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

namespace shape {

/// In the current design, a "shape" is a Part with some set of requirements that endow the entities of that part with
/// some shape. For example,
///   - a point particle can be represented as having a PARTICLE topology with one node at its center (it need not be
///   orientable)
///   - a line particle can be represented as having a LINE_3 topology with three nodes (one at each end and one at its
///   center).
///   - a sphere is a point particle with an element radius.
///   - an ellipsoid is a point particle with three element axis lengths and an element orientation.
///   - a spherocylinder is a point particle with an element radius, length, and orientation.
///   - a spherocylidner_segment is a line segment with element radius and length.
///   - a NURBS is a SUPERTOPOLOGY<N> with N nodes corresponding to the control points.
///
/// Each shape can be uniquely identified by either the shape's part or a fast unique identifier, namely shape_t.
/// \note shape_t is simply the agent_t associated with the shape. As a result, a shape_t will never equate to, for
/// example, a constraint_t since they are both agent_t's. You can think of this as a runtime extensible class enum.
using shape_t = agent_t;

}  // namespace shape

}  // namespace mundy

#endif  // MUNDY_SHAPE_SHAPE_HPP_
