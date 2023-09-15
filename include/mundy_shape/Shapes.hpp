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

#ifndef MUNDY_SHAPE_SHAPES_HPP_
#define MUNDY_SHAPE_SHAPES_HPP_

/// \file Shapes.hpp
/// \brief Declaration of the Shapes class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_agent/AgentRegistry.hpp>          // for MUNDY_REGISTER_AGENT
#include <mundy_meta/FieldRequirements.hpp>       // for mundy::meta::FieldRequirements
#include <mundy_meta/MeshRequirements.hpp>        // for mundy::meta::MeshRequirements
#include <mundy_meta/PartRequirements.hpp>        // for mundy::meta::PartRequirements
#include <mundy_shape/shapes/Sphere.hpp>          // for mundy::shape::Sphere
#include <mundy_shape/shapes/Spherocylinder.hpp>  // for mundy::shape::Spherocylinder

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
using shape_t = mundy::agent::agent_t;

class Shapes {
 public:
  //! \name Getters
  //@{

  /// \brief Get the name of our part.
  static inline std::string get_name() {
    return std::string(our_name_);
  }

  /// @brief Get the name of our parent part.
  static inline std::string get_parent_name() {
    return std::string(our_parents_name_);
  }

  /// \brief Get the topology of our part.
  static constexpr inline stk::topology::topology_t get_topology() {
    return our_topology_;
  }

  /// \brief Get the rank of our part.
  static constexpr inline stk::topology::rank_t get_rank() {
    return our_rank_;
  }

  /// \brief Get if our part has a topology or not.
  static constexpr inline bool has_topology() {
    return our_has_topology_;
  }

  /// \brief Get if our part has a rank or not.
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

  /// \brief Get our mesh requirements.
  static inline std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements() {
    // Shapes is an assembly part containing all shapes.

    // Declare our part as a subpart of our parent part.
    mundy::agent::AgentHierarchy::add_subpart_reqs(our_part_reqs_ptr_, std::string(our_parents_name_),
                                                   std::string(our_grandparents_name_));

    // Fetch our parent's requirements.
    // If done correctly, this call will result in a upward tree traversal. Our part is declared as a subpart of our
    // parent, which is declared as a subpart of its parent. This process repeated until we reach a root node. The
    // combined requirements for all parts touched in this traversal are then returned here.
    return mundy::agent::AgentHierarchy::get_mesh_requirements(std::string(our_parents_name_),
                                                               std::string(our_grandparents_name_));
  }

 private:
  //! \name Member variable definitions
  //@{

  /// \brief The name of the our part.
  static constexpr inline std::string_view our_name_ = "SHAPES";

  /// \brief The name of the our parent part.
  static constexpr inline std::string_view our_parents_name_ = "AGENTS";

  /// \brief The name of the our grandparent part.
  static constexpr inline std::string_view our_grandparents_name_ = "";

  /// \brief Our topology (we don't have a topology, so this should never be used).
  static constexpr inline stk::topology::topology_t our_topology_ = stk::topology::INVALID_TOPOLOGY;

  /// \brief Our rank (we don't have a rank, so this should never be used).
  static constexpr inline stk::topology::rank_t our_rank_ = stk::topology::INVALID_RANK;

  /// \brief If our part has a topology or not.
  static constexpr inline bool our_has_topology_ = false;

  /// \brief If our part has a rank or not.
  static constexpr inline bool our_has_rank_ = false;

  /// @brief Our part requirements.
  static inline std::shared_ptr<mundy::meta::PartRequirements> our_part_reqs_ptr_ =
      std::make_shared<mundy::meta::PartRequirements>(std::string(our_name_));
  //@}
};  // Shapes

}  // namespace shape

}  // namespace mundy

MUNDY_REGISTER_AGENT(mundy::shape::Shapes)

#endif  // MUNDY_SHAPE_SHAPES_HPP_
