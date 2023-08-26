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

#ifndef MUNDY_SHAPE_SHAPES_SPHEROCYLINDER_HPP_
#define MUNDY_SHAPE_SHAPES_SPHEROCYLINDER_HPP_

/// \file Spherocylinder.hpp
/// \brief Declaration of the Spherocylinder class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy includes
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements
#include <mundy_meta/MeshRequirements.hpp>   // for mundy::meta::MeshRequirements
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements
#include <mundy_shape/ShapeRegistry.hpp>     // for MUNDY_REGISTER_SHAPE

namespace mundy {

namespace shape {

namespace shapes {

/// \class Spherocylinder
/// \brief The static interface for all of Mundy's Spherocylinder shapes.
///
/// The design of this class is in accordance with the static interface requirements of
/// mundy::shape::ShapeFactory.
class Spherocylinder {
 public:
  //! \name Getters
  //@{

  /// \brief Get the Spherocylinder's name.
  /// This name must be unique and not shared by any other shape.
  static constexpr inline std::string_view get_name() {
    return our_part_name_;
  }

  /// \brief Get the Spherocylinder's topology.
  static constexpr inline stk::topology::topology_t get_topology() {
    return our_topology_;
  }

  /// \brief Get the mesh requirements for the Spherocylinder.
  static inline mundy::meta::MeshRequirements get_mesh_requirements() {
    // By default, we assume that the Spherocylinder is a point particle with a radius.
    static auto all_shapes_part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    all_shapes_part_reqs->set_part_name("SHAPES");
    all_shapes_part_reqs->set_part_rank(stk::topology::ELEMENT_RANK);

    static auto our_part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    our_part_reqs->set_part_name(our_part_name_);
    our_part_reqs->set_part_topology(our_topology_);
    our_part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(our_node_coord_field_name_,
                                                                                       stk::topology::NODE_RANK, 3, 1));
    our_part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        our_element_radius_field_name_, stk::topology::ELEMENT_RANK, 1, 1));
    our_part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        our_element_length_field_name_, stk::topology::ELEMENT_RANK, 1, 1));
    all_shapes_part_reqs->add_subpart_reqs(our_part_reqs);

    static auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_reqs(all_shapes_part_reqs);
    return mesh_reqs;
  }

  /// \brief Get the set of default field names for the Spherocylinder.
  static inline std::vector<std::string> get_default_field_names() {
    return {our_node_coord_field_name_, our_element_radius_field_name_};
  }

  /// \brief Get the default radius field name for the Spherocylinder.
  static constexpr inline std::string_view get_element_radius_field_name() {
    return our_element_radius_field_name_;
  }

  /// \brief Get the default node coordinate field name for the Spherocylinder.
  static constexpr inline std::string_view get_node_coord_field_name() {
    return our_node_coord_field_name_;
  }

 private:
  //! \name Internal members
  //@{

  static constexpr stk::topology our_topology_ = stk::topology::PARTICLE;
  static constexpr std::string_view our_part_name_ = "SPHEROCYLINDER";
  static constexpr std::string_view our_element_radius_field_name_ = "ELEMENT_RADIUS";
  static constexpr std::string_view our_element_length_field_name_ = "ELEMENT_LENGTH";
  static constexpr std::string_view our_node_coord_field_name_ = "NODE_COORD";
  //@}
};  // Spherocylinder

}  // namespace shapes

}  // namespace shape

}  // namespace mundy

MUNDY_REGISTER_SHAPE(mundy::shape::shapes::Spherocylinder)

#endif  // MUNDY_SHAPE_SHAPES_SPHEROCYLINDER_HPP_
