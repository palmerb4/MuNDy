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

/// \file Spherocylinder.cpp
/// \brief Definition of the ComputeAABB's Spherocylinder kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>          // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                               // for mundy::mesh::BulkData
#include <mundy_shapes/Spherocylinders.hpp>                      // for mundy::shapes::Spherocylinders
#include <mundy_shapes/compute_aabb/kernels/Spherocylinder.hpp>  // for mundy::shapes::compute_aabb::kernels::Spherocylinder

namespace mundy {

namespace shapes {

namespace compute_aabb {

namespace kernels {

// \name Constructors and destructor
//{

Spherocylinder::Spherocylinder(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "Spherocylinder: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(Spherocylinder::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shapes::Spherocylinders::get_node_coord_field_name();
  const std::string element_radius_field_name = mundy::shapes::Spherocylinders::get_element_radius_field_name();
  const std::string element_length_field_name = mundy::shapes::Spherocylinders::get_element_length_field_name();
  const std::string element_aabb_field_name = valid_fixed_params.get<std::string>("element_aabb_field_name");

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  element_length_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_length_field_name);
  element_aabb_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_aabb_field_name);

  MUNDY_THROW_ASSERT(node_coord_field_ptr_ != nullptr, std::invalid_argument,
                     "Spherocylinder: node_coord_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(element_radius_field_ptr_ != nullptr, std::invalid_argument,
                     "Spherocylinder: element_radius_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(element_length_field_ptr_ != nullptr, std::invalid_argument,
                     "Spherocylinder: element_length_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(element_aabb_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: element_aabb_field_ptr cannot be a nullptr. Check that the field exists.");

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(part != nullptr, std::invalid_argument,
                         "Sphere: Part " << part_name << " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> Spherocylinder::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void Spherocylinder::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(Spherocylinder::get_valid_mutable_params());

  // Fill the internal members using the given parameter list.
  buffer_distance_ = valid_mutable_params.get<double>("buffer_distance");
}
//}

// \name Actions
//{

void Spherocylinder::execute(const stk::mesh::Selector &spherocylinder_selector) {
  // Get references to internal members so we aren't passing around *this
  stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
  stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
  stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
  const double buffer_distance = buffer_distance_;

  stk::mesh::Selector locally_owned_intersection_with_valid_entity_parts =
      stk::mesh::selectIntersection(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() &
      spherocylinder_selector;
  stk::mesh::for_each_entity_run(
      *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), stk::topology::ELEMENT_RANK,
      locally_owned_intersection_with_valid_entity_parts,
      [&node_coord_field, &element_radius_field, &element_aabb_field, &buffer_distance](
          [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &spherocylinder_element) {
        stk::mesh::Entity const *nodes = bulk_data.begin_nodes(spherocylinder_element);
        double *left_endpt_coords = stk::mesh::field_data(node_coord_field, nodes[0]);
        double *right_endpt_coords = stk::mesh::field_data(node_coord_field, nodes[2]);
        double *radius = stk::mesh::field_data(element_radius_field, spherocylinder_element);
        // double *length = stk::mesh::field_data(element_length_field, spherocylinder_element);
        double *aabb = stk::mesh::field_data(element_aabb_field, spherocylinder_element);

        aabb[0] = std::min(left_endpt_coords[0], right_endpt_coords[0]) - radius[0] - buffer_distance;
        aabb[1] = std::min(left_endpt_coords[1], right_endpt_coords[1]) - radius[0] - buffer_distance;
        aabb[2] = std::min(left_endpt_coords[2], right_endpt_coords[2]) - radius[0] - buffer_distance;
        aabb[3] = std::max(left_endpt_coords[0], right_endpt_coords[0]) + radius[0] + buffer_distance;
        aabb[4] = std::max(left_endpt_coords[1], right_endpt_coords[1]) + radius[0] + buffer_distance;
        aabb[5] = std::max(left_endpt_coords[2], right_endpt_coords[2]) + radius[0] + buffer_distance;
      });
}
//}

}  // namespace kernels

}  // namespace compute_aabb

}  // namespace shapes

}  // namespace mundy
