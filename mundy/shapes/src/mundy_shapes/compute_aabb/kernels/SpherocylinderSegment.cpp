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

/// \file SpherocylinderSegment.cpp
/// \brief Definition of the ComputeAABB's SpherocylinderSegment kernel.

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
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>    // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data, mundy::mesh::matrix3_field_data
#include <mundy_shapes/SpherocylinderSegments.hpp>                      // for mundy::shapes::SpherocylinderSegments
#include <mundy_shapes/compute_aabb/kernels/SpherocylinderSegment.hpp>  // for mundy::shapes::compute_aabb::kernels::SpherocylinderSegment

namespace mundy {

namespace shapes {

namespace compute_aabb {

namespace kernels {

// \name Constructors and destructor
//{

SpherocylinderSegment::SpherocylinderSegment(mundy::mesh::BulkData *const bulk_data_ptr,
                                             const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "SpherocylinderSegment: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(SpherocylinderSegment::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shapes::SpherocylinderSegments::get_node_coord_field_name();
  const std::string element_radius_field_name = mundy::shapes::SpherocylinderSegments::get_element_radius_field_name();
  const std::string element_aabb_field_name = valid_fixed_params.get<std::string>("element_aabb_field_name");

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  element_aabb_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_aabb_field_name);

  MUNDY_THROW_ASSERT(node_coord_field_ptr_ != nullptr, std::invalid_argument,
                     "SpherocylinderSegment: node_coord_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(
      element_radius_field_ptr_ != nullptr, std::invalid_argument,
      "SpherocylinderSegment: element_radius_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(element_aabb_field_ptr_ != nullptr, std::invalid_argument,
                     "SpherocylinderSegment: element_aabb_field_ptr cannot be a nullptr. Check that the field exists.");

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(
          part != nullptr, std::invalid_argument,
          "SpherocylinderSegment: Part " << part_name << " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SpherocylinderSegment::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void SpherocylinderSegment::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(SpherocylinderSegment::get_valid_mutable_params());

  // Fill the internal members using the given parameter list.
  buffer_distance_ = valid_mutable_params.get<double>("buffer_distance");
}
//}

// \name Actions
//{

void SpherocylinderSegment::execute(const stk::mesh::Selector &spherocylinder_segment_selector) {
  // Get references to internal members so we aren't passing around *this
  stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
  stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
  stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
  const double buffer_distance = buffer_distance_;

  stk::mesh::Selector locally_owned_intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() &
      spherocylinder_segment_selector;
  stk::mesh::for_each_entity_run(
      *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), stk::topology::ELEMENT_RANK,
      locally_owned_intersection_with_valid_entity_parts,
      [&node_coord_field, &element_radius_field, &element_aabb_field, &buffer_distance](
          [[maybe_unused]] const stk::mesh::BulkData &bulk_data,
          const stk::mesh::Entity &spherocylinder_segment_element) {
        // Element data
        const double radius = stk::mesh::field_data(element_radius_field, spherocylinder_segment_element)[0];

        // Node data
        const stk::mesh::Entity *nodes = bulk_data.begin_nodes(spherocylinder_segment_element);
        const stk::mesh::Entity &left_node = nodes[0];
        const stk::mesh::Entity &right_node = nodes[1];
        const auto left_node_coord = mundy::mesh::vector3_field_data(node_coord_field, left_node);
        const auto right_node_coord = mundy::mesh::vector3_field_data(node_coord_field, right_node);

        // Populate the AABB.
        double *aabb = stk::mesh::field_data(element_aabb_field, spherocylinder_segment_element);
        auto min_xyz = mundy::math::get_vector3_view<double>(aabb);
        auto max_xyz = mundy::math::get_vector3_view<double>(aabb + 3);

        min_xyz[0] = std::min(left_node_coord[0], right_node_coord[0]) - radius - buffer_distance;
        min_xyz[1] = std::min(left_node_coord[1], right_node_coord[1]) - radius - buffer_distance;
        min_xyz[2] = std::min(left_node_coord[2], right_node_coord[2]) - radius - buffer_distance;
        max_xyz[0] = std::max(left_node_coord[0], right_node_coord[0]) + radius + buffer_distance;
        max_xyz[1] = std::max(left_node_coord[1], right_node_coord[1]) + radius + buffer_distance;
        max_xyz[2] = std::max(left_node_coord[2], right_node_coord[2]) + radius + buffer_distance;
      });
}
//}

}  // namespace kernels

}  // namespace compute_aabb

}  // namespace shapes

}  // namespace mundy
