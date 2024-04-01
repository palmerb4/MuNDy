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

/// \file SphereSpherocylinderSegmentLinker.cpp
/// \brief Definition of the ComputeSignedSeparationDistanceAndContactNormal's SphereSpherocylinderSegmentLinker kernel.

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
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_linkers/compute_signed_separation_distance_and_contact_normal/kernels/SphereSpherocylinderSegmentLinker.hpp>  // for mundy::linkers::...::kernels::SphereSpherocylinderSegmentLinker
#include <mundy_math/Vector3.hpp>                  // for mundy::math::Vector3
#include <mundy_math/distance/SegmentSegment.hpp>  // for mundy::math::distance::distance_sq_from_point_to_line_segment
#include <mundy_mesh/BulkData.hpp>                 // for mundy::mesh::BulkData
#include <mundy_shapes/Spheres.hpp>                // for mundy::shapes::Spheres
#include <mundy_shapes/Spherocylinders.hpp>        // for mundy::shapes::Spherocylinders

namespace mundy {

namespace linkers {

namespace compute_signed_separation_distance_and_contact_normal {

namespace kernels {

// \name Constructors and destructor
//{

SphereSpherocylinderSegmentLinker::SphereSpherocylinderSegmentLinker(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                     const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "SphereSpherocylinderSegmentLinker: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(SphereSpherocylinderSegmentLinker::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shapes::Spheres::get_node_coord_field_name();
  const std::string element_radius_field_name = mundy::shapes::Spheres::get_element_radius_field_name();
  const std::string linker_contact_normal_field_name =
      valid_fixed_params.get<std::string>("linker_contact_normal_field_name");
  const std::string linker_signed_separation_distance_field_name =
      valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  linker_contact_normal_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_normal_field_name);
  linker_signed_separation_distance_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_signed_separation_distance_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_ASSERT(field_ptr != nullptr, std::invalid_argument,
                       "SphereSpherocylinderSegmentLinker: Field "
                           << field_name << " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(node_coord_field_ptr_, node_coord_field_name);
  field_exists(element_radius_field_ptr_, element_radius_field_name);
  field_exists(linker_contact_normal_field_ptr_, linker_contact_normal_field_name);
  field_exists(linker_signed_separation_distance_field_ptr_, linker_signed_separation_distance_field_name);

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  Teuchos::Array<std::string> valid_sphere_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_sphere_part_names");
  Teuchos::Array<std::string> valid_spherocylinder_segment_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_spherocylinder_segment_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(part != nullptr, std::invalid_argument,
                         "SphereSpherocylinderSegmentLinker: Part "
                             << part_name << " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  valid_sphere_parts_ = parts_from_names(*meta_data_ptr_, valid_sphere_part_names);
  valid_spherocylinder_segment_parts_ = parts_from_names(*meta_data_ptr_, valid_spherocylinder_segment_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SphereSpherocylinderSegmentLinker::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void SphereSpherocylinderSegmentLinker::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(SphereSpherocylinderSegmentLinker::get_valid_mutable_params());
}
//}

// \name Actions
//{

void SphereSpherocylinderSegmentLinker::execute(
    const stk::mesh::Selector &sphere_spherocylinder_segment_linker_selector) {
  // Get references to internal members so we aren't passing around *this
  stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
  stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
  stk::mesh::Field<double> &linker_contact_normal_field = *linker_contact_normal_field_ptr_;
  stk::mesh::Field<double> &linker_signed_separation_distance_field = *linker_signed_separation_distance_field_ptr_;

  stk::mesh::Selector locally_owned_intersection_with_valid_entity_parts =
      stk::mesh::selectIntersection(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() &
      sphere_spherocylinder_segment_linker_selector;
  stk::mesh::for_each_entity_run(
      *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), stk::topology::CONSTRAINT_RANK,
      locally_owned_intersection_with_valid_entity_parts,
      [&node_coord_field, &element_radius_field, &linker_contact_normal_field,
       &linker_signed_separation_distance_field](const stk::mesh::BulkData &bulk_data,
                                                 const stk::mesh::Entity &sphere_spherocylinder_segment_linker) {
        // Use references to avoid copying entities
        const stk::mesh::Entity &sphere_element = bulk_data.begin_elements(sphere_spherocylinder_segment_linker)[0];
        const stk::mesh::Entity &spherocylinder_segment_element =
            bulk_data.begin_elements(sphere_spherocylinder_segment_linker)[1];
        const stk::mesh::Entity &sphere_node = bulk_data.begin_nodes(sphere_element)[0];
        const stk::mesh::Entity &spherocylinder_segment_left_node =
            bulk_data.begin_nodes(spherocylinder_segment_element)[0];
        const stk::mesh::Entity &spherocylinder_segment_right_node =
            bulk_data.begin_nodes(spherocylinder_segment_element)[1];

        // Get the sphere data
        const auto sphere_center_coord =
            mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_coord_field, sphere_node));
        const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];

        // Get the spherocylinder_segment data
        const auto spherocylinder_segment_left_endpoint_coord =
            mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_coord_field, spherocylinder_segment_node));
        const auto spherocylinder_segment_right_endpoint_coord =
            mundy::math::get_vector3_view<double>(stk::mesh::field_data(node_coord_field, spherocylinder_segment_node));
        const double spherocylinder_segment_radius =
            stk::mesh::field_data(element_radius_field, spherocylinder_segment_element)[0];

        // Compute the separation distance and contact point along the center line of the spherocylinder_segment
        mundy::math::Vector3<double> closest_point;
        const double distance = std::sqrt(mundy::math::distance::distance_sq_from_point_to_line_segment(
            sphere_center_coord, spherocylinder_segment_left_endpoint_coord,
            spherocylinder_segment_right_endpoint_coord, &closest_point));

        // Compute the separation distance and contact normal
        const auto left_to_right_vector = closest_point - sphere_center_coord;
        const double radius_sum = sphere_radius + spherocylinder_segment_radius;
        const double separation_distance = distance - radius_sum;
        const double inv_distance = 1.0 / distance;

        // Set the separation distance and contact normal
        // Notice that the contact normal points from the left sphere to the right sphere
        // It is the normal to the left sphere and negative the normal of the right sphere.
        auto contact_normal = mundy::math::get_vector3_view<double>(
            stk::mesh::field_data(linker_contact_normal_field, sphere_spherocylinder_segment_linker));
        double *signed_separation_distance =
            stk::mesh::field_data(linker_signed_separation_distance_field, sphere_spherocylinder_segment_linker);
        signed_separation_distance[0] = separation_distance;
        contact_normal = left_to_right_vector * inv_distance;
      });
}
//}

}  // namespace kernels

}  // namespace compute_signed_separation_distance_and_contact_normal

}  // namespace linkers

}  // namespace mundy
