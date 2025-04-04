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

/// \file SpherocylinderSpherocylinderLinker.cpp
/// \brief Definition of the ComputeSignedSeparationDistanceAndContactNormal's SpherocylinderSpherocylinderLinker
/// kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>          // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SpherocylinderSpherocylinderLinker.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSpherocylinderLinker
#include <mundy_math/Quaternion.hpp>               // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>                  // for mundy::math::Vector3
#include <mundy_math/distance/SegmentSegment.hpp>  // for mundy::math::distance::distance_sq_between_line_segments
#include <mundy_mesh/BulkData.hpp>                 // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data, mundy::mesh::matrix3_field_data
#include <mundy_shapes/Spherocylinders.hpp>  // for mundy::shapes::Spherocylinders

namespace mundy {

namespace linkers {

namespace compute_signed_separation_distance_contact_normal_and_contact_points {

namespace kernels {

// \name Constructors and destructor
//{

SpherocylinderSpherocylinderLinker::SpherocylinderSpherocylinderLinker(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                       const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "SpherocylinderSpherocylinderLinker: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(SpherocylinderSpherocylinderLinker::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shapes::Spherocylinders::get_node_coord_field_name();
  const std::string element_radius_field_name = mundy::shapes::Spherocylinders::get_element_radius_field_name();
  const std::string element_length_field_name = mundy::shapes::Spherocylinders::get_element_length_field_name();
  const std::string element_orientation_field_name =
      mundy::shapes::Spherocylinders::get_element_orientation_field_name();
  const std::string linker_contact_normal_field_name =
      valid_fixed_params.get<std::string>("linker_contact_normal_field_name");
  const std::string linker_signed_separation_distance_field_name =
      valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");
  const std::string linker_contact_points_field_name =
      valid_fixed_params.get<std::string>("linker_contact_points_field_name");
  const std::string linked_entities_field_name = NeighborLinkers::get_linked_entities_field_name();

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  element_length_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_length_field_name);
  element_orientation_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_orientation_field_name);
  linker_contact_normal_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_normal_field_name);
  linker_signed_separation_distance_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_signed_separation_distance_field_name);
  linker_contact_points_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_points_field_name);
  linked_entities_field_ptr_ = meta_data_ptr_->get_field<LinkedEntitiesFieldType::value_type>(
      stk::topology::CONSTRAINT_RANK, linked_entities_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_REQUIRE(field_ptr != nullptr, std::invalid_argument,
                        std::string("SpherocylinderSpherocylinderLinker: Field ") + field_name +
                            " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(node_coord_field_ptr_, node_coord_field_name);
  field_exists(element_radius_field_ptr_, element_radius_field_name);
  field_exists(element_length_field_ptr_, element_length_field_name);
  field_exists(element_orientation_field_ptr_, element_orientation_field_name);
  field_exists(linker_contact_normal_field_ptr_, linker_contact_normal_field_name);
  field_exists(linker_signed_separation_distance_field_ptr_, linker_signed_separation_distance_field_name);
  field_exists(linker_contact_points_field_ptr_, linker_contact_points_field_name);
  field_exists(linked_entities_field_ptr_, linked_entities_field_name);

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  Teuchos::Array<std::string> valid_spherocylinder_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_spherocylinder_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_REQUIRE(part != nullptr, std::invalid_argument,
                          std::string("SpherocylinderSpherocylinderLinker: Part ") + part_name +
                              " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  valid_spherocylinder_parts_ = parts_from_names(*meta_data_ptr_, valid_spherocylinder_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SpherocylinderSpherocylinderLinker::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void SpherocylinderSpherocylinderLinker::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(SpherocylinderSpherocylinderLinker::get_valid_mutable_params());
}
//}

// \name Actions
//{

void SpherocylinderSpherocylinderLinker::execute(
    const stk::mesh::Selector &spherocylinder_spherocylinder_linker_selector) {
  // Communicate ghosted fields.
  stk::mesh::communicate_field_data(*bulk_data_ptr_, {node_coord_field_ptr_, element_radius_field_ptr_,
                                                      element_length_field_ptr_, element_orientation_field_ptr_});

  // Get references to internal members so we aren't passing around *this
  const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
  const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
  const stk::mesh::Field<double> &element_length_field = *element_length_field_ptr_;
  const stk::mesh::Field<double> &element_orientation_field = *element_orientation_field_ptr_;
  const LinkedEntitiesFieldType &linked_entities_field = *linked_entities_field_ptr_;
  stk::mesh::Field<double> &linker_contact_normal_field = *linker_contact_normal_field_ptr_;
  stk::mesh::Field<double> &linker_contact_points_field = *linker_contact_points_field_ptr_;
  stk::mesh::Field<double> &linker_signed_separation_distance_field = *linker_signed_separation_distance_field_ptr_;

  // At the end of this loop, all locally owned and ghosted linkers will be up-to-date.
  stk::mesh::Selector intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & spherocylinder_spherocylinder_linker_selector;
  mundy::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, intersection_with_valid_entity_parts,
      [&node_coord_field, &element_radius_field, &element_length_field, &element_orientation_field,
       &linker_contact_normal_field, &linker_contact_points_field, &linker_signed_separation_distance_field,
       &linked_entities_field](const stk::mesh::BulkData &bulk_data,
                               const stk::mesh::Entity &spherocylinder_spherocylinder_linker) {
        // Use references to avoid copying entities
        const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(linked_entities_field, spherocylinder_spherocylinder_linker));
        const stk::mesh::Entity &spherocylinder1_element = bulk_data.get_entity(key_t_ptr[0]);
        const stk::mesh::Entity &spherocylinder2_element = bulk_data.get_entity(key_t_ptr[1]);

        MUNDY_THROW_ASSERT(bulk_data.is_valid(spherocylinder1_element), std::invalid_argument,
                           "SpherocylinderSpherocylinderLinker: spherocylinder1_element entity is not valid.");
        MUNDY_THROW_ASSERT(bulk_data.is_valid(spherocylinder2_element), std::invalid_argument,
                           "SpherocylinderSpherocylinderLinker: spherocylinder2_element entity is not valid.");

        const stk::mesh::Entity &spherocylinder1_node = bulk_data.begin_nodes(spherocylinder1_element)[0];
        const stk::mesh::Entity &spherocylinder2_node = bulk_data.begin_nodes(spherocylinder2_element)[0];

        // Get the spherocylinder data
        const auto spherocylinder1_center_coord =
            mundy::mesh::vector3_field_data(node_coord_field, spherocylinder1_node);
        const auto spherocylinder2_center_coord =
            mundy::mesh::vector3_field_data(node_coord_field, spherocylinder2_node);

        const double spherocylinder1_radius = stk::mesh::field_data(element_radius_field, spherocylinder1_element)[0];
        const double spherocylinder2_radius = stk::mesh::field_data(element_radius_field, spherocylinder2_element)[0];

        const double spherocylinder1_length = stk::mesh::field_data(element_length_field, spherocylinder1_element)[0];
        const double spherocylinder2_length = stk::mesh::field_data(element_length_field, spherocylinder2_element)[0];

        const auto spherocylinder1_orientation = mundy::math::get_quaternion_view<double>(
            stk::mesh::field_data(element_orientation_field, spherocylinder1_element));
        const auto spherocylinder2_orientation = mundy::math::get_quaternion_view<double>(
            stk::mesh::field_data(element_orientation_field, spherocylinder2_element));

        // Find the endpoints of the spherocylinder
        // Note, the orientation maps the reference configuration to the current configuration and in the reference
        // configuration the spherocylinder is aligned with the x-axis.
        const auto tangent_vector1 = spherocylinder1_orientation * mundy::math::Vector3<double>(1.0, 0.0, 0.0);
        const auto tangent_vector2 = spherocylinder2_orientation * mundy::math::Vector3<double>(1.0, 0.0, 0.0);
        const auto left_endpoint1 = spherocylinder1_center_coord - 0.5 * tangent_vector1 * spherocylinder1_length;
        const auto left_endpoint2 = spherocylinder2_center_coord - 0.5 * tangent_vector2 * spherocylinder2_length;
        const auto right_endpoint1 = spherocylinder1_center_coord + 0.5 * tangent_vector1 * spherocylinder1_length;
        const auto right_endpoint2 = spherocylinder2_center_coord + 0.5 * tangent_vector2 * spherocylinder2_length;

        // Compute the separation distance and contact point along the center line of each spherocylinder
        mundy::math::Vector3<double> closest_point1;
        mundy::math::Vector3<double> closest_point2;
        double t1;
        double t2;
        const double distance = Kokkos::sqrt(mundy::math::distance::distance_sq_between_line_segments(
            left_endpoint1, right_endpoint1, left_endpoint2, right_endpoint2, closest_point1, closest_point2, t1, t2));

        // Compute the separation distance and contact normal
        const auto left_to_right_vector = closest_point2 - closest_point1;
        const double radius_sum = spherocylinder1_radius + spherocylinder2_radius;
        const double separation_distance = distance - radius_sum;
        const double inv_distance = 1.0 / distance;

        // Set the separation distance and contact normal
        // Notice that the contact normal points from the left sphere to the right sphere
        // It is the normal to the left sphere and negative the normal of the right sphere.
        auto contact_normal = mundy::math::get_vector3_view<double>(
            stk::mesh::field_data(linker_contact_normal_field, spherocylinder_spherocylinder_linker));
        auto spherocylinder1_contact_point = mundy::math::get_vector3_view<double>(
            stk::mesh::field_data(linker_contact_points_field, spherocylinder_spherocylinder_linker));
        auto spherocylinder2_contact_point = mundy::math::get_vector3_view<double>(
            stk::mesh::field_data(linker_contact_points_field, spherocylinder_spherocylinder_linker) + 3);
        double *signed_separation_distance =
            stk::mesh::field_data(linker_signed_separation_distance_field, spherocylinder_spherocylinder_linker);
        signed_separation_distance[0] = separation_distance;
        contact_normal = left_to_right_vector * inv_distance;
        // spherocylinder1_contact_point = closest_point1 + spherocylinder1_radius * contact_normal;
        // spherocylinder2_contact_point = closest_point2 - spherocylinder2_radius * contact_normal;
        spherocylinder1_contact_point = closest_point1;
        spherocylinder2_contact_point = closest_point2;
      });
}
//}

}  // namespace kernels

}  // namespace compute_signed_separation_distance_contact_normal_and_contact_points

}  // namespace linkers

}  // namespace mundy
