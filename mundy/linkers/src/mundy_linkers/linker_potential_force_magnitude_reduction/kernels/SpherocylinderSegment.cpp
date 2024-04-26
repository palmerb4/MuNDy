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
/// \brief Definition of the ComputeSignedSeparationDistanceAndContactNormal's SpherocylinderSegment kernel.

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
#include <mundy_linkers/linker_potential_force_magnitude_reduction/kernels/SpherocylinderSegment.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSegment
#include <mundy_math/Vector3.hpp>   // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data, mundy::mesh::matrix3_field_data
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace linkers {

namespace linker_potential_force_magnitude_reduction {

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

  valid_fixed_params.print(std::cout, Teuchos::ParameterList::PrintOptions().showDoc(true).indent(2).showTypes(true));

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shapes::SpherocylinderSegments::get_node_coord_field_name();
  const std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
  const std::string linker_contact_normal_field_name =
      valid_fixed_params.get<std::string>("linker_contact_normal_field_name");
  const std::string linker_contact_points_field_name =
      valid_fixed_params.get<std::string>("linker_contact_points_field_name");
  const std::string linker_potential_force_magnitude_field_name =
      valid_fixed_params.get<std::string>("linker_potential_force_magnitude_field_name");

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name);
  linker_contact_normal_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_normal_field_name);
  linker_contact_points_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_points_field_name);
  linker_potential_force_magnitude_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_potential_force_magnitude_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_ASSERT(
        field_ptr != nullptr, std::invalid_argument,
        "SpherocylinderSegment: Field " << field_name << " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(linker_contact_normal_field_ptr_, linker_contact_normal_field_name);
  field_exists(linker_contact_points_field_ptr_, linker_contact_points_field_name);
  field_exists(linker_potential_force_magnitude_field_ptr_, linker_potential_force_magnitude_field_name);
  field_exists(node_coord_field_ptr_, node_coord_field_name);
  field_exists(node_force_field_ptr_, node_force_field_name);

  // Get the part pointers.
  const std::string name_of_linker_part_to_reduce_over =
      valid_fixed_params.get<std::string>("name_of_linker_part_to_reduce_over");
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
  linkers_part_to_reduce_over_ = meta_data_ptr_->get_part(name_of_linker_part_to_reduce_over);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SpherocylinderSegment::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void SpherocylinderSegment::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  // We don't have any valid mutable params, so this seems pointless but it's useful in that it will throw if the user
  // gives us parameters. This is useful for catching user errors.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(SpherocylinderSegment::get_valid_mutable_params());
}
//}

// \name Actions
//{

void SpherocylinderSegment::execute(const stk::mesh::Selector &spherocylinder_segment_selector) {
  // Communicate the linker fields.
  stk::mesh::communicate_field_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr_),
                                    {linker_contact_normal_field_ptr_, linker_potential_force_magnitude_field_ptr_});

  // Get references to internal members so we aren't passing around *this
  const stk::mesh::Field<double> &linker_contact_normal_field = *linker_contact_normal_field_ptr_;
  const stk::mesh::Field<double> &linker_contact_points_field = *linker_contact_points_field_ptr_;
  const stk::mesh::Field<double> &linker_potential_force_magnitude_field = *linker_potential_force_magnitude_field_ptr_;
  const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
  const stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
  stk::mesh::Part &linkers_part_to_reduce_over = *linkers_part_to_reduce_over_;

  stk::mesh::Selector locally_owned_intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() &
      spherocylinder_segment_selector;
  stk::mesh::for_each_entity_run(
      *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), stk::topology::ELEMENT_RANK,
      locally_owned_intersection_with_valid_entity_parts,
      [&linker_contact_normal_field, &linker_contact_points_field, &linker_potential_force_magnitude_field,
       &node_coord_field, &node_force_field, &linkers_part_to_reduce_over](
          const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &spherocylinder_segment) {
        // Get our nodes and their force

        // The force experienced by each node is determined such that the induced center of mass force and torque are
        // the same as that felt by a spherocylinder with the same geometry and force distribution.
        //
        // This requires that the force applied to each node be scaled by some scalars left_sf and right_sf, where
        // left_sf + right_sf = 1. These scale factors will depend on the ratio of the distance from the contact point
        // (p) to the left and right endpoints (x1, x2) of the rod. This ratio can be attained without square roots only
        // vectors in the lab frame:
        //  distance_ratio = (x2 - p) dot (x2 - c) / ((x1 - p) dot (x1 - c)) If we denote the
        // where c is the center of mass of the spherocylinder. We'll assume that the spherocylinder has uniform density
        // such that c lies at the geometric center of the rod. c = (x1 + x2) / 2
        //  distance_ratio = (x2 - p) dot (x2 - (x1 + x2) / 2) / ((x1 - p) dot (x1 - (x1 + x2) / 2))
        //                 = -(x2 - p) dot (x2 - x1) / ((x1 - p) dot (x2 - x1))
        // The scale factors are given by:
        //  left_sf = distance_ratio * right_sf
        //  right_sf = 1 / (1 + distance_ratio)
        // Of course, there are special degeneracies:
        //   1. If the numerator (x2 - p) dot (x2 - x1) and denominator (x1 - p) dot (x2 - x1) are both zero, then
        //   left_sf = right_sf = 1/2
        //   2. If the numerator is non-zero but the denominator is zero, then left_sf = 1 and right_sf = 0
        const stk::mesh::Entity &left_node = bulk_data.begin_nodes(spherocylinder_segment)[0];
        const stk::mesh::Entity &right_node = bulk_data.begin_nodes(spherocylinder_segment)[1];

        auto left_node_coord = mundy::mesh::vector3_field_data(node_coord_field, left_node);
        auto right_node_coord = mundy::mesh::vector3_field_data(node_coord_field, right_node);

        auto left_node_force = mundy::mesh::vector3_field_data(node_force_field, left_node);
        auto right_node_force = mundy::mesh::vector3_field_data(node_force_field, right_node);

        // Loop over the connected constraint rank entities
        const unsigned num_constraint_rank_conn =
            bulk_data.num_connectivity(spherocylinder_segment, stk::topology::CONSTRAINT_RANK);
        const stk::mesh::Entity *connected_linkers =
            bulk_data.begin(spherocylinder_segment, stk::topology::CONSTRAINT_RANK);

        for (unsigned i = 0; i < num_constraint_rank_conn; i++) {
          const stk::mesh::Entity &connected_linker = connected_linkers[i];
          MUNDY_THROW_ASSERT(bulk_data.is_valid(connected_linker), std::logic_error,
                             "SpherocylinderSegment: connected_linker is not valid.");

          const bool is_reduction_linker = bulk_data.bucket(connected_linker).member(linkers_part_to_reduce_over);

          if (is_reduction_linker) {
            // The contact normal stored on a linker points from the left element to the right element. This is
            // important, as it means we should multiply by -1 if we are the right element.
            const bool are_we_the_left_spherocylinder_segment =
                (bulk_data.begin(connected_linker, stk::topology::ELEMENT_RANK)[0] == spherocylinder_segment);
            const double sign = are_we_the_left_spherocylinder_segment ? 1.0 : -1.0;
            auto contact_normal = mundy::math::get_vector3_view<double>(
                stk::mesh::field_data(linker_contact_normal_field, connected_linker));
            auto contact_point = mundy::math::get_vector3_view<double>(
                stk::mesh::field_data(linker_contact_points_field, connected_linker) +
                3 * !are_we_the_left_spherocylinder_segment);
            double potential_force_magnitude =
                stk::mesh::field_data(linker_potential_force_magnitude_field, connected_linker)[0];

            const double numerator =
                -mundy::math::dot(right_node_coord - contact_point, right_node_coord - left_node_coord);
            const double denominator =
                mundy::math::dot(left_node_coord - contact_point, right_node_coord - left_node_coord);
            const bool numerator_near_zero = std::abs(numerator) < 1e-12;      // TODO(replace with fancy tol)
            const bool denominator_near_zero = std::abs(denominator) < 1e-12;  // TODO(replace with fancy tol)

            const auto contact_force = sign * contact_normal * potential_force_magnitude;

            if (numerator_near_zero && denominator_near_zero) {
              // Special case 1. Rod of length zero.
              const double left_sf = 0.5;
              const double right_sf = 0.5;
              left_node_force += contact_force * left_sf;
              right_node_force += contact_force * right_sf;
            } else if (denominator_near_zero) {
              // Special case 2. Perfect contact with the left node.
              left_node_force += contact_force;
            } else {
              const double distance_ratio = numerator / denominator;
              const double right_sf = 1.0 / (1.0 + distance_ratio);
              const double left_sf = distance_ratio * right_sf;

              left_node_force += contact_force * left_sf;
              right_node_force += contact_force * right_sf;
            }
          }
        }
      });
}
//}

}  // namespace kernels

}  // namespace linker_potential_force_magnitude_reduction

}  // namespace linkers

}  // namespace mundy
