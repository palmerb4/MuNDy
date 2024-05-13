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

/// \file SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact.cpp
/// \brief Definition of the EvaluateLinkerPotentials'
/// SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact kernel.

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
#include <mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact
#include <mundy_math/Vector3.hpp>                  // for mundy::math::Vector3
#include <mundy_math/distance/SegmentSegment.hpp>  // for mundy::math::distance::distance_sq_from_point_to_line_segment
#include <mundy_mesh/BulkData.hpp>                 // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace linkers {

namespace evaluate_linker_potentials {

namespace kernels {

// \name Constructors and destructor
//{

SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::
    SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                        const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(
      bulk_data_ptr_ != nullptr, std::invalid_argument,
      "SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(
      SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coords_field_name =
      mundy::shapes::SpherocylinderSegments::get_node_coord_field_name();
  const std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
  const std::string element_radius_field_name = mundy::shapes::SpherocylinderSegments::get_element_radius_field_name();
  const std::string linker_potential_force_field_name =
      valid_fixed_params.get<std::string>("linker_potential_force_field_name");
  const std::string linker_signed_separation_distance_field_name =
      valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");
  const std::string linker_tangential_displacement_field_name =
      valid_fixed_params.get<std::string>("linker_tangential_displacement_field_name");
  const std::string linker_contact_normal_field_name =
      valid_fixed_params.get<std::string>("linker_contact_normal_field_name");
  const std::string linker_contact_points_field_name =
      valid_fixed_params.get<std::string>("linker_contact_points_field_name");

  node_coords_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coords_field_name);
  node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  linker_potential_force_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_potential_force_field_name);
  linker_signed_separation_distance_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_signed_separation_distance_field_name);
  linker_tangential_displacement_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_tangential_displacement_field_name);
  linker_contact_normal_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_normal_field_name);
  linker_contact_points_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_points_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_ASSERT(field_ptr != nullptr, std::invalid_argument,
                       "SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact: Field "
                           << field_name << " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(node_coords_field_ptr_, node_coords_field_name);
  field_exists(node_velocity_field_ptr_, node_velocity_field_name);
  field_exists(element_radius_field_ptr_, element_radius_field_name);
  field_exists(linker_signed_separation_distance_field_ptr_, linker_signed_separation_distance_field_name);
  field_exists(linker_tangential_displacement_field_ptr_, linker_tangential_displacement_field_name);
  field_exists(linker_potential_force_field_ptr_, linker_potential_force_field_name);
  field_exists(linker_contact_normal_field_ptr_, linker_contact_normal_field_name);
  field_exists(linker_contact_points_field_ptr_, linker_contact_points_field_name);

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  Teuchos::Array<std::string> valid_sy_seg_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_spherocylinder_segment_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(part != nullptr, std::invalid_argument,
                         "SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact: Part "
                             << part_name << " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };  // parts_from_names

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  valid_sy_seg_parts_ = parts_from_names(*meta_data_ptr_, valid_sy_seg_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *>
SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::set_mutable_params(
    const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  // We don't have any valid mutable params, so this seems pointless but it's useful in that it will throw if the user
  // gives us parameters. This is useful for catching user errors.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(
      SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::get_valid_mutable_params());
}
//}

// \name Actions
//{

void SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::execute(
    const stk::mesh::Selector &sy_seg_sy_seg_linker_selector) {
  // Communicate the fields of downward connected entities.
  stk::mesh::communicate_field_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), {element_radius_field_ptr_});

  // Get references to internal members so we aren't passing around *this
  const stk::mesh::Field<double> &node_coords_field = *node_coords_field_ptr_;
  const stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
  const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
  const stk::mesh::Field<double> &linker_potential_force_field = *linker_potential_force_field_ptr_;
  const stk::mesh::Field<double> &linker_signed_separation_distance_field =
      *linker_signed_separation_distance_field_ptr_;
  const stk::mesh::Field<double> &linker_tangential_displacement_field = *linker_tangential_displacement_field_ptr_;
  const stk::mesh::Field<double> &linker_contact_normal_field = *linker_contact_normal_field_ptr_;
  const stk::mesh::Field<double> &linker_contact_points_field = *linker_contact_points_field_ptr_;

  // TODO(palmerb4): For now, we hardcode some of the parameters. We'll need to take them in as mutable params.
  const double mass = 1.0;
  const double youngs_modulus = 1.0e6;
  const double poissons_ratio = 0.3;
  const double shear_modulus = 0.5 * youngs_modulus / (1.0 + poissons_ratio);
  const double normal_spring_coeff = 4.0 / 3.0 * shear_modulus / (1.0 - poissons_ratio);
  const double tang_spring_coeff = 4.0 * shear_modulus / (2.0 - poissons_ratio);
  const double friction_coeff = 0.5;  // Typically between 0 and 1
  const double normal_damping_coeff = 0.0;
  const double tang_damping_coeff = 0.0;  // A good choice is 0.5 * normal_damping_coeff
  const double time_step_size = 1.0e-6;
  const double effective_mass = 0.5 * mass;
  const double effective_youngs_modulus = 0.5 * (youngs_modulus) / (1.0 - poissons_ratio * poissons_ratio);

  stk::mesh::Selector locally_owned_intersection_with_valid_entity_parts = stk::mesh::selectUnion(valid_entity_parts_) &
                                                                           meta_data_ptr_->locally_owned_part() &
                                                                           sy_seg_sy_seg_linker_selector;
  stk::mesh::for_each_entity_run(
      *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), stk::topology::CONSTRAINT_RANK,
      locally_owned_intersection_with_valid_entity_parts,
      [&node_coords_field, &node_velocity_field, &element_radius_field, &linker_potential_force_field,
       &linker_signed_separation_distance_field, &linker_tangential_displacement_field,
       &linker_contact_normal_field, &linker_contact_points_field, &time_step_size, &effective_mass,
       &effective_youngs_modulus, &normal_spring_coeff, &tang_spring_coeff,
       &normal_damping_coeff, &tang_damping_coeff, &friction_coeff]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                  const stk::mesh::Entity &sy_seg_sy_seg_linker) {
        // This is an expensive kernel, so we only run it if the particles actually overlap.
        const double linker_signed_separation_distance =
            stk::mesh::field_data(linker_signed_separation_distance_field, sy_seg_sy_seg_linker)[0];
        auto tang_disp =
            mundy::mesh::vector3_field_data(linker_tangential_displacement_field, sy_seg_sy_seg_linker);
        if (linker_signed_separation_distance > 0) {
          // No contact, reset the tangential displacement
          tang_disp.set(0.0, 0.0, 0.0);
        } else {
          // Contact, compute the contact forces

          // Fetch the attached entities. Use references to avoid copying or pointer dereferencing.
          const stk::mesh::Entity &left_sy_seg_element = bulk_data.begin_elements(sy_seg_sy_seg_linker)[0];
          const stk::mesh::Entity &right_sy_seg_element = bulk_data.begin_elements(sy_seg_sy_seg_linker)[1];
          const stk::mesh::Entity *left_sy_seg_nodes = bulk_data.begin_nodes(left_sy_seg_element);
          const stk::mesh::Entity *right_sy_seg_nodes = bulk_data.begin_nodes(right_sy_seg_element);

          // Determine the velocity of the contact points
          auto get_contact_point_velocity = [](const mundy::math::Vector3<double, auto> &contact_point,
                                               const stk::mesh::Entity *nodes,
                                               const stk::mesh::Field<double> &node_velocity_field,
                                               const stk::mesh::Field<double> &node_coords_field) {
            //        x
            //       /
            // p0---c---p1
            //
            // x translates and rotates with rigid body motion caused by rotation/translation of the center of mass of
            // line p0--p1. It also translates based on the extension/contraction of this line. To compute this motion,
            // we need to know the translational/rotational velocity of the center, the rate of change of length of the
            // rod, and the archlength of the point along the rod. There are some nasty degeneracies here, so we rely on
            // some helper functions to do the necessary checks. One the archlength is know, we can compute the velocity
            // of the the contact point using \dot{x} = (\dot{p1} - \dot{p0}) * archlength + \dot{p0}.
            //
            // TODO(palmerb4): We could store the contact archlength for each particle when we compute the contact
            // invariants. This would avoid the expensive point-to-line eveluation here.
            const auto pos0 = mundy::mesh::vector3_field_data(node_coords_field, nodes[0]);
            const auto pos1 = mundy::mesh::vector3_field_data(node_coords_field, nodes[1]);
            const auto vel0 = mundy::mesh::vector3_field_data(node_velocity_field, nodes[0]);
            const auto vel1 = mundy::mesh::vector3_field_data(node_velocity_field, nodes[1]);

            double archlength;  // in [0, 1] with 0 representing p0 and 1 representing p1.
            mundy::math::distance::distance_sq_from_point_to_line_segment(contact_point, pos0, pos1, nullptr,
                                                                          &archlength);
            return (vel1 - vel0) * archlength + vel0;
          };  // get_contact_point_velocity

          const auto left_contact_normal =
              mundy::mesh::vector3_field_data(linker_contact_normal_field, sy_seg_sy_seg_linker);
          const auto left_cp =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(linker_contact_points_field, sy_seg_sy_seg_linker));
          const auto right_cp =
              mundy::math::get_vector3_view<double>(stk::mesh::field_data(linker_contact_points_field, sy_seg_sy_seg_linker) + 3);
          const auto left_cp_vel =
              get_contact_point_velocity(left_cp, left_sy_seg_nodes, node_velocity_field, node_coords_field);
          const auto right_cp_vel =
              get_contact_point_velocity(right_cp, right_sy_seg_nodes, node_velocity_field, node_coords_field);

          // Compute the relative normal and tangential velocities
          const auto rel_cp_vel = right_cp_vel - left_cp_vel;
          const auto rel_vel_normal = mundy::math::dot(rel_cp_vel, left_contact_normal) * left_contact_normal;
          const auto rel_vel_tang = rel_cp_vel - rel_vel_normal;

          // Compute the tangential displacement (history variable)
          // First add on the current tangential displacement, then project onto the tangent plane.
          tang_disp += rel_vel_tang * time_step_size;
          tang_disp -= mundy::math::dot(tang_disp, left_contact_normal) * left_contact_normal;
          const double tang_disp_mag = mundy::math::norm(tang_disp);

          // Compute the contact force
          // Note, for LAMMPS' delta is the negative of our signed separation distance.
          const double left_radius = stk::mesh::field_data(element_radius_field, left_sy_seg_element)[0];
          const double right_radius = stk::mesh::field_data(element_radius_field, right_sy_seg_element)[0];
          const double effective_radius = (left_radius * right_radius) / (left_radius + right_radius);
          const double hertz_poly = std::sqrt(effective_radius * linker_signed_separation_distance);
          auto normal_force =
              hertz_poly * (-normal_spring_coeff * linker_signed_separation_distance * left_contact_normal -
                            effective_mass * normal_damping_coeff * rel_vel_normal);
          auto tang_force =
              hertz_poly * (tang_spring_coeff * tang_disp - effective_mass * tang_damping_coeff * rel_vel_tang);

          // Rescale frictional displacements and forces if needed to satisfy the Coulomb friction law
          // Ft = min(friction_coeff*Fn, Ft)
          const double normal_force_mag = mundy::math::norm(normal_force);
          const double tang_force_mag = mundy::math::norm(tang_force);
          const double scaled_normal_force_mag = friction_coeff * normal_force_mag;
          if (tang_force_mag > scaled_normal_force_mag) {
            if (tang_disp_mag != 0.0) {  // TODO(palmerb4): Exact comparison to 0.0 is bad. Use a tol.
              tang_disp = (scaled_normal_force_mag / tang_force_mag) *
                              (tang_disp + effective_mass * tang_damping_coeff * rel_vel_tang / tang_spring_coeff) -
                          effective_mass * tang_damping_coeff * rel_vel_tang / tang_spring_coeff;
              tang_force *= scaled_normal_force_mag / tang_force_mag;
            } else {
              tang_force.set(0.0, 0.0, 0.0);
            }
          }

          // Save the contact force (Forces are equal and opposite, so we only save the left force)
          mundy::mesh::vector3_field_data(linker_potential_force_field, sy_seg_sy_seg_linker) +=
              normal_force + tang_force;
        }
      });
}

//}

}  // namespace kernels

}  // namespace evaluate_linker_potentials

}  // namespace linkers

}  // namespace mundy
