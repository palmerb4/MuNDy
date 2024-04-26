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

/// \file SpherocylinderSegmentSpherocylinderSegmentHertzianContact.cpp
/// \brief Definition of the EvaluateLinkerPotentials' SpherocylinderSegmentSpherocylinderSegmentHertzianContact kernel.

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
#include <mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentHertzianContact.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSegmentSpherocylinderSegmentHertzianContact
#include <mundy_mesh/BulkData.hpp>                  // for mundy::mesh::BulkData
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace linkers {

namespace evaluate_linker_potentials {

namespace kernels {

// \name Constructors and destructor
//{

SpherocylinderSegmentSpherocylinderSegmentHertzianContact::SpherocylinderSegmentSpherocylinderSegmentHertzianContact(
    mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "SpherocylinderSegmentSpherocylinderSegmentHertzianContact: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(
      SpherocylinderSegmentSpherocylinderSegmentHertzianContact::get_valid_fixed_params());

  // Get the field pointers.
  const std::string element_radius_field_name = mundy::shapes::SpherocylinderSegments::get_element_radius_field_name();
  const std::string element_youngs_modulus_field_name =
      valid_fixed_params.get<std::string>("element_youngs_modulus_field_name");
  const std::string element_poissons_ratio_field_name =
      valid_fixed_params.get<std::string>("element_poissons_ratio_field_name");

  const std::string linker_potential_force_magnitude_field_name =
      valid_fixed_params.get<std::string>("linker_potential_force_magnitude_field_name");
  const std::string linker_signed_separation_distance_field_name =
      valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");

  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  element_youngs_modulus_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_youngs_modulus_field_name);
  element_poissons_ratio_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_poissons_ratio_field_name);

  linker_potential_force_magnitude_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_potential_force_magnitude_field_name);
  linker_signed_separation_distance_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_signed_separation_distance_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_ASSERT(field_ptr != nullptr, std::invalid_argument,
                       "SpherocylinderSegmentSpherocylinderSegmentHertzianContact: Field "
                           << field_name << " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(element_radius_field_ptr_, element_radius_field_name);
  field_exists(element_youngs_modulus_field_ptr_, element_youngs_modulus_field_name);
  field_exists(element_poissons_ratio_field_ptr_, element_poissons_ratio_field_name);
  field_exists(linker_potential_force_magnitude_field_ptr_, linker_potential_force_magnitude_field_name);
  field_exists(linker_signed_separation_distance_field_ptr_, linker_signed_separation_distance_field_name);

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  Teuchos::Array<std::string> valid_spherocylinder_segment_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_spherocylinder_segment_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(part != nullptr, std::invalid_argument,
                         "SpherocylinderSegmentSpherocylinderSegmentHertzianContact: Part "
                             << part_name << " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };  // parts_from_names

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  valid_spherocylinder_segment_parts_ = parts_from_names(*meta_data_ptr_, valid_spherocylinder_segment_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SpherocylinderSegmentSpherocylinderSegmentHertzianContact::get_valid_entity_parts()
    const {
  return valid_entity_parts_;
}

void SpherocylinderSegmentSpherocylinderSegmentHertzianContact::set_mutable_params(
    const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  // We don't have any valid mutable params, so this seems pointless but it's useful in that it will throw if the user
  // gives us parameters. This is useful for catching user errors.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(
      SpherocylinderSegmentSpherocylinderSegmentHertzianContact::get_valid_mutable_params());
}
//}

// \name Actions
//{

void SpherocylinderSegmentSpherocylinderSegmentHertzianContact::execute(
    const stk::mesh::Selector &spherocylinder_segment_spherocylinder_segment_linker_selector) {
  // Communicate the fields of downward connected entities.
  stk::mesh::communicate_field_data(
      *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_),
      {element_radius_field_ptr_, element_youngs_modulus_field_ptr_, element_poissons_ratio_field_ptr_});

  // Get references to internal members so we aren't passing around *this
  const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
  const stk::mesh::Field<double> &element_youngs_modulus_field = *element_youngs_modulus_field_ptr_;
  const stk::mesh::Field<double> &element_poissons_ratio_field = *element_poissons_ratio_field_ptr_;
  const stk::mesh::Field<double> &linker_potential_force_magnitude_field = *linker_potential_force_magnitude_field_ptr_;
  const stk::mesh::Field<double> &linker_signed_separation_distance_field =
      *linker_signed_separation_distance_field_ptr_;

  stk::mesh::Selector locally_owned_intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() &
      spherocylinder_segment_spherocylinder_segment_linker_selector;
  stk::mesh::for_each_entity_run(
      *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), stk::topology::CONSTRAINT_RANK,
      locally_owned_intersection_with_valid_entity_parts,
      [&element_radius_field, &element_youngs_modulus_field, &element_poissons_ratio_field,
       &linker_potential_force_magnitude_field, &linker_signed_separation_distance_field](
          [[maybe_unused]] const stk::mesh::BulkData &bulk_data,
          const stk::mesh::Entity &spherocylinder_segment_spherocylinder_segment_linker) {
        // Use references to avoid copying entities
        const stk::mesh::Entity &left_spherocylinder_segment_element =
            bulk_data.begin_elements(spherocylinder_segment_spherocylinder_segment_linker)[0];
        const stk::mesh::Entity &right_spherocylinder_segment_element =
            bulk_data.begin_elements(spherocylinder_segment_spherocylinder_segment_linker)[1];

        const double left_radius = stk::mesh::field_data(element_radius_field, left_spherocylinder_segment_element)[0];
        const double right_radius =
            stk::mesh::field_data(element_radius_field, right_spherocylinder_segment_element)[0];
        const double left_youngs_modulus =
            stk::mesh::field_data(element_youngs_modulus_field, left_spherocylinder_segment_element)[0];
        const double right_youngs_modulus =
            stk::mesh::field_data(element_youngs_modulus_field, right_spherocylinder_segment_element)[0];
        const double left_poissons_ratio =
            stk::mesh::field_data(element_poissons_ratio_field, left_spherocylinder_segment_element)[0];
        const double right_poissons_ratio =
            stk::mesh::field_data(element_poissons_ratio_field, right_spherocylinder_segment_element)[0];
        const double linker_signed_separation_distance = stk::mesh::field_data(
            linker_signed_separation_distance_field, spherocylinder_segment_spherocylinder_segment_linker)[0];

        const double effective_radius = (left_radius * right_radius) / (left_radius + right_radius);
        const double effective_youngs_modulus =
            (left_youngs_modulus * right_youngs_modulus) /
            (right_youngs_modulus - right_youngs_modulus * left_poissons_ratio * left_poissons_ratio +
             left_youngs_modulus - left_youngs_modulus * right_poissons_ratio * right_poissons_ratio);

        // Only apply force to overlapping particles
        // Note, signed separation distance is negative when particles overlap, so delta = -signed_separation_distance.
        double *linker_potential_force_magnitude = stk::mesh::field_data(
            linker_potential_force_magnitude_field, spherocylinder_segment_spherocylinder_segment_linker);
        const bool do_particles_overlap = linker_signed_separation_distance < 0;
        linker_potential_force_magnitude[0] =
            do_particles_overlap ? (4.0 / 3.0) * effective_youngs_modulus * std::sqrt(effective_radius) *
                                       std::pow(-linker_signed_separation_distance, 1.5)
                                 : 0.0;
      });
}

//}

}  // namespace kernels

}  // namespace evaluate_linker_potentials

}  // namespace linkers

}  // namespace mundy
