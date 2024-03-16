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

/// \file SphereSphereHertzianContact.cpp
/// \brief Definition of the ComputeSignedSeparationDistanceAndContactNormal's SphereSphereHertzianContact kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_core/throw_assert.hpp>                                                      // for MUNDY_THROW_ASSERT
#include <mundy_linker/evaluate_linker_potentials/kernels/SphereSphereHertzianContact.hpp>  // for mundy::linker::...::kernels::SphereSphereHertzianContact
#include <mundy_mesh/BulkData.hpp>                                                          // for mundy::mesh::BulkData
#include <mundy_shape/shapes/Spheres.hpp>  // for mundy::shape::shapes::Spheres

/// F = \frac{4}{3} E \sqrt{R} \delta^{3/2}
/// where:
/// - F is the contact force,
/// - E is the effective modulus of elasticity, calculated as
///   E = \left( \frac{1 - \nu_1^2}{E_1} + \frac{1 - \nu_2^2}{E_2} \right)^{-1},
/// - R is the effective radius of contact, defined as
///   \frac{1}{R} = \frac{1}{R_1} + \frac{1}{R_2},
/// - \delta is the deformation at the contact point,
/// - R_1 and R_2 are the radii of the two spheres,
/// - E_1 and E_2 are the Young's moduli of the materials,
/// - \nu_1 and \nu_2 are the Poisson's ratios of the materials.
///
/// The formula assumes isotropic and linearly elastic materials and small deformations. There are more complex Hertzian
/// contact models, so make sure this model is appropriate for your use case.
///
/// In terms of fixed parameters, this class requires
/// - "valid_entity_part_names" (Teuchos::Array<std::string>): The list of valid linker entity part names for the
/// kernel.
/// - "valid_sphere_part_names" (Teuchos::Array<std::string>): The list of valid sphere part names for the kernel.
/// - "linker_potential_force_magnitude_field_name" (std::string): The name of the field in which to write the linker's
/// potential force magnitude.
/// - "linker_signed_separation_distance_field_name" (std::string): The name of the field in which to write the signed
/// separation distance.
/// - "element_youngs_modulus_field_name" (std::string): The name of the field in which to read the Young's modulus.
/// - "element_poissons_ratio_field_name" (std::string): The name of the field in which to read the Poisson's ratio.

namespace mundy {

namespace linker {

namespace evaluate_linker_potentials {

namespace kernels {

// \name Constructors and destructor
//{

SphereSphereHertzianContact::SphereSphereHertzianContact(mundy::mesh::BulkData *const bulk_data_ptr,
                                                         const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "SphereSphereHertzianContact: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(SphereSphereHertzianContact::get_valid_fixed_params());

  // Get the field pointers.
  const std::string element_radius_field_name = mundy::shape::shapes::Spheres::get_element_radius_field_name();
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
    MUNDY_THROW_ASSERT(
        field_ptr != nullptr, std::invalid_argument,
        "SphereSphereHertzianContact: Field " << field_name << " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(element_radius_field_ptr_, element_radius_field_name);
  field_exists(element_youngs_modulus_field_ptr_, element_youngs_modulus_field_name);
  field_exists(element_poissons_ratio_field_ptr_, element_poissons_ratio_field_name);
  field_exists(linker_potential_force_magnitude_field_ptr_, linker_potential_force_magnitude_field_name);
  field_exists(linker_signed_separation_distance_field_ptr_, linker_signed_separation_distance_field_name);

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  Teuchos::Array<std::string> valid_sphere_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_sphere_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(
          part != nullptr, std::invalid_argument,
          "SphereSphereHertzianContact: Part " << part_name << " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };  // parts_from_names

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  valid_sphere_parts_ = parts_from_names(*meta_data_ptr_, valid_sphere_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SphereSphereHertzianContact::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

stk::topology::rank_t SphereSphereHertzianContact::get_entity_rank() const {
  return stk::topology::CONSTRAINT_RANK;
}

void SphereSphereHertzianContact::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  // We don't have any valid mutable params, so this seems pointless but it's useful in that it will throw if the user
  // gives us parameters. This is useful for catching user errors.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(SphereSphereHertzianContact::get_valid_mutable_params());
}
//}

// \name Actions
//{

void SphereSphereHertzianContact::setup() {
}

void SphereSphereHertzianContact::execute(const stk::mesh::Entity &sphere_sphere_linker) const {
  // Use references to avoid copying entities
  const stk::mesh::Entity &left_sphere_element = bulk_data_ptr_->begin_elements(sphere_sphere_linker)[0];
  const stk::mesh::Entity &right_sphere_element = bulk_data_ptr_->begin_elements(sphere_sphere_linker)[1];

  const double *left_radius = stk::mesh::field_data(*element_radius_field_ptr_, left_sphere_element);
  const double *right_radius = stk::mesh::field_data(*element_radius_field_ptr_, right_sphere_element);
  const double *left_youngs_modulus = stk::mesh::field_data(*element_youngs_modulus_field_ptr_, left_sphere_element);
  const double *right_youngs_modulus = stk::mesh::field_data(*element_youngs_modulus_field_ptr_, right_sphere_element);
  const double *left_poissons_ratio = stk::mesh::field_data(*element_poissons_ratio_field_ptr_, left_sphere_element);
  const double *right_poissons_ratio = stk::mesh::field_data(*element_poissons_ratio_field_ptr_, right_sphere_element);

  double *linker_potential_force_magnitude =
      stk::mesh::field_data(*linker_potential_force_magnitude_field_ptr_, sphere_sphere_linker);
  const double *linker_signed_separation_distance =
      stk::mesh::field_data(*linker_signed_separation_distance_field_ptr_, sphere_sphere_linker);

  const double effective_radius = (left_radius[0] * right_radius[0]) / (left_radius[0] + right_radius[0]);
  const double effective_youngs_modulus =
      (left_youngs_modulus[0] * right_youngs_modulus[0]) /
      (right_youngs_modulus[0] - right_youngs_modulus[0] * left_poissons_ratio[0] * left_poissons_ratio[0] +
       left_youngs_modulus[0] - left_youngs_modulus[0] * right_poissons_ratio[0] * right_poissons_ratio[0]);

  // Only apply force to overlapping particles
  // Note, signed separation distance is negative when particles overlap, so delta = -signed_separation_distance.
  const bool do_particles_overlap = linker_signed_separation_distance[0] < 0;
  linker_potential_force_magnitude[0] = do_particles_overlap
                                            ? (4.0 / 3.0) * effective_youngs_modulus * std::sqrt(effective_radius) *
                                                  std::pow(-linker_signed_separation_distance[0], 1.5)
                                            : 0.0;
}

void SphereSphereHertzianContact::finalize() {
}
//}

}  // namespace kernels

}  // namespace evaluate_linker_potentials

}  // namespace linker

}  // namespace mundy
