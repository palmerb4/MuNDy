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

/// \file Sphere.cpp
/// \brief Definition of the ComputeSignedSeparationDistanceAndContactNormal's Sphere kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_core/throw_assert.hpp>                                                 // for MUNDY_THROW_ASSERT
#include <mundy_linker/linker_potential_force_magnitude_reduction/kernels/Sphere.hpp>  // for mundy::linker::...::kernels::Sphere
#include <mundy_mesh/BulkData.hpp>                                                     // for mundy::mesh::BulkData
#include <mundy_shape/shapes/Spheres.hpp>  // for mundy::shape::shapes::Spheres

namespace mundy {

namespace linker {

namespace linker_potential_force_magnitude_reduction {

namespace kernels {

// \name Constructors and destructor
//{

Sphere::Sphere(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Sphere: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(Sphere::get_valid_fixed_params());

  valid_fixed_params.print(std::cout, Teuchos::ParameterList::PrintOptions().showDoc(true).indent(2).showTypes(true));

  // Get the field pointers.
  const std::string linker_contact_normal_field_name =
      valid_fixed_params.get<std::string>("linker_contact_normal_field_name");
  const std::string linker_potential_force_magnitude_field_name =
      valid_fixed_params.get<std::string>("linker_potential_force_magnitude_field_name");

  const std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");

  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name);
  linker_contact_normal_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_normal_field_name);
  linker_potential_force_magnitude_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_potential_force_magnitude_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_ASSERT(field_ptr != nullptr, std::invalid_argument,
                       "Sphere: Field " << field_name << " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(linker_contact_normal_field_ptr_, linker_contact_normal_field_name);
  field_exists(linker_potential_force_magnitude_field_ptr_, linker_potential_force_magnitude_field_name);
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
      MUNDY_THROW_ASSERT(part != nullptr, std::invalid_argument,
                         "Sphere: Part " << part_name << " cannot be a nullptr. Check that the part exists.");
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

std::vector<stk::mesh::Part *> Sphere::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

stk::topology::rank_t Sphere::get_entity_rank() const {
  return stk::topology::ELEMENT_RANK;
}

void Sphere::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  // We don't have any valid mutable params, so this seems pointless but it's useful in that it will throw if the user
  // gives us parameters. This is useful for catching user errors.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(Sphere::get_valid_mutable_params());
}
//}

// \name Actions
//{

void Sphere::setup() {
}

void Sphere::execute(const stk::mesh::Entity &sphere) {
  // Get our node and its force
  const stk::mesh::Entity &node = bulk_data_ptr_->begin_nodes(sphere)[0];
  double *node_force = stk::mesh::field_data(*node_force_field_ptr_, node);

  // Loop over the connected constraint rank entities
  const unsigned num_constraint_rank_conn = bulk_data_ptr_->num_connectivity(sphere, stk::topology::CONSTRAINT_RANK);
  const stk::mesh::Entity *connected_linkers = bulk_data_ptr_->begin(sphere, stk::topology::CONSTRAINT_RANK);

  for (unsigned i = 0; i < num_constraint_rank_conn; i++) {
    const stk::mesh::Entity &connected_linker = connected_linkers[i];
    const bool is_reduction_linker = bulk_data_ptr_->bucket(connected_linker).member(*linkers_part_to_reduce_over_);

    if (is_reduction_linker) {
      // The contact normal stored on a linker points from the left element to the right element. This is important, as it means we should multiply by -1 if we are the right element.
      const bool are_we_the_left_sphere = (bulk_data_ptr_->begin(connected_linker, stk::topology::ELEMENT_RANK)[0] == sphere);
      const double sign = are_we_the_left_sphere ? 1.0 : -1.0;
      double *contact_normal = stk::mesh::field_data(*linker_contact_normal_field_ptr_, connected_linker);
      double *potential_force_magnitude =
          stk::mesh::field_data(*linker_potential_force_magnitude_field_ptr_, connected_linker);
      node_force[0] -= sign * contact_normal[0] * potential_force_magnitude[0];
      node_force[1] -= sign * contact_normal[1] * potential_force_magnitude[0];
      node_force[2] -= sign * contact_normal[2] * potential_force_magnitude[0];
    }
  }
}

void Sphere::finalize() {
}
//}

}  // namespace kernels

}  // namespace linker_potential_force_magnitude_reduction

}  // namespace linker

}  // namespace mundy
