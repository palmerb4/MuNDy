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

/// \file Collision.cpp
/// \brief Definition of the ComputeConstraintForcing's Collision kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_constraint/compute_constraint_forcing/kernels/Collision.hpp>  // for mundy::constraint::...::kernels::Collision
#include <mundy_core/throw_assert.hpp>                                        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                                            // for mundy::mesh::BulkData

namespace mundy {

namespace constraint {

namespace compute_constraint_forcing {

namespace kernels {

// \name Constructors and destructor
//{

Collision::Collision(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Collision: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Store the valid entity parts for the kernel.
  Teuchos::Array<std::string> input_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("input_part_names");
  for (const std::string &part_name : input_part_names) {
    valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
    MUNDY_THROW_ASSERT(
        valid_entity_parts_.back() != nullptr, std::invalid_argument,
        "Collision: Part '" << part_name << "' from the input_part_names does not exist in the meta data.");
  }

  // Fetch the fields.
  const std::string node_normal_field_name = valid_fixed_params.get<std::string>("node_normal_field_name");
  const std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
  const std::string element_lagrange_multiplier_field_name =
      valid_fixed_params.get<std::string>("element_lagrange_multiplier_field_name");

  // Get the field pointers.
  node_normal_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_normal_field_name);
  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name);
  element_lagrange_multiplier_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_lagrange_multiplier_field_name);

  // Check that the fields exist
  MUNDY_THROW_ASSERT(node_normal_field_ptr_ != nullptr, std::invalid_argument,
                     "Collision: node_normal_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(node_force_field_ptr_ != nullptr, std::invalid_argument,
                     "Collision: node_force_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(
      element_lagrange_multiplier_field_ptr_ != nullptr, std::invalid_argument,
      "Collision: element_lagrange_multiplier_field_ptr cannot be a nullptr. Check that the field exists.");
}
//}

// \name Setters
//{

void Collision::set_mutable_params([[maybe_unused]] const Teuchos::ParameterList &mutable_params) {
}
//}

// \name Getters
//{

std::vector<stk::mesh::Part *> Collision::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

// \name Actions
//{

void Collision::setup() {
  // TODO(palmerb4): Populate our ghost collision elements.
}

void Collision::execute(const stk::mesh::Entity &collision_node) {
  const int num_connected_elements = bulk_data_ptr_->num_elements(collision_node);
  stk::mesh::Entity const *connected_elements = bulk_data_ptr_->begin_elements(collision_node);
  double *node_normal = stk::mesh::field_data(*node_normal_field_ptr_, collision_node);
  double *node_force = stk::mesh::field_data(*node_force_field_ptr_, collision_node);

  // Fetch the attached collision constraint's information.
  for (int i = 0; i < num_connected_elements; i++) {
    bool is_collision_constraint = bulk_data_ptr_->bucket(connected_elements[i]).member(collision_part_ordinal_);
    if (is_collision_constraint) {
      const double linker_lag_mult =
          stk::mesh::field_data(*element_lagrange_multiplier_field_ptr_, connected_elements[i])[0];

      node_force[0] += -linker_lag_mult * node_normal[0];
      node_force[1] += -linker_lag_mult * node_normal[1];
      node_force[2] += -linker_lag_mult * node_normal[2];
    }
  }
}

void Collision::finalize() {
  // TODO(palmerb4): We need to reduce over the shared nodes.
}
//}

}  // namespace kernels

}  // namespace compute_constraint_forcing

}  // namespace constraint

}  // namespace mundy
