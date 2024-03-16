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
/// \brief Definition ComputeConstraintViolation's Collision kernel

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_constraint/compute_constraint_violation/kernels/Collision.hpp>  // for mundy::constraint::...::kernels::Collision
#include <mundy_core/throw_assert.hpp>                                          // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                                              // for mundy::mesh::BulkData

namespace mundy {

namespace constraint {

namespace compute_constraint_violation {

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

  // Fill the internal members using the given parameter list.
  element_signed_separation_dist_field_name_ =
      valid_fixed_params.get<std::string>("element_constraint_violation_on_dist_field_name");
  element_lagrange_multiplier_field_name_ =
      valid_fixed_params.get<std::string>("element_lagrange_multiplier_field_name");
  element_constraint_violation_field_name_ =
      valid_fixed_params.get<std::string>("element_constraint_violation_field_name");

  // Get the field pointers.
  element_signed_separation_dist_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_signed_separation_dist_field_name_);
  element_lagrange_multiplier_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_lagrange_multiplier_field_name_);
  element_constraint_violation_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_constraint_violation_field_name_);

  // Check that the fields exist.
  MUNDY_THROW_ASSERT(
      element_signed_separation_dist_field_ptr_ != nullptr, std::invalid_argument,
      "Collision: element_signed_separation_dist_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(
      element_lagrange_multiplier_field_ptr_ != nullptr, std::invalid_argument,
      "Collision: element_lagrange_multiplier_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(
      element_constraint_violation_field_ptr_ != nullptr, std::invalid_argument,
      "Collision: element_constraint_violation_field_ptr cannot be a nullptr. Check that the field exists.");
}
//}

// \name MetaKernel interface implementation
//{

void Collision::set_mutable_params([[maybe_unused]] const Teuchos::ParameterList &mutable_params) {
}
//}

// \name Actions
//{

void Collision::setup() {
}

void Collision::execute(const stk::mesh::Entity &collision_element) const {
  const double *signed_separation_dist =
      stk::mesh::field_data(*element_signed_separation_dist_field_ptr_, collision_element);
  const double *lagrange_mult = stk::mesh::field_data(*element_lagrange_multiplier_field_ptr_, collision_element);
  double *constraint_violation = stk::mesh::field_data(*element_constraint_violation_field_ptr_, collision_element);

  // Minimum map constraint violation.
  constraint_violation[0] = std::min(signed_separation_dist[0], lagrange_mult[0]);
}

void Collision::finalize() {
}
//}

}  // namespace kernels

}  // namespace compute_constraint_violation

}  // namespace constraint

}  // namespace mundy
