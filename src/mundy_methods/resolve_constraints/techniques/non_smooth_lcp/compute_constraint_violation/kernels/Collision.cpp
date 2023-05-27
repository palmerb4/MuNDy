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
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_methods/resolve_constraints/techniques/non_smooth_lcp/compute_constraint_violation/kernels/Collision.hpp>  // for mundy::methods::...::kernels::Collision

namespace mundy {

namespace methods {

namespace resolve_constraints {

namespace techniques {

namespace non_smooth_lcp {

namespace compute_constraint_violation {

namespace kernels {

// \name Constructors and destructor
//{

Collision::Collision(stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_parameter_list = parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_params());

  // Fill the internal members using the internal parameter list.
  signed_sep_dist_field_name_ =
      valid_parameter_list.get<std::string>("element_constraint_violation_on_dist_field_name");
  element_lagrange_multiplier_field_name_ =
      valid_parameter_list.get<std::string>("element_lagrange_multiplier_field_name");
  element_constraint_violation_field_name_ =
      valid_parameter_list.get<std::string>("element_constraint_violation_field_name");

  // Store the input params.
  element_signed_separation_dist_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, element_signed_separation_dist_field_name_);
  element_lagrange_multiplier_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, element_lagrange_multiplier_field_name_);
  element_constraint_violation_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, element_constraint_violation_field_name_);
}
//}

// \name Actions
//{

void Collision::execute(const stk::mesh::Entity &element) {
  stk::mesh::Entity const *nodes = bulk_data_ptr_->begin_nodes(element);
  const double *signed_separation_dist = stk::mesh::field_data(*element_signed_separation_dist_field_ptr_, element);
  const double *lagrange_mult = stk::mesh::field_data(*element_lagrange_multiplier_field_ptr_, element);
  double *constraint_violation = stk::mesh::field_data(*element_constraint_violation_field_ptr_, element);

  // Minimum map constraint violation.
  constraint_violation[0] = std::min(separation_dist[0], lagrange_mult[0]);
}
//}

}  // namespace kernels

}  // namespace compute_constraint_violation

}  // namespace non_smooth_lcp

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace methods

}  // namespace mundy
