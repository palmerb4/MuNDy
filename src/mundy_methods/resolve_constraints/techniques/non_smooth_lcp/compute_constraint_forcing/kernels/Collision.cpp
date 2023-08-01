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
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData
#include <mundy_methods/resolve_constraints/techniques/non_smooth_lcp/compute_constraint_forcing/kernels/Collision.hpp>  // for mundy::methods::...::kernels::Collision

namespace mundy {

namespace methods {

namespace resolve_constraints {

namespace techniques {

namespace non_smooth_lcp {

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

  // Fill the internal members using the given parameter list.
  node_normal_field_name_ = valid_fixed_params.get<std::string>("node_normal_field_name");
  node_force_field_name_ = valid_fixed_params.get<std::string>("node_force_field_name");
  element_lagrange_multiplier_field_name_ =
      valid_fixed_params.get<std::string>("element_lagrange_multiplier_field_name");

  // Store the input params.
  node_normal_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_normal_field_name_);
  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name_);
  element_lagrange_multiplier_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_lagrange_multiplier_field_name_);

  // Prefetch the collision part.
  collision_part_ptr_ = meta_data_ptr_->get_part("COLLISION");
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
  // TODO(palmerb4): Populate our ghost collision elements.
}

void Collision::execute(const stk::mesh::Entity &collision_node) {
  const size_t num_connected_elements = bulk_data_ptr->num_elements(collision_node);
  stk::mesh::Entity const *connected_elements = bulk_data_ptr->begin_elements(collision_node);
  double *node_normal = stk::mesh::field_data(linkerLagMultField, collision_node);
  double *node_force = stk::mesh::field_data(linkerLagMultField, collision_node);

  // Fetch the attached collision constraint's information.
  for (int i = 0; i < num_connected_elements; i++) {
    bool is_collision_constraint = bulk_data_ptr->bucket(connected_elements[i]).member(collision_part_ptr_);
    if (is_collision_constraint) {
      const double linker_lag_mult = stk::mesh::field_data(linkerLagMultField, connected_elements[i])[0];

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

}  // namespace non_smooth_lcp

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace methods

}  // namespace mundy
