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

/// \file CollisionSphere.cpp
/// \brief Definition of the ComputeConstraintForcing's CollisionSphere kernel.

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
#include <mundy_methods/resolve_constraints/techniques/non_smooth_lcp/compute_constraint_forcing/kernels/CollisionSphere.hpp>  // for mundy::methods::...::kernels::CollisionSphere

namespace mundy {

namespace methods {

namespace resolve_constraints {

namespace techniques {

namespace non_smooth_lcp {

namespace compute_constraint_forcing {

namespace kernels {

// \name Constructors and destructor
//{

CollisionSphere::CollisionSphere(stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_parameter_list = parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_params());

  // Fill the internal members using the internal parameter list.
  node_coord_field_name_ = valid_parameter_list.get<std::string>("node_coord_field_name");
  node_force_field_name_ = valid_parameter_list.get<std::string>("node_force_field_name");
  node_torque_field_name_ = valid_parameter_list.get<std::string>("node_torque_field_name");
  lagrange_multiplier_field_name_ = valid_parameter_list.get<std::string>("lagrange_multiplier_field_name");

  // Store the input params.
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name_);
  node_torque_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_torque_field_name_);
  lagrange_multiplier_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, lagrange_multiplier_field_name_);
}
//}

// \name Actions
//{

void CollisionSphere::execute(const stk::mesh::Entity &collision_element, const stk::mesh::Entity &sphere_element) {
  stk::mesh::Entity const *nodes = bulk_data_ptr_->begin_nodes(element);
  double *coords = stk::mesh::field_data(*node_coord_field_ptr_, nodes[0]);
  double *radius = stk::mesh::field_data(*radius_field_ptr_, element);
  double *aabb = stk::mesh::field_data(*aabb_field_ptr_, element);

  aabb[0] = coords[0] - radius[0] - buffer_distance_;
  aabb[1] = coords[1] - radius[0] - buffer_distance_;
  aabb[2] = coords[2] - radius[0] - buffer_distance_;
  aabb[3] = coords[0] + radius[0] + buffer_distance_;
  aabb[4] = coords[1] + radius[0] + buffer_distance_;
  aabb[5] = coords[2] + radius[0] + buffer_distance_;
}
//}

}  // namespace kernels

}  // namespace compute_constraint_forcing

}  // namespace non_smooth_lcp

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace methods

}  // namespace mundy