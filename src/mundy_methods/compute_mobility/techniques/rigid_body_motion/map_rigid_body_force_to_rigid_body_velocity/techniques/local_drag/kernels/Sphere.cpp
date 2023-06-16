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
/// \brief Definition of LocalDrag's Sphere kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>               // for Teuchos::ParameterList
#include <stk_expreval/stk_expreval/Constant.hpp>  // for stk::stk_expreval::pi()
#include <stk_mesh/base/Entity.hpp>                // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>                 // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                                                  // for mundy::mesh::BulkData
#include <mundy_methods/compute_mobility/techniques/local_drag/kernels/Sphere.hpp>  // for mundy::methods::...::kernels::Sphere.hpp
#include <mundy_methods/utils/Quaternion.hpp>                                       // for mundy::utils::Quaternion

namespace mundy {

namespace methods {

namespace compute_mobility {

namespace techniques {

namespace local_drag {

namespace kernels {

// \name Constructors and destructor
//{

Sphere::Sphere(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_fixed_parameter_list = fixed_parameter_list;
  valid_fixed_parameter_list.validateParametersAndSetDefaults(this->get_valid_fixed_params());

  // Fill the internal members using the internal parameter list.
  node_force_field_name_ = valid_fixed_parameter_list.get<std::string>("node_force_field_name");
  node_torque_field_name_ = valid_fixed_parameter_list.get<std::string>("node_torque_field_name");
  node_velocity_field_name_ = valid_fixed_parameter_list.get<std::string>("node_velocity_field_name");
  node_omega_field_name_ = valid_fixed_parameter_list.get<std::string>("node_omega_field_name");
  element_radius_field_name_ = valid_fixed_parameter_list.get<std::string>("element_radius_field_name");

  // Store the input params.
  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, node_force_field_name_);
  node_torque_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, node_torque_field_name_);
  node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, node_velocity_field_name_);
  node_omega_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, node_omega_field_name_);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, element_radius_field_name_);
}
//}

// \name MetaKernel interface implementation
//{

Teuchos::ParameterList Sphere::set_transient_params(const Teuchos::ParameterList &transient_parameter_list) const {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_transient_parameter_list = transient_parameter_list;
  valid_transient_parameter_list.validateParametersAndSetDefaults(this->get_valid_transient_params());

  // Fill the internal members using the internal parameter list.
  viscosity_ = valid_transient_parameter_list.get<double>("viscosity");
}
//}

// \name Actions
//{

void Sphere::execute(const stk::mesh::Entity &element) {
  // Fetch the sphere's fields.
  stk::mesh::Entity const center_node = bulk_data_ptr_->begin_nodes(element)[0];

  double *node_force = stk::mesh::field_data(node_force_field_ptr_, center_node);
  double *node_torque = stk::mesh::field_data(node_torque_field_ptr_, center_node);
  double *node_velocity = stk::mesh::field_data(node_velocity_field_ptr_, center_node);
  double *node_omega = stk::mesh::field_data(node_omega_field_ptr_, center_node);
  double *radius = stk::mesh::field_data(element_radius_field_ptr_, element);

  // Compute the mobility matrix for the sphere using local drag.
  const double drag_trans = 6 * stk::stk_expreval::pi() * radius[0] * viscosity_;
  const double drag_rot = 8 * stk::stk_expreval::pi() * radius[0] * radius[0] * radius[0] * viscosity_;
  const double drag_trans_inv = 1.0 / drag_trans;
  const double drag_rot_inv = 1.0 / drag_rot;

  // solve for the induced node_velocity and node_omega
  node_velocity[0] = drag_trans_inv * node_force[0];
  node_velocity[1] = drag_trans_inv * node_force[1];
  node_velocity[2] = drag_trans_inv * node_force[2];
  node_omega[0] = drag_rot_inv * node_torque[0];
  node_omega[1] = drag_rot_inv * node_torque[1];
  node_omega[2] = drag_rot_inv * node_torque[2];
}
//}

}  // namespace kernels

}  // namespace local_drag

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy
