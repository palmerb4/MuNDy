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
#include <stk_mesh/base/BulkData.hpp>              // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>                // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>                 // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
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

Sphere::Sphere(stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_parameter_list = parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_params());

  // Fill the internal members using the internal parameter list.
  viscosity_ = valid_parameter_list.get<double>("viscosity");
  force_field_name_ = valid_parameter_list.get<std::string>("force_field_name");
  torque_field_name_ = valid_parameter_list.get<std::string>("torque_field_name");
  velocity_field_name_ = valid_parameter_list.get<std::string>("velocity_field_name");
  omega_field_name_ = valid_parameter_list.get<std::string>("omega_field_name");
  orientation_field_name_ = valid_parameter_list.get<std::string>("orientation_field_name");
  radius_field_name_ = valid_parameter_list.get<std::string>("radius_field_name");

  // Store the input params.
  force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, force_field_name_);
  torque_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, torque_field_name_);
  velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, velocity_field_name_);
  omega_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, omega_field_name_);
  orientation_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, orientation_field_name_);
  radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, radius_field_name_);
}
//}

// \name Actions
//{

void Sphere::execute(const stk::mesh::Entity &element) {
  // Fetch the sphere's fields.
  double *force = stk::mesh::field_data(force_field_ptr_, element);
  double *torque = stk::mesh::field_data(torque_field_ptr_, element);
  double *velocity = stk::mesh::field_data(velocity_field_ptr_, element);
  double *omega = stk::mesh::field_data(omega_field_ptr_, element);
  double *orientation = stk::mesh::field_data(orientation_field_ptr_, element);
  double *radius = stk::mesh::field_data(radius_field_ptr_, element);

  // Compute the mobility matrix for the sphere using local drag.
  Quaternion quat(orientation[0], orientation[1], orientation[2], orientation[3]);
  const stk::math::Vec<double, 3> q = quat.rotate(stk::math::Vec<double, 3>({0, 0, 1}));
  const double qq[3][3] = {{q[0] * q[0], q[0] * q[1], q[0] * q[2]},
                           {q[1] * q[0], q[1] * q[1], q[1] * q[2]},
                           {q[2] * q[0], q[2] * q[1], q[2] * q[2]}};
  const double Imqq[3][3] = {
      {1 - qq[0][0], -qq[0][1], -qq[0][2]}, {-qq[1][0], 1 - qq[1][1], -qq[1][2]}, {-qq[2][0], -qq[2][1], 1 - qq[2][2]}};
  const double drag_para = 6 * stk::stk_expreval::pi() * radius[0] * viscosity_;
  const double drag_perp = drag_para;
  const double drag_rot = 8 * stk::stk_expreval::pi() * radius[0] * radius[0] * radius[0] * viscosity_;
  const double drag_para_inv = 1.0 / drag_para;
  const double drag_perp_inv = 1.0 / drag_perp;
  const double drag_rot_inv = 1.0 / drag_rot;
  const double mob_trans[3][3] = {
      {drag_para_inv * qq[0][0] + drag_perp_inv * Imqq[0][0], drag_para_inv * qq[0][1] + drag_perp_inv * Imqq[0][1],
       drag_para_inv * qq[0][2] + drag_perp_inv * Imqq[0][2]},
      {drag_para_inv * qq[1][0] + drag_perp_inv * Imqq[1][0], drag_para_inv * qq[1][1] + drag_perp_inv * Imqq[1][1],
       drag_para_inv * qq[1][2] + drag_perp_inv * Imqq[1][2]},
      {drag_para_inv * qq[2][0] + drag_perp_inv * Imqq[2][0], drag_para_inv * qq[2][1] + drag_perp_inv * Imqq[2][1],
       drag_para_inv * qq[2][2] + drag_perp_inv * Imqq[2][2]}};
  const double mob_rot[3][3] = {
      {drag_rot_inv * qq[0][0] + drag_rot_inv * Imqq[0][0], drag_rot_inv * qq[0][1] + drag_rot_inv * Imqq[0][1],
       drag_rot_inv * qq[0][2] + drag_rot_inv * Imqq[0][2]},
      {drag_rot_inv * qq[1][0] + drag_rot_inv * Imqq[1][0], drag_rot_inv * qq[1][1] + drag_rot_inv * Imqq[1][1],
       drag_rot_inv * qq[1][2] + drag_rot_inv * Imqq[1][2]},
      {drag_rot_inv * qq[2][0] + drag_rot_inv * Imqq[2][0], drag_rot_inv * qq[2][1] + drag_rot_inv * Imqq[2][1],
       drag_rot_inv * qq[2][2] + drag_rot_inv * Imqq[2][2]}};

  // solve for the induced velocity and omega
  velocity[0] = mob_trans[0][0] * force[0] + mob_trans[0][1] * force[1] + mob_trans[0][2] * force[2];
  velocity[1] = mob_trans[1][0] * force[0] + mob_trans[1][1] * force[1] + mob_trans[1][2] * force[2];
  velocity[2] = mob_trans[2][0] * force[0] + mob_trans[2][1] * force[1] + mob_trans[2][2] * force[2];
  omega[0] = mob_rot[0][0] * torque[0] + mob_rot[0][1] * torque[1] + mob_rot[0][2] * torque[2];
  omega[1] = mob_rot[1][0] * torque[0] + mob_rot[1][1] * torque[1] + mob_rot[1][2] * torque[2];
  omega[2] = mob_rot[2][0] * torque[0] + mob_rot[2][1] * torque[1] + mob_rot[2][2] * torque[2];
}
//}

}  // namespace kernels

}  // namespace local_drag

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy
