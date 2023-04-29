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
/// \brief Definition of the NodeEuler's Sphere kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field, stl::mesh::field_data

// Mundy libszhan1908
#include <mundy_methods/compute_mobility/techniques/node_euler/kernels/Sphere.hpp>  // for mundy::methods::...::kernels::Sphere.hpp
#include <mundy_methods/utils/Quaternion.hpp>                                       // for mundy::utils::Quaternion

namespace mundy {

namespace methods {

namespace compute_mobility {

namespace techniques {

namespace node_euler {

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
  time_step_size_ = valid_parameter_list.get<double>("time_step_size");
  node_coord_field_name_ = valid_parameter_list.get<std::string>("node_coord_field_name");
  node_orientation_field_name_ = valid_parameter_list.get<std::string>("node_orientation_field_name");
  node_velocity_field_name_ = valid_parameter_list.get<std::string>("node_velocity_field_name");
  node_omega_field_name_ = valid_parameter_list.get<std::string>("node_omega_field_name");

  // Store the input params.
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
  node_orientation_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_orientation_field_name_);
  node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, node_velocity_field_name_);
  node_omega_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, node_omega_field_name_);
}
//}

// \name Actions
//{

void Sphere::execute(const stk::mesh::Entity &element) {
  stk::mesh::Entity const *nodes = bulk_data_ptr_->begin_nodes(element);
  double *coords = stk::mesh::field_data(*node_coord_field_ptr_, nodes[0]);
  double *orienntation = stk::mesh::field_data(*node_orientation_field_ptr_, nodes[0]);
  double *node_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, nodes[0]);
  double *node_omega = stk::mesh::field_data(*node_omega_field_ptr_, nodes[0]);

  // Euler step position.
  coords[0] += time_step_size_ * node_velocity[0];
  coords[1] += time_step_size_ * node_velocity[1];
  coords[2] += time_step_size_ * node_velocity[2];

  // Euler step orientation.
  mundy::methods::utils::Quaternion quat(orienntation[0], orienntation[1], orienntation[2], orienntation[3]);
  quat.rotate_self(node_omega[0], node_omega[1], node_omega[2], time_step_size_);
  orientation[0] = quat.w;
  orientation[1] = quat.x;
  orientation[2] = quat.y;
  orientation[3] = quat.z;
}
//}

}  // namespace kernels

}  // namespace node_euler

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy
