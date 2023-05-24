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

/// \file LinkerSphere.cpp
/// \brief Definition of the NodeEuler's LinkerSphere kernel.

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
#include <mundy_methods/compute_time_integration/techniques/node_euler/kernels/LinkerSphere.hpp>  // for mundy::methods::...::kernels::LinkerSphere.hpp
#include <mundy_methods/utils/Quaternion.hpp>  // for mundy::utils::Quaternion

namespace mundy {

namespace methods {

namespace compute_time_integration {

namespace techniques {

namespace node_euler {

namespace kernels {

// \name Constructors and destructor
//{

LinkerSphere::LinkerSphere(stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_parameter_list = parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_params());

  // Fill the internal members using the internal parameter list.
  time_step_size_ = valid_parameter_list.get<double>("time_step_size");
  element_orientation_field_name_ = valid_parameter_list.get<std::string>("element_orientation_field_name");
  node_coord_field_name_ = valid_parameter_list.get<std::string>("node_coord_field_name");
  node_velocity_field_name_ = valid_parameter_list.get<std::string>("node_velocity_field_name");
  node_omega_field_name_ = valid_parameter_list.get<std::string>("node_omega_field_name");

  // Store the input params.
  element_orientation_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_orientation_field_name_);
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
  node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name_);
  node_omega_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_omega_field_name_);
}
//}

// \name Actions
//{

void LinkerSphere::execute(const stk::mesh::Entity &linker, const stk::mesh::Entity &element) {
  // TODO(palmerb4): For now the element and linker are decoupled but this means that we may not satisfy rigid body
  // motion due to rounding errors and that the surface nodes may leave the surface over time.

  // Euler step position for each node in the fixed connectivity.
  stk::mesh::Entity sphere_node = bulk_data_ptr_->begin_nodes(element)[0];
  double *sphere_node_omega = stk::mesh::field_data(*node_omega_field_ptr_, sphere_node);
  double *sphere_node_coords = stk::mesh::field_data(*node_coord_field_ptr_, sphere_node);
  double *sphere_node_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, sphere_node);
  sphere_node_coords[0] += time_step_size_ * sphere_node_velocity[0];
  sphere_node_coords[1] += time_step_size_ * sphere_node_velocity[1];
  sphere_node_coords[2] += time_step_size_ * sphere_node_velocity[2];

  // Euler step element orientation.
  double *orientation = stk::mesh::field_data(*element_orientation_field_name_, element);
  mundy::methods::utils::Quaternion quat(orientation[0], orientation[1], orientation[2], orientation[3]);
  quat.rotate_self(sphere_node_omega[0], sphere_node_omega[1], sphere_node_omega[2], time_step_size_);
  orientation[0] = quat.w;
  orientation[1] = quat.x;
  orientation[2] = quat.y;
  orientation[3] = quat.z;

  // Euler step position for each node in the dynamic connectivity.
  unsigned num_surface_nodes = bulk_data_ptr_->num_nodes(linker);
  stk::mesh::Entity const *surface_nodes = bulk_data_ptr_->begin_nodes(linker);
  for (int i = 0; i < num_surface_nodes; i++) {
    double *surface_node_coords = stk::mesh::field_data(*node_coord_field_ptr_, surface_nodes[i]);
    double *surface_node_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, surface_nodes[i]);
    surface_node_coords[0] += time_step_size_ * surface_node_velocity[0];
    surface_node_coords[1] += time_step_size_ * surface_node_velocity[1];
    surface_node_coords[2] += time_step_size_ * surface_node_velocity[2];
  }
}
//}

}  // namespace kernels

}  // namespace node_euler

}  // namespace techniques

}  // namespace compute_time_integration

}  // namespace methods

}  // namespace mundy
