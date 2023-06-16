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
/// \brief Definition of the MapRigidBodyVelocityToSurfaceVelocity's Sphere kernel.

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
#include <mundy_methods/compute_mobility/map_rigid_body_velocity_to_surface_velocity/kernels/Sphere.hpp>  // for mundy::methods::compute_aabb::kernels::Sphere

namespace mundy {

namespace methods {

namespace compute_mobility {

namespace map_rigid_body_velocity_to_surface_velocity {

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
  node_coord_field_name_ = valid_fixed_parameter_list.get<std::string>("node_coord_field_name");
  node_velocity_field_name_ = valid_fixed_parameter_list.get<std::string>("node_velocity_field_name");
  node_omega_field_name_ = valid_fixed_parameter_list.get<std::string>("node_omega_field_name");

  // Store the input params.
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
  node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name_);
  node_omega_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_omega_field_name_);
}
//}

// \name MetaKernel interface implementation
//{

Teuchos::ParameterList Sphere::set_transient_params(const Teuchos::ParameterList &transient_parameter_list) const {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_transient_parameter_list = transient_parameter_list;
  valid_transient_parameter_list.validateParametersAndSetDefaults(this->get_valid_transient_params());
}
//}

// \name Actions
//{

void Sphere::execute(const stk::mesh::Entity &linker) {
  stk::mesh::Entity const *surface_nodes = bulk_data_ptr_->begin_nodes(linker);
  stk::mesh::Entity const sphere = bulk_data_ptr_->begin_elements(linker)[0];
  stk::mesh::Entity const body_node = bulk_data_ptr_->begin_nodes(sphere)[0];

  double *body_node_coords = stk::mesh::field_data(*node_coord_field_ptr_, body_node);
  double *body_node_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, body_node);
  double *body_node_omega = stk::mesh::field_data(*node_omega_field_ptr_, body_node);
  unsigned num_surface_nodes = bulk_data_ptr_->num_nodes();
  for (int i = 0; i < num_surface_nodes; i++) {
    double *surface_node_coords = stk::mesh::field_data(*node_coord_field_ptr_, surface_nodes[i]);
    double *surface_node_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, surface_nodes[i]);
    double *surface_node_omega = stk::mesh::field_data(*node_omega_field_ptr_, surface_nodes[i]);

    double relative_pos[3] = {surface_node_coords[0] - body_node_coords[0],
                              surface_node_coords[1] - body_node_coords[1],
                              surface_node_coords[2] - body_node_coords[2]};

    surface_node_velocity[0] =
        body_node_velocity[0] + body_node_omega[1] * relative_pos[2] - body_node_omega[2] * relative_pos[1];
    surface_node_velocity[1] =
        body_node_velocity[1] + body_node_omega[2] * relative_pos[0] - body_node_omega[0] * relative_pos[2];
    surface_node_velocity[2] =
        body_node_velocity[2] + body_node_omega[0] * relative_pos[1] - body_node_omega[1] * relative_pos[0];
    surface_node_omega[0] = body_node_omega[0];
    surface_node_omega[1] = body_node_omega[1];
    surface_node_omega[2] = body_node_omega[2];
  }
}
//}

}  // namespace kernels

}  // namespace map_rigid_body_velocity_to_surface_velocity

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy
