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

/// \file CollisionSphereSphere.cpp
/// \brief Definition of the ComputeAABB's CollisionSphereSphere kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                                                         // for mundy::mesh::BulkData
#include <mundy_methods/generate_collision_constraints/kernels/CollisionSphereSphere.hpp>  // for mundy::methods::...::CollisionSphereSphere

namespace mundy {

namespace methods {

namespace compute_aabb {

namespace kernels {

// \name Constructors and destructor
//{

CollisionSphereSphere::CollisionSphereSphere(mundy::mesh::BulkData *const bulk_data_ptr,
                                             const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "Sphere: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  static_validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Fill the internal members using the internal parameter list.
  node_coord_field_name_ = valid_fixed_params.get<std::string>("node_coord_field_name");
  radius_field_name_ = valid_fixed_params.get<std::string>("radius_field_name");
  aabb_field_name_ = valid_fixed_params.get<std::string>("aabb_field_name");

  // Store the input params.
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
  radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, radius_field_name_);
  aabb_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, aabb_field_name_);
}
//}

// \name MetaKernel interface implementation
//{

Teuchos::ParameterList CollisionSphereSphere::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  static_validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fill the internal members using the internal parameter list.
  buffer_distance_ = valid_mutable_params.get<double>("buffer_distance");
}
//}

// \name Actions
//{

void CollisionSphereSphere::setup() {
}

void CollisionSphereSphere::execute(const stk::mesh::Entity &collision, const stk::mesh::Entity &left_sphere,
                                    const stk::mesh::Entity &right_sphere) {
  // Fetch the connected nodes.
  stk::mesh::Entity left_node = bulk_data_ptr_->begin_nodes(collision)[0];
  stk::mesh::Entity right_node = bulk_data_ptr_->begin_nodes(collision)[1];

  // Fetch the required fields.
  const double *const left_sphere_pos = stk::mesh::field_data(*node_coord_field_ptr_, left_node);
  const double *const right_sphere_pos = stk::mesh::field_data(*node_coord_field_ptr_, right_node);
  const double *const left_sphere_radius = stk::mesh::field_data(element_radius_field_ptr_, left_sphere);
  const double *const right_sphere_radius = stk::mesh::field_data(element_radius_field_ptr_, right_sphere);
  double *element_signed_separation_dist = stk::mesh::field_data(element_signed_separation_dist_field_ptr_, collision);
  double *left_contact_node_pos = stk::mesh::field_data(*node_coord_field_ptr_, left_node);
  double *right_contact_node_pos = stk::mesh::field_data(*node_coord_field_ptr_, right_node);
  double *left_contact_node_normal = stk::mesh::field_data(*node_normal_field_ptr_, left_node);
  double *right_contact_node_normal = stk::mesh::field_data(*node_normal_field_ptr_, right_node);

  // Fill the constraint information.
  const stk::math::Vec<double, 3> dist_left_to_right({right_sphere_pos[0] - left_sphere_pos[0],
                                                      right_sphere_pos[1] - left_sphere_pos[1],
                                                      right_sphere_pos[2] - left_sphere_pos[2]});
  const double dist_left_to_right_mag =
      std::sqrt(dist_left_to_right[0] * dist_left_to_right[0] + dist_left_to_right[1] * dist_left_to_right[1] +
                dist_left_to_right[2] * dist_left_to_right[2]);
  const stk::math::Vec<double, 3> left_contact_normal = dist_left_to_right / dist_left_to_right_mag;

  element_signed_separation_dist[0] = dist_left_to_right_mag - left_sphere_radius[0] - right_sphere_radius[0];

  left_contact_node_pos[0] = left_sphere_pos[0] + left_sphere_radius[0] * left_contact_normal[0];
  left_contact_node_pos[1] = left_sphere_pos[1] + left_sphere_radius[0] * left_contact_normal[1];
  left_contact_node_pos[2] = left_sphere_pos[2] + left_sphere_radius[0] * left_contact_normal[2];
  right_contact_node_pos[0] = right_sphere_pos[0] - right_sphere_radius[0] * left_contact_normal[0];
  right_contact_node_pos[1] = right_sphere_pos[1] - right_sphere_radius[0] * left_contact_normal[1];
  right_contact_node_pos[2] = right_sphere_pos[2] - right_sphere_radius[0] * left_contact_normal[2];

  left_contact_node_normal[0] = left_contact_normal[0];
  left_contact_node_normal[1] = left_contact_normal[1];
  left_contact_node_normal[2] = left_contact_normal[2];
  right_contact_node_normal[0] = -left_contact_normal[0];
  right_contact_node_normal[1] = -left_contact_normal[1];
  right_contact_node_normal[2] = -left_contact_normal[2];
}

void CollisionSphereSphere::finalize() {
}
//}

}  // namespace kernels

}  // namespace compute_aabb

}  // namespace methods

}  // namespace mundy
