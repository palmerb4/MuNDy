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

/// \file Spherocylinder.cpp
/// \brief Definition of the ComputeAABB's Spherocylinder kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                                // for mundy::mesh::BulkData
#include <mundy_methods/ComputeAABB.hpp>                          // for mundy::methods::ComputeAABB
#include <mundy_methods/compute_aabb/kernels/Spherocylinder.hpp>  // for mundy::methods::compute_aabb::kernels::Spherocylinder

// \name Registration
//{

/// @brief Register ComputeAABB with the global MetaMethodFactory.
MUNDY_REGISTER_METACLASS(mundy::methods::compute_aabb::kernels::Spherocylinder,
                         mundy::methods::ComputeAABB::OurKernelFactory)
//}

namespace mundy {

namespace methods {

namespace compute_aabb {

namespace kernels {

// \name Constructors and destructor
//{

Spherocylinder::Spherocylinder(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "Spherocylinder: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Fill the internal members using the given parameter list.
  node_coord_field_name_ = valid_fixed_params.get<std::string>("node_coord_field_name");
  element_radius_field_name_ = valid_fixed_params.get<std::string>("element_radius_field_name");
  element_aabb_field_name_ = valid_fixed_params.get<std::string>("element_aabb_field_name");

  // Store the input params.
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
  element_radius_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name_);
  element_aabb_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_aabb_field_name_);
}
//}

// \name MetaKernel interface implementation
//{

void Spherocylinder::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fill the internal members using the given parameter list.
  buffer_distance_ = valid_mutable_params.get<double>("buffer_distance");
}
//}

// \name Actions
//{

void Spherocylinder::setup() {
}

void Spherocylinder::execute(const stk::mesh::Entity &spherocylinder_element) {
  stk::mesh::Entity const *nodes = bulk_data_ptr_->begin_nodes(spherocylinder_element);
  double *left_endpt_coords = stk::mesh::field_data(*node_coord_field_ptr_, nodes[0]);
  double *right_endpt_coords = stk::mesh::field_data(*node_coord_field_ptr_, nodes[2]);
  double *radius = stk::mesh::field_data(*element_radius_field_ptr_, spherocylinder_element);
  double *aabb = stk::mesh::field_data(*aabb_field_ptr_, spherocylinder_element);

  aabb[0] = std::min(left_endpt_coords[0], right_endpt_coords[0]) - radius[0] - buffer_distance_;
  aabb[1] = std::min(left_endpt_coords[1], right_endpt_coords[1]) - radius[0] - buffer_distance_;
  aabb[2] = std::min(left_endpt_coords[2], right_endpt_coords[2]) - radius[0] - buffer_distance_;
  aabb[3] = std::max(left_endpt_coords[0], right_endpt_coords[0]) + radius[0] + buffer_distance_;
  aabb[4] = std::max(left_endpt_coords[1], right_endpt_coords[1]) + radius[0] + buffer_distance_;
  aabb[5] = std::max(left_endpt_coords[2], right_endpt_coords[2]) + radius[0] + buffer_distance_;
}

void Spherocylinder::finalize() {
}
//}

}  // namespace kernels

}  // namespace compute_aabb

}  // namespace methods

}  // namespace mundy
