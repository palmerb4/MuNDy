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
/// \brief Definition of the ComputeOBB's Sphere kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                       // for mundy::mesh::BulkData
#include <mundy_methods/compute_obb/kernels/Sphere.hpp>  // for mundy::methods::compute_obb::kernels::Sphere

namespace mundy {

namespace methods {

namespace compute_obb {

namespace kernels {

// \name Constructors and destructor
//{

Sphere::Sphere(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "Sphere: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  static_validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Fill the internal members using the internal parameter list
  obb_field_name_ = valid_fixed_params.get<std::string>("obb_field_name");
  radius_field_name_ = valid_fixed_params.get<std::string>("radius_field_name");
  node_coord_field_name_ = valid_fixed_params.get<std::string>("node_coordinate_field_name");

  // Store the input params.
  obb_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, obb_field_name_);
  radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, radius_field_name_);
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
}
//}

// \name MetaKernel interface implementation
//{

Teuchos::ParameterList Sphere::set_mutable_params(const Teuchos::ParameterList &mutable_params) const {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  static_validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fill the internal members using the internal parameter list.
  buffer_distance_ = valid_fixed_params.get<double>("buffer_distance");
}
//}

// \name Actions
//{

void Sphere::execute(const stk::mesh::Entity &element) {
  stk::mesh::Entity const *nodes = bulk_data_ptr_->begin_nodes(element);
  double *coords = stk::mesh::field_data(*node_coord_field_ptr_, nodes[0]);
  double *radius = stk::mesh::field_data(*radius_field_ptr_, element);
  double *obb = stk::mesh::field_data(*obb_field_ptr_, element);

  obb[0] = coords[0] - radius[0] - buffer_distance_;
  obb[1] = coords[1] - radius[0] - buffer_distance_;
  obb[2] = coords[2] - radius[0] - buffer_distance_;
  obb[3] = coords[0] + radius[0] + buffer_distance_;
  obb[4] = coords[1] + radius[0] + buffer_distance_;
  obb[5] = coords[2] + radius[0] + buffer_distance_;
}
//}

}  // namespace kernels

}  // namespace compute_obb

}  // namespace methods

}  // namespace mundy
