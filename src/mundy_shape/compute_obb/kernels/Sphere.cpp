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
#include <mundy_mesh/BulkData.hpp>                     // for mundy::mesh::BulkData
#include <mundy_shape/compute_obb/kernels/Sphere.hpp>  // for mundy::shape::compute_obb::kernels::Sphere

namespace mundy {

namespace shape {

namespace compute_obb {

namespace kernels {

// \name Constructors and destructor
//{

Sphere::Sphere(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "Sphere: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Fill the internal members using the valid parameter list
  const std::string_view element_obb_field_name_ = valid_fixed_params.get<std::string>("element_obb_field_name");
  const std::string_view element_radius_field_name = mundy::shape::shapes::Sphere::get_element_radius_field_name();
  const std::string_view node_coord_field_name = mundy::shape::shapes::Sphere::get_node_coord_field_name();

  // Get the field pointers.
  obb_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_obb_field_name_);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name_);
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);

  // Check that the fields exist.
  MUNDY_THROW_ASSERT(obb_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: obb_field_ptr_ cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(element_radius_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: element_radius_field_ptr_ cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(node_coord_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: node_coord_field_ptr_ cannot be a nullptr. Check that the field exists.");
}
//}

// \name MetaKernel interface implementation
//{

void Sphere::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fill the internal members using the given parameter list.
  buffer_distance_ = valid_mutable_params.get<double>("buffer_distance");
}
//}

// \name Actions
//{

void Sphere::setup() {
}

void Sphere::execute(const stk::mesh::Entity &sphere_element) {
  stk::mesh::Entity const *nodes = bulk_data_ptr_->begin_nodes(sphere_element);
  double *coords = stk::mesh::field_data(*node_coord_field_ptr_, nodes[0]);
  double *radius = stk::mesh::field_data(*element_radius_field_ptr_, sphere_element);
  double *obb = stk::mesh::field_data(*obb_field_ptr_, sphere_element);

  obb[0] = coords[0] - radius[0] - buffer_distance_;
  obb[1] = coords[1] - radius[0] - buffer_distance_;
  obb[2] = coords[2] - radius[0] - buffer_distance_;
  obb[3] = coords[0] + radius[0] + buffer_distance_;
  obb[4] = coords[1] + radius[0] + buffer_distance_;
  obb[5] = coords[2] + radius[0] + buffer_distance_;
}

void Sphere::finalize() {
}
//}

}  // namespace kernels

}  // namespace compute_obb

}  // namespace shape

}  // namespace mundy
