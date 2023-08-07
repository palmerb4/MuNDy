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
/// \brief Definition of the ComputeBoundingRadius's Sphere kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                                 // for mundy::mesh::BulkData
#include <mundy_shape/compute_bounding_radius/kernels/Sphere.hpp>  // for mundy::shape::compute_bounding_radius::kernels::Sphere

namespace mundy {

namespace shape {

namespace compute_bounding_radius {

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

  // Fill the internal members using the given parameter list.
  radius_field_name_ = valid_fixed_params.get<std::string>("radius_field_name");
  bounding_radius_field_name_ = valid_fixed_params.get<std::string>("bounding_radius_field_name");

  // Store the input params.
  radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, radius_field_name_);
  bounding_radius_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, bounding_radius_field_name_);

  // Check that the fields exist.
  MUNDY_THROW_ASSERT(radius_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: radius_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(bounding_radius_field_ptr_ != nullptr, std::invalid_argument,
                     "Sphere: bounding_radius_field_ptr cannot be a nullptr. Check that the field exists.");
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
  double *radius = stk::mesh::field_data(*radius_field_ptr_, sphere_element);
  double *bounding_radius = stk::mesh::field_data(*bounding_radius_field_ptr_, sphere_element);
  bounding_radius[0] = radius[0] + buffer_distance_;
}

void Sphere::finalize() {
}
//}

}  // namespace kernels

}  // namespace compute_bounding_radius

}  // namespace shape

}  // namespace mundy
