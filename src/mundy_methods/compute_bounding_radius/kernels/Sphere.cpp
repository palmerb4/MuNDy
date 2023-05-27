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
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field, stl::mesh::field_data

// Mundy libszhan1908
#include <mundy_methods/compute_bounding_radius/kernels/Sphere.hpp>  // for mundy::methods::compute_bounding_radius::kernels::Sphere

namespace mundy {

namespace methods {

namespace compute_bounding_radius {

namespace kernels {

// \name Constructors and destructor
//{

Sphere::Sphere(stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_fixed_parameter_list = fixed_parameter_list;
  valid_fixed_parameter_list.validateParametersAndSetDefaults(this->get_valid_fixed_params());

  // Fill the internal members using the internal parameter list.
  radius_field_name_ = valid_fixed_parameter_list.get<std::string>("radius_field_name");
  bounding_radius_field_name_ = valid_fixed_parameter_list.get<std::string>("bounding_radius_field_name");

  // Store the input params.
  radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, radius_field_name_);
  bounding_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, bounding_radius_field_name_);
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
  buffer_distance_ = valid_transient_parameter_list.get<double>("buffer_distance");
}
//}

// \name Actions
//{

void Sphere::execute(const stk::mesh::Entity &element) {
  double *radius = stk::mesh::field_data(*radius_field_ptr_, element);
  double *bounding_radius = stk::mesh::field_data(*bounding_radius_field_ptr_, element);
  bounding_radius[0] = radius[0] + buffer_distance_;
}
//}

}  // namespace kernels

}  // namespace compute_bounding_radius

}  // namespace methods

}  // namespace mundy
