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

/// \file ComputeBoundingRadiusSphereKernel.cpp
/// \brief Definition of the ComputeBoundingRadiusSphereKernel class

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
#include <mundy_meta/FieldRequirements.hpp>   // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaKernel.hpp>          // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaKernelFactory.hpp>   // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernelRegistry.hpp>  // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartRequirements.hpp>    // for mundy::meta::PartRequirements
#include <mundy_methods/ComputeBoundingRadiusSphereKernel.hpp>  // for mundy::methods::ComputeBoundingRadiusSphereKernel

namespace mundy {

namespace methods {

// \name Constructors and destructor
//{

ComputeBoundingRadiusSphereKernel::ComputeBoundingRadiusSphereKernel(stk::mesh::BulkData *const bulk_data_ptr,
                                                                     const Teuchos::ParameterList &parameter_list) {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_parameter_list = parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_params());

  // Fill the internal members using the internal parameter list.
  buffer_distance_ = valid_parameter_list.get<double>("buffer_distance");
  radius_field_name_ = valid_parameter_list.get<std::string>("radius_field_name");
  bounding_radius_field_name_ = valid_parameter_list.get<std::string>("bounding_radius_field_name");

  // Store the input params.
  radius_field_ptr_ = *bulk_data_ptr->get_field<double>(stk::topology::ELEM_RANK, radius_field_name_);
  bounding_radius_field_ptr_ = *bulk_data_ptr->get_field<double>(stk::topology::ELEM_RANK, bounding_radius_field_name_);
}
//}

// \name Actions
//{

void ComputeBoundingRadiusSphereKernel::execute(const stk::mesh::Entity &element) {
  double *radius = stk::mesh::field_data(*radius_field_ptr_, element);
  double *bounding_sphere = stk::mesh::field_data(*bounding_sphere_field_ptr_, element);
  bounding_sphere[0] = radius[0] + buffer_distance_;
}
//}

}  // namespace methods

}  // namespace mundy
