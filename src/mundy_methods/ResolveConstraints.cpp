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

/// \file ResolveConstraints.cpp
/// \brief Definition of the ResolveConstraints class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector

// Mundy libs
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_meta/MetaFactory.hpp>            // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>             // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>             // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>           // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>       // for mundy::meta::PartRequirements
#include <mundy_methods/ResolveConstraints.hpp>  // for mundy::methods::ResolveConstraints
#include <mundy/throw_assert.hpp>   // for MUNDY_THROW_ASSERT

namespace mundy {

namespace methods {

// \name Constructors and destructor
//{

ResolveConstraints::ResolveConstraints(mundy::mesh::BulkData *const bulk_data_ptr,
                                       const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                             "ResolveConstraints: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  static_validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Fetch the technique sublist and return its parameters.
  Teuchos::ParameterList &technique_params = valid_fixed_params.sublist("technique");
  const std::string technique_name = technique_params.get<std::string>("name");
  technique_ptr_ = OurMethodFactory::create_new_instance(technique_name, bulk_data_ptr_, technique_params);
}
//}

// \name MetaMethod interface implementation
//{

void ResolveConstraints::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  static_validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fetch the technique sublist and return its parameters.
  Teuchos::ParameterList &technique_params = valid_mutable_params.sublist("technique");
  const std::string technique_name = technique_params.get<std::string>("name");
  technique_ptr_->set_mutable_params(technique_params);
}
//}

// \name Actions
//{

void ResolveConstraints::execute(const stk::mesh::Selector &input_selector) {
  technique_ptr_->execute(input_selector);
}
//}

}  // namespace methods

}  // namespace mundy
