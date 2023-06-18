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

/// \file MapRigidBodyForceToRigidBodyVelocity.cpp
/// \brief Definition of the MapRigidBodyForceToRigidBodyVelocity class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>     // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector

// Mundy libs
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_methods/compute_mobility/techniques/rigid_body_motion/MapRigidBodyForceToRigidBodyVelocity.hpp>  // for mundy::methods::...::MapRigidBodyForceToRigidBodyVelocity

namespace mundy {

namespace methods {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

// \name Constructors and destructor
//{

MapRigidBodyForceToRigidBodyVelocity::MapRigidBodyForceToRigidBodyVelocity(
    mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "MapRigidBodyForceToRigidBodyVelocity: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_fixed_parameter_list = fixed_parameter_list;
  valid_fixed_parameter_list.validateParametersAndSetDefaults(this->get_valid_fixed_params());

  // Fetch the technique sublist and return its parameters.
  Teuchos::ParameterList &technique_parameter_list = valid_fixed_parameter_list.sublist("technique");
  const std::string technique_name = technique_parameter_list.get<std::string>("name");

  technique_ptr_ = mundy::meta::MetaMethodFactory<void, MapRigidBodyForceToRigidBodyVelocity>::create_new_instance(
      technique_name, bulk_data_ptr_, technique_parameter_list);
}
//}

// \name MetaMethod interface implementation
//{

Teuchos::ParameterList MapRigidBodyForceToRigidBodyVelocity::set_mutable_params(
    const Teuchos::ParameterList &mutable_parameter_list) const {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_mutable_parameter_list = mutable_parameter_list;
  valid_mutable_parameter_list.validateParametersAndSetDefaults(this->get_valid_mutable_params());

  // Fill the internal mutable parameters and set the mutable parameters of each registered kernel.
  // In this case, all kernels have the same mutable parameter (viscosity), so we simply pass along our list.
  for (int i = 0; i < kernel_ptrs_.size(); i++) {
    kernel_ptrs_[i]->set_mutable_params(valid_mutable_parameter_list);
  }
}
//}

// \name Actions
//{

void MapRigidBodyForceToRigidBodyVelocity::execute(const stk::mesh::Selector &input_selector) {
  technique_ptr_->execute();
}
//}

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy
