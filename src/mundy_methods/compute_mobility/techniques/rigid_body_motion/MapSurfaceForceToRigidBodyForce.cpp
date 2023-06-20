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

/// \file MapSurfaceForceToRigidBodyForce.cpp
/// \brief Definition of the MapSurfaceForceToRigidBodyForce class

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
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaTwoWayKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::MetaMethodRegistry
#include <mundy_methods/compute_mobility/techniques/rigid_body_motion/MapSurfaceForceToRigidBodyForce.hpp>  // for mundy::methods::...::MapSurfaceForceToRigidBodyForce

namespace mundy {

namespace methods {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

// \name Constructors and destructor
//{

MapSurfaceForceToRigidBodyForce::MapSurfaceForceToRigidBodyForce(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                 const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "MapSurfaceForceToRigidBodyForce: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  static_validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Parse the parameters
  Teuchos::ParameterList &kernels_sublist = valid_fixed_params.sublist("kernels", true);
  const unsigned num_multibody_types_ = kernels_sublist.get<unsigned>("count");
  multibody_part_ptr_vector_.reserve(num_multibody_types_);
  multibody_kernel_ptrs_.reserve(num_multibody_types_);
  for (size_t i = 0; i < num_multibody_types_; i++) {
    Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
    const std::string kernel_name = kernel_params.get<std::string>("name");
    multibody_part_ptr_vector_.push_back(meta_data_ptr_->get_part(kernel_name));
    multibody_kernel_ptrs_.push_back(OurKernelFactory::create_new_instance(kernel_name, bulk_data_ptr_, kernel_params));
  }
}
//}

// \name MetaMethod interface implementation
//{

void MapSurfaceForceToRigidBodyForce::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  static_validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Parse the parameters
  Teuchos::ParameterList &kernels_sublist = valid_mutable_params.sublist("kernels", true);
  TEUCHOS_TEST_FOR_EXCEPTION(num_multibody_types_ == kernels_sublist.get<unsigned>("count"), std::invalid_argument,
                             "MapSurfaceForceToRigidBodyForce: Internal error. Mismatch between the stored kernel "
                             "count and the parameter list kernel count.\n"
                                 << "Odd... Please contact the development team.");
  for (size_t i = 0; i < num_multibody_types_; i++) {
    Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
    multibody_kernel_ptrs_[i]->set_mutable_params(kernel_params);
  }
}
//}

// \name Actions
//{

void MapSurfaceForceToRigidBodyForce::execute(const stk::mesh::Selector &input_selector) {
  // TODO(palmerb4): The following won't function correctly if the center body nodes are shared.
  // Is there a way to assert that an entity is not shared?

  // Currently we sum into the body force. Shall we add alpha and beta (like Tpetra) to let users choose
  // if the summation will occur or not. This also makes clear that the summation occurs.
  for (size_t i = 0; i < num_part_pairs_; i++) {
    std::shared_ptr<mundy::meta::MetaTwoWayKernelBase<void>> kernel_ptr = kernel_ptrs_[i];

    stk::mesh::Selector locally_owned_linker_part =
        meta_data_ptr_->locally_owned_part() & *part_pair_ptr_vector_[i].first;
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEM_RANK, locally_owned_linker_part,
        [&kernel_ptr]([[maybe_unused]] const mundy::mesh::BulkData &bulk_data, stk::mesh::Entity linker) {
          stk::mesh::Entity linked_element = bulk_data_ptr_->begin_elements(linker)[0];
          kernel_ptr->execute(linker, linked_element);
        });
  }
}
//}

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy
