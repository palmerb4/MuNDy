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

/// \file ComputeAABB.cpp
/// \brief Definition of the ComputeAABB class

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
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements
#include <mundy_methods/ComputeAABB.hpp>    // for mundy::methods::ComputeAABB
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData

namespace mundy {

namespace methods {

// \name Constructors and destructor
//{

ComputeAABB::ComputeAABB(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "ComputeAABB: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_fixed_parameter_list = fixed_parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_fixed_params());

  // Parse the parameters
  Teuchos::ParameterList &parts_parameter_list = valid_fixed_parameter_list.sublist("input_parts");
  num_parts_ = parts_parameter_list.get<unsigned>("count");
  part_ptr_vector_.resize(num_parts_);
  for (size_t i = 0; i < num_parts_; i++) {
    // Fetch the i'th part and its parameters
    Teuchos::ParameterList &part_parameter_list = parts_parameter_list.sublist("input_part_" + std::to_string(i));
    const std::string part_name = part_parameter_list.get<std::string>("name");
    part_ptr_vector_[i] = meta_data_ptr_->get_part(part_name);

    // Fetch the parameters for this part's kernel
    const Teuchos::ParameterList &part_kernel_parameter_list =
        part_parameter_list.sublist("kernels").sublist("compute_aabb");

    // Create the kernel instance.
    const std::string kernel_name = part_kernel_parameter_list.get<std::string>("name");
    compute_aabb_kernel_ptrs_.push_back(mundy::meta::MetaKernelFactory<void, ComputeAABB>::create_new_instance(
        kernel_name, bulk_data_ptr_, part_kernel_parameter_list));
  }

  // For this method, the parts cannot intersect, if they did the result could be non-deterministic.
  for (size_t i = 0; i < num_parts_; i++) {
    for (size_t j = 0; j < num_parts_; j++) {
      if (i != j) {
        const bool parts_intersect = stk::mesh::intersect(*part_ptr_vector_[i], *part_ptr_vector_[j]);
        TEUCHOS_TEST_FOR_EXCEPTION(parts_intersect, std::invalid_argument,
                                   "ComputeAABB: Part " << part_ptr_vector_[i]->name() << " and "
                                                        << "Part " << part_ptr_vector_[j]->name() << "intersect.");
      }
    }
  }
}
//}

// \name MetaMethod interface implementation
//{

Teuchos::ParameterList ComputeAABB::set_transient_params(const Teuchos::ParameterList &transient_parameter_list) const {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_transient_parameter_list = transient_parameter_list;
  valid_transient_parameter_list.validateParametersAndSetDefaults(this->get_valid_transient_params());
}
//}

// \name Actions
//{

void ComputeAABB::execute(const stk::mesh::Selector &input_selector) {
  for (size_t i = 0; i < num_parts_; i++) {
    std::shared_ptr<mundy::meta::MetaKernelBase<void>> compute_aabb_kernel_ptr = compute_aabb_kernel_ptrs_[i];

    stk::mesh::Selector locally_owned_part = meta_data_ptr_->locally_owned_part() & *part_ptr_vector_[i];
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEM_RANK, locally_owned_part,
        [&compute_aabb_kernel_ptr]([[maybe_unused]] const mundy::mesh::BulkData &bulk_data, stk::mesh::Entity element) {
          compute_aabb_kernel_ptr->execute(element);
        });
  }
}
//}

}  // namespace methods

}  // namespace mundy
