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

/// \file ComputeConstraintProjection.cpp
/// \brief Definition of the ComputeConstraintProjection class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/BulkData.hpp>    // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>      // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>    // for stk::mesh::Selector

// Mundy libs
#include <mundy_meta/MetaKernel.hpp>                      // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaKernelFactory.hpp>               // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaMethod.hpp>                      // for mundy::meta::MetaMethod
#include <mundy_meta/MetaMethodRegistry.hpp>              // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>                // for mundy::meta::PartRequirements
#include <mundy_methods/ComputeConstraintProjection.hpp>  // for mundy::methods::ComputeConstraintProjection

namespace mundy {

namespace methods {

// \name Constructors and destructor
//{

ComputeConstraintProjection::ComputeConstraintProjection(stk::mesh::BulkData *const bulk_data_ptr,
                                                         const Teuchos::ParameterList &parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), part_ptr_vector_(part_ptr_vector), num_parts_(part_ptr_vector_.size()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "mundy::methods::ComputeConstraintProjection: bulk_data_ptr cannot be a nullptr.");

  // The parts cannot intersect.
  for (int i = 0; i < num_parts_; i++) {
    for (int j = 0; j < num_parts_; j++) {
      if (i = !j) {
        const bool parts_intersect = stk::mesh::intersect(*part_ptr_vector[i], *part_ptr_vector[j]);
        TEUCHOS_TEST_FOR_EXCEPTION(parts_intersect, std::invalid_argument,
                                   "mundy::methods::ComputeConstraintProjection: Part "
                                       << part_ptr_vector[i]->name() << " and "
                                       << "Part " << part_ptr_vector[j]->name() << "intersect.");
      }
    }
  }

  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_parameter_list = parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_params());

  // Create and store the required kernels.
  for (int i = 0; i < num_parts_; i++) {
    // Fetch the parameters for this part's kernel
    const std::string part_name = part_ptr_vector_[i]->name();
    const Teuchos::ParameterList &part_parameter_list = valid_parameter_list.sublist(part_name);
    const Teuchos::ParameterList &part_kernel_parameter_list =
        part_parameter_list.sublist("kernels").sublist("compute_constraint_projection");

    // Create the kernel instance.
    const std::string kernel_name = part_kernel_parameter_list.get<std::string>("name");
    compute_constraint_projection_kernels_.push_back(
        MetaKernelFactory<ComputeConstraintProjection>::create_new_instance(kernel_name, bulk_data_ptr,
                                                                            part_kernel_parameter_list));
  }
}
//}

// \name Actions
//{

/// \brief Run the method's core calculation.
void ComputeConstraintProjection::execute() {
  for (int i = 0; i < num_parts_; i++) {
    const MetaKernel &compute_constraint_projection_kernel = compute_constraint_projection_kernels_[i];

    stk::mesh::Selector locally_owned_part =
        bulk_data_ptr->mesh_meta_data().locally_owned_part() && *part_ptr_vector_[i];
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr, stk::topology::ELEM_RANK, locally_owned_part,
        [&compute_constraint_projection_kernel](const stk::mesh::BulkData &bulk_data, stk::mesh::Entity element) {
          compute_constraint_projection_kernel->execute(element);
        });
  }
}
//}

}  // namespace methods

}  // namespace mundy
