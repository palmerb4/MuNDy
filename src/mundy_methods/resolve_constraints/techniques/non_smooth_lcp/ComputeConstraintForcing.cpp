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

/// \file ComputeConstraintForcing.cpp
/// \brief Definition of the ComputeConstraintForcing class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>     // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/BulkData.hpp>       // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector

// Mundy libs
#include <mundy_meta/MetaKernel.hpp>          // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaKernelFactory.hpp>   // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaMethod.hpp>          // for mundy::meta::MetaMethod
#include <mundy_meta/MetaMethodRegistry.hpp>  // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>    // for mundy::meta::PartRequirements
#include <mundy_methods/resolve_constraints/techniques/non_smooth_lcp/ComputeConstraintForcing.hpp>  // for mundy::methods::...::non_smooth_lcp::ComputeConstraintForcing

namespace mundy {

namespace methods {

namespace resolve_constraints {

namespace techniques {

namespace non_smooth_lcp {

// \name Constructors and destructor
//{

ComputeConstraintForcing::ComputeConstraintForcing(stk::mesh::BulkData *const bulk_data_ptr,
                                                   const Teuchos::ParameterList &parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "ComputeConstraintForcing: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_parameter_list = parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_params());

  // Parse the parameters
  Teuchos::ParameterList &parts_parameter_list = valid_parameter_list.sublist("input_part_pairs");
  num_part_pairs_ = parts_parameter_list.get<unsigned>("count");
  part_pair_ptr_vector_.resize(num_part_pairs_);
  for (int i = 0; i < num_part_pairs_; i++) {
    // Fetch the i'th part and its parameters.
    Teuchos::ParameterList &part_parameter_list = parts_parameter_list.sublist("input_part_pair_" + std::to_string(i));
    const Teuchos::Array<std::string> pair_names = part_pair_parameter_list.get<Teuchos::Array<std::string>>("name");
    part_pair_ptr_vector_[i] =
        std::make_pair(meta_data_ptr_->get_part(pair_names[0]), meta_data_ptr_->get_part(pair_names[1]));

    // Fetch the parameters for this part's kernel.
    const Teuchos::ParameterList &part_kernel_parameter_list =
        part_parameter_list.sublist("kernels").sublist("compute_constraint_forcing");

    // Create the kernel instance.
    const std::string kernel_name = part_kernel_parameter_list.get<std::string>("name");
    compute_constraint_forcing_kernel_ptrs_.push_back(
        mundy::meta::MetaKernelFactory<void, ComputeConstraintForcing>::create_new_instance(
            kernel_name, bulk_data_ptr_, part_kernel_parameter_list));
  }

  // For this method, none of the source parts can intersect; if they did the result could be non-determinaistic.
  // The same is true for the target parts.
  for (int i = 0; i < num_part_pairs_; i++) {
    for (int j = 0; j < num_part_pairs_; j++) {
      if (i != j) {
        const bool source_parts_intersect =
            stk::mesh::intersect(*part_pair_ptr_vector_[i].first, *part_pair_ptr_vector_[j].first);
        TEUCHOS_TEST_FOR_EXCEPTION(source_parts_intersect, std::invalid_argument,
                                   "ComputeConstraintForcing: Source Part "
                                       << part_pair_ptr_vector_[i].first->name() << " and "
                                       << "Part " << part_pair_ptr_vector_[j].first->name() << "intersect.");
        const bool target_parts_intersect =
            stk::mesh::intersect(*part_pair_ptr_vector_[i].second, *part_pair_ptr_vector_[j].second);
        TEUCHOS_TEST_FOR_EXCEPTION(target_parts_intersect, std::invalid_argument,
                                   "ComputeConstraintForcing: Target Part "
                                       << part_pair_ptr_vector_[i].second->name() << " and "
                                       << "Part " << part_pair_ptr_vector_[j].second->name() << "intersect.");
      }
    }
  }
}
//}

// \name Actions
//{

void ComputeConstraintForcing::execute() {
  for (int i = 0; i < num_part_pairs_; i++) {
    std::shared_ptr<mundy::meta::MetaKernelBase<void>> &compute_constraint_forcing_kernel_ptr =
        compute_constraint_forcing_kernel_ptrs_[i];

    stk::mesh::Selector locally_owned_source_part =
        meta_data_ptr_->locally_owned_part() & *part_pair_ptr_vector_[i].first;
    stk::mesh::Selector locally_owned_target_part =
        meta_data_ptr_->locally_owned_part() & *part_pair_ptr_vector_[i].second;
    stk::mesh::for_each_entity_run(*bulk_data_ptr_, stk::topology::NODE_RANK, locally_owned_target_part,
                                   [&compute_constraint_forcing_kernel_ptr, &locally_owned_source_part](
                                       const stk::mesh::BulkData &bulk_data, stk::mesh::Entity sphere_node) {
                                     // Run the forcing kernel on neighbors of the source element that are in the target
                                     // part.
                                     int num_connected_elems = bulk_data.num_elements(sphere_node);
                                     const stk::mesh::Entity *connected_elems = bulk_data.begin_elements(sphere_node);
                                     for (unsigned i = 0; i < num_connected_elems; ++i) {
                                       if (bulk_data.bucket(connected_elems[i]).member(locally_owned_source_part)) {
                                         compute_constraint_forcing_kernel_ptr->execute(connected_elems[i],
                                                                                        sphere_node);
                                       }
                                     }
                                   });
  }
}
//}

}  // namespace non_smooth_lcp

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace methods

}  // namespace mundy
