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

/// \file ComputeConstraintResidual.cpp
/// \brief Definition of the ComputeConstraintResidual class

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
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::MetaMethodRegistry
#include <mundy_methods/resolve_constraints/techniques/non_smooth_lcp/ComputeConstraintResidual.hpp>  // for mundy::methods::...::non_smooth_lcp::ComputeConstraintResidual

namespace mundy {

namespace methods {

namespace resolve_constraints {

namespace techniques {

namespace non_smooth_lcp {

// \name Constructors and destructor
//{

ComputeConstraintResidual::ComputeConstraintResidual(mundy::mesh::BulkData *const bulk_data_ptr,
                                                     const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "Sphere: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  static_validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Fill the internal members using the given parameter list.
  std::string element_constraint_violation_field_name_ =
      valid_fixed_params.get<std::string>("element_constraint_violation_field_name");
  element_constraint_violation_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_constraint_violation_field_name_);

  // Prefetch the constraint part.
  constraint_part_ptr_ = meta_data_ptr_->get_part("CONSTRAINT");
}
//}

// \name MetaMethod interface implementation
//{

void ComputeConstraintResidual::set_mutable_params(
    [[maybe_unused]] const Teuchos::ParameterList &mutable_params) {
}
//}

// \name Actions
//{

double ComputeConstraintResidual::execute(const stk::mesh::Selector &input_selector) {
  // TODO(palmerb4): Break the following into techniques.
  // The following returns the constraint residual as the L1 norm of the constraint violation over the given parts.
  // Another technique would be to use the L2 norm.
  double local_residual = 0.0;
  stk::mesh::Selector locally_owned_intersection_with_constraints =
      stk::mesh::Selector(meta_data_ptr_->locally_owned_part()) & stk::mesh::Selector(*constraint_part_ptr_) &
      input_selector;

  // TODO(palmerb4): Replace this function with for_each_entity_reduce (only possible after the ngp update).
  // In the meantime, we use an internal stk function that doesn't use thread parallelism, lest we have a race
  // condition.
  stk::mesh::impl::for_each_selected_entity_run_no_threads(
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK, locally_owned_intersection_with_constraints,
      [&local_residual]([[maybe_unused]] const mundy::mesh::BulkData &bulk_data, stk::mesh::Entity element) {
        // This is the  norm of the constraint violation.
        const double constraint_violation =
            stk::mesh::field_data(*element_constraint_violation_field_ptr_, element)[0];
        local_residual = std::max(local_residual, constraint_violation);
      });

  // compute the global maximum absolute projected sep
  global_residual = 0.0;
  stk::all_reduce_max(bulk_data_ptr_->parallel(), &local_residual, &global_residual, 1);

  return global_residual;
}
//}

}  // namespace non_smooth_lcp

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace methods

}  // namespace mundy
