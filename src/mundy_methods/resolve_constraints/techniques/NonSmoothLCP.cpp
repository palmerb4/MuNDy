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

/// \file NonSmoothLCP.cpp
/// \brief Definition of the NonSmoothLCP class

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
#include <mundy_methods/compute_mobility/techniques/NonSmoothLCP.hpp>  // for mundy::methods::...::NonSmoothLCP

namespace mundy {

namespace methods {

namespace compute_mobility {

namespace techniques {

// \name Constructors and destructor
//{

NonSmoothLCP::NonSmoothLCP(stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "NonSmoothLCP: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_parameter_list = parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_params());

  // Fetch the parameters for this part's sub-methods.
  Teuchos::ParameterList &technique_parameter_list = valid_parameter_list.sublist("technique");
  const std::string technique_name = technique_parameter_list.get<std::string>("name");
  Teuchos::ParameterList &part_map_rbf_to_rbv_parameter_list =
      part_parameter_list.sublist("methods").sublist("map_rigid_body_force_to_rigid_body_velocity");
  Teuchos::ParameterList &part_map_rbv_to_sv_parameter_list =
      part_parameter_list.sublist("methods").sublist("map_rigid_body_velocity_to_surface_velocity");
  Teuchos::ParameterList &part_map_sf_to_rbf_parameter_list =
      part_parameter_list.sublist("methods").sublist("map_surface_force_to_rigid_body_force");

  // Initialize and store the sub-methods.
  const std::string rbf_to_rbv_class_id = part_map_rbf_to_rbv_parameter_list.get<std::string>("class_id");
  const std::string rbv_to_sv_class_id = part_map_rbv_to_sv_parameter_list.get<std::string>("class_id");
  const std::string sf_to_rbf_class_id = part_map_sf_to_rbf_parameter_list.get<std::string>("class_id");
  map_rigid_body_force_to_rigid_body_velocity_method_ptr_ =
      mundy::meta::MetaMethodFactory<void, NonSmoothLCP>::create_new_instance(rbf_to_rbv_class_id, bulk_data_ptr_,
                                                                              part_map_rbf_to_rbv_parameter_list);
  map_rigid_body_velocity_to_surface_velocity_method_ptr_ =
      mundy::meta::MetaMethodFactory<void, NonSmoothLCP>::create_new_instance(rbv_to_sv_class_id, bulk_data_ptr_,
                                                                              part_map_rbv_to_sv_parameter_list);
  map_surface_force_to_rigid_body_force_method_ptr_ =
      mundy::meta::MetaMethodFactory<void, NonSmoothLCP>::create_new_instance(sf_to_rbf_class_id, bulk_data_ptr_,
                                                                              part_map_sf_to_rbf_parameter_list);
}
//}

// \name Actions
//{

void NonSmoothLCP::execute() {
  // The following is the BBPGD solution to the linear complemenarity problem

  // Fill the Lagreange multipliers xkm1 with our initial guess. Our choice of initial guess is zero.
  for (size_t i = 0; i < num_parts_; i++) {
    stk::mesh::Selector locally_owned_part = meta_data_ptr_->locally_owned_part() & *part_ptr_vector_[i];
    // Here, we use an internal stk function that doesn't use thread parallelism, lest we have a race condition.
    // TODO(palmerb4): Replace this function with for_each_entity_reduce (only possible after the ngp update).
    stk::mesh::impl::for_each_selected_entity_run_no_threads(
        *bulk_data_ptr_, stk::topology::ELEM_RANK, locally_owned_part,
        []([[maybe_unused]] const stk::mesh::BulkData &bulk_data, stk::mesh::Entity element) {
          stk::mesh::field_data(*element_constraint_violation_field_name_, element)[0] = 0.0;
        });
  }

  // Iterate until the residual is below the threshold or until we surpass the maximum number of iterations.
  double alpha;
  int ite_count = 0;
  while (ite_count < max_num_iterations_) {
    if (ite_count > 0) {
      // Take a projected gradient step.
      // TODO(palmerb4): How do we pass alpha to this method?
      // Well, we need to break our parameters into those that impact the field values and those that don't.
      // These are the fixed and transient parameters.
      // alpha is a transient parameter and can therefore be passed into execute.
      // Ok, having execute accept and parse the parameters could be detimental to performance,
      // Why don't we add a set_transient_parameters function.
      cpmpute_gradient_step_method_ptr_->execute();
      compute_constraint_projection_method_ptr_->execute();
    }

    // Compute the new gradient using gk = D^T M D xk. This involves three steps.
    // Step 1: Compute the force induced by each constraint on its nodes.
    compute_constraint_forcing_method_ptr_->execute();

    // Step 2: Compute the the velocity of each particle's nodes.
    compute_mobility_method_ptr_->execute();

    // Step 3: Map the velocity of each constraint's nodes to that constraint's linearized rate of change of constraint
    // violation.
    compute_linearized_rate_of_change_of_constraint_violation->execute();

    // Compute the global constraint residuial.
    compute_constraint_violation_method_ptr_->execute();
    double residual = compute_constraint_residual_method_ptr_->execute();

    // Check for early termination.
    if (residual < tolerance) {
      // Success, the current set of Lagrange multipliers is correct.
      break;
    }

    // Compute the new Barzilai-Borwein step size.
    if (ite_count == 0) {
      // Initial guess for Barzilai-Borwein step size.
      alpha = 1.0 / residual;
    } else if (ite_count % 2 ==) {
      // Barzilai-Borwein step size Choice 1.
      a = xkdiff_dot_xkdiff;
      b = xkdiff_dot_gkdiff;
    } else {
      // Barzilai-Borwein step size Choice 2.
      a = xkdiff_dot_gkdiff;
      b = gkdiff_dot_gkdiff;
    }

    // Rotate the state of the xk and gk.
    bulk_data_ptr_->update_field_data_states(element_lagrange_multiplier_field_ptr_);
    bulk_data_ptr_->update_field_data_states(element_constraint_violation_gradient_field_ptr_);
    ite_count++;
  }

  // If the maximum number of iterations is surpassed, we optionally throw an error.
  if (ite_count >= con_ite_max && throw_on_failed_to_converge_) {
    ThrowRequireMsg(false, "NonSmoothLCP: Failed to converge in "
                               << max_num_iterations_ << " iterations. \n Current residual is " << residual);
  }
}
//}

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy
