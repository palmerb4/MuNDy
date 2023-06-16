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

/// \file NodeEuler.cpp
/// \brief Definition of ComputeTimeIntegration's NodeEuler technique.

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
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaTwoWayKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements
#include <mundy_methods/compute_time_integration/NodeEuler.hpp>  // for mundy::methods::compute_time_integration::NodeEuler

namespace mundy {

namespace methods {

namespace compute_time_integration {

// \name Constructors and destructor
//{

NodeEuler::NodeEuler(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                             "NodeEuler: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_fixed_parameter_list = fixed_parameter_list;
  valid_fixed_parameter_list.validateParametersAndSetDefaults(this->get_valid_fixed_params());

  // Parse the parameters
  Teuchos::ParameterList &parts_parameter_list = valid_fixed_parameter_list.sublist("input_parts");
  num_parts_ = parts_parameter_list.get<unsigned>("count");
  part_ptr_vector_.resize(num_parts_);
  for (size_t i = 0; i < num_parts_; i++) {
    // Fetch the i'th part and its parameters
    Teuchos::ParameterList &part_parameter_list = parts_parameter_list.sublist("input_part_" + std::to_string(i));
    const std::string part_name = part_parameter_list.get<std::string>("name");
    part_ptr_vector_[i] = meta_data_ptr_->get_part(part_name);
  }

  // Fill the internal members using the internal parameter list.
  node_coord_field_name_ = valid_fixed_parameter_list.get<std::string>("node_coord_field_name");
  node_velocity_field_name_ = valid_fixed_parameter_list.get<std::string>("node_velocity_field_name");

  // Store the input params.
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
  node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name_);
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
  time_step_size_ = valid_transient_parameter_list.get<double>("time_step_size");
}
//}

// \name Actions
//{

void NodeEuler::execute(const stk::mesh::Selector &input_selector) {
  for (size_t i = 0; i < num_parts_; i++) {
    stk::mesh::Selector locally_owned_part = meta_data_ptr_->locally_owned_part();
    stk::mesh::for_each_entity_run(*bulk_data_ptr_, stk::topology::NODE_RANK, locally_owned_part,
                                   [](const mundy::mesh::BulkData &bulk_data, stk::mesh::Entity node) {
                                     // TODO(palmerb4): Add a flag for specifying that node position has changed
                                     // This is the best way to indicate that things like the normal vector need
                                     // updated. Does STK have an observer that lets us check if fields need updated?

                                     // Euler step position.
                                     double *node_coords = stk::mesh::field_data(*node_coord_field_ptr_, node);
                                     double *node_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, node);
                                     node_coords[0] += time_step_size_ * node_velocity[0];
                                     node_coords[1] += time_step_size_ * node_velocity[1];
                                     node_coords[2] += time_step_size_ * node_velocity[2];
                                   });
  }
}
//}

}  // namespace compute_time_integration

}  // namespace methods

}  // namespace mundy
