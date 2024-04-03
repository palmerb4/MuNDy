// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
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

/// \file Driver.cpp
/// \brief Definition of the Driver class

// C++ core libs

// Trilinos libs
#include <Teuchos_YamlParameterListHelpers.hpp>

// Mundy libs
#include <mundy_driver/Driver.hpp>  // for mundy::driver::Configurator

namespace mundy {

namespace driver {

/// \name Using directives
//@{

// Shorthand names for the different meta method factories we use later
using FactoryMM = mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodExecutionInterface<void>>;
using FactoryMMS = mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>;
using FactoryMMPS =
    mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>;

//@}

/// \name Constructors and destructors
//@{

Driver::Driver(const stk::ParallelMachine& communicator) {
  set_communicator(communicator);
}

//@}

//! \name Queries of registered "methods"
//@{

std::string Driver::get_registered_meta_method_execution_interface() {
  return FactoryMM::get_keys_as_string();
}

std::string Driver::get_registered_meta_method_subset_execution_interface() {
  return FactoryMMS::get_keys_as_string();
}

std::string Driver::get_registered_meta_method_pairwise_subset_execution_interface() {
  return FactoryMMPS::get_keys_as_string();
}

std::string Driver::get_registered_classes() {
  return get_registered_meta_method_execution_interface() + get_registered_meta_method_subset_execution_interface() +
         get_registered_meta_method_pairwise_subset_execution_interface();
}

void Driver::print_mesh_requirements() {
  mesh_reqs_ptr_->print_reqs(std::cout, 0);
}

//@}

//! \name Setters and Getters
//@{

void Driver::set_n_dimensions(const int n_dim) {
  n_dim_ = n_dim;
}

void Driver::set_communicator(const stk::ParallelMachine& communicator) {
  communicator_ = communicator;
}

void Driver::set_node_coordinates_field_name(const std::string& node_coordinates_field_name) {
  node_coordinates_field_name_ = node_coordinates_field_name;
}

//@}

void Driver::build_mesh_requirements() {
  mesh_reqs_ptr_ = std::make_shared<mundy::meta::MeshRequirements>(communicator_);
  mesh_reqs_ptr_->set_spatial_dimension(n_dim_);
  mesh_reqs_ptr_->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
}

void Driver::add_mesh_requirement(const std::string& method_type, const std::string& method_name,
                                  const Teuchos::ParameterList& fixed_params) {
  // Check to make sure we've already called build_mesh_requirements (mesh_reqs_ptr_ isn't null)
  MUNDY_THROW_ASSERT(mesh_reqs_ptr_ != nullptr, std::invalid_argument,
                     "Cannot add a mesh requirement before mesh is created in Driver.");
  // Add a single requirement to the mesh, taking into account what factory it came from
  // TODO(cje): At some point having this be a single factory, or making this easier, would be nice
  if (method_type == "meta_method_execution_interface") {
    mesh_reqs_ptr_->merge(FactoryMM::get_mesh_requirements(method_name, fixed_params));
  } else if (method_type == "meta_method_subset_execution_interface") {
    mesh_reqs_ptr_->merge(FactoryMMS::get_mesh_requirements(method_name, fixed_params));
  } else if (method_type == "meta_method_pairwise_subset_execution_interface") {
    mesh_reqs_ptr_->merge(FactoryMMPS::get_mesh_requirements(method_name, fixed_params));
  }
}

void Driver::declare_mesh() {
  // Check to make sure we've already called build_mesh_requirements (mesh_reqs_ptr_ isn't null)
  MUNDY_THROW_ASSERT(mesh_reqs_ptr_ != nullptr, std::invalid_argument,
                     "Cannot declare a mesh before mesh is created in Driver.");
  // This does not commit the mesh as it is supposed to play nicely with IO, which sometimes commits the mesh on a
  // restart.
  bulk_data_ptr_ = mesh_reqs_ptr_->declare_mesh();
  meta_data_ptr_ = bulk_data_ptr_->mesh_meta_data_ptr();
  // Also set to using simple fields
  meta_data_ptr_->use_simple_fields();
  // Set the coordinate field name
  meta_data_ptr_->set_coordinate_field_name(node_coordinates_field_name_);

  // TODO(cje): Remove later
  stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);
}

void Driver::commit_mesh() {
  MUNDY_THROW_ASSERT(meta_data_ptr_ != nullptr, std::invalid_argument,
                     "Cannot commit a mesh without valid meta_data in Driver.");
  // Ask the meta data to commit the mesh for us
  meta_data_ptr_->commit();
}

void Driver::add_meta_class_instance(const std::string& method_type, const std::string& method_name,
                                     const Teuchos::ParameterList& fixed_params,
                                     const Teuchos::ParameterList& mutable_params) {
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "Cannot create a class instance without vaild BulkData in Driver.");
  if (method_type == "meta_method_execution_interface") {
    // Create a new class instance
    std::shared_ptr<mundy::meta::MetaMethodExecutionInterface<void>> new_meta_method =
        FactoryMM::create_new_instance(method_name, bulk_data_ptr_.get(), fixed_params);
    new_meta_method->set_mutable_params(mutable_params);
    meta_methods_.push_back(new_meta_method);
    meta_method_string_to_id_[method_name] = static_cast<unsigned>(meta_methods_.size()) - 1;
  } else if (method_type == "meta_method_subset_execution_interface") {
    // Create a new class instance
    std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> new_meta_method =
        FactoryMMS::create_new_instance(method_name, bulk_data_ptr_.get(), fixed_params);
    new_meta_method->set_mutable_params(mutable_params);
    meta_methods_subset_.push_back(new_meta_method);
    meta_method_subset_string_to_id_[method_name] = static_cast<unsigned>(meta_methods_subset_.size()) - 1;
  } else if (method_type == "meta_method_pairwise_subset_execution_interface") {
    // Create a new class instance
    std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>> new_meta_method =
        FactoryMMPS::create_new_instance(method_name, bulk_data_ptr_.get(), fixed_params);
    new_meta_method->set_mutable_params(mutable_params);
    meta_methods_pairwise_subset_.push_back(new_meta_method);
    meta_method_pairwise_subset_string_to_id_[method_name] =
        static_cast<unsigned>(meta_methods_pairwise_subset_.size()) - 1;
  }
}

}  // namespace driver

}  // namespace mundy
