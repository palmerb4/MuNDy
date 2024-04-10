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

//@}

//! \name Setters and Getters
//@{

void Driver::set_n_dimensions(const int n_dim) {
  n_dim_ = n_dim;
}

void Driver::set_communicator(const stk::ParallelMachine& communicator) {
  communicator_ = communicator;
}

//@}

void Driver::build_mesh_requirements() {
  mesh_reqs_ptr_ = std::make_shared<mundy::meta::MeshRequirements>(communicator_);
  mesh_reqs_ptr_->set_spatial_dimension(n_dim_);
  mesh_reqs_ptr_->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
}

}  // namespace driver

}  // namespace mundy
