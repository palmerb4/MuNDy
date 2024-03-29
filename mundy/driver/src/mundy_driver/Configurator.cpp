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

/// \file Configurator.cpp
/// \brief Definition of the Configurator class

// C++ core libs

// Trilinos libs
#include <Teuchos_YamlParameterListHelpers.hpp>

// Mundy libs
#include <mundy_driver/Configurator.hpp>  // for mundy::driver::Configurator

namespace mundy {

namespace driver {

/// \name Constructors and destructors
//@{

Configurator::Configurator(const std::string& input_format, const std::string& input_filename) {
  // Check to see what input file format we have
  if (input_format == "yaml") {
    param_list_ = *Teuchos::getParametersFromYamlFile(input_filename);
  } else if (input_format == "xml") {
    MUNDY_THROW_ASSERT(false, std::invalid_argument,
                       "mundy::driver::Configurator XML files are not implemented for reading yet.");
  } else {
    MUNDY_THROW_ASSERT(false, std::invalid_argument,
                       "mundy::driver::Configurator file_format " + input_format + " not recognized.");
  }
}

//@}

//! \name Queries of registered "methods"
//@{

std::string Configurator::get_registered_MetaMethodExecutionInterface() {
  return mundy::driver::ConfigurableMetaMethodFactory<
      mundy::meta::MetaMethodExecutionInterface<void>>::get_keys_as_string();
}

std::string Configurator::get_registered_MetaMethodSubsetExecutionInterface() {
  return mundy::driver::ConfigurableMetaMethodFactory<
      mundy::meta::MetaMethodSubsetExecutionInterface<void>>::get_keys_as_string();
}

std::string Configurator::get_registered_MetaMethodPairwiseSubsetExecutionInterface() {
  return mundy::driver::ConfigurableMetaMethodFactory<
      mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>::get_keys_as_string();
}

std::string Configurator::get_registered_classes() {
  return get_registered_MetaMethodExecutionInterface() + get_registered_MetaMethodSubsetExecutionInterface() +
         get_registered_MetaMethodPairwiseSubsetExecutionInterface();
}

//@}

//! \name Parse configuration
//@{

void Configurator::ParseParameters() {
  // At this point we are expecting to have a valid param_list_. Get into the Configuration section first, and configure
  // the MetaMethod* that we need
  Teuchos::ParameterList configuration_list = param_list_.sublist("Configuration");

  std::cout << "Configuration sublist:\n" << configuration_list << std::endl;
}

//@}

}  // namespace driver

}  // namespace mundy
