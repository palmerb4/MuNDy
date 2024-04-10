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
#include <mundy_driver/Driver.hpp>        // for mundy::driver::Driver

namespace mundy {

namespace driver {

/// \name Using directives
//@{

// Shorthand names for the different meta method factories we use later
// TODO(cje): There must be a better way to get the MetaMethod factories and check/create from them in a general way.
// Right now I have ugly if-else blocks both here and the Driver, which are a pain, and make the code more fragile if we
// add things.
using FactoryMM = mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodExecutionInterface<void>>;
using FactoryMMS = mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>;
using FactoryMMPS =
    mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>;

//@}

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

std::string Configurator::get_registered_meta_method_execution_interface() {
  return FactoryMM::get_keys_as_string();
}

std::string Configurator::get_registered_meta_method_subset_execution_interface() {
  return FactoryMMS::get_keys_as_string();
}

std::string Configurator::get_registered_meta_method_pairwise_subset_execution_interface() {
  return FactoryMMPS::get_keys_as_string();
}

std::string Configurator::get_registered_classes() {
  return get_registered_meta_method_execution_interface() + get_registered_meta_method_subset_execution_interface() +
         get_registered_meta_method_pairwise_subset_execution_interface();
}

//@}

//! \name Parse
//@{

void Configurator::parse_parameters() {
  // At this point we are expecting to have a valid param_list_. Get into the Configuration section first, and configure
  // the MetaMethod* that we need
  MUNDY_THROW_ASSERT(param_list_.isSublist("configuration"), std::invalid_argument,
                     "mundy::driver::ParseParameters parameters do not contain an 'configuration' sublist.");
  const Teuchos::ParameterList config_params = param_list_.sublist("configuration");
  parse_configuration(config_params);
}

void Configurator::parse_configuration(const Teuchos::ParameterList& config_params) {
  // TODO(cje): Remove later
  std::cout << "Configuration sublist:\n" << config_params << std::endl;

  // Get simulation variables that don't belong to a specific Meta*
  n_dim_ = config_params.get<int>("n_dim");

  // Loop over known MetaMethod types and parse_and_configure them
  for (auto metamethod_type = metamethod_types_.begin(); metamethod_type != metamethod_types_.end();
       ++metamethod_type) {
    std::string metamethod_str(*metamethod_type);
    // Configure the MetaMethod interface
    if (config_params.isSublist(metamethod_str)) {
      const Teuchos::ParameterList metamethod_params = config_params.sublist(metamethod_str);
      parse_metamethod(metamethod_str, metamethod_params);
    }
  }
}

void Configurator::parse_metamethod(const std::string& method_type, const Teuchos::ParameterList& method_params) {
  // TODO(cje): Remove later
  std::cout << "meta_method " << method_type << " sublist:\n" << method_params << std::endl;
  // Loop over MetaMethod sublist
  for (auto pit = method_params.begin(); pit != method_params.end(); ++pit) {
    const std::string& param_name = pit->first;
    const Teuchos::ParameterEntry& entry = pit->second;

    // TODO(cje): Remove later
    std::cout << "Processing MetaMethod " << param_name << std::endl;

    // Dive into the sublist of this method and get the method it's trying to call
    MUNDY_THROW_ASSERT(
        entry.isList(), std::invalid_argument,
        "mundy::driver::Configurator::parse_and_configure_metamethod Invalid specification of method " + param_name);
    const Teuchos::ParameterList method_sublist = Teuchos::getValue<Teuchos::ParameterList>(entry);

    // Check to see if the method is a valid one (registered with MetaFactory). Unfortunately, without reflection, we
    // have to make this an ugly if/else statement
    const std::string method_name = method_sublist.get<std::string>("method");
    if (method_type == "meta_method_execution_interface") {
      bool is_valid_metamethod = FactoryMM::is_valid_key(method_name);
      MUNDY_THROW_ASSERT(is_valid_metamethod, std::invalid_argument,
                         "mundy::driver::Configurator::parse_and_configure_metamethod Could not find MetaMethod " +
                             method_name + " for name " + param_name);
    } else if (method_type == "meta_method_subset_execution_interface") {
      bool is_valid_metamethod = FactoryMMS::is_valid_key(method_name);
      MUNDY_THROW_ASSERT(is_valid_metamethod, std::invalid_argument,
                         "mundy::driver::Configurator::parse_and_configure_metamethod Could not find MetaMethod " +
                             method_name + " for name " + param_name);
    } else if (method_type == "meta_method_pairwise_subset_execution_interface") {
      bool is_valid_metamethod = FactoryMMPS::is_valid_key(method_name);
      MUNDY_THROW_ASSERT(is_valid_metamethod, std::invalid_argument,
                         "mundy::driver::Configurator::parse_and_configure_metamethod Could not find MetaMethod " +
                             method_name + " for name " + param_name);
    }

    // Now that we have the name of the method, etc, we can slice off the fixed/mutable parameters and validate them. Do
    // both together, as we are going to have an icky if block again I believe. Also, create a default structure to
    // load, as sometimes we won't have both types of paramters specified.
    Teuchos::ParameterList fixed_params;
    Teuchos::ParameterList mutable_params;
    if (method_sublist.isSublist("fixed_params")) {
      fixed_params = method_sublist.sublist("fixed_params");
    }
    if (method_sublist.isSublist("mutable_params")) {
      mutable_params = method_sublist.sublist("mutable_params");
    }
    // Create a mirrored copy of the params to validate against
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    Teuchos::ParameterList valid_mutable_params = mutable_params;

    // Validate based on the type of the factory
    if (method_type == "meta_method_execution_interface") {
      valid_fixed_params.validateParametersAndSetDefaults(FactoryMM::get_valid_fixed_params(method_name));
      valid_mutable_params.validateParametersAndSetDefaults(FactoryMM::get_valid_mutable_params(method_name));
    } else if (method_type == "meta_method_subset_execution_interface") {
      valid_fixed_params.validateParametersAndSetDefaults(FactoryMMS::get_valid_fixed_params(method_name));
      valid_mutable_params.validateParametersAndSetDefaults(FactoryMMS::get_valid_mutable_params(method_name));
    } else if (method_type == "meta_method_pairwise_subset_execution_interface") {
      valid_fixed_params.validateParametersAndSetDefaults(FactoryMMPS::get_valid_fixed_params(method_name));
      valid_mutable_params.validateParametersAndSetDefaults(FactoryMMPS::get_valid_mutable_params(method_name));
    }

    // Build the enabled package to pass to Driver later. First, check to see if we already have a unique user name for
    // this action, and if so, throw an error up that the user cannot do this, each name they give to a meta method must
    // be unique.
    if (enabled_meta_methods_.find(param_name) != enabled_meta_methods_.end()) {
      MUNDY_THROW_ASSERT(false, std::invalid_argument,
                         "User-defined MetaMethod name " + param_name +
                             " already exist, check configuration for duplicate MetaMethod names");
    }
    // Now that we have a unique name, create the structure for the map for later (this is two commands because getting
    // the type coersion for std::make_tuple doesn't seem to like things for some reason...)
    std::tuple mtuple(method_type, method_name, valid_fixed_params, valid_mutable_params);
    enabled_meta_methods_[param_name] = mtuple;

    ///////////////////////////////////////////////////////////////////////////////
    //// Mesh interaction
    ///////////////////////////////////////////////////////////////////////////////
    //// Merge the requirements onto the mesh
    // if (method_type == "meta_method_execution_interface") {
    //   mesh_reqs_ptr_->merge(FactoryMM::get_mesh_requirements(method_name, valid_fixed_params));
    // } else if (method_type == "meta_method_subset_execution_interface") {
    //   mesh_reqs_ptr_->merge(FactoryMMS::get_mesh_requirements(method_name, valid_fixed_params));
    // } else if (method_type == "meta_method_pairwise_subset_execution_interface") {
    //   mesh_reqs_ptr_->merge(FactoryMMPS::get_mesh_requirements(method_name, valid_fixed_params));
    // }
  }  // for loop over user-given method names
}

//@}

}  // namespace driver

}  // namespace mundy
