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

// External libs
#include <gmock/gmock.h>  // for EXPECT_THAT
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <iostream>
#include <string>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <Teuchos_YamlParameterListHelpers.hpp>
#include <Teuchos_YamlParser_decl.hpp>

// Mundy libs
#include <mundy_driver/Configurator.hpp>

namespace mundy {

namespace driver {

namespace {

/* What tests should we run?

IoBroker is the main IO class for mundy, and we want to make sure it works in both serial and parallel environments,
as well as for the accuracy of what it is reading and writing. Based loosely off of the BrownianIORestart.cpp
example, as well as the MundyLinker examples.

This class is a priviledged class, as it needs to be able to modify the mesh, and in the RESTART capacity, can
actually commit the meta data.
*/

//! \name Configurator unit tests
//@{
TEST(Configurator, ConstructConfiguratorFromParameterList) {
  // ComputeAABB fixed parameters
  Teuchos::ParameterList compute_aabb_fixed_params;
  compute_aabb_fixed_params
      .set<Teuchos::Array<std::string>>("enabled_kernel_names", Teuchos::tuple<std::string>(std::string("SPHERE")))
      .set("element_aabb_field_name", "ELEMENT_AABB");
  compute_aabb_fixed_params.sublist("SPHERE").set<Teuchos::Array<std::string>>(
      "valid_entity_part_names", Teuchos::tuple<std::string>(std::string("SPHERES")));

  // GenerateNeighborLinkers fixed parameters
  Teuchos::ParameterList generate_neighbor_linkers_fixed_params;
  generate_neighbor_linkers_fixed_params.set("enabled_technique_name", "STK_SEARCH")
      .set<Teuchos::Array<std::string>>("specialized_neighbor_linkers_part_names",
                                        Teuchos::tuple<std::string>("SPHERE_SPHERE_LINKERS"));
  generate_neighbor_linkers_fixed_params.sublist("STK_SEARCH")
      .set<Teuchos::Array<std::string>>("valid_source_entity_part_names",
                                        Teuchos::tuple<std::string>(std::string("SPHERES")))
      .set<Teuchos::Array<std::string>>("valid_target_entity_part_names",
                                        Teuchos::tuple<std::string>(std::string("SPHERES")))
      .set("element_aabb_field_name", std::string("ELEMENT_AABB"));

  // ComputeAABB mutable params
  Teuchos::ParameterList compute_aabb_mutable_params;
  compute_aabb_mutable_params.set("buffer_distance", 0.0);

  // This test isn't MPI threaded so just write out the parameter lists
  std::cout << "ComputeAABB fixed params:\n" << compute_aabb_fixed_params << std::endl;
  std::cout << "ComputeAABB mutable params:\n" << compute_aabb_mutable_params << std::endl;
  std::cout << "GenerateNeighborLinkers fixed params:\n" << generate_neighbor_linkers_fixed_params << std::endl;

  // Validate the fixed params and check what is going on in the yaml files
  compute_aabb_fixed_params.validateParametersAndSetDefaults(mundy::shapes::ComputeAABB::get_valid_fixed_params());
  generate_neighbor_linkers_fixed_params.validateParametersAndSetDefaults(
      mundy::linkers::GenerateNeighborLinkers::get_valid_fixed_params());

  // Validate the mutable params
  compute_aabb_mutable_params.validateParametersAndSetDefaults(mundy::shapes::ComputeAABB::get_valid_mutable_params());

  // Dump these to yaml to see what they look like to inform out behavior
  {
    std::ostringstream computeaabb_yaml_stream;
    Teuchos::YAMLParameterList::writeYamlStream(computeaabb_yaml_stream, compute_aabb_fixed_params);
    std::cout << computeaabb_yaml_stream.str() << std::endl;

    std::ostringstream computeaabb_mutable_yaml_stream;
    Teuchos::YAMLParameterList::writeYamlStream(computeaabb_mutable_yaml_stream, compute_aabb_mutable_params);
    std::cout << computeaabb_mutable_yaml_stream.str() << std::endl;

    std::ostringstream generateneighborlinkers_yaml_stream;
    Teuchos::YAMLParameterList::writeYamlStream(generateneighborlinkers_yaml_stream,
                                                generate_neighbor_linkers_fixed_params);
    std::cout << generateneighborlinkers_yaml_stream.str() << std::endl;
  }

  // Try to combine the fixed parameters into a single thing...
  Teuchos::ParameterList compute_aabb_method("sphereComputeAABB");
  compute_aabb_method.set("method", "COMPUTE_AABB");
  compute_aabb_method.set("fixed_params", compute_aabb_fixed_params);
  compute_aabb_method.set("mutable_params", compute_aabb_mutable_params);

  // Add the neighbor linkers so we have something else to look at...
  Teuchos::ParameterList generate_neighbor_linkers_method("sphereNeighborLinkers");
  generate_neighbor_linkers_method.set("method", "GenerateNeighborLinkers");
  generate_neighbor_linkers_method.set("fixed_params", generate_neighbor_linkers_fixed_params);

  // Add another layer to have multiple things...
  Teuchos::ParameterList method_params("Simulation");
  Teuchos::ParameterList method_params_sublist = method_params.sublist("Actions");

  {
    std::ostringstream computeaabb_yaml_stream;
    Teuchos::YAMLParameterList::writeYamlStream(computeaabb_yaml_stream, compute_aabb_method);
    std::cout << computeaabb_yaml_stream.str() << std::endl;

    std::ostringstream generateneighborlinkers_yaml_stream;
    Teuchos::YAMLParameterList::writeYamlStream(generateneighborlinkers_yaml_stream, generate_neighbor_linkers_method);
    std::cout << generateneighborlinkers_yaml_stream.str() << std::endl;
  }
}

TEST(Configurator, ConstructConfiguratorFromYAMLFile) {
  // This should be in the local test directory if things were set up correctly
  const std::string yaml_file = "./test_configurator_basic.yaml";

  // Construct a configurator from the YAML file
  Configurator configurator("yaml", yaml_file);

  // Get a teuchos parameter list from this and print it
  Teuchos::ParameterList param_list = *Teuchos::getParametersFromYamlFile(yaml_file);

  std::cout << param_list << std::endl;

  // Get the keys that we can talk to as the configurator
  std::string registered_metamethodsubsetexecutioninterface = mundy::driver::ConfigurableMetaMethodFactory<
      mundy::meta::MetaMethodSubsetExecutionInterface<void>>::get_keys_as_string();

  std::cout << registered_metamethodsubsetexecutioninterface << std::endl;
}

TEST(Configurator, ParseParametersFromYAMLFileBasic) {
  // This should be in the local test directory if things were set up correctly
  const std::string yaml_file = "./test_configurator_basic.yaml";

  // Construct a configurator from the YAML file
  Configurator configurator("yaml", yaml_file);

  // Run the parse command
  configurator.ParseParameters();
}

//@}

//! \name Configurator exposure tests
//@{
TEST(Configurator, ExposeRegisteredClasses) {
  // Create a dummy configurator, this will still trigger the registration procedure
  Configurator configurator;

  // Ask it to expose the registered MetaMethodExecutionInterface
  std::string registered_mmei = configurator.get_registered_MetaMethodExecutionInterface();
  // Ask it to expose the registered MetaMethodSubsetExecutionInterface
  std::string registered_mmsei = configurator.get_registered_MetaMethodSubsetExecutionInterface();
  // Ask it to expose the registered MetaMethodPairwiseSubsetExecutionInterface
  std::string registered_mmpsei = configurator.get_registered_MetaMethodPairwiseSubsetExecutionInterface();
  // Ask for all the registered classes
  std::string registered_classes = configurator.get_registered_classes();

  EXPECT_THAT(registered_mmei, ::testing::ContainsRegex("DECLARE_AND_INIT_SHAPES"));
  EXPECT_THAT(registered_mmsei, ::testing::ContainsRegex("COMPUTE_OBB"));
  EXPECT_THAT(registered_mmpsei, ::testing::ContainsRegex("GENERATE_NEIGHBOR_LINKERS"));
  EXPECT_THAT(registered_classes, ::testing::ContainsRegex("COMPUTE_AABB"));
}

//@}

}  // namespace

}  // namespace driver

}  // namespace mundy
