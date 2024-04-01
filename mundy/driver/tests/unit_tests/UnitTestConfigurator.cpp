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

The Configurator is responsible for pushing the information from the configuration file into the driver, including the
configuration of the MetaMethods and the ordering of operations.

*/

//! \name Configurator unit tests
//@{
TEST(Configurator, ParseParametersFromYAMLFileBasic) {
  // This should be in the local test directory if things were set up correctly
  const std::string yaml_file = "./unit_test_configurator_basic.yaml";

  // Construct a configurator from the YAML file
  Configurator configurator("yaml", yaml_file);

  // Run the parse command
  configurator.parse_parameters();
}

//@}

//! \name Configurator exposure tests
//@{
TEST(Configurator, ExposeRegisteredClasses) {
  // Create a dummy configurator, this will still trigger the registration procedure
  Configurator configurator;

  // Ask it to expose the registered MetaMethodExecutionInterface
  std::string registered_mmei = configurator.get_registered_meta_method_execution_interface();
  // Ask it to expose the registered MetaMethodSubsetExecutionInterface
  std::string registered_mmsei = configurator.get_registered_meta_method_subset_execution_interface();
  // Ask it to expose the registered MetaMethodPairwiseSubsetExecutionInterface
  std::string registered_mmpsei = configurator.get_registered_meta_method_pairwise_subset_execution_interface();
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
