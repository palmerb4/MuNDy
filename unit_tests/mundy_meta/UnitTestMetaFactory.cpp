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
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Field.hpp>         // for stk::mesh::Field
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>            // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>      // for mundy::meta::FieldRequirements
#include <mundy_meta/FieldRequirementsBase.hpp>  // for mundy::meta::FieldRequirementsBase
#include <mundy_meta/MetaFactory.hpp>            // for mundy::meta::MetaFactory

// Mundy test libs
#include <mundy_meta/utils/ExampleMetaMethod.hpp>  // for mundy::meta::utils::ExampleMetaMethod

namespace mundy {

namespace meta {

namespace {

//! \name MetaFactory object registration tests
//@{

TEST(MetaFactoryRegistration, RegistrationWorksProperly) {
  // Registration of a class derived from \c HasMeshRequirementsAndIsRegisterable with \c MetaFactory should store the
  // class's identifier, instance generator, requirements generator, fixed parameters validator, and mutable parameters
  // validator.

  // Create our example meta factory.
  // To avoid contaminating other tests and mundy itself, we'll use a unique identifier for all tests and reset the
  // factory before and after each test.
  using ReturnType = void;
  using RegistrationType = int;
  using RegistrationIdentifier = int;
  using ExampleMetaFactory = MetaMethodFactory<ReturnType, RegistrationIdentifier, RegistrationType>;
  ExampleMetaFactory::reset();

  // Create out example class to register.
  // This class must be derived from \c HasMeshRequirementsAndIsRegisterable.
  // TODO(palmerb4): Is there a succinct way to check this at compile time?
  constexpr int class_identifier = 1;
  using ClassToRegister = utils::ExampleMetaMethod<class_identifier>;
  ASSERT_TRUE(ClassToRegister::static_get_class_identifier() == class_identifier);

  // Reset the test counter within our classes to register.
  // We'll use these counters to ensure that MetaFactory is properly calling our internal methods.
  ClassToRegister::reset_counters();

  // Attempt to register a class derived from \c HasMeshRequirementsAndIsRegisterable with \c MetaFactory.
  EXPECT_EQ(ExampleMetaFactory::num_registered_classes(), 0);
  EXPECT_FALSE(ExampleMetaFactory::is_valid_key(class_identifier));
  ExampleMetaFactory::register_new_class<ClassToRegister>();
  EXPECT_EQ(ExampleMetaFactory::num_registered_classes(), 1);
  EXPECT_TRUE(ExampleMetaFactory::is_valid_key(class_identifier));

  // Try to use the factory to access the internal member functions of our registered class.
  // We'll use these counters to ensure that MetaFactory is properly calling our internal methods.
  mundy::mesh::BulkData* bulk_data_ptr = nullptr;
  Teuchos::ParameterList fixed_params;
  Teuchos::ParameterList mutable_params;

  EXPECT_EQ(ClassToRegister::num_get_mesh_requirements_calls(), 0);
  ExampleMetaFactory::get_mesh_requirements(class_identifier, fixed_params);
  EXPECT_EQ(ClassToRegister::num_get_mesh_requirements_calls(), 1);

  EXPECT_EQ(ClassToRegister::num_validate_fixed_parameters_and_set_defaults_calls(), 0);
  ExampleMetaFactory::validate_fixed_parameters_and_set_defaults(class_identifier, &fixed_params);
  EXPECT_EQ(ClassToRegister::num_validate_fixed_parameters_and_set_defaults_calls(), 1);

  EXPECT_EQ(ClassToRegister::num_validate_mutable_parameters_and_set_defaults_calls(), 0);
  ExampleMetaFactory::validate_mutable_parameters_and_set_defaults(class_identifier, &mutable_params);
  EXPECT_EQ(ClassToRegister::num_validate_mutable_parameters_and_set_defaults_calls(), 1);

  EXPECT_EQ(ClassToRegister::num_create_new_instance_calls(), 0);
  ExampleMetaFactory::create_new_instance(class_identifier, bulk_data_ptr, fixed_params);
  EXPECT_EQ(ClassToRegister::num_create_new_instance_calls(), 1);

  ExampleMetaFactory::reset();
}

TEST(MetaFactoryRegistration, Reregistration) {
  // Ensure that attempting to register a class with a key that already exists throws an exception.

  // Create our example meta factory.
  // To avoid contaminating other tests and mundy itself, we'll use a unique identifier for all tests and reset the
  // factory before and after each test.
  using ReturnType = void;
  using RegistrationType = int;
  using RegistrationIdentifier = int;
  using ExampleMetaFactory = MetaMethodFactory<ReturnType, RegistrationIdentifier, RegistrationType>;
  ExampleMetaFactory::reset();

  // Create out example classes to register.
  // These classes must be derived from \c HasMeshRequirementsAndIsRegisterable.
  constexpr int class_identifier = 1;
  using ClassToRegister1 = utils::ExampleMetaMethod<class_identifier, 1>;
  using ClassToRegister2 = utils::ExampleMetaMethod<class_identifier, 2>;
  bool classes_are_different = !std::is_same_v<ClassToRegister1, ClassToRegister2>;
  ASSERT_TRUE(classes_are_different);
  ASSERT_TRUE(ClassToRegister1::static_get_class_identifier() == class_identifier);
  ASSERT_TRUE(ClassToRegister2::static_get_class_identifier() == class_identifier);

  // Reset the test counter within our classes to register.
  // We'll use these counters to ensure that MetaFactory is properly calling our internal methods.
  ClassToRegister1::reset_counters();
  ClassToRegister2::reset_counters();

  // Attempt to register a class derived from \c HasMeshRequirementsAndIsRegisterable with \c MetaFactory.
  EXPECT_EQ(ExampleMetaFactory::num_registered_classes(), 0);
  ExampleMetaFactory::register_new_class<ClassToRegister1>();
  ASSERT_EQ(ExampleMetaFactory::num_registered_classes(), 1);

  // Attempting to register a class with a key that already exists should throw an exception
  EXPECT_THROW(ExampleMetaFactory::register_new_class<ClassToRegister1>(), std::logic_error);

  ExampleMetaFactory::reset();
}

TEST(MetaFactoryRegistration, RegistrationWithDifferentRegistrationIdentifier) {
  /* Ensure that MetaFactories with different registration identifiers do not share registered classes.
  The setup for this test is as follows:
    Register ClassToRegister with MetaFactory1 with registration identifier RegistrationIdentifier1.
    Register ClassToRegister with MetaFactory2 with registration identifier RegistrationIdentifier2.
  */

  // Create our example meta factories.
  // To avoid contaminating other tests and mundy itself, we'll use a unique identifier for all tests and reset the
  // factories before and after each test.
  using ReturnType = void;
  using RegistrationType = int;
  using RegistrationIdentifier1 = int;
  using RegistrationIdentifier2 = double;
  using ExampleMetaFactory1 = MetaMethodFactory<ReturnType, RegistrationIdentifier1, RegistrationType>;
  using ExampleMetaFactory2 = MetaMethodFactory<ReturnType, RegistrationIdentifier2, RegistrationType>;
  ExampleMetaFactory1::reset();
  ExampleMetaFactory2::reset();

  // Create out example class to register.
  constexpr int class_identifier = 1;
  using ClassToRegister = utils::ExampleMetaMethod<class_identifier>;
  ASSERT_TRUE(ClassToRegister::static_get_class_identifier() == class_identifier);

  // Register with the first factory.
  EXPECT_EQ(ExampleMetaFactory1::num_registered_classes(), 0);
  EXPECT_EQ(ExampleMetaFactory2::num_registered_classes(), 0);
  EXPECT_FALSE(ExampleMetaFactory1::is_valid_key(class_identifier));
  ExampleMetaFactory1::register_new_class<ClassToRegister>();
  EXPECT_EQ(ExampleMetaFactory1::num_registered_classes(), 1);
  EXPECT_EQ(ExampleMetaFactory2::num_registered_classes(), 0);

  // Register with the second factory.
  EXPECT_FALSE(ExampleMetaFactory2::is_valid_key(class_identifier));
  ExampleMetaFactory2::register_new_class<ClassToRegister>();
  EXPECT_EQ(ExampleMetaFactory1::num_registered_classes(), 1);
  EXPECT_EQ(ExampleMetaFactory2::num_registered_classes(), 1);

  ExampleMetaFactory1::reset();
  ExampleMetaFactory2::reset();
}
//@}

}  // namespace

}  // namespace meta

}  // namespace mundy
