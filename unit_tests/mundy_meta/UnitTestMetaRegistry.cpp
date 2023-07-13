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
#include <mundy_meta/MetaRegistry.hpp>           // for mundy::meta::MetaRegistry

// Mundy test libs
#include <mundy_meta/utils/ExampleMetaMethod.hpp>  // for mundy::meta::ExampleMetaMethod

namespace mundy {

namespace meta {

namespace {

// MetaRegistry has a fascinating, albeit odd design. It is designed to automatically register classes derived from
// \c HasMeshRequirementsAndIsRegisterable with some \c MetaFactory. To do so, one simply inherits from MetaRegistry
// with the appropriate template parameters. This class is rather abstract but allows us to use partial specialization
// to generate many different registries without duplicating code. Here, we directly test MetaRegistry's functionality
// and some of its partial specializations.

//! \name MetaRegistry automatic registration tests
//@{

using RegistrationIdentifier1 = int;
using RegistrationIdentifier2 = double;

/// @brief Register with GlobalMetaMethodFactory
class ExampleMetaMethodRegisteredWithGlobal
    : public ExampleMetaMethod<0>,
      public GlobalMetaMethodRegistry<void, ExampleMetaMethodRegisteredWithGlobal, int> {
};  // class ExampleMetaMethodRegisteredWithGlobal

/// @brief Register with a MetaMethodFactory with a given RegistrationIdentifier
class ExampleMetaMethodRegisteredWithFactory1
    : public ExampleMetaMethod<1>,
      public MetaMethodRegistry<void, ExampleMetaMethodRegisteredWithFactory1, RegistrationIdentifier1, int> {
};  // class ExampleMetaMethodRegisteredWithSomeFactory

/// @brief Register with a MetaMethodFactory with a given RegistrationIdentifier
class ExampleMetaMethodRegisteredWithFactory2
    : public ExampleMetaMethod<2>,
      public MetaMethodRegistry<void, ExampleMetaMethodRegisteredWithFactory2, RegistrationIdentifier2, int> {
};  // class ExampleMetaMethodRegisteredWithFactory2

/// @brief ReRegister with a MetaMethodFactory with a given RegistrationIdentifier, overwriting any previous
/// registrations
class ExampleMetaMethodReRegisteredWithFactory2
    : public ExampleMetaMethod<2>,
      public MetaMethodRegistry<void, ExampleMetaMethodReRegisteredWithFactory2, RegistrationIdentifier2, int, true> {
};  // class ExampleMetaMethodReRegisteredWithFactory2

TEST(MetaRegistry, AutoRegistration) {
  // Test that the MetaRegistry performed the registration with the GlobalMetaMethodFactory
  using OurGlobalMetaMethodFactory = GlobalMetaMethodFactory<void, int>;
  EXPECT_EQ(OurGlobalMetaMethodFactory::num_registered_classes(), 1);
  EXPECT_TRUE(
      OurGlobalMetaMethodFactory::is_valid_key(ExampleMetaMethodRegisteredWithGlobal::static_get_class_identifier()));

  // Test that the MetaRegistry performed the registration with the MetaMethodFactory with a given
  // RegistrationIdentifier
  using OurMetaMethodFactory1 = MetaMethodFactory<void, RegistrationIdentifier1, int>;
  EXPECT_EQ(OurMetaMethodFactory1::num_registered_classes(), 1);
  EXPECT_TRUE(
      OurMetaMethodFactory1::is_valid_key(ExampleMetaMethodRegisteredWithFactory1::static_get_class_identifier()));

  // Test that the MetaRegistry performed the reregistration with the MetaMethodFactory with a given
  // RegistrationIdentifier.
  using OurMetaMethodFactory2 = MetaMethodFactory<void, RegistrationIdentifier2, int>;
  auto key = ExampleMetaMethodRegisteredWithFactory2::static_get_class_identifier();
  EXPECT_EQ(OurMetaMethodFactory2::num_registered_classes(), 1);
  ASSERT_TRUE(OurMetaMethodFactory2::is_valid_key(key));

  // To test that the old registration was overwritten, we need to check that the correct internal methods are called.
  ExampleMetaMethodRegisteredWithFactory2::reset_counters();
  ExampleMetaMethodReRegisteredWithFactory2::reset_counters();
  ASSERT_EQ(ExampleMetaMethodRegisteredWithFactory2::num_get_mesh_requirements_calls(), 0);
  ASSERT_EQ(ExampleMetaMethodReRegisteredWithFactory2::num_get_mesh_requirements_calls(), 0);
  Teuchos::ParameterList fixed_params;
  OurMetaMethodFactory2::get_mesh_requirements(key, fixed_params);
  EXPECT_EQ(ExampleMetaMethodRegisteredWithFactory2::num_get_mesh_requirements_calls(), 0);
  EXPECT_EQ(ExampleMetaMethodReRegisteredWithFactory2::num_get_mesh_requirements_calls(), 1);
}
//@}

}  // namespace

}  // namespace meta

}  // namespace mundy
