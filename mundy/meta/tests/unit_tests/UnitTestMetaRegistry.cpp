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
#include "utils/ExampleMetaMethod.hpp"  // for mundy::meta::utils::ExampleMetaMethod

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

struct DummyRegistrationIdentifier {};  // Dummy registration identifier;

TEST(MetaRegistry, AutoRegistration) {
  // Test that the MUNDY_REGISTER_METACLASS macro performed the registration with the MetaMethodFactory with a given
  // RegistrationIdentifier
  using OurMetaMethodFactory = MetaMethodFactory<void, DummyRegistrationIdentifier, int>;
  EXPECT_EQ(OurMetaMethodFactory::num_registered_classes(), 2);
  EXPECT_TRUE(OurMetaMethodFactory::is_valid_key(mundy::meta::utils::ExampleMetaMethod<1>::get_registration_id()));
  EXPECT_TRUE(OurMetaMethodFactory::is_valid_key(mundy::meta::utils::ExampleMetaMethod<2>::get_registration_id()));
}
//@}

}  // namespace

}  // namespace meta

}  // namespace mundy

// Registration shouldn't need to explicitly come before TEST, since it will be registered at compile time.

// Register a class with the MetaMethodFactory with a given RegistrationIdentifier.
MUNDY_REGISTER_METACLASS(mundy::meta::utils::ExampleMetaMethod<1>,
                         mundy::meta::MetaMethodFactory<void, DummyRegistrationIdentifier, int>);

// Register a different class with the same MetaMethodFactory with a given RegistrationIdentifier.
MUNDY_REGISTER_METACLASS(mundy::meta::utils::ExampleMetaMethod<2>,
                         mundy::meta::MetaMethodFactory<void, DummyRegistrationIdentifier, int>);
