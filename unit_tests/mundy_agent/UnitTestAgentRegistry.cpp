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

// Mundy libs
#include <mundy_agent/AgentHierarchy.hpp>   // for mundy::agent::AgentHierarchy
#include <mundy_agent/AgentRegistry.hpp>  // for mundy::agent::AgentRegistry

// Mundy test libs
#include <mundy_agent/utils/ExampleAgent.hpp>  // for mundy::agent::ExampleAgent

namespace mundy {

namespace agent {

namespace {

// AgentRegistry has a fascinating, albeit odd design. It is designed to automatically register classes derived from
// \c HasMeshRequirementsAndIsRegisterable with some \c AgentHierarchy. To do so, one simply inherits from
// AgentRegistry with the appropriate template parameters. This class is rather abstract but allows us to use
// partial specialization to generate many different registries without duplicating code. Here, we directly test
// AgentRegistry's functionality and some of its partial specializations.

//! \name AgentRegistry automatic registration tests
//@{

struct DummyRegistrationIdentifier {};  // Dummy registration identifier;

TEST(AgentRegistry, AutoRegistration) {
  // Test that the MUNDY_REGISTER_METACLASS macro performed the registration with the AgentHierarchy
  EXPECT_TRUE(AgentHierarchy::is_valid(mundy::agent::utils::ExampleAgent<1>::get_name()));
  EXPECT_TRUE(AgentHierarchy::is_valid(mundy::agent::utils::ExampleAgent<2>::get_name()));
}
//@}

}  // namespace

}  // namespace agent

}  // namespace mundy

// Registration shouldn't need to explicitly come before TEST, since it will be registered at compile time.

// Register a class with AgentHierarchy
MUNDY_REGISTER_AGENT(mundy::agent::utils::ExampleAgent<1>)

// Register a different class with AgentHierarchy
MUNDY_REGISTER_AGENT(mundy::agent::utils::ExampleAgent<2>)
