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

/// \file FieldRequirements.cpp
/// \brief Definition of the FieldRequirements class

// C++ core libs
#include <algorithm>    // for std::max
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible

// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/Field.hpp>       // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>    // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part
#include <stk_topology/topology.hpp>     // for stk::topology

// Mundy libs
#include <mundy_meta/FieldRequirementsBase.hpp>  // for mundy::meta::FieldRequirementsBase

namespace mundy {

namespace meta {

// \name Helper functions
//{

stk::topology::rank_t map_string_to_rank(const std::string &rank_string) {
  if (rank_string == "NODE_RANK") {
    return stk::topology::NODE_RANK;
  } else if (rank_string == "EDGE_RANK") {
    return stk::topology::EDGE_RANK;
  } else if (rank_string == "FACE_RANK") {
    return stk::topology::FACE_RANK;
  } else if (rank_string == "ELEMENT_RANK") {
    return stk::topology::ELEMENT_RANK;
  } else if (rank_string == "CONSTRAINT_RANK") {
    return stk::topology::CONSTRAINT_RANK;
  } else if (rank_string == "INVALID_RANK") {
    return stk::topology::INVALID_RANK;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                               "The provided rank string " << rank_string << " is not valid.");
  }
}
//}

}  // namespace meta

}  // namespace mundy
