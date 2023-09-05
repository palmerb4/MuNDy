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

#ifndef UNIT_TESTS_MUNDY_MULTIBODY_UTILS_EXAMPLEMULTIBODYTYPE_HPP_
#define UNIT_TESTS_MUNDY_MULTIBODY_UTILS_EXAMPLEMULTIBODYTYPE_HPP_

/// \file ExampleMetaMethod.hpp
/// \brief Declaration of the ExampleMetaMethod class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::GlobalMetaMethodRegistry

namespace mundy {

namespace multibody {

namespace utils {

/// \class ExampleMultibodyType
/// \brief The static interface for all of Mundy's multibody ExampleMultibodyType objects.
///
/// The design of this class is in accordance with the static interface requirements of
/// mundy::multibody::MultibodyFactory.
template <int I>
class ExampleMultibodyType {
 public:
  //! \name Getters
  //@{

  /// \brief Get the ExampleMultibodyType's name.
  /// This name must be unique and not shared by any other multibody object.
  static constexpr inline std::string_view get_name() {
    if constexpr (I == 0) {
      return "FAKE_NAME_0";
    } else if constexpr (I == 1) {
      return "FAKE_NAME_1";
    } else if constexpr (I == 2) {
      return "FAKE_NAME_2";
    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid ExampleMultibodyType index.");
      return "FAKE_NAME_N";
    }
  }

  /// \brief Get the ExampleMultibodyType's topology.
  static constexpr inline stk::topology::topology_t get_topology() {
    return stk::topology::INVALID_TOPOLOGY;
  }

  /// \brief Get the ExampleMultibodyType's rank.
  static constexpr inline stk::topology::rank_t get_rank() {
    return stk::topology::INVALID_RANK;
  }

  /// \brief Get if the ExampleMultibodyType has a parent multibody type.
  static constexpr inline bool has_parent() {
    return false;
  }

  /// \brief Get the parent multibody type of the ExampleMultibodyType.
  static constexpr inline std::string_view get_parent_name() {
    return "INVALID";
  }
};  // ExampleMultibodyType

}  // namespace utils

}  // namespace multibody

}  // namespace mundy

#endif  // UNIT_TESTS_MUNDY_MULTIBODY_UTILS_EXAMPLEMULTIBODYTYPE_HPP_
