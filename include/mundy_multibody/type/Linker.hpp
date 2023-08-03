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

#ifndef MUNDY_MULTIBODY_TYPE_LINKER_HPP_
#define MUNDY_MULTIBODY_TYPE_LINKER_HPP_

/// \file Linker.hpp
/// \brief Declaration of the Linker class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy includes
#include "mundy_multibody/MultibodyRegistry.hpp"  // for MUNDY_REGISTER_MULTIBODYTYPE

namespace mundy {

namespace multibody {

namespace type {

/// \class Linker
/// \brief The static interface for all of Mundy's multibody Linker objects.
///
/// The design of this class is in accordance with the static interface requirements of
/// mundy::multibody::MultibodyFactory.
class Linker {
 public:
  //! \name Getters
  //@{

  /// \brief Get the Linker's name.
  /// This name must be unique and not shared by any other multibody object.
  static constexpr inline std::string_view get_name() {
    return "LINKER";
  }

  /// \brief Get the Linker's topology.
  static constexpr inline stk::topology::topology_t get_topology() {
    return stk::topology::INVALID_TOPOLOGY;
  }

  /// \brief Get the Linker's rank.
  static constexpr inline stk::topology::rank_t get_rank() {
    return stk::topology::CONSTRAINT_RANK;
  }

  /// \brief Get if the Linker has a parent multibody type.
  static constexpr inline bool has_parent() {
    return false;
  }

  /// \brief Get the parent multibody type of the Linker.
  static constexpr inline std::string_view get_parent_name() {
    return "INVALID";
  }
};  // Linker

}  // namespace type

}  // namespace multibody

}  // namespace mundy

MUNDY_REGISTER_MULTIBODYTYPE(mundy::multibody::type::Linker)

#endif  // MUNDY_MULTIBODY_TYPE_LINKER_HPP_
