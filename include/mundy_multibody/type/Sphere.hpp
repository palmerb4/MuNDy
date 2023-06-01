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

#ifndef MUNDY_MULTIBODY_TYPE_SPHERE_HPP_
#define MUNDY_MULTIBODY_TYPE_SPHERE_HPP_

/// \file Sphere.hpp
/// \brief Declaration of the Sphere class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

namespace mundy {

namespace multibody {

namespace type {

/// \class Sphere
/// \brief The static interface for all of Mundy's multibody Sphere objects.
class Sphere {
  //! \name Getters
  //@{

  /// \brief Get the Sphere's name.
  /// This name must be unique and not shared by any other Sphere object.
  static constexpr inline std::string_view details_get_name() {
    return "SPHERE";
  }

  /// \brief Get the Sphere's topology.
  static constexpr inline stk::topology details_get_topology() {
    return stk::topology::PARTICLE;
  }

  /// \brief Get the Sphere's rank.
  static constexpr inline stk::topology details_get_rank() {
    return stk::topology::ELEMENT_RANK;
  }

  /// \brief Get if the Sphere has a parent multibody type.
  static constexpr inline bool details_has_parent() {
    return false;
  }

  /// \brief Get the parent multibody type of the Sphere.
  static constexpr inline bool details_get_parent_name() {
    return "INVALID";
  }
};  // Sphere

}  // namespace type

}  // namespace multibody

}  // namespace mundy

#endif  // MUNDY_MULTIBODY_TYPE_SPHERE_HPP_
