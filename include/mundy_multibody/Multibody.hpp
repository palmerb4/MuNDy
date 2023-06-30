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

#ifndef MUNDY_MULTIBODY_MULTIBODY_HPP_
#define MUNDY_MULTIBODY_MULTIBODY_HPP_

/// \file Multibody.hpp
/// \brief Declaration of the Multibody class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

namespace mundy {

namespace multibody {

using multibody_t = unsigned;

/// \class Multibody
/// \brief The static interface for all of Mundy's multibody objects, be they rigid bodies and/or constraints.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c Multibody must
/// implement the following static member functions
/// - \c details_get_name             implementation of the \c get_name             interface.
/// - \c details_get_topology         implementation of the \c get_topology         interface (both versions).
/// - \c details_has_parent_multibody implementation of the \c has_parent_multibody interface (both versions).
/// - \c details_get_parent_multibody implementation of the \c get_parent_multibody interface (both versions).
///
/// To keep these out of the public interface, we suggest that each details function be made private and
/// \c Multibody<DerivedClass> be made a friend of \c DerivedClass.
///
/// \tparam DerivedClass A class derived from \c Multibody that implements the desired
/// interface.
template <class DerivedClass>
class Multibody {
  //! \name Getters
  //@{

  /// \brief Get the name associated with this multibody object.
  /// This name must be unique and not shared by any other multibody object.
  static constexpr inline std::string_view get_name() {
    return DerivedClass::details_get_name();
  }

  /// \brief Get the topology of the multibody object.
  static constexpr inline stk::topology get_topology() {
    return DerivedClass::details_get_topology();
  }

  /// \brief Get the rank of the multibody object.
  static constexpr inline stk::topology get_rank() {
    return DerivedClass::details_get_rank();
  }

  /// \brief Get if this multibody object is a subset of another.
  static constexpr inline bool has_parent() {
    return DerivedClass::details_has_parent_multibody();
  }

  /// \brief Get the parent to this multibody object.
  static constexpr inline bool get_parent_name() {
    return DerivedClass::details_get_parent_multibody();
  }
};  // Multibody

}  // namespace multibody

}  // namespace mundy

#endif  // MUNDY_MULTIBODY_MULTIBODY_HPP_
