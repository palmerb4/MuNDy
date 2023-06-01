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

#ifndef MUNDY_MULTIBODY_REGISTRY_HPP_
#define MUNDY_MULTIBODY_REGISTRY_HPP_

/// \file Registry.hpp
/// \brief Declaration of the Registry class

// C++ core libs
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <utility>      // for std::pair

// Mundy libs
#include <mundy_multibody/Factory.hpp>  // for mundy::meta::Factory
#include <mundy_multibody/MultibodyType.hpp>     // for mundy::meta::MultibodyType

namespace mundy {

namespace multibody {

/// \class Registry
/// \brief A class for registering new \c MultibodyType types within \c Factory.
///
/// All classes derived from \c MultibodyType, which wish to be registered within the \c Factory should inherit
/// from this class where the template parameter is the derived type itself (follows the Curiously Recurring Template
/// Pattern).
///
/// \tparam ClassToRegister A class derived from \c MultibodyType.
template <class ClassToRegister>
struct Registry {
  //! \name Actions
  //@{

  /// @brief Register \c ClassToRegister with the \c Factory.
  ///
  /// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
  /// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
  static inline bool register_type() {
    Factory::template register_new_method<ClassToRegister>();
    return true;
  }
  //@}

  //! \name Member variables
  //@{

  /// @brief A flag for if the given type has been registered with the \c Factory or not.
  static const bool is_registered;
  //@}
};  // Registry

/// @brief Perform the static registration.
///
/// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
/// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
template <class ClassToRegister>
const bool Registry<ClassToRegister>::is_registered = Registry<ClassToRegister>::register_type();

}  // namespace multibody

}  // namespace mundy

#endif  // MUNDY_MULTIBODY_REGISTRY_HPP_
