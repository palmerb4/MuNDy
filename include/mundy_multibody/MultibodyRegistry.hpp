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

#ifndef MUNDY_MULTIBODY_MULTIBODYREGISTRY_HPP_
#define MUNDY_MULTIBODY_MULTIBODYREGISTRY_HPP_

/// \file MultibodyRegistry.hpp
/// \brief Declaration of the MultibodyRegistry class

// C++ core libs
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <utility>      // for std::pair

// Mundy libs
#include <mundy_multibody/Multibody.hpp>         // for mundy::meta::Multibody
#include <mundy_multibody/MultibodyFactory.hpp>  // for mundy::meta::MultibodyFactory

namespace mundy {

/// \class MultibodyRegistry
/// \brief A class for registering new \c Multibody types within \c MultibodyFactory.
///
/// All classes derived from \c Multibody, which wish to be registered within the \c MultibodyFactory should inherit
/// from this class where the template parameter is the derived type itself (follows the Curiously Recurring Template
/// Pattern).
///
/// \tparam ClassToRegister A class derived from \c Multibody.
template <class ClassToRegister>
struct MultibodyRegistry {
  //! \name Actions
  //@{

  /// @brief Register \c ClassToRegister with the \c MultibodyFactory.
  ///
  /// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
  /// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
  static inline bool register_type() {
    MultibodyFactory::template register_new_method<ClassToRegister>();
    return true;
  }
  //@}

  //! \name Member variables
  //@{

  /// @brief A flag for if the given type has been registered with the \c MultibodyFactory or not.
  static const bool is_registered;
  //@}
};  // MultibodyRegistry

/// @brief Perform the static registration.
///
/// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
/// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
template <class ClassToRegister>
const bool MultibodyRegistry<ClassToRegister>::is_registered = MultibodyRegistry<ClassToRegister>::register_type();

}  // namespace mundy

#endif  // MUNDY_MULTIBODY_MULTIBODYREGISTRY_HPP_
