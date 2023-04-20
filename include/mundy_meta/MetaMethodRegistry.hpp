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

#ifndef MUNDY_META_METAMETHODREGISTRY_HPP_
#define MUNDY_META_METAMETHODREGISTRY_HPP_

/// \file MetaMethodRegistry.hpp
/// \brief Declaration of the MetaMethodRegistry class

// C++ core libs
#include <type_traits>  // for std::enable_if, std::is_base_of

// Mundy libs
#include <mundy_meta/MetaMethod.hpp>         // for mundy::meta::MetaMethod
#include <mundy_meta/MetaMethodFactory.hpp>  // for mundy::meta::MetaMethodFactory

namespace mundy {

namespace meta {

/// \class MetaMethodRegistry
/// \brief A class for registering \c MetaMethods within \c MetaMethodFactory.
///
/// All classes derived from \c MetaMethod, which wish to be registered within the \c MetaMethodFactory should inherit
/// from this class where the template parameter is the derived type itself (follows the Curiously Recurring Template
/// Pattern).
///
/// \tparam DerivedMetaMethod A class derived from \c MetaMethod.
/// \tparam RegistryIdentifier A template type used to create different independent instances of MetaMethodFactory.
template <class DerivedMetaMethod, typename RegistryIdentifier = DefaultMethodIdentifier,
          typename std::enable_if<std::is_base_of<MetaMethodBase, DerivedMetaMethod>::value, void>::type>
struct MetaMethodRegistry {
  //! \name Actions
  //@{

  /// @brief Register DerivedMetaMethod with the MetaMethodFactory.
  ///
  /// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
  /// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
  static inline bool register_type() {
    MetaMethodFactory::register_new_method<DerivedMetaMethod>();
    return true;
  }
  //@}

  //! \name Member variables
  //@{

  /// @brief A flag for if the given type has been registered with the \c MetaMethodFactory or not.
  static const bool is_registered;
  //@}
};  // MetaMethodRegistry

/// @brief Perform the static registration.
///
/// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
/// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
///
/// \tparam DerivedMetaMethod A class derived from \c MetaMethod.
template <class DerivedMetaMethod>
const bool MetaMethodRegistry<DerivedMetaMethod>::is_registered =
    MetaMethodRegistry<DerivedMetaMethod>::register_type();

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAMETHODREGISTRY_HPP_
