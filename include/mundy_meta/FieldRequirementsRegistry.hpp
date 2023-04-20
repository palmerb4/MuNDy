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

#ifndef MUNDY_META_FIELDREQUIREMENTSREGISTRY_HPP_
#define MUNDY_META_FIELDREQUIREMENTSREGISTRY_HPP_

/// \file FieldRequirementsRegistry.hpp
/// \brief Declaration of the FieldRequirementsRegistry class

// C++ core libs
#include <type_traits>  // for std::enable_if, std::is_base_of

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>         // for mundy::meta::FieldRequirements
#include <mundy_meta/FieldRequirementsFactory.hpp>  // for mundy::meta::FieldRequirementsFactory

namespace mundy {

namespace meta {

/// \class FieldRequirementsRegistry
/// \brief A class for registering \c FieldRequirementss within \c FieldRequirementsFactory.
///
/// All valid field types that can be passed to \c FieldRequirements should be registered within the
/// \c FieldRequirementsFactory. This registry aids in the registration process. The actual registration is carried out
/// in \c FieldRequirements.cpp to guarentee that the registration is carried out by any code that uses
/// \c FieldRequirements.
///
/// \tparam FieldTypeToRegister A trivially copyable type to be registered with the \c FieldRequirementsFactory.
template <class FieldTypeToRegister,
          typename std::enable_if<std::is_trivially_copyable<FieldTypeToRegister>::value, void>::type = 0>
struct FieldRequirementsRegistry {
  //! \name Actions
  //@{

  /// @brief Register \c FieldTypeToRegister with the \c FieldRequirementsFactory.
  ///
  /// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
  /// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
  ///
  /// \param field_type_string [in] The field type string to associate with \c FieldTypeToRegister.
  static inline bool register_type(const std::string& field_type_string) {
    FieldRequirementsFactory::register_new_field_type<FieldTypeToRegister>(field_type_string);
    return true;
  }
  //@}

  //! \name Member variables
  //@{

  /// @brief A flag for if the given type has been registered with the \c FieldRequirementsFactory or not.
  static const bool is_registered;
  //@}
};  // FieldRequirementsRegistry

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQUIREMENTSREGISTRY_HPP_
