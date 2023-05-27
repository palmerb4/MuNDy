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
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_trivially_copyable

// Mundy libs
#include <mundy_meta/FieldRequirementsFactory.hpp>  // for mundy::meta::FieldRequirementsFactory

namespace mundy {

namespace meta {

/// \class FieldRequirementsRegistry
/// \brief A class for registering \c FieldRequirementss within \c FieldRequirementsFactory.
///
/// All valid field types that can be passed to \c FieldRequirements should be registered within the
/// \c FieldRequirementsFactory. This registry aids in the registration process.
///
/// \tparam FieldTypeToRegister A trivially copyable type to be registered with the \c FieldRequirementsFactory.
template <class FieldTypeToRegister,
          std::enable_if_t<std::is_trivially_copyable<FieldTypeToRegister>::value, bool> = true>
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

/// @brief Perform the static registration of the desired FieldRequirements FieldTypes.
///
/// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
/// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
// clang-format off
const bool FieldRequirementsRegistry<short>::is_registered = FieldRequirementsRegistry<short>::register_type("SHORT");
const bool FieldRequirementsRegistry<unsigned short>::is_registered = FieldRequirementsRegistry<unsigned short>::register_type("UNSIGNED_SHORT");
const bool FieldRequirementsRegistry<int>::is_registered = FieldRequirementsRegistry<int>::register_type("INT");
const bool FieldRequirementsRegistry<unsigned int>::is_registered = FieldRequirementsRegistry<unsigned int>::register_type("UNSIGNED_INT");
const bool FieldRequirementsRegistry<long>::is_registered = FieldRequirementsRegistry<long>::register_type("LONG");
const bool FieldRequirementsRegistry<unsigned long>::is_registered = FieldRequirementsRegistry<unsigned long>::register_type("UNSIGNED_LONG");
const bool FieldRequirementsRegistry<long long>::is_registered = FieldRequirementsRegistry<long long>::register_type("LONG_LONG");
const bool FieldRequirementsRegistry<unsigned long long>::is_registered = FieldRequirementsRegistry<unsigned long long>::register_type("UNSIGNED_LONG_LONG");
const bool FieldRequirementsRegistry<float>::is_registered = FieldRequirementsRegistry<float>::register_type("FLOAT");
const bool FieldRequirementsRegistry<double>::is_registered = FieldRequirementsRegistry<double>::register_type("DOUBLE");
const bool FieldRequirementsRegistry<long double>::is_registered = FieldRequirementsRegistry<long double>::register_type("LONG_DOUBLE");
const bool FieldRequirementsRegistry<std::complex<float>>::is_registered = FieldRequirementsRegistry<std::complex<float>>::register_type("COMPLES_FLOAT");
const bool FieldRequirementsRegistry<std::complex<double>>::is_registered = FieldRequirementsRegistry<std::complex<double>>::register_type("COMPLEX_DOUBLE");
// clang-format on

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQUIREMENTSREGISTRY_HPP_
