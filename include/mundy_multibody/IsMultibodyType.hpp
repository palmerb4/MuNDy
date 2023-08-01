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

#ifndef MUNDY_MULTIBODY_ISMULTIBODYTYPE_HPP_
#define MUNDY_MULTIBODY_ISMULTIBODYTYPE_HPP_

/// \file IsMultibodyType.hpp
/// \brief Declaration of the IsMultibodyType class

// C++ core libs
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

namespace mundy {

namespace multibody {

/// \class IsMultibodyType
/// \brief A traits class for checking if a given type has the desired multibody static interface.
///
/// \tparam T The type to check.
template <typename T>
struct IsMultibodyType {
 private:
  /// TODO(palmerb4): Come C++20, we can use concepts to simplify this code. For now, we have to use SFINAE.
  /// I know it's odd to have the private functions at the top, but these are used by the public functions below.
  //! \name SFINAE helpers
  //@{

  /// \brief Helper for checking if \c U has a \c get_name function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_name function, \c std::false_type otherwise.
  template <typename U>
  static auto check_get_name([[maybe_unused]] int unused) -> decltype(U::get_name(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_name function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_name function.
  template <typename>
  static auto check_get_name(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c get_topology function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_topology function, \c std::false_type
  /// otherwise.
  template <typename U>
  static auto check_get_topology([[maybe_unused]] int unused) -> decltype(U::get_topology(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_topology function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_topology function.
  template <typename>
  static auto check_get_topology(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c get_rank function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_rank function, \c
  /// std::false_type otherwise.
  template <typename U>
  static auto check_get_rank([[maybe_unused]] int unused) -> decltype(U::get_rank(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_rank function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_rank function.
  template <typename>
  static auto check_get_rank(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c has_parent function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c has_parent function, \c std::false_type otherwise.
  template <typename U>
  static auto check_has_parent([[maybe_unused]] int unused) -> decltype(U::has_parent(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c has_parent function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c has_parent function.
  template <typename>
  static auto check_has_parent(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c get_parent_name function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_parent_name function, \c std::false_type otherwise.
  template <typename U>
  static auto check_get_parent_name([[maybe_unused]] int unused) -> decltype(U::get_parent_name(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_parent_name function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_parent_name function.
  template <typename>
  static auto check_get_parent_name(...) -> std::false_type;
  //@}

 public:
  //! \name Getters
  //@{

  /// \brief Check for the existence of a \c get_name function.
  /// \return \c true if \c T has a \c get_name function, \c false otherwise.
  ///
  /// The specific signature of the \c get_name function is:
  /// \code
  /// static constexpr inline std::string_view get_name();
  /// \endcode
  static constexpr bool has_get_name =
      decltype(check_get_name<T>(0))::value && std::is_same_v<decltype(T::get_name()), std::string_view>;

  /// \brief Check for the existence of a \c get_topology function.
  /// \return \c true if \c T has a \c get_topology function, \c false otherwise.
  ///
  /// The specific signature of the \c get_topology function is:
  /// \code
  /// static constexpr inline stk::topology::topology_t get_topology();
  /// \endcode
  static constexpr bool has_get_topology = decltype(check_get_topology<T>(0))::value &&
                                           std::is_same_v<decltype(T::get_topology()), stk::topology::topology_t>;

  /// \brief Check for the existence of a \c get_rank function.
  /// \return \c true if \c T has a \c get_rank function, \c false otherwise.
  ///
  /// The specific signature of the \c get_rank function is:
  /// \code
  /// static constexpr inline stk::topology::rank_t get_rank();
  /// \endcode
  static constexpr bool has_get_rank =
      decltype(check_get_rank<T>(0))::value && std::is_same_v<decltype(T::get_rank()), stk::topology::rank_t>;

  /// \brief Check for the existence of a \c has_parent function.
  /// \return \c true if \c T has a \c has_parent function, \c false otherwise.
  ///
  /// The specific signature of the \c has_parent function is:
  /// \code
  /// static constexpr inline bool has_parent();
  /// \endcode
  static constexpr bool has_has_parent =
      decltype(check_has_parent<T>(0))::value && std::is_same_v<decltype(T::has_parent()), bool>;

  /// \brief Check for the existence of a \c get_parent_name function.
  /// \return \c true if \c T has a \c get_parent_name function, \c false otherwise.
  ///
  /// The specific signature of the \c get_parent_name function is:
  /// \code
  /// static constexpr inline std::string_view get_parent_name();
  /// \endcode
  static constexpr bool has_get_parent_name =
      decltype(check_get_parent_name<T>(0))::value && std::is_same_v<decltype(T::get_parent_name()), std::string_view>;

  /// \brief Value type semantics for checking \c T meets all the requirements to have mesh requirements and be
  /// registerable. \return \c true if \c T meets all the requirements to have mesh requirements and be registerable, \c
  /// false otherwise.
  static constexpr bool value =
      has_get_name && has_get_topology && has_get_rank && has_has_parent && has_get_parent_name;
};  // IsMultibodyType

}  // namespace multibody

}  // namespace mundy

#endif  // MUNDY_MULTIBODY_ISMULTIBODYTYPE_HPP_
