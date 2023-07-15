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

#ifndef MUNDY_META_HASMESHREQUIREMENTSANDISREGISTERABLE_HPP_
#define MUNDY_META_HASMESHREQUIREMENTSANDISREGISTERABLE_HPP_

/// \file HasMeshRequirementsAndIsRegisterable.hpp
/// \brief Declaration of the HasMeshRequirementsAndIsRegisterable class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part

// Mundy libs
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements

namespace mundy {

namespace meta {

/// \class HasMeshRequirementsAndIsRegisterable
/// \brief A traits class for checking if a given type has the desired static interface.
///
/// \tparam T The type to check.
template <typename T>
struct HasMeshRequirementsAndIsRegisterable {
  //! \name Getters
  //@{

  /// \brief Check for the existence of a \c get_requirements_generator function.
  /// \return \c true if \c T has a \c get_requirements_generator function, \c false otherwise.
  ///
  /// The specific signature of the \c get_requirements_generator function is:
  /// \code
  /// static std::shared_ptr<RequirementsGenerator> get_requirements_generator();
  /// \endcode
  static constexpr bool has_get_requirements_generator =
      decltype(check_get_requirements_generator<T>(0))::value &&
      std::is_same<decltype(T::get_requirements_generator()), std::shared_ptr<RequirementsGenerator>>::value;

  /// \brief Check for the existence of a \c get_params_validator function.
  /// \return \c true if \c T has a \c get_params_validator function, \c false otherwise.
  ///
  /// The specific signature of the \c get_params_validator function is:
  /// \code
  /// static std::shared_ptr<ParamsValidator> get_params_validator();
  /// \endcode
  static constexpr bool has_get_params_validator =
      decltype(check_get_params_validator<T>(0))::value &&
      std::is_same<decltype(T::get_params_validator()), std::shared_ptr<ParamsValidator>>::value;

  /// \brief Check for the existence of a \c RegistrationType type alias.
  /// \return \c true if \c T has a \c RegistrationType type alias, \c false otherwise.
  ///
  /// The specific signature of the \c RegistrationType type alias is:
  /// \code
  /// T::RegistrationType
  /// \endcode
  static constexpr bool has_registration_type = decltype(check_registration_type<T>(0))::value;

  /// \brief Check for the existence of a \c get_registration_id function.
  /// \return \c true if \c T has a \c get_registration_id function, \c false otherwise.
  ///
  /// The specific signature of the \c get_registration_id function is:
  /// \code
  /// static RegistrationType get_registration_id();
  /// \endcode
  static constexpr bool has_get_registration_id =
      decltype(check_get_registration_id<T>(0))::value && decltype(check_registration_type<T>(0))::value &&
      std::is_same<decltype(T::get_registration_id()), typename T::RegistrationType>::value;

  /// \brief Check for the existence of a \c get_new_class_generator function.
  /// \return \c true if \c T has a \c get_new_class_generator function, \c false otherwise.
  ///
  /// The specific signature of the \c get_new_class_generator function is:
  /// \code
  /// static std::unique_ptr<NewClassGenerator> get_new_class_generator();
  /// \endcode
  static constexpr bool has_new_class_generator =
      decltype(check_new_class_generator<T>(0))::value &&
      std::is_same<decltype(T::get_new_class_generator()), std::unique_ptr<typename T::NewClassGenerator>>::value;

 private:
  /// TODO(palmerb4): Come C++20, we can use concepts to simplify this code. For now, we have to use SFINAE.
  //! \name SFINAE helpers
  //@{

  /// \brief Helper for checking if \c U has a \c get_mesh_requirements function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_mesh_requirements function, \c std::false_type otherwise.
  static auto check_get_mesh_requirements([[maybe_unused]] int unused)
      -> decltype(U::get_mesh_requirements(std::declval<const Teuchos::ParameterList &>()), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_mesh_requirements function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_mesh_requirements function.
  template <typename>
  static auto check_get_mesh_requirements(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c get_valid_fixed_params function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_valid_fixed_params function, \c std::false_type otherwise.
  static auto check_get_params_validator([[maybe_unused]] int unused)
      -> decltype(U::get_valid_fixed_params(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_valid_fixed_params function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_valid_fixed_params function.
  template <typename>
  static auto check_get_params_validator(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c get_valid_mutable_params function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_valid_mutable_params function, \c std::false_type otherwise.
  static auto check_get_valid_mutable_params([[maybe_unused]] int unused)
      -> decltype(U::get_valid_mutable_params(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_valid_mutable_params function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_valid_mutable_params function.
  template <typename>
  static auto check_get_valid_mutable_params(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c RegistrationType type alias.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c RegistrationType type alias, \c std::false_type otherwise.
  static auto check_registration_type([[maybe_unused]] int unused)
      -> decltype(std::declval<typename U::RegistrationType>(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c RegistrationType type alias.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c RegistrationType type alias.
  template <typename>
  static auto check_registration_type(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c get_registration_id function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_registration_id function, \c std::false_type otherwise.
  static auto check_get_registration_id([[maybe_unused]] int unused)
      -> decltype(U::get_registration_id(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_registration_id function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_registration_id function.
  template <typename>
  static auto check_get_registration_id(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c get_new_class_generator function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_new_class_generator function, \c std::false_type otherwise.
  static auto check_new_class_generator([[maybe_unused]] int unused)
      -> decltype(U::get_new_class_generator(std::declval<const Teuchos::ParameterList &>()), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_new_class_generator function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_new_class_generator function.
  template <typename>
  static auto check_new_class_generator(...) -> std::false_type;
  //@}
};  // HasMeshRequirementsAndIsRegisterable

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_HASMESHREQUIREMENTSANDISREGISTERABLE_HPP_
