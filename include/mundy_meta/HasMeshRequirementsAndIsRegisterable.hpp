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
#include <memory>       // for std::shared_ptr, std::shared_ptr
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
 private:
  /// TODO(palmerb4): Come C++20, we can use concepts to simplify this code. For now, we have to use SFINAE.
  /// I know it's odd to have the private functions at the top, but these are used by the public functions below.
  //! \name SFINAE helpers
  //@{

  /// \brief Helper for checking if \c U has a \c get_mesh_requirements function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_mesh_requirements function, \c std::false_type otherwise.
  template <typename U>
  static auto check_get_mesh_requirements([[maybe_unused]] int unused)
      -> decltype(U::get_mesh_requirements(std::declval<Teuchos::ParameterList>()), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_mesh_requirements function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_mesh_requirements function.
  template <typename>
  static auto check_get_mesh_requirements(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c validate_fixed_parameters_and_set_defaults function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c validate_fixed_parameters_and_set_defaults function, \c std::false_type
  /// otherwise.
  template <typename U>
  static auto check_validate_fixed_parameters_and_set_defaults([[maybe_unused]] int unused)
      -> decltype(U::validate_fixed_parameters_and_set_defaults(std::declval<Teuchos::ParameterList *>()),
                  std::true_type{});

  /// \brief Helper for checking if \c U has a \c validate_fixed_parameters_and_set_defaults function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c validate_fixed_parameters_and_set_defaults function.
  template <typename>
  static auto check_validate_fixed_parameters_and_set_defaults(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c validate_mutable_parameters_and_set_defaults function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c validate_mutable_parameters_and_set_defaults function, \c
  /// std::false_type otherwise.
  template <typename U>
  static auto check_validate_mutable_parameters_and_set_defaults([[maybe_unused]] int unused)
      -> decltype(U::validate_mutable_parameters_and_set_defaults(std::declval<Teuchos::ParameterList *>()),
                  std::true_type{});

  /// \brief Helper for checking if \c U has a \c validate_mutable_parameters_and_set_defaults function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c validate_mutable_parameters_and_set_defaults function.
  template <typename>
  static auto check_validate_mutable_parameters_and_set_defaults(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c RegistrationType type alias.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c RegistrationType type alias, \c std::false_type otherwise.
  template <typename U>
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
  template <typename U>
  static auto check_get_registration_id([[maybe_unused]] int unused)
      -> decltype(U::get_registration_id(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_registration_id function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_registration_id function.
  template <typename>
  static auto check_get_registration_id(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c PolymorphicBaseType type alias.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c PolymorphicBaseType type alias, \c std::false_type otherwise.
  template <typename U>
  static auto check_polymorphic_base_type([[maybe_unused]] int unused)
      -> decltype(std::declval<typename U::PolymorphicBaseType>(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c PolymorphicBaseType type alias.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c PolymorphicBaseType type alias.
  template <typename>
  static auto check_polymorphic_base_type(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c create_new_instance function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c create_new_instance function, \c std::false_type otherwise.
  template <typename U>
  static auto check_create_new_instance([[maybe_unused]] int unused)
      -> decltype(U::create_new_instance(std::declval<mundy::mesh::BulkData *>(),
                                         std::declval<Teuchos::ParameterList>()),
                  std::true_type{});

  /// \brief Helper for checking if \c U has a \c create_new_instance function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c create_new_instance function.
  template <typename>
  static auto check_create_new_instance(...) -> std::false_type;
  //@}

 public:
  //! \name Getters
  //@{

  /// \brief Check for the existence of a \c get_mesh_requirements function.
  /// \return \c true if \c T has a \c get_mesh_requirements function, \c false otherwise.
  ///
  /// The specific signature of the \c get_mesh_requirements function is:
  /// \code
  /// static std::shared_ptr<MeshRequirements> get_mesh_requirements();
  /// \endcode
  static constexpr bool has_get_mesh_requirements =
      decltype(check_get_mesh_requirements<T>(0))::value &&
      std::is_same_v<decltype(T::get_mesh_requirements(std::declval<Teuchos::ParameterList>())),
                     std::shared_ptr<MeshRequirements>>;

  /// \brief Check for the existence of a \c validate_fixed_parameters_and_set_defaults function.
  /// \return \c true if \c T has a \c validate_fixed_parameters_and_set_defaults function, \c false otherwise.
  ///
  /// The specific signature of the \c validate_fixed_parameters_and_set_defaults function is:
  /// \code
  /// static void validate_fixed_parameters_and_set_defaults(Teuchos::ParameterList *const fixed_params_ptr);
  /// \endcode
  static constexpr bool has_validate_fixed_parameters_and_set_defaults =
      decltype(check_validate_fixed_parameters_and_set_defaults<T>(0))::value &&
      std::is_same_v<decltype(T::validate_fixed_parameters_and_set_defaults(std::declval<Teuchos::ParameterList *>())),
                     void>;

  /// \brief Check for the existence of a \c validate_mutable_parameters_and_set_defaults function.
  /// \return \c true if \c T has a \c validate_mutable_parameters_and_set_defaults function, \c false otherwise.
  ///
  /// The specific signature of the \c validate_mutable_parameters_and_set_defaults function is:
  /// \code
  /// static void validate_mutable_parameters_and_set_defaults(Teuchos::ParameterList *const mutable_params_ptr);
  /// \endcode
  static constexpr bool has_validate_mutable_parameters_and_set_defaults =
      decltype(check_validate_mutable_parameters_and_set_defaults<T>(0))::value &&
      std::is_same_v<
          decltype(T::validate_mutable_parameters_and_set_defaults(std::declval<Teuchos::ParameterList *>())), void>;

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
      std::is_same_v<decltype(T::get_registration_id()), typename T::RegistrationType>;

  /// \brief Check for the existence of a \c PolymorphicBaseType type alias.
  /// \return \c true if \c T has a \c PolymorphicBaseType type alias, \c false otherwise.
  ///
  /// The specific signature of the \c PolymorphicBaseType type alias is:
  /// \code
  /// T::PolymorphicBaseType
  /// \endcode
  static constexpr bool has_polymorphic_base_type = decltype(check_polymorphic_base_type<T>(0))::value;

  /// \brief Check for the existence of a \c create_new_instance function.
  /// \return \c true if \c T has a \c create_new_instance function, \c false otherwise.
  ///
  /// The specific signature of the \c create_new_instance function is:
  /// \code
  /// static std::shared_ptr<PolymorphicBaseType> create_new_instance();
  /// \endcode
  static constexpr bool has_create_new_instance =
      decltype(check_create_new_instance<T>(0))::value && decltype(check_polymorphic_base_type<T>(0))::value &&
      std::is_same_v<decltype(T::create_new_instance(std::declval<mundy::mesh::BulkData *>(),
                                                     std::declval<Teuchos::ParameterList>())),
                     std::shared_ptr<typename T::PolymorphicBaseType>>;

  /// \brief Value type semantics for checking \c T meets all the requirements to have mesh requirements and be
  /// registerable. \return \c true if \c T meets all the requirements to have mesh requirements and be registerable, \c
  /// false otherwise.
  static constexpr bool value = has_get_mesh_requirements && has_validate_fixed_parameters_and_set_defaults &&
                                has_validate_mutable_parameters_and_set_defaults && has_registration_type &&
                                has_get_registration_id && has_polymorphic_base_type && has_create_new_instance;
};  // HasMeshRequirementsAndIsRegisterable

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_HASMESHREQUIREMENTSANDISREGISTERABLE_HPP_
