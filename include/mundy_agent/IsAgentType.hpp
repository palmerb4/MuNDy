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

#ifndef MUNDY_AGENT_ISAGENTTYPE_HPP_
#define MUNDY_AGENT_ISAGENTTYPE_HPP_

/// \file IsAgentType.hpp
/// \brief Declaration of the IsAgentType class

// C++ core libs
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

namespace mundy {

namespace agent {

/// \class IsAgentType
/// \brief A traits class for checking if a given type has the desired agent static interface.
///
/// Here is an example that satisfies the desired interface:
/// \code{.cpp}
/// class ExampleAgent {
///  public:
///   //! \name Getters
///   //@{
///
///   /// \brief Get the ExampleAgent's name.
///   static constexpr inline std::string_view get_name();
///
///   /// \brief Get the ExampleAgent's parent's name.
///   static constexpr inline std::string_view get_parent_name();
///
///   /// \brief Get the ExampleAgent's topology (throws if the part doesn't constrain topology)
///   static constexpr inline stk::topology::topology_t get_topology();
///
///   /// \brief Get the ExampleAgent's rank (throws if the part doesn't constrain rank)
///   static constexpr inline stk::topology::rank_t get_rank();
///
///   /// \brief Get if the ExampleAgent constrains the part's topology.
///   static constexpr inline bool constrains_topology();
///
///   /// \brief Get if the ExampleAgent constrains the part's rank.
///   static constexpr inline bool constrains_rank();
///
///   /// \brief Add new part requirements to ALL members of this agent part.
///   /// These modifications are reflected in our mesh requirements.
///   static inline void add_part_reqs(std::shared_ptr<mundy::meta::PartRequirements> part_reqs_ptr);
///
///   /// \brief Add sub-part requirements.
///   /// These modifications are reflected in our mesh requirements.
///   static inline void add_subpart_reqs(std::shared_ptr<mundy::meta::PartRequirements> subpart_reqs_ptr);
///
///   /// \brief Get the mesh requirements for the ExampleAgent.
///   static inline std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements();
/// };  // ExampleAgent
/// \endcode
/// \tparam T The type to check.
template <typename T>
struct IsAgentType {
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
  /// \return \c std::true_type if \c U has a \c get_rank function, \c std::false_type
  /// otherwise.
  template <typename U>
  static auto check_get_rank([[maybe_unused]] int unused) -> decltype(U::get_rank(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_rank function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_rank function.
  template <typename>
  static auto check_get_rank(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c constrains_topology function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c constrains_topology function, \c std::false_type
  /// otherwise.
  template <typename U>
  static auto check_constrains_topology([[maybe_unused]] int unused)
      -> decltype(U::constrains_topology(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c constrains_topology function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c constrains_topology function.
  template <typename>
  static auto check_constrains_topology(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c constrains_rank function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c constrains_rank function, \c std::false_type
  /// otherwise.
  template <typename U>
  static auto check_constrains_rank([[maybe_unused]] int unused) -> decltype(U::constrains_rank(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c constrains_rank function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c constrains_rank function.
  template <typename>
  static auto check_constrains_rank(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c add_part_reqs function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c add_part_reqs function, \c std::false_type
  /// otherwise.
  template <typename U>
  static auto check_add_part_reqs([[maybe_unused]] int unused)
      -> decltype(U::add_part_reqs(std::declval<td::shared_ptr<mundy::meta::PartRequirements>>()), std::true_type{});

  /// \brief Helper for checking if \c U has a \c add_part_reqs function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c add_part_reqs function.
  template <typename>
  static auto check_add_part_reqs(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c add_subpart_reqs function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c add_subpart_reqs function, \c std::false_type
  /// otherwise.
  template <typename U>
  static auto check_add_subpart_reqs([[maybe_unused]] int unused)
      -> decltype(U::add_subpart_reqs(std::declval<td::shared_ptr<mundy::meta::PartRequirements>>()), std::true_type{});

  /// \brief Helper for checking if \c U has a \c add_subpart_reqs function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c add_subpart_reqs function.
  template <typename>
  static auto check_add_subpart_reqs(...) -> std::false_type;

  /// \brief Helper for checking if \c U has a \c get_mesh_requirements function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::true_type if \c U has a \c get_mesh_requirements function, \c std::false_type
  /// otherwise.
  template <typename U>
  static auto check_get_mesh_requirements([[maybe_unused]] int unused)
      -> decltype(U::get_mesh_requirements(), std::true_type{});

  /// \brief Helper for checking if \c U has a \c get_mesh_requirements function.
  /// \tparam U The type to check.
  /// \param[in] unused An unused parameter to allow SFINAE to work.
  /// \return \c std::false_type if \c U does not have a \c get_mesh_requirements function.
  template <typename>
  static auto check_get_mesh_requirements(...) -> std::false_type;
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

  /// \brief Check for the existence of a \c constrains_topology function.
  /// \return \c true if \c T has a \c constrains_topology function, \c false otherwise.
  ///
  /// The specific signature of the \c constrains_topology function is:
  /// \code
  /// static constexpr inline bool constrains_topology();
  /// \endcode
  static constexpr bool has_constrains_topology =
      decltype(check_constrains_topology<T>(0))::value && std::is_same_v<decltype(T::constrains_topology()), bool>;

  /// \brief Check for the existence of a \c constrains_rank function.
  /// \return \c true if \c T has a \c constrains_rank function, \c false otherwise.
  ///
  /// The specific signature of the \c constrains_rank function is:
  /// \code
  /// static constexpr inline bool constrains_rank();
  /// \endcode
  static constexpr bool has_constrains_rank =
      decltype(check_constrains_rank<T>(0))::value && std::is_same_v<decltype(T::constrains_rank()), bool>;

  /// \brief Check for the existence of a \c add_part_reqs function.
  /// \return \c true if \c T has a \c add_part_reqs function, \c false otherwise.
  ///
  /// The specific signature of the \c add_part_reqs function is:
  /// \code
  /// static inline void add_part_reqs(std::shared_ptr<mundy::meta::PartRequirements> part_reqs_ptr);
  /// \endcode
  static constexpr bool has_add_part_reqs = decltype(check_add_part_reqs<T>(0))::value;

  /// \brief Check for the existence of a \c add_subpart_reqs function.
  /// \return \c true if \c T has a \c add_subpart_reqs function, \c false otherwise.
  ///
  /// The specific signature of the \c add_subpart_reqs function is:
  /// \code
  /// static inline void add_subpart_reqs(std::shared_ptr<mundy::meta::PartRequirements> subpart_reqs_ptr);
  /// \endcode
  static constexpr bool has_add_subpart_reqs = decltype(check_add_subpart_reqs<T>(0))::value;

  /// \brief Check for the existence of a \c get_mesh_requirements function.
  /// \return \c true if \c T has a \c get_mesh_requirements function, \c false otherwise.
  ///
  /// The specific signature of the \c get_mesh_requirements function is:
  /// \code
  /// static inline std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements();
  /// \endcode
  static constexpr bool has_get_mesh_requirements = decltype(check_get_mesh_requirements<T>(0))::value;

  /// \brief Value type semantics for checking \c T meets all the requirements to have mesh requirements and be
  /// registerable. \return \c true if \c T meets all the requirements to have mesh requirements and be registerable, \c
  /// false otherwise.
  static constexpr bool value = has_get_name && has_get_topology && has_get_rank && has_constrains_topology &&
                                has_constrains_rank && has_add_part_reqs && has_add_subpart_reqs &&
                                has_get_mesh_requirements;
};  // IsAgentType

}  // namespace agent

}  // namespace mundy

#endif  // MUNDY_AGENT_ISAGENTTYPE_HPP_
