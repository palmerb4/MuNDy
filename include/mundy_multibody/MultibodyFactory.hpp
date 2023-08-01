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

#ifndef MUNDY_MULTIBODY_MULTIBODYFACTORY_HPP_
#define MUNDY_MULTIBODY_MULTIBODYFACTORY_HPP_

/// \file MultibodyFactory.hpp
/// \brief Declaration of the MultibodyFactory class

// C++ core libs
#include <functional>  // for std::function
#include <iostream>
#include <map>        // for std::map
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <utility>    // for std::make_pair

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>               // for MUNDY_THROW_ASSERT
#include <mundy_multibody/IsMultibodyType.hpp>  // for mundy::multibody::IsMultibodyType
#include <mundy_multibody/Multibody.hpp>        // for mundy::multibody::Multibody

namespace mundy {

namespace multibody {

/// \class MultibodyFactory
/// \brief A factory containing generation routines for classes derived from \c MultibodyType.
///
/// The goal of \c MultibodyFactory, as with most factories, is to provide an abstraction for case switches between
/// different methods. This factory is a bit different in that it always users to register new classes (that match the
/// desired interface) and associate them with corresponding keys. These classes can then be fetched based using their
/// registration id. Most importantly, this enables users to register their own derived classes without modifying
/// Mundy's source code.
///
/// Any class that wishes to be registered with this factory must must implement the following static interface. Don't
/// worry, if your class fails to meet one of these requirements, register_new_class will throw a human-readable compile
/// time error telling you which functions you need to implement/modify. The specific signatures of these functions are
/// given below.
///
/// \code{.cpp}
/// // Get the objects's name.
/// // This name must be unique and not shared by any other multibody object.
/// static constexpr inline std::string_view get_name();
///
/// // Get the objects's topology.
/// static constexpr inline stk::topology::topology_t get_topology();
///
/// // Get the objects's rank.
/// static constexpr inline stk::topology::topology_t get_rank()
///
/// // Get if the objects has a parent multibody type.
/// static constexpr inline bool has_parent()
///
/// // Get the parent multibody type of the objects.
/// static constexpr inline std::string_view get_parent_name()
/// \endcode
///
/// \note Credit where credit is due: The design for this class originates from Andreas Zimmerer and his
/// self-registering types design (albeit with heavy modifications).
/// https://www.jibbow.com/posts/cpp-header-only-self-registering-types/
class MultibodyFactory {
 public:
  //! \name Typedefs
  //@{

  /// \brief A function that returns an std::string_view.
  using StringViewGenerator = std::function<std::string_view()>;

  /// \brief A function that returns an stk::topology::topology_t.
  using TopologyGenerator = std::function<stk::topology::topology_t()>;

  /// \brief A function that returns an stk::topology::rank_t.
  using RankGenerator = std::function<stk::topology::rank_t()>;

  /// \brief A function that returns a boolean.
  using BooleanGenerator = std::function<bool()>;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of classes this factory recognizes.
  static size_t get_number_of_registered_types() {
    return number_of_registered_types_;
  }

  /// \brief Get if the provided name is valid or not
  /// \param name [in] A string name that may or may not correspond to a registered class.
  static bool is_valid(const std::string_view& name) {
    return get_name_to_type_map().count(name) != 0;
  }

  /// \brief Get if the provided multibody_type is valid or not
  /// \param multibody_type [in] A multibody_type that may or may not correspond to a registered class.
  static bool is_valid(const multibody_t multibody_type) {
    return get_name_map().count(multibody_type) != 0;
  }

  /// \brief Throw if the provided name is invalid
  /// \param name [in] A string name that may or may not correspond to a registered class.
  static void assert_is_valid(const std::string_view& name) {
    MUNDY_THROW_ASSERT(is_valid(name), std::invalid_argument,
                       "MultibodyFactory: The provided class's name '"
                           << name << "' is not valid.\n"
                           << "There are currently " << get_number_of_registered_types() << " registered classes.\n"
                           << "Their names are:" << get_valid_names_as_a_string() << "\n"
                           << "Their ids are:" << get_valid_ids_as_a_string() << "\n");
  }

  /// \brief Throw if the provided multibody_type is invalid
  /// \param multibody_type [in] A multibody_type that may or may not correspond to a registered class.
  static void assert_is_valid(const multibody_t multibody_type) {
    MUNDY_THROW_ASSERT(is_valid(multibody_type), std::invalid_argument,
                       "MultibodyFactory: The provided class's id '"
                           << multibody_type << "' is not valid."
                           << "There are currently " << get_number_of_registered_types() << " registered classes.\n"
                           << "Their names are:" << get_valid_names_as_a_string() << "\n"
                           << "Their ids are:" << get_valid_ids_as_a_string() << "\n");
  }

  /// \brief Get the multibody_type corresponding to a registered class with the given name.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static multibody_t get_multibody_type(const std::string_view& name) {
    assert_is_valid(name);
    return get_name_to_type_map()[name];
  }

  /// \brief Get the name corresponding to a registered class with the given multibody_type.
  /// \param multibody_type [in] A multibody_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(multibody_type) returns false
  static std::string_view get_name(const multibody_t multibody_type) {
    assert_is_valid(multibody_type);
    return get_name_map()[multibody_type];
  }

  /// \brief Get the topology corresponding to a registered class with the given multibody_type.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static stk::topology::topology_t get_topology(const std::string_view& name) {
    assert_is_valid(name);
    const multibody_t multibody_type = get_multibody_type(name);
    return get_topology(multibody_type);
  }

  /// \brief Get the topology corresponding to a registered class with the given multibody_type.
  /// \param multibody_type [in] A multibody_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(multibody_type) returns false
  static stk::topology::topology_t get_topology(const multibody_t multibody_type) {
    assert_is_valid(multibody_type);
    return get_topology_generator_map()[multibody_type]();
  }

  /// \brief Get the rank corresponding to a registered class with the given multibody_type.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static stk::topology::rank_t get_rank(const std::string_view& name) {
    assert_is_valid(name);
    const multibody_t multibody_type = get_multibody_type(name);
    return get_rank(multibody_type);
  }

  /// \brief Get the rank corresponding to a registered class with the given multibody_type.
  /// \param multibody_type [in] A multibody_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(multibody_type) returns false
  static stk::topology::rank_t get_rank(const multibody_t multibody_type) {
    assert_is_valid(multibody_type);
    return get_rank_generator_map()[multibody_type]();
  }

  /// \brief Get the if the registered class with the given multibody_type has a parent multibody type.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static bool has_parent(const std::string_view& name) {
    assert_is_valid(name);
    const multibody_t multibody_type = get_multibody_type(name);
    return has_parent(multibody_type);
  }

  /// \brief Get the if the registered class with the given multibody_type has a parent multibody type.
  /// \param multibody_type [in] A multibody_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(multibody_type) returns false
  static bool has_parent(const multibody_t multibody_type) {
    assert_is_valid(multibody_type);
    return has_parent_generator_map()[multibody_type]();
  }

  /// \brief Get the parent multibody type's multibody_type corresponding to a registered class with the given
  /// multibody_type. \param name [in] A string name that correspond to a registered class. Throws an error if this name
  /// is not registered to an existing class, i.e., is_valid(name) returns false
  static bool get_parent_multibody_type(const std::string_view& name) {
    assert_is_valid(name);
    const multibody_t multibody_type = get_multibody_type(name);
    return get_parent_multibody_type(multibody_type);
  }

  /// \brief Get the parent multibody type's multibody_type corresponding to a registered class with the given
  /// multibody_type. \param multibody_type [in] A multibody_type that correspond to a registered class. Throws an error
  /// if this id is not registered to an existing class, i.e., is_valid(multibody_type) returns false.
  static bool get_parent_multibody_type(const multibody_t multibody_type) {
    assert_is_valid(multibody_type);
    std::string_view parent_name = get_parent_name(multibody_type);
    return get_multibody_type(parent_name);
  }

  /// \brief Get the parent multibody type's name corresponding to a registered class with the given multibody_type.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static std::string_view get_parent_name(const std::string_view& name) {
    assert_is_valid(name);
    const multibody_t multibody_type = get_multibody_type(name);
    return get_parent_name(multibody_type);
  }

  /// \brief Get the parent multibody type's name corresponding to a registered class with the given multibody_type.
  /// \param multibody_type [in] A multibody_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(multibody_type) returns false
  static std::string_view get_parent_name(const multibody_t multibody_type) {
    assert_is_valid(multibody_type);
    return get_parent_name_generator_map()[multibody_type]();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Register a new class. The key for the class is determined by its class identifier.
  template <typename ClassToRegister>
  static inline bool register_new_class() {
    // Check that the ClassToRegister has the desired interface.
    using Checker = IsMultibodyType<ClassToRegister>;
    static_assert(Checker::has_get_name,
                  "MultibodyFactory: The class to register doesn't have the correct has_get_name function.\n"
                  "See the documentation of MultibodyFactory for more information about the expected interface.");
    static_assert(Checker::has_get_topology,
                  "MultibodyFactory: The class to register doesn't have the correct has_get_topology function.\n"
                  "See the documentation of MultibodyFactory for more information about the expected interface.");
    static_assert(Checker::has_get_rank,
                  "MultibodyFactory: The class to register doesn't have the correct has_get_rank function.\n"
                  "See the documentation of MultibodyFactory for more information about the expected interface.");
    static_assert(Checker::has_has_parent,
                  "MultibodyFactory: The class to register doesn't have the correct has_has_parent function.\n"
                  "See the documentation of MultibodyFactory for more information about the expected interface.");
    static_assert(Checker::has_get_parent_name,
                  "MultibodyFactory: The class to register doesn't have the correct has_get_parent_name function.\n"
                  "See the documentation of MultibodyFactory for more information about the expected interface.");

    std::cout << "MultibodyFactory: Registering class " << ClassToRegister::get_name() << " with id "
              << number_of_registered_types_ << std::endl;

    // Register the class.
    const std::string_view name = ClassToRegister::get_name();
    MUNDY_THROW_ASSERT(!is_valid(name), std::invalid_argument,
                       "MultibodyFactory: The provided class's name '" << name << "' already exists.");

    number_of_registered_types_++;
    const multibody_t multibody_type = number_of_registered_types_ - 1;

    get_name_to_type_map().insert(std::make_pair(name, multibody_type));
    get_name_map().insert(std::make_pair(multibody_type, name));
    get_topology_generator_map().insert(std::make_pair(multibody_type, ClassToRegister::get_topology));
    get_rank_generator_map().insert(std::make_pair(multibody_type, ClassToRegister::get_rank));
    has_parent_generator_map().insert(std::make_pair(multibody_type, ClassToRegister::has_parent));
    get_parent_name_generator_map().insert(std::make_pair(multibody_type, ClassToRegister::get_parent_name));

    return true;
  }
  //@}

 private:
  //! \name Internal member
  //@{

  /// \brief The number of registered multibody types.
  /// \note This is initialized to zero outside the class declaration.
  static multibody_t number_of_registered_types_;
  //@}

  //! \name Typedefs
  //@{

  /// \brief A map from a string_view to multibody_type.
  using NameToFastIdMap = std::map<std::string_view, multibody_t>;

  /// \brief A map from a string_view to multibody_type.
  using FastIdToNameMap = std::map<multibody_t, std::string_view>;

  /// \brief A map from multibody_type to a function that returns a string view.
  using FastIdToStringViewGeneratorMap = std::map<multibody_t, StringViewGenerator>;

  /// \brief A map from multibody_type to a function that returns an stk::topology::topology_t.
  using FastIdToTopologyGeneratorMap = std::map<multibody_t, TopologyGenerator>;

  /// \brief A map from multibody_type to a function that returns an stk::topology::rank_t.
  using FastIdToRankGeneratorMap = std::map<multibody_t, RankGenerator>;

  /// \brief A function that takes in a multibody_t and returns a boolean.
  using FastIdToBooleanGeneratorMap = std::map<multibody_t, BooleanGenerator>;
  //@}

  //! \name Attributes
  //@{
  static std::string get_valid_names_as_a_string() {
    std::ostringstream oss;
    for (const auto& pair : get_name_to_type_map()) {
      oss << " " << pair.first;
    }
    return oss.str();
  }

  static std::string get_valid_ids_as_a_string() {
    std::ostringstream oss;
    for (const auto& pair : get_name_map()) {
      oss << " " << pair.first;
    }
    return oss.str();
  }

  static NameToFastIdMap& get_name_to_type_map() {
    // Static: One and the same instance for all function calls.
    static NameToFastIdMap name_to_id_map;
    return name_to_id_map;
  }

  static FastIdToNameMap& get_name_map() {
    // Static: One and the same instance for all function calls.
    static FastIdToNameMap name_map;
    return name_map;
  }

  static FastIdToTopologyGeneratorMap& get_topology_generator_map() {
    // Static: One and the same instance for all function calls.
    static FastIdToTopologyGeneratorMap topology_generator_map;
    return topology_generator_map;
  }

  static FastIdToRankGeneratorMap& get_rank_generator_map() {
    // Static: One and the same instance for all function calls.
    static FastIdToRankGeneratorMap rank_generator_map;
    return rank_generator_map;
  }

  static FastIdToBooleanGeneratorMap& has_parent_generator_map() {
    // Static: One and the same instance for all function calls.
    static FastIdToBooleanGeneratorMap has_parent_generator_map;
    return has_parent_generator_map;
  }

  static FastIdToStringViewGeneratorMap& get_parent_name_generator_map() {
    // Static: One and the same instance for all function calls.
    static FastIdToStringViewGeneratorMap parent_name_generator_map;
    return parent_name_generator_map;
  }
  //@}
};  // MultibodyFactory

}  // namespace multibody

}  // namespace mundy

#endif  // MUNDY_MULTIBODY_MULTIBODYFACTORY_HPP_
