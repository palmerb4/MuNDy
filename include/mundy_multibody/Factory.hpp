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

#ifndef MUNDY_MULTIBODY_FACTORY_HPP_
#define MUNDY_MULTIBODY_FACTORY_HPP_

/// \file Factory.hpp
/// \brief Declaration of the Factory class

// C++ core libs
#include <functional>  // for std::function
#include <map>         // for std::map
#include <stdexcept>   // for std::logic_error, std::invalid_argument
#include <string>      // for std::string
#include <utility>     // for std::make_pair

// Trilinos libs
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION

// Mundy libs
#include <mundy_multibody/Multibody.hpp>  // for mundy::multibody::Multibody

namespace mundy {

namespace multibody {

/// \class Factory
/// \brief A factory containing generation routines for classes derived from \c MultibodyType.
///
/// The goal of this factory, as with most factories, is to provide an abstraction for case switches between
/// different methods. This factory is a bit different in that it always users to register new classes derived from
/// \c MultibodyType and associate them with corresponding keys without modifying Mundy's source code.
///
/// \note Credit where credit is due: The design for this class originates from Andreas Zimmerer and his
/// self-registering types design. https://www.jibbow.com/posts/cpp-header-only-self-registering-types/
class Factory {
 public:
  //! \name Typedefs
  //@{

  /// \brief A function that returns an std::string_view.
  using StringViewGenerator = std::function<std::string_view()>;

  /// \brief A function that returns an stk::topology.
  using TopologyGenerator = std::function<stk::topology()>;

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
    return get_name_to_id_map().count(name) != 0;
  }

  /// \brief Get if the provided fast id is valid or not
  /// \param fast_id [in] A fast id that may or may not correspond to a registered class.
  static bool is_valid(const multibody_t fast_id) {
    return get_name_generator_map().count(fast_id) != 0;
  }

  /// \brief Get the fast id corresponding to a registered class with the given name.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static multibody_t get_fast_id(const std::string_view& name) {
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(name), std::invalid_argument,
                               "Factory: The provided class's name '" << name << "' is not valid.");
    return get_name_to_id_map()[name];
  }

  /// \brief Get the name corresponding to a registered class with the given fast id.
  /// \param fast_id [in] A fast id that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(fast_id) returns false
  static std::string_view get_name(const multibody_t fast_id) {
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(fast_id), std::invalid_argument,
                               "Factory: The provided class's id '" << fast_id << "' is not valid.");
    return get_name_generator_map()[fast_id]();
  }

  /// \brief Get the topology corresponding to a registered class with the given fast id.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static std::topology get_topology(const std::string_view& name) {
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(name), std::invalid_argument,
                               "Factory: The provided class's name '" << name << "' is not valid.");
    const multibody_t fast_id = get_fast_id(name);
    return get_topology(fast_id);
  }

  /// \brief Get the topology corresponding to a registered class with the given fast id.
  /// \param fast_id [in] A fast id that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(fast_id) returns false
  static std::topology get_topology(const multibody_t fast_id) {
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(fast_id), std::invalid_argument,
                               "Factory: The provided class's id '" << fast_id << "' is not valid.");
    return get_topology_generator_map()[fast_id]();
  }

  /// \brief Get the if the registered class with the given fast id has a parent multibody type.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static bool has_parent(const std::string_view& name) {
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(name), std::invalid_argument,
                               "Factory: The provided class's name '" << name << "' is not valid.");
    const multibody_t fast_id = get_fast_id(name);
    return has_parent(fast_id);
  }

  /// \brief Get the if the registered class with the given fast id has a parent multibody type.
  /// \param fast_id [in] A fast id that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(fast_id) returns false
  static bool has_parent(const multibody_t fast_id) {
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(fast_id), std::invalid_argument,
                               "Factory: The provided class's id '" << fast_id << "' is not valid.");
    return get_has_parent_generator_map()[fast_id]();
  }

  /// \brief Get the parent multibody type's fast id corresponding to a registered class with the given fast id.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static bool get_parent_fast_id(const std::string_view& name) {
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(name), std::invalid_argument,
                               "Factory: The provided class's name '" << name << "' is not valid.");
    const multibody_t fast_id = get_fast_id(name);
    return get_parent_fast_id(fast_id);
  }

  /// \brief Get the parent multibody type's fast id corresponding to a registered class with the given fast id.
  /// \param fast_id [in] A fast id that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(fast_id) returns false.
  static bool get_parent_fast_id(const multibody_t fast_id) {
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(fast_id), std::invalid_argument,
                               "Factory: The provided class's id '" << fast_id << "' is not valid.");
    std::string_view parent_name = get_parent_name(fast_id);
    return get_fast_id(parent_name);
  }

  /// \brief Get the parent multibody type's name corresponding to a registered class with the given fast id.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static bool get_parent_name(const std::string_view& name) {
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(name), std::invalid_argument,
                               "Factory: The provided class's name '" << name << "' is not valid.");
    const multibody_t fast_id = get_fast_id(name);
    return get_parent_name(fast_id);
  }

  /// \brief Get the parent multibody type's name corresponding to a registered class with the given fast id.
  /// \param fast_id [in] A fast id that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(fast_id) returns false
  static bool get_parent_name(const multibody_t fast_id) {
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(fast_id), std::invalid_argument,
                               "Factory: The provided class's id '" << fast_id << "' is not valid.");
    return get_parent_name_generator_map()[fast_id]();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Register a new class. The key for the class is determined by its class identifier.
  template <typename ClassToRegister>
  static void register_new_class() {
    const std::string_view name = ClassToRegister::get_name();
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid(name), std::invalid_argument,
                               "Factory: The provided class's name '" << name << "' already exists.");
    number_of_registered_types_++;

    multibody_t fast_id = number_of_registered_types_ - 1;
    get_name_to_id_map().insert(std::make_pair(name, fast_id));
    get_name_generator_map().insert(std::make_pair(fast_id, name));
    get_topology_generator_map().insert(std::make_pair(fast_id, ClassToRegister::get_topology));
    get_has_parent_generator_map().insert(std::make_pair(fast_id, ClassToRegister::has_parent));
    get_parent_name_generator_map().insert(std::make_pair(fast_id, ClassToRegister::get_parent_name));
  }
  //@}

 private:
  //! \name Internal member
  //@{

  static multibody_t number_of_registered_types_ = 0;
  //@}

  //! \name Typedefs
  //@{

  /// \brief A map from a string_view to fast id.
  using NameToFastIdMap = std::map<std::string_view, multibody_t>;

  /// \brief A map from fast id to a function that returns a string view.
  using FastIdToStringViewGeneratorMap = std::map<multibody_t, StringViewGenerator>;

  /// \brief A map from fast id to a function that returns an stk::topology.
  using FastIdToTopologyGeneratorMap = std::map<multibody_t, TopologyGenerator>;

  /// \brief A function that takes in a multibody_t and returns a boolean.
  using FastIdToBooleanGeneratorMap = std::map<multibody_t, BooleanGenerator>;
  //@}

  //! \name Attributes
  //@{

  static NameToFastIdMap& get_name_to_id_map() {
    // Static: One and the same instance for all function calls.
    static NameToFastIdMap name_to_id_map;
    return name_to_id_map;
  }

  static FastIdToStringViewGeneratorMap& get_name_generator_map() {
    // Static: One and the same instance for all function calls.
    static FastIdToStringViewGeneratorMap name_generator_map;
    return name_generator_map;
  }

  static FastIdToTopologyGeneratorMap& get_topology_generator_map() {
    // Static: One and the same instance for all function calls.
    static FastIdToTopologyGeneratorMap topology_generator_map;
    return topology_generator_map;
  }

  static FastIdToBooleanGeneratorMap& get_has_parent_generator_map() {
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

  //! \name Friends
  //@{

  /// \brief Every concrete class that inherits from the MultibodyRegistry will be added to this factory's
  /// registry. This process requires friendship <3.
  template <class>
  friend class Registry;
  //@}
};  // Factory

}  // namespace multibody

}  // namespace mundy

#endif  // MUNDY_MULTIBODY_FACTORY_HPP_
