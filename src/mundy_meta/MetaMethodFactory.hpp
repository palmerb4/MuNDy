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

#ifndef MUNDY_META_METAMETHODFACTORY_HPP_
#define MUNDY_META_METAMETHODFACTORY_HPP_

/// \file MetaMethodFactory.hpp
/// \brief Declaration of the MetaMethodFactory class

// C++ core libs
#include <functional>   // for std::function
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/BulkData.hpp>    // for stk::mesh::BulkData
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part

// Mundy libs
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

/// \brief An empty struct to symbolize an unused template parameter.
struct DefaultMethodIdentifier {};  // DefaultMethodIdentifier

/// \class MetaMethodFactory
/// \brief A factory containing generation routines for all of Mundy's \c MetaMethods.
///
/// The goal of \c MetaMethodFactory, as with most factories, is to provide an abstraction for case switches between
/// different methods. This factory is a bit different in that it always users to register new \c MetaMethods and
/// associate them with their corresponding keys. This allows a method to be created based on a string. Most
/// importantly, it enables users to add their own \c MetaMethods without modifying Mundy's source code.
///
/// It's important to note that the static members of this factory will be shared between any \c MetaMethodFactories
/// with the same template \c RegistryIdentifier. We use the default identifier to register all of Mundy's
/// \c MetaMethods.
///
/// \note This factory does not store an instance of \c MetaMethod; rather, it stores maps from a string to some of
/// \c MetaMethod's static member functions.
///
/// \note Credit where credit is due: The design for this class originates from Andreas Zimmerer and his
/// self-registering types design. https://www.jibbow.com/posts/cpp-header-only-self-registering-types/
///
/// \tparam RegistryIdentifier A template type used to create different independent instances of MetaMethodFactory.
template <typename RegistryIdentifier = DefaultMethodIdentifier>
class MetaMethodFactory {
 public:
  //! \name Typedefs
  //@{

  /// \brief A function type that takes a parameter list and produces a shared pointer to an object derived from
  /// \c MetaMethod.
  using NewMetaMethodGenerator = std::function<std::unique_ptr<MetaMethodBase>(
      const stk::mesh::BulkData*, const std::vector<*stk::mesh::Part>&, const Teuchos::ParameterList&)>;

  /// \brief A function type that takes a parameter list and produces a PartRequirements instance.
  using NewRequirementsGenerator = std::function<PartRequirements>(const Teuchos::ParameterList&);

  /// \brief A function type that produces a Teuchos::ParameterList instance.
  using NewDefaultParamsGenerator = std::function<Teuchos::ParameterList>();
  //@}

  //! \name Attributes
  //@{

  /// \brief Get the number of \c MetaMethod classes this factory recognizes.
  static size_t get_number_of_subclasses() {
    return get_instance_generator_map().size();
  }

  /// \brief Get the requirements that this a registered \c MetaMethod imposes upon each particle and/or constraint.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made a registered \c
  /// MetaMethod with respect to the parts, topology, and fields input into the \c execute function. These assumptions
  /// may vary based parameters in the \c parameter_list.
  ///
  /// The registered \c MetaMethod accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a \c MetaMethod with this factory, use the
  /// provided \c register_new_method function.
  ///
  /// \param key [in] A key corresponding to a registered \c MetaMethod.
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A default parameter list
  /// is accessible via \c get_valid_params.
  static std::unique_ptr<PartRequirements> get_part_requirements(const std::string& key,
                                                                 const Teuchos::ParameterList& parameter_list) {
    return get_instance_generator_map()[key](parameter_list);
  }

  /// \brief Get the default parameter list for a registered \c MetaMethod.
  ///
  /// The registered \c MetaMethod accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a \c MetaMethod with this factory, use the
  /// provided \c register_new_method function.
  ///
  /// \note This function does not cache its return value, so
  /// each time you call this function, a new \c Teuchos::ParameterList will be created. You can save the result
  /// yourself if you wish to reuse it.
  ///
  /// \param key [in] A key corresponding to a registered \c MetaMethod.
  static Teuchos::ParameterList get_valid_params(const std::string& key) {
    TEUCHOS_TEST_FOR_EXCEPTION(is_valid_key(key), std::invalid_argument,
                               "The provided key " << key << " is not valid.");
    return get_valid_params_generator_map()[key]();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Register a method. The key for the method is determined by its class identifier.
  template <typename MethodToRegister,
            typename std::enable_if<std::is_base_of<MetaMethodBase, MethodToRegister>::value, void>::type>
  std::unique_ptr<MetaMethodFactory> register_new_method() {
    const std::string key = MethodToRegister::get_class_identifier();
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid_key(key), std::invalid_argument,
                               "The provided key " << key << " already exists.");
    get_instance_generator_map().insert(std::make_pair(key, MethodToRegister::create_new_instance));
    get_requirement_generator_map().insert(std::make_pair(key, MethodToRegister::get_part_requirements));
    get_valid_params_generator_map().insert(std::make_pair(key, MethodToRegister::get_valid_params));
  }

  /// \brief Generate a new instance of a registered \c MetaMethod.
  ///
  /// The registered \c MetaMethod accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a \c MetaMethod with this factory, use the
  /// provided \c register_new_method function.
  ///
  /// \param key [in] A key corresponding to a registered \c MetaMethod.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::unique_ptr<MetaMethodFactory> create_new_instance(const std::string& key,
                                                                const stk::mesh::BulkData* bulk_data_ptr,
                                                                const std::vector<*stk::mesh::Part>& part_ptr_vector,
                                                                const Teuchos::ParameterList& parameter_list) {
    return get_instance_generator_map(key)(bulk_data_ptr, part_ptr_vector, parameter_list);
  }
  //@}

 private:
  //! \name Typedefs
  //@{

  /// \brief A map from key to a function for generating a new \c MetaMethod.
  using InstanceGeneratorMap = std::map<std::string, NewMetaMethodGenerator>;

  /// \brief A map from key to a function for generating a \c MetaMethod's part requirements.
  using RequirementGeneratorMap = std::map<std::string, NewRequirementsGenerator>;

  /// \brief A map from key to a function for generating a \c MetaMethod's part default requirements.
  using DefaultParamsGeneratorMap = std::map<std::string, NewDefaultParamsGenerator>;
  //@}

  //! \name Attributes
  //@{
  static InstanceGeneratorMap& get_instance_generator_map() {
    // Static: One and the same instance for all function calls.
    static InstanceGeneratorMap instance_generator_map;
    return instance_generator_map;
  }

  static RequirementGeneratorMap& get_requirement_generator_map() {
    // Static: One and the same instance for all function calls.
    static RequirementGeneratorMap requirement_generator_map;
    return requirement_generator_map;
  }

  static DefaultParamsGeneratorMap& get_valid_params_generator_map() {
    // Static: One and the same instance for all function calls.
    static DefaultParamsGeneratorMap default_params_generator_map;
    return default_params_generator_map;
  }
  //@}

  //! \name Friends
  //@{

  /// \brief Every concrete MetaMethod that inherits from the MetaMethodRegistry will be added to this factory's
  /// registry. This process requires friendship <3.
  template <typename T>
  friend class MetaMethodRegistry<T>;
  //@}
};  // MetaMethodFactory

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_METAMETHODFACTORY_HPP_
