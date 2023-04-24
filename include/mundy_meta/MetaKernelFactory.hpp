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

#ifndef MUNDY_META_METAKERNELFACTORY_HPP_
#define MUNDY_META_METAKERNELFACTORY_HPP_

/// \file MetaKernelFactory.hpp
/// \brief Declaration of the MetaKernelFactory class

// C++ core libs
#include <functional>   // for std::function
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <utility>      // for std::make_pair
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
struct DefaultKernelIdentifier {};  // DefaultKernelIdentifier

/// \class MetaKernelFactory
/// \brief A factory containing generation routines for all of Mundy's \c MetaKernels.
///
/// The goal of \c MetaKernelFactory, as with most factories, is to provide an abstraction for case switches between
/// different methods. This factory is a bit different in that it always users to register new \c MetaKernels and
/// associate them with their corresponding keys. This allows a method to be created based on a string. Most
/// importantly, it enables users to add their own \c MetaKernels without modifying Mundy's source code.
///
/// It's important to note that the static members of this factory will be shared between any \c MetaKernelFactories
/// with the same template \c RegistryIdentifier.
///
/// \note This factory does not store an instance of \c MetaKernel; rather, it stores maps from a string to some of
/// \c MetaKernel's static member functions.
///
/// \note Credit where credit is due: The design for this class originates from Andreas Zimmerer and his
/// self-registering types design. https://www.jibbow.com/posts/cpp-header-only-self-registering-types/
///
/// \tparam RegistryIdentifier A template type used to create different independent instances of MetaKernelFactory.
template <typename RegistryIdentifier = DefaultKernelIdentifier>
class MetaKernelFactory {
 public:
  //! \name Typedefs
  //@{

  /// \brief A function type that takes a parameter list and produces a shared pointer to an object derived from
  /// \c MetaKernel.
  using NewMetaKernelGenerator =
      std::function<std::shared_ptr<MetaKernelBase>(stk::mesh::BulkData* const, const Teuchos::ParameterList&)>;

  /// \brief A function type that takes a parameter list and produces a PartRequirements instance.
  using NewRequirementsGenerator = std::function<PartRequirements>(const Teuchos::ParameterList&);

  /// \brief A function type that produces a Teuchos::ParameterList instance.
  using NewDefaultParamsGenerator = std::function<Teuchos::ParameterList()>;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of \c MetaKernel classes this factory recognizes.
  static size_t get_number_of_subclasses() {
    return get_instance_generator_map().size();
  }

  /// \brief Get if the provided key is valid or not
  /// \param key [in] A key that may or may not correspond to a registered \c MetaKernel.
  static bool is_valid_key(const std::string& key) {
    return get_instance_generator_map().count(key) != 0;
  }

  /// \brief Get the requirements that this a registered \c MetaKernel imposes upon each particle and/or constraint.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made a registered
  /// \c MetaKernel with respect to the parts, topology, and fields input into the \c execute function. These
  /// assumptions may vary based parameters in the \c parameter_list.
  ///
  /// The registered \c MetaKernel accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, \c is_valid_key(key) must return true. To register a \c MetaKernel with this factory, use the
  /// provided \c register_new_method function.
  ///
  /// \param key [in] A key corresponding to a registered \c MetaKernel.
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A default parameter list
  /// is accessible via \c get_valid_params.
  static std::shared_ptr<PartRequirements> get_part_requirements(const std::string& key,
                                                                 const Teuchos::ParameterList& parameter_list) {
    return get_instance_generator_map()[key](parameter_list);
  }

  /// \brief Get the default parameter list for a registered \c MetaKernel.
  ///
  /// The registered \c MetaKernel accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a \c MetaKernel with this factory, use the
  /// provided \c register_new_method function.
  ///
  /// \note This function does not cache its return value, so
  /// each time you call this function, a new \c Teuchos::ParameterList will be created. You can save the result
  /// yourself if you wish to reuse it.
  ///
  /// \param key [in] A key corresponding to a registered \c MetaKernel.
  static Teuchos::ParameterList get_valid_params(const std::string& key) {
    TEUCHOS_TEST_FOR_EXCEPTION(is_valid_key(key), std::invalid_argument,
                               "The provided key " << key << " is not valid.");
    return get_valid_params_generator_map()[key]();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Register a method. The key for the method is determined by its class identifier.
  template <typename KernelToRegister,
            std::enable_if_t<std::is_base_of<MetaKernelBase, KernelToRegister>::value, bool> = true>
  void register_new_kernel();

  /// \brief Generate a new instance of a registered \c MetaKernel.
  ///
  /// The registered \c MetaKernel accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, \c is_valid_key(key) must return true. To register a \c MetaKernel with this factory, use the
  /// provided \c register_new_method function.
  ///
  /// \param key [in] A key corresponding to a registered \c MetaKernel.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::shared_ptr<MetaKernelBase> create_new_instance(const std::string& key,
                                                             stk::mesh::BulkData* const bulk_data_ptr,
                                                             const Teuchos::ParameterList& parameter_list) {
    return get_instance_generator_map()[key](bulk_data_ptr, parameter_list);
  }
  //@}

 private:
  //! \name Typedefs
  //@{

  /// \brief A map from key to a function for generating a new \c MetaKernel.
  using InstanceGeneratorMap = std::map<std::string, NewMetaKernelGenerator>;

  /// \brief A map from key to a function for generating a \c MetaKernel's part requirements.
  using RequirementGeneratorMap = std::map<std::string, NewRequirementsGenerator>;

  /// \brief A map from key to a function for generating a \c MetaKernel's part default requirements.
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

  /// \brief Every concrete \c MetaKernel that inherits from the \c MetaKernelRegistry will be added to this factory's
  /// registry. This process requires friendship <3.
  ///
  /// For devs, the templating here is strategic such that only \c MetaKernelRegistry's with the same identifier should
  /// be friends with this factory.
  template <ReturnType SameReturnType, typename AnyKernel, RegistryIdentifier SameRegistryIdentifier,
            std::enable_if_t<std::is_base_of<MetaKernelBase<ReturnType>, DerivedMetaKernel>::value, bool>>
  friend class MetaKernelRegistry;
  //@}
};  // MetaKernelFactory

//! \name template implementations
//@{

template <typename KernelToRegister, std::enable_if_t<std::is_base_of<MetaKernelBase, KernelToRegister>::value, bool>>
void MetaKernelFactory::register_new_kernel() {
  const std::string key = KernelToRegister::get_class_identifier();
  TEUCHOS_TEST_FOR_EXCEPTION(!is_valid_key(key), std::invalid_argument,
                             "The provided key " << key << " already exists.");
  get_instance_generator_map().insert(std::make_pair(key, KernelToRegister::create_new_instance));
  get_requirement_generator_map().insert(std::make_pair(key, KernelToRegister::get_part_requirements));
  get_valid_params_generator_map().insert(std::make_pair(key, KernelToRegister::get_valid_params));
}
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAKERNELFACTORY_HPP_
