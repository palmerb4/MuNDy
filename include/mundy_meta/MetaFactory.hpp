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

#ifndef MUNDY_META_METAFACTORY_HPP_
#define MUNDY_META_METAFACTORY_HPP_

/// \file MetaFactory.hpp
/// \brief Declaration of the MetaFactory class

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
#include <stk_topology/topology.hpp>     // for stk::topology

// Mundy libs
#include <mundy_meta/MetaKernel.hpp>          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethod.hpp>          // for mundy::meta::MetaMethod
#include <mundy_meta/MetaPairwiseKernel.hpp>  // for mundy::meta::MetaPairwiseKernel
#include <mundy_meta/PartRequirements.hpp>    // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

/// \brief An empty struct to symbolize an unused template parameter.
struct GlobalIdentifier {};  // GlobalIdentifier

/// \class MetaFactory
/// \brief A factory containing generation routines for classes derived from \c HasMeshRequirementsAndIsRegisterable.
///
/// The goal of \c MetaFactory, as with most factories, is to provide an abstraction for case switches between
/// different methods. This factory is a bit different in that it always users to register new classes derived from
/// \c HasMeshRequirementsAndIsRegisterable and associate them with corresponding keys. These classes can then be
/// fetched based using their class identifier. Most importantly, it enables users to register their own derived classes
/// without modifying Mundy's source code.
///
/// It's important to note that the static members of this factory will be shared between any \c MetaFactory
/// with the same set of template parameters. As a result, we can create a new factory with its own set of registered
/// classes by simply changing the \c RegistryIdentifier. For methods that should be globally accessible, we offer the
/// default \c GlobalIdentifier.
///
/// \note Credit where credit is due: The design for this class originates from Andreas Zimmerer and his
/// self-registering types design. https://www.jibbow.com/posts/cpp-header-only-self-registering-types/
///
/// \tparam BaseType A polymorphic base type shared by each registered class.
/// \tparam RegistrationType The type of each class's identifier.
/// \tparam RegistryIdentifier A template type used to create different independent instances of \c MetaFactory.
template <typename BaseType, typename RegistrationType = std::string, typename RegistryIdentifier = GlobalIdentifier>
class MetaFactory {
 public:
  //! \name Typedefs
  //@{

  /// \brief A function type that takes a parameter list and produces a shared pointer to an object derived from
  /// class.
  using NewClassGenerator =
      std::function<std::shared_ptr<BaseType>(stk::mesh::BulkData* const, const Teuchos::ParameterList&)>;

  /// \brief A function type that takes a parameter list and produces a vector of shared pointers to PartRequirements
  /// instances.
  using NewRequirementsGenerator =
      std::function<std::vector<std::shared_ptr<PartRequirements>>(const Teuchos::ParameterList&)>;

  /// \brief A function type that produces a Teuchos::ParameterList instance.
  using NewDefaultParamsGenerator = std::function<Teuchos::ParameterList()>;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of classes this factory recognizes.
  static size_t get_number_of_subclasses() {
    return get_instance_generator_map().size();
  }

  /// \brief Get if the provided key is valid or not
  /// \param key [in] A key that may or may not correspond to a registered class.
  static bool is_valid_key(const RegistrationType& key) {
    return get_instance_generator_map().count(key) != 0;
  }

  /// \brief Get the requirements that this a registered class imposes upon each part.
  ///
  /// The set of part requirements returned by this function are meant to encode the assumptions made by this class
  /// with respect to the structure, topology, and fields of the STK mesh. These assumptions may vary
  /// based on parameters in the \c fixed_parameter_list but not the \c transient_parameter_list.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A default fixed
  /// parameter list is accessible via \c get_valid_fixed_params.
  static std::vector<std::shared_ptr<PartRequirements>> get_part_requirements(
      const RegistrationType& key, const Teuchos::ParameterList& fixed_parameter_list) {
    return get_requirement_generator_map()[key](fixed_parameter_list);
  }

  /// \brief Get the default fixed parameter list for a registered class.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \note This function does not cache its return value, so each time you call this function, a new
  /// \c Teuchos::ParameterList will be created. You can save the result yourself if you wish to reuse it.
  ///
  /// \param key [in] A key corresponding to a registered class.
  static Teuchos::ParameterList get_valid_fixed_params(const RegistrationType& key) {
    TEUCHOS_TEST_FOR_EXCEPTION(is_valid_key(key), std::invalid_argument,
                               "MetaFactory: The provided key " << key << " is not valid.");
    return get_valid_fixed_params_generator_map()[key]();
  }

  /// \brief Get the default transient parameter list for a registered class.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \note This function does not cache its return value, so each time you call this function, a new
  /// \c Teuchos::ParameterList will be created. You can save the result yourself if you wish to reuse it.
  ///
  /// \param key [in] A key corresponding to a registered class.
  static Teuchos::ParameterList get_valid_transient_params(const RegistrationType& key) {
    TEUCHOS_TEST_FOR_EXCEPTION(is_valid_key(key), std::invalid_argument,
                               "MetaFactory: The provided key " << key << " is not valid.");
    return get_valid_transient_params_generator_map()[key]();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Register a new class. The key for the class is determined by its class identifier.
  template <typename ClassToRegister>
  static void register_new_class() {
    const RegistrationType key = ClassToRegister::static_get_class_identifier();
    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid_key(key), std::invalid_argument,
                               "MetaFactory: The provided key " << key << " already exists.");
    get_instance_generator_map().insert(std::make_pair(key, ClassToRegister::static_create_new_instance));
    get_requirement_generator_map().insert(std::make_pair(key, ClassToRegister::static_get_part_requirements));
    get_valid_fixed_params_generator_map().insert(std::make_pair(key, ClassToRegister::static_get_valid_fixed_params));
    get_valid_transient_params_generator_map().insert(
        std::make_pair(key, ClassToRegister::static_get_valid_transient_params));
  }

  /// \brief Generate a new instance of a registered class.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  ///
  /// \param fixed_parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<BaseType> create_new_instance(const RegistrationType& key,
                                                       stk::mesh::BulkData* const bulk_data_ptr,
                                                       const Teuchos::ParameterList& fixed_parameter_list) {
    return get_instance_generator_map()[key](bulk_data_ptr, fixed_parameter_list);
  }
  //@}

 private:
  //! \name Typedefs
  //@{

  /// \brief A map from key to a function for generating a new class.
  using InstanceGeneratorMap = std::map<RegistrationType, NewClassGenerator>;

  /// \brief A map from key to a function for generating a class's part requirements.
  using RequirementGeneratorMap = std::map<RegistrationType, NewRequirementsGenerator>;

  /// \brief A map from key to a function for generating a class's part default requirements.
  using DefaultParamsGeneratorMap = std::map<RegistrationType, NewDefaultParamsGenerator>;
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

  static DefaultParamsGeneratorMap& get_valid_fixed_params_generator_map() {
    // Static: One and the same instance for all function calls.
    static DefaultParamsGeneratorMap default_fixed_params_generator_map;
    return default_fixed_params_generator_map;
  }

  static DefaultParamsGeneratorMap& get_valid_transient_params_generator_map() {
    // Static: One and the same instance for all function calls.
    static DefaultParamsGeneratorMap default_transient_params_generator_map;
    return default_transient_params_generator_map;
  }
  //@}

  //! \name Friends
  //@{

  /// \brief Every concrete class that inherits from the classRegistry will be added to this factory's
  /// registry. This process requires friendship <3.
  ///
  /// \note For devs: Unfortunately, "Friend declarations cannot refer to partial specializations," so there is no way
  /// to only have classRegistry with the same identifier be friends with this factory. Instead, ALL
  /// classRegistry are friends, including the ones we don't want. TODO(palmerb4): Find a workaround.
  template <typename, class, typename>
  friend class MetaMethodRegistry;
  //@}
};  // MetaFactory

/// \brief Partial specialization for MetaMethods.
template <typename RegistrationType = std::string, typename RegistryIdentifier = GlobalIdentifier>
using MetaMethodFactory = MetaFactory<MetaMethodBase<ReturnType>, RegistrationType, RegistryIdentifier>;

/// \brief Partial specialization for MetaKernels.
template <typename RegistrationType = std::string, typename RegistryIdentifier = GlobalIdentifier>
using MetaKernelFactory = MetaFactory<MetaKernelBase<ReturnType>, RegistrationType, RegistryIdentifier>;

/// \brief Partial specialization for MetaPairwiseKernels.
template <typename RegistrationType = std::string, typename RegistryIdentifier = GlobalIdentifier>
using MetaPairwiseKernelFactory = MetaFactory<MetaPairwiseKernelBase<ReturnType>, RegistrationType, RegistryIdentifier>;

/// \brief Partial specialization for MetaKernels, identified by a mundy multibody type.
template <typename RegistryIdentifier = GlobalIdentifier>
using MetaTopologyKernelFactory =
    MetaFactory<MetaKernelBase<ReturnType>, mundy::multibody::multibody_t, RegistryIdentifier>;

/// \brief Partial specialization for MetaPairwiseKernels, identified by a mundy multibody type.
template <typename RegistryIdentifier = GlobalIdentifier>
using MetaTopologyPairwiseKernelFactory =
    MetaFactory<MetaPairwiseKernelBase<ReturnType>, stk::multibody::multibody_t, RegistryIdentifier>;

/// \brief Partial specialization for MetaKernels, identified by an stk topology type.
template <typename RegistryIdentifier = GlobalIdentifier>
using MetaTopologyKernelFactory =
    MetaFactory<MetaKernelBase<ReturnType>, stk::topology::topology_t, RegistryIdentifier>;

/// \brief Partial specialization for MetaPairwiseKernels, identified by an stk topology type.
template <typename RegistryIdentifier = GlobalIdentifier>
using MetaTopologyPairwiseKernelFactory =
    MetaFactory<MetaPairwiseKernelBase<ReturnType>, stk::topology::topology_t, RegistryIdentifier>;

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAFACTORY_HPP_
