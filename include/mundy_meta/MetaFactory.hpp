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
#include <set>          // for std::set
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <utility>      // for std::make_pair
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part
#include <stk_topology/topology.hpp>     // for stk::topology

// Mundy libs
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaKWayKernel.hpp>    // for mundy::meta::MetaKWayKernel
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod

namespace mundy {

namespace meta {

/// \brief An empty struct used to define a global MetaFactory.
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
/// classes by simply changing the \c RegistryIdentifier. For methods that should be globally accessible, we offer a
/// GlobalMetaFactory type specialization.
///
/// \note Credit where credit is due: The design for this class originates from Andreas Zimmerer and his
/// self-registering types design. https://www.jibbow.com/posts/cpp-header-only-self-registering-types/
///
/// \tparam PolymorphicBaseType_t A polymorphic base type shared by each registered class.
/// \tparam RegistrationType_t The type of each class's identifier.
/// \tparam RegistryIdentifier_t A template type used to create different independent instances of \c MetaFactory.
template <typename PolymorphicBaseType_t, typename RegistryIdentifier_t, typename RegistrationType_t = std::string>
class MetaFactory {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = PolymorphicBaseType_t;
  using RegistrationType = RegistrationType_t;
  using RegistryIdentifier = RegistryIdentifier_t;

  /// \brief A function type that takes a parameter list and produces a shared pointer to an object derived from
  /// class.
  using NewClassGenerator =
      std::function<std::shared_ptr<PolymorphicBaseType>(mundy::mesh::BulkData* const, const Teuchos::ParameterList&)>;

  /// \brief A function type that takes a parameter list and produces a vector of shared pointers to PartRequirements
  /// instances.
  using NewRequirementsGenerator = std::function<std::shared_ptr<MeshRequirements>(const Teuchos::ParameterList&)>;

  /// \brief A function type that accepts a Teuchos::ParameterList pointer.
  using NewParamsValidatorGenerator = std::function<void(Teuchos::ParameterList const *)>;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of classes this factory recognizes.
  static size_t num_registered_classes() {
    return get_internal_keys().size();
  }

  /// \brief Get the set of all registered keys.
  static std::set<RegistrationType> get_keys() {
    return get_internal_keys();
  }

  /// \brief Get if the provided key is valid or not.
  /// \param key [in] A key that may or may not correspond to a registered class.
  static bool is_valid_key(const RegistrationType& key) {
    return get_instance_generator_map().count(key) != 0;
  }

  /// \brief Get the requirements that this a registered class imposes upon each part.
  ///
  /// The set of part requirements returned by this function are meant to encode the assumptions made by this class
  /// with respect to the structure, topology, and fields of the STK mesh. These assumptions may vary
  /// based on parameters in the \c fixed_params but not the \c mutable_params.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A default fixed
  /// parameter list is accessible via \c get_valid_fixed_params.
  static std::vector<std::shared_ptr<MeshRequirements>> get_mesh_requirements(
      const RegistrationType& key, const Teuchos::ParameterList& fixed_params) {
    return get_requirement_generator_map()[key](fixed_params);
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  static void validate_fixed_parameters_and_set_defaults(const RegistrationType& key,
                                                         Teuchos::ParameterList const* fixed_params_ptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(is_valid_key(key), std::invalid_argument,
                               "MetaFactory: The provided key " << key << " is not valid.");
    return get_validate_fixed_params_generator_map()[key](fixed_params_ptr);
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  static void validate_mutable_parameters_and_set_defaults(const RegistrationType& key,
                                                           Teuchos::ParameterList const* mutable_params_ptr) {
    TEUCHOS_TEST_FOR_EXCEPTION(is_valid_key(key), std::invalid_argument,
                               "MetaFactory: The provided key " << key << " is not valid.");
    return get_validate_mutable_params_generator_map()[key](mutable_params_ptr);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Register a new class. The key for the class is determined by its class identifier.
  template <typename ClassToRegister>
  static void register_new_class(const bool overwrite_existing = false) {
    const RegistrationType key = ClassToRegister::static_get_class_identifier();
    if (overwrite_existing) {
      get_internal_keys().erase(key);
      get_instance_generator_map().erase(key);
      get_requirement_generator_map().erase(key);
      get_validate_fixed_params_generator_map().erase(key);
      get_validate_mutable_params_generator_map().erase(key);
    }

    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid_key(key), std::invalid_argument,
                               "MetaFactory: The provided key " << key << " already exists.");
    get_internal_keys().insert(key);
    get_instance_generator_map().insert(std::make_pair(key, ClassToRegister::static_create_new_instance));
    get_requirement_generator_map().insert(std::make_pair(key, ClassToRegister::static_get_mesh_requirements));
    get_validate_fixed_params_generator_map().insert(
        std::make_pair(key, ClassToRegister::static_validate_fixed_parameters_and_set_defaults));
    get_validate_mutable_params_generator_map().insert(
        std::make_pair(key, ClassToRegister::static_validate_mutable_parameters_and_set_defaults));
  }

  /// \brief Generate a new instance of a registered class.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  ///
  /// \param fixed_params [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(const RegistrationType& key,
                                                                  mundy::mesh::BulkData* const bulk_data_ptr,
                                                                  const Teuchos::ParameterList& fixed_params) {
    return get_instance_generator_map()[key](bulk_data_ptr, fixed_params);
  }
  //@}

 private:
  //! \name Typedefs
  //@{

  /// \brief A set of keys.
  using SetOfKeys = std::set<RegistrationType>;

  /// \brief A map from key to a function for generating a new class.
  using InstanceGeneratorMap = std::map<RegistrationType, NewClassGenerator>;

  /// \brief A map from key to a function for generating a class's part requirements.
  using RequirementGeneratorMap = std::map<RegistrationType, NewRequirementsGenerator>;

  /// \brief A map from key to a function for generating a class's part default requirements.
  using ParamsValidatorGeneratorMap = std::map<RegistrationType, NewParamsValidatorGenerator>;
  //@}

  //! \name Internal getters
  //@{
  static SetOfKeys& get_internal_keys() {
    // Static: One and the same instance for all function calls.
    static SetOfKeys keys;
    return keys;
  }

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

  static ParamsValidatorGeneratorMap& get_validate_fixed_params_generator_map() {
    // Static: One and the same instance for all function calls.
    static ParamsValidatorGeneratorMap fixed_params_validator_generator_map;
    return fixed_params_validator_generator_map;
  }

  static ParamsValidatorGeneratorMap& get_validate_mutable_params_generator_map() {
    // Static: One and the same instance for all function calls.
    static ParamsValidatorGeneratorMap mutable_params_validator_generator_map;
    return mutable_params_validator_generator_map;
  }
  //@}

  //! \name Friends
  //@{

  /// \brief Every concrete class that inherits from the MetaRegistry will be added to this factory's
  /// registry. This process requires friendship <3.
  ///
  /// \note For devs: Unfortunately, "Friend declarations cannot refer to partial specializations," so there is no way
  /// to only have MetaRegistry with the same identifier be friends with this factory. Instead, ALL
  /// MetaRegistry are friends, including the ones we don't want. TODO(palmerb4): Find a workaround.
  template <typename, class, typename, typename, bool>
  friend class MetaRegistry;
  //@}
};  // MetaFactory

/// \name Type specializations for generating \c MetaMethods.
//@{

/// \brief Partial specialization for \c MetaMethods.
template <typename ReturnType, typename RegistryIdentifier, typename RegistrationType = std::string>
using MetaMethodFactory =
    MetaFactory<MetaMethodBase<ReturnType, RegistryIdentifier>, RegistryIdentifier, RegistrationType>;

/// \brief Partial specialization for global \c MetaMethods.
template <typename ReturnType, typename RegistrationType = std::string>
using GlobalMetaMethodFactory =
    MetaFactory<MetaMethodBase<ReturnType, GlobalIdentifier>, GlobalIdentifier, RegistrationType>;
//@}

/// \name Type specializations for generating \c MetaKernels.
//@{

/// \brief Partial specialization for \c MetaKernels.
template <typename ReturnType, typename RegistryIdentifier, typename RegistrationType = std::string>
using MetaKernelFactory =
    MetaFactory<MetaKernelBase<ReturnType, RegistryIdentifier>, RegistryIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaKernels, identified by a mundy multibody type.
template <typename ReturnType, typename RegistryIdentifier>
using MetaMultibodyKernelFactory = MetaKernelFactory<mundy::multibody::multibody_t, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaKernels, identified by an stk topology type.
template <typename ReturnType, typename RegistryIdentifier>
using MetaTopologyKernelFactory = MetaKernelFactory<stk::topology::topology_t, RegistryIdentifier>;

/// \brief Partial specialization for global \c MetaKernels.
template <typename ReturnType, typename RegistrationType = std::string>
using GlobalMetaKernelFactory =
    MetaFactory<MetaKernelBase<ReturnType, GlobalIdentifier>, GlobalIdentifier, RegistrationType>;

/// \brief Partial specialization for global \c MetaKernels, identified by a mundy multibody type.
template <typename ReturnType>
using GlobalMetaMultibodyKernelFactory = MetaKernelFactory<mundy::multibody::multibody_t, GlobalIdentifier>;

/// \brief Partial specialization for global \c MetaKernels, identified by an stk topology type.
template <typename ReturnType>
using GlobalMetaTopologyKernelFactory = MetaKernelFactory<stk::topology::topology_t, GlobalIdentifier>;
//@}

/// \name Type specializations for generating \c MetaKWayKernels.
//@{

/// \brief Partial specialization for \c MetaKWayKernel.
template <std::size_t K, typename ReturnType, typename RegistryIdentifier, typename RegistrationType = std::string>
using MetaKWayKernelFactory =
    MetaFactory<MetaKWayKernelBase<K, ReturnType, RegistryIdentifier>, RegistryIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaKWayKernels, identified by a mundy multibody type.
template <std::size_t K, typename ReturnType, typename RegistryIdentifier>
using MetaKWayMultibodyKernelFactory =
    MetaKWayKernelFactory<K, std::array<mundy::multibody::multibody_t, K>, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaKWayKernels, identified by an stk topology type.
template <std::size_t K, typename ReturnType, typename RegistryIdentifier>
using MetaKWayTopologyKernelFactory =
    MetaKWayKernelFactory<K, std::array<stk::topology::topology_t, K>, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaKWayKernel.
template <std::size_t K, typename ReturnType, typename RegistrationType = std::string>
using GlobalMetaKWayKernelFactory =
    MetaFactory<MetaKWayKernelBase<K, ReturnType, GlobalIdentifier>, GlobalIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaKWayKernels, identified by a mundy multibody type.
template <std::size_t K, typename ReturnType>
using GlobalMetaKWayMultibodyKernelFactory =
    MetaKWayKernelFactory<K, std::array<mundy::multibody::multibody_t, K>, GlobalIdentifier>;

/// \brief Partial specialization for \c MetaKWayKernels, identified by an stk topology type.
template <std::size_t K, typename ReturnType>
using GlobalMetaKWayTopologyKernelFactory =
    MetaKWayKernelFactory<K, std::array<stk::topology::topology_t, K>, GlobalIdentifier>;
//@}

/// \name Type specializations for generating \c MetaTwoWayKernels.
//@{

/// \brief Partial specialization for \c MetaTwoWayKernels.
template <typename ReturnType, typename RegistryIdentifier, typename RegistrationType = std::string>
using MetaTwoWayKernelFactory =
    MetaFactory<MetaTwoWayKernelBase<ReturnType, RegistryIdentifier>, RegistryIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaTwoWayKernels, identified by two mundy multibody types.
template <typename ReturnType, typename RegistryIdentifier>
using MetaMultibodyTwoWayKernelFactory =
    MetaTwoWayKernelFactory<std::array<mundy::multibody::multibody_t, 2>, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaTwoWayKernels, identified by a two stk topology types.
template <typename ReturnType, typename RegistryIdentifier>
using MetaTopologyTwoWayKernelFactory =
    MetaTwoWayKernelFactory<std::array<mundy::multibody::multibody_t, 2>, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaTwoWayKernels.
template <typename ReturnType, typename RegistrationType = std::string>
using GlobalMetaTwoWayKernelFactory =
    MetaFactory<MetaTwoWayKernelBase<ReturnType, GlobalIdentifier>, GlobalIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaTwoWayKernels, identified by two mundy multibody types.
template <typename ReturnType>
using GlobalMetaMultibodyTwoWayKernelFactory =
    MetaTwoWayKernelFactory<std::array<mundy::multibody::multibody_t, 2>, GlobalIdentifier>;

/// \brief Partial specialization for \c MetaTwoWayKernels, identified by a two stk topology types.
template <typename ReturnType>
using GlobalMetaTopologyTwoWayKernelFactory =
    MetaTwoWayKernelFactory<std::array<mundy::multibody::multibody_t, 2>, GlobalIdentifier>;
//@}

/// \name Type specializations for generating \c MetaThreeWayKernels.
//@{

/// \brief Partial specialization for \c MetaThreeWayKernels.
template <typename ReturnType, typename RegistryIdentifier, typename RegistrationType = std::string>
using MetaThreeWayKernelFactory =
    MetaFactory<MetaThreeWayKernelBase<ReturnType, RegistryIdentifier>, RegistryIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaThreeWayKernels, identified by three mundy multibody types.
template <typename ReturnType, typename RegistryIdentifier>
using MetaMultibodyThreeWayKernelFactory =
    MetaThreeWayKernelFactory<std::array<mundy::multibody::multibody_t, 3>, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaThreeWayKernels, identified by a three stk topology types.
template <typename ReturnType, typename RegistryIdentifier>
using MetaTopologyThreeWayKernelFactory =
    MetaThreeWayKernelFactory<std::array<mundy::multibody::multibody_t, 3>, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaThreeWayKernels.
template <typename ReturnType, typename RegistrationType = std::string>
using GlobalMetaThreeWayKernelFactory =
    MetaFactory<MetaThreeWayKernelBase<ReturnType, GlobalIdentifier>, GlobalIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaThreeWayKernels, identified by three mundy multibody types.
template <typename ReturnType>
using GlobalMetaMultibodyThreeWayKernelFactory =
    MetaThreeWayKernelFactory<std::array<mundy::multibody::multibody_t, 3>, GlobalIdentifier>;

/// \brief Partial specialization for \c MetaThreeWayKernels, identified by a three stk topology types.
template <typename ReturnType>
using GlobalMetaTopologyThreeWayKernelFactory =
    MetaThreeWayKernelFactory<std::array<mundy::multibody::multibody_t, 3>, GlobalIdentifier>;
//@}
}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAFACTORY_HPP_
