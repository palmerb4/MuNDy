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
#include <iostream>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaKWayKernel.hpp>    // for mundy::meta::MetaKWayKernel
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <stk_util/util/ReportHandler.hpp>  // for STK_ThrowAssertMsg

namespace mundy {

namespace meta {

/// \brief An empty struct used to define a global MetaFactory.
struct GlobalIdentifier {};  // GlobalIdentifier

/// \class MetaFactory
/// \brief A factory containing generation routines for classes that have mesh requirements and are registerable.
///
/// The goal of \c MetaFactory, as with most factories, is to provide an abstraction for case switches between
/// different methods. This factory is a bit different in that it always users to register new classes (that match the
/// desired interface) and associate them with corresponding keys. These classes can then be fetched based using their
/// registration id. Most importantly, this enables users to register their own derived classes without modifying
/// Mundy's source code.
///
/// It's important to note that the static members of this factory will be shared between any \c MetaFactory
/// with the same set of template parameters. As a result, we can create a new factory with its own set of registered
/// classes by simply changing the \c RegistryIdentifier. For methods that should be globally accessible, we offer a
/// GlobalMetaFactory type specialization.
///
/// Any class that wishes to be registered with this factory must must implement the following static interface. Don't
/// worry, if your class fails to meet one of these requirements, register_new_class will throw a human-readable compile
/// time error telling you which functions you need to implement/modify. The specific signatures of these functions are
/// given below.
///
/// \code{.cpp}
/// // The type of this class's registration id (e.g. std::string_view).
/// using RegistrationType = ...;
///
/// // The type of this class's polymorphic base type (e.g. MetaKernel).
/// using PolymorphicBaseType = ...;
///
/// // Get the requirements generator for this class. May be nullptr.
/// static std::shared_ptr<std::shared_ptr<MeshRequirements>> get_mesh_requirements();
///
/// // Validate the fixed params and set their defaults.
/// static void validate_fixed_parameters_and_set_defaults(Teuchos::ParameterList *const fixed_params_ptr);
///
/// // Validate the mutable params and set their defaults.
/// static void validate_mutable_parameters_and_set_defaults(Teuchos::ParameterList *const mutable_params_ptr);
///
/// // Get the new class generator for this class. May be nullptr.
/// static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData* const bulk_data_ptr, const
///                                                                 Teuchos::ParameterList& fixed_params);
///
/// // Get the registration id for this class.
/// static RegistrationType get_registration_id();
/// \endcode
///
/// \note Credit where credit is due: The design for this class originates from Andreas Zimmerer and his
/// self-registering types design (albeit with heavy modifications).
/// https://www.jibbow.com/posts/cpp-header-only-self-registering-types/
///
/// \tparam PolymorphicBaseType_t A polymorphic base type shared by each registered class.
/// \tparam RegistrationType_t The type to register each class with (defaults to std::string_view).
/// \tparam RegistryIdentifier_t A template type used to create different independent instances of \c MetaFactory.
template <typename PolymorphicBaseType_t, typename RegistryIdentifier_t, typename RegistrationType_t = std::string_view>
class MetaFactory {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = PolymorphicBaseType_t;
  using RegistryIdentifier = RegistryIdentifier_t;
  using RegistrationType = RegistrationType_t;

  /// \brief A function type that takes a parameter list and produces a shared pointer to an object derived from
  /// class.
  using NewClassGenerator =
      std::function<std::shared_ptr<PolymorphicBaseType>(mundy::mesh::BulkData* const, const Teuchos::ParameterList&)>;

  /// \brief A function type that takes a parameter list and produces a vector of shared pointers to PartRequirements
  /// instances.
  using NewRequirementsGenerator = std::function<std::shared_ptr<MeshRequirements>(const Teuchos::ParameterList&)>;

  /// \brief A function type that accepts a Teuchos::ParameterList pointer.
  using NewParamsValidatorGenerator = std::function<void(Teuchos::ParameterList* const)>;
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
  static std::shared_ptr<MeshRequirements> get_mesh_requirements(const RegistrationType& key,
                                                                 const Teuchos::ParameterList& fixed_params) {
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
                                                         Teuchos::ParameterList* const fixed_params_ptr) {
    MUNDY_THROW_ASSERT(is_valid_key(key), std::invalid_argument,
                       "MetaFactory: The provided key " << key << " is not valid.");
    get_validate_fixed_params_generator_map()[key](fixed_params_ptr);
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  static void validate_mutable_parameters_and_set_defaults(const RegistrationType& key,
                                                           Teuchos::ParameterList* const mutable_params_ptr) {
    MUNDY_THROW_ASSERT(is_valid_key(key), std::invalid_argument,
                       "MetaFactory: The provided key " << key << " is not valid.");
    get_validate_mutable_params_generator_map()[key](mutable_params_ptr);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Reset the factory to its initial state.
  ///
  /// This function removes all registered classes and clears all internal data structures.
  static void reset() {
    get_internal_keys().clear();
    get_instance_generator_map().clear();
    get_requirement_generator_map().clear();
    get_validate_fixed_params_generator_map().clear();
    get_validate_mutable_params_generator_map().clear();
  }

  /// \brief Register a new class. The key for the class is determined by its class identifier.
  template <typename ClassToRegister>
  static void register_new_class() {
    // Check that the ClassToRegister has the desired interface.
    using Checker = HasMeshRequirementsAndIsRegisterable<ClassToRegister>;
    static_assert(Checker::has_get_mesh_requirements,
                  "MetaFactory: The class to register doesn't have the correct get_mesh_requirements function.\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(Checker::has_validate_fixed_parameters_and_set_defaults,
                  "MetaFactory: The class to register doesn't have the correct "
                  "validate_fixed_parameters_and_set_defaults function.\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(Checker::has_validate_mutable_parameters_and_set_defaults,
                  "MetaFactory: The class to register doesn't have the correct "
                  "validate_mutable_parameters_and_set_defaults function.\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(Checker::has_registration_type,
                  "MetaFactory: The class to register doesn't have the correct RegistrationType type alias.\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(Checker::has_get_registration_id,
                  "MetaFactory: The class to register doesn't have the correct get_registration_id function.\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(Checker::has_polymorphic_base_type,
                  "MetaFactory: The class to register doesn't have a PolymorphicBaseType type alias .\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(Checker::has_create_new_instance,
                  "MetaFactory: The class to register doesn't have the correct create_new_instance function.\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(std::is_same_v<typename ClassToRegister::RegistrationType, RegistrationType>,
                  "MetaFactory: The class to register has a different RegistrationType type alias\n "
                  "than the RegistrationType of this factory.");

    // Register the class.
    const RegistrationType key = ClassToRegister::get_registration_id();

    std::cout << "MetaFactory: Registering class " << key << std::endl;

    MUNDY_THROW_ASSERT(!is_valid_key(key), std::invalid_argument,
                       "MetaFactory: The provided key " << key << " already exists.");
    get_internal_keys().insert(key);
    get_instance_generator_map().insert(std::make_pair(key, ClassToRegister::create_new_instance));
    get_requirement_generator_map().insert(std::make_pair(key, ClassToRegister::get_mesh_requirements));
    get_validate_fixed_params_generator_map().insert(
        std::make_pair(key, ClassToRegister::validate_fixed_parameters_and_set_defaults));
    get_validate_mutable_params_generator_map().insert(
        std::make_pair(key, ClassToRegister::validate_mutable_parameters_and_set_defaults));
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
};  // MetaFactory

/// \name Type specializations for generating \c MetaMethods.
//@{

/// \brief Partial specialization for \c MetaMethods.
template <typename ReturnType, typename RegistryIdentifier, typename RegistrationType = std::string_view>
using MetaMethodFactory = MetaFactory<MetaMethod<ReturnType>, RegistryIdentifier, RegistrationType>;

/// \brief Partial specialization for global \c MetaMethods.
template <typename ReturnType, typename RegistrationType = std::string_view>
using GlobalMetaMethodFactory = MetaMethodFactory<ReturnType, GlobalIdentifier, RegistrationType>;
//@}

/// \name Type specializations for generating \c MetaKernels.
//@{

/// \brief Partial specialization for \c MetaKernels.
template <typename ReturnType, typename RegistryIdentifier, typename RegistrationType = std::string_view>
using MetaKernelFactory = MetaFactory<MetaKernel<ReturnType>, RegistryIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaKernels, identified by a mundy multibody type.
template <typename ReturnType, typename RegistryIdentifier>
using MetaMultibodyKernelFactory = MetaKernelFactory<ReturnType, RegistryIdentifier, mundy::multibody::multibody_t>;

/// \brief Partial specialization for \c MetaKernels, identified by an stk topology type.
template <typename ReturnType, typename RegistryIdentifier>
using MetaTopologyKernelFactory = MetaKernelFactory<ReturnType, RegistryIdentifier, stk::topology::topology_t>;

/// \brief Partial specialization for global \c MetaKernels.
template <typename ReturnType, typename RegistrationType = std::string_view>
using GlobalMetaKernelFactory = MetaKernelFactory<ReturnType, GlobalIdentifier, RegistrationType>;

/// \brief Partial specialization for global \c MetaKernels, identified by a mundy multibody type.
template <typename ReturnType>
using GlobalMetaMultibodyKernelFactory = GlobalMetaKernelFactory<ReturnType, mundy::multibody::multibody_t>;

/// \brief Partial specialization for global \c MetaKernels, identified by an stk topology type.
template <typename ReturnType>
using GlobalMetaTopologyKernelFactory = GlobalMetaKernelFactory<ReturnType, stk::topology::topology_t>;
//@}

/// \name Type specializations for generating \c MetaKWayKernels.
//@{

/// \brief Partial specialization for \c MetaKWayKernels.
template <std::size_t K, typename ReturnType, typename RegistryIdentifier, typename RegistrationType = std::string_view>
using MetaKWayKernelFactory = MetaFactory<MetaKWayKernel<K, ReturnType>, RegistryIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaKWayKernels, identified by a mundy multibody type.
template <std::size_t K, typename ReturnType, typename RegistryIdentifier>
using MetaMultibodyKWayKernelFactory =
    MetaKWayKernelFactory<K, ReturnType, RegistryIdentifier, mundy::multibody::multibody_t>;

/// \brief Partial specialization for \c MetaKWayKernels, identified by an stk topology type.
template <std::size_t K, typename ReturnType, typename RegistryIdentifier>
using MetaTopologyKWayKernelFactory =
    MetaKWayKernelFactory<K, ReturnType, RegistryIdentifier, stk::topology::topology_t>;

/// \brief Partial specialization for global \c MetaKWayKernels.
template <std::size_t K, typename ReturnType, typename RegistrationType = std::string_view>
using GlobalMetaKWayKernelFactory = MetaKWayKernelFactory<K, ReturnType, GlobalIdentifier, RegistrationType>;

/// \brief Partial specialization for global \c MetaKWayKernels, identified by a mundy multibody type.
template <std::size_t K, typename ReturnType>
using GlobalMetaMultibodyKWayKernelFactory = GlobalMetaKWayKernelFactory<K, ReturnType, mundy::multibody::multibody_t>;

/// \brief Partial specialization for global \c MetaKWayKernels, identified by an stk topology type.
template <std::size_t K, typename ReturnType>
using GlobalMetaTopologyKWayKernelFactory = GlobalMetaKWayKernelFactory<K, ReturnType, stk::topology::topology_t>;
//@}

/// \name Type specializations for generating \c MetaTwoWayKernels.
//@{

/// \brief Partial specialization for \c MetaTwoWayKernels.
template <typename ReturnType, typename RegistryIdentifier, typename RegistrationType = std::string_view>
using MetaTwoWayKernelFactory = MetaKWayKernelFactory<2, ReturnType, RegistryIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaTwoWayKernels, identified by a mundy multibody type.
template <typename ReturnType, typename RegistryIdentifier>
using MetaMultibodyTwoWayKernelFactory = MetaMultibodyKWayKernelFactory<2, ReturnType, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaTwoWayKernels, identified by an stk topology type.
template <typename ReturnType, typename RegistryIdentifier>
using MetaTopologyTwoWayKernelFactory = MetaTopologyKWayKernelFactory<2, ReturnType, RegistryIdentifier>;

/// \brief Partial specialization for global \c MetaTwoWayKernels.
template <typename ReturnType, typename RegistrationType = std::string_view>
using GlobalMetaTwoWayKernelFactory = GlobalMetaKWayKernelFactory<2, ReturnType, RegistrationType>;

/// \brief Partial specialization for global \c MetaTwoWayKernels, identified by a mundy multibody type.
template <typename ReturnType>
using GlobalMetaMultibodyTwoWayKernelFactory = GlobalMetaMultibodyKWayKernelFactory<2, ReturnType>;

/// \brief Partial specialization for global \c MetaTwoWayKernels, identified by an stk topology type.
template <typename ReturnType>
using GlobalMetaTopologyTwoWayKernelFactory = GlobalMetaTopologyKWayKernelFactory<2, ReturnType>;
//@}

/// \name Type specializations for generating \c MetaThreeWayKernels.
//@{

/// \brief Partial specialization for \c MetaThreeWayKernels.
template <typename ReturnType, typename RegistryIdentifier, typename RegistrationType = std::string_view>
using MetaThreeWayKernelFactory = MetaKWayKernelFactory<2, ReturnType, RegistryIdentifier, RegistrationType>;

/// \brief Partial specialization for \c MetaThreeWayKernels, identified by a mundy multibody type.
template <typename ReturnType, typename RegistryIdentifier>
using MetaMultibodyThreeWayKernelFactory = MetaMultibodyKWayKernelFactory<2, ReturnType, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaThreeWayKernels, identified by an stk topology type.
template <typename ReturnType, typename RegistryIdentifier>
using MetaTopologyThreeWayKernelFactory = MetaTopologyKWayKernelFactory<2, ReturnType, RegistryIdentifier>;

/// \brief Partial specialization for global \c MetaThreeWayKernels.
template <typename ReturnType, typename RegistrationType = std::string_view>
using GlobalMetaThreeWayKernelFactory = GlobalMetaKWayKernelFactory<2, ReturnType, RegistrationType>;

/// \brief Partial specialization for global \c MetaThreeWayKernels, identified by a mundy multibody type.
template <typename ReturnType>
using GlobalMetaMultibodyThreeWayKernelFactory = GlobalMetaMultibodyKWayKernelFactory<2, ReturnType>;

/// \brief Partial specialization for global \c MetaThreeWayKernels, identified by an stk topology type.
template <typename ReturnType>
using GlobalMetaTopologyThreeWayKernelFactory = GlobalMetaTopologyKWayKernelFactory<2, ReturnType>;
//@}
}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAFACTORY_HPP_
