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
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements

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
/// \tparam PolymorphicBaseType A polymorphic base type shared by each registered class.
/// \tparam RegistrationType The type of each class's identifier.
/// \tparam RegistryIdentifier A template type used to create different independent instances of \c MetaFactory.
template <typename PolymorphicBaseType, typename RegistrationType = std::string,
          typename RegistryIdentifier = GlobalIdentifier>
class MetaFactory {
 public:
  //! \name Typedefs
  //@{

  using polymorphic_base_type = PolymorphicBaseType;
  using registration_type = RegistrationType;
  using registration_id_type = RegistryIdentifier;

  /// \brief A function type that takes a parameter list and produces a shared pointer to an object derived from
  /// class.
  using NewClassGenerator =
      std::function<std::shared_ptr<PolymorphicBaseType>(mundy::mesh::BulkData* const, const Teuchos::ParameterList&)>;

  /// \brief A function type that takes a parameter list and produces a vector of shared pointers to PartRequirements
  /// instances.
  using NewRequirementsGenerator =
      std::function<std::vector<std::shared_ptr<MeshRequirements>>(const Teuchos::ParameterList&)>;

  /// \brief A function type that produces a Teuchos::ParameterList instance.
  using NewDefaultParamsGenerator = std::function<Teuchos::ParameterList()>;
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
  /// based on parameters in the \c fixed_parameter_list but not the \c mutable_parameter_list.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A default fixed
  /// parameter list is accessible via \c get_valid_fixed_params.
  static std::vector<std::shared_ptr<MeshRequirements>> get_mesh_requirements(
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

  /// \brief Get the default mutable parameter list for a registered class.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \note This function does not cache its return value, so each time you call this function, a new
  /// \c Teuchos::ParameterList will be created. You can save the result yourself if you wish to reuse it.
  ///
  /// \param key [in] A key corresponding to a registered class.
  static Teuchos::ParameterList get_valid_mutable_params(const RegistrationType& key) {
    TEUCHOS_TEST_FOR_EXCEPTION(is_valid_key(key), std::invalid_argument,
                               "MetaFactory: The provided key " << key << " is not valid.");
    return get_valid_mutable_params_generator_map()[key]();
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
      get_valid_fixed_params_generator_map().erase(key);
      get_valid_mutable_params_generator_map().erase(key);
    }

    TEUCHOS_TEST_FOR_EXCEPTION(!is_valid_key(key), std::invalid_argument,
                               "MetaFactory: The provided key " << key << " already exists.");
    get_internal_keys().insert(key);
    get_instance_generator_map().insert(std::make_pair(key, ClassToRegister::static_create_new_instance));
    get_requirement_generator_map().insert(std::make_pair(key, ClassToRegister::static_get_part_requirements));
    get_valid_fixed_params_generator_map().insert(std::make_pair(key, ClassToRegister::static_get_valid_fixed_params));
    get_valid_mutable_params_generator_map().insert(
        std::make_pair(key, ClassToRegister::static_get_valid_mutable_params));
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
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(const RegistrationType& key,
                                                                  mundy::mesh::BulkData* const bulk_data_ptr,
                                                                  const Teuchos::ParameterList& fixed_parameter_list) {
    return get_instance_generator_map()[key](bulk_data_ptr, fixed_parameter_list);
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
  using DefaultParamsGeneratorMap = std::map<RegistrationType, NewDefaultParamsGenerator>;
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

  static DefaultParamsGeneratorMap& get_valid_fixed_params_generator_map() {
    // Static: One and the same instance for all function calls.
    static DefaultParamsGeneratorMap default_fixed_params_generator_map;
    return default_fixed_params_generator_map;
  }

  static DefaultParamsGeneratorMap& get_valid_mutable_params_generator_map() {
    // Static: One and the same instance for all function calls.
    static DefaultParamsGeneratorMap default_mutable_params_generator_map;
    return default_mutable_params_generator_map;
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
  friend class MetaRegistry;
  //@}
};  // MetaFactory

/// \name Type specializations for generating \c MetaMethods.
//@{

/// \brief Partial specialization for MetaMethods.
template <typename ReturnType, typename RegistrationType = std::string, typename RegistryIdentifier = GlobalIdentifier>
using MetaMethodFactory =
    MetaFactory<MetaMethodBase<ReturnType, RegistryIdentifier>, RegistrationType, RegistryIdentifier>;
//@}

/// \name Type specializations for generating \c MetaKernels.
//@{

/// \brief Partial specialization for \c MetaKernels.
template <typename ReturnType, typename RegistrationType = std::string, typename RegistryIdentifier = GlobalIdentifier>
using MetaKernelFactory =
    MetaFactory<MetaKernelBase<ReturnType, RegistryIdentifier>, RegistrationType, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaKernels, identified by a mundy multibody type.
template <typename ReturnType, typename RegistryIdentifier = GlobalIdentifier>
using MetaMultibodyKernelFactory = MetaKernelFactory<mundy::multibody::multibody_t, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaKernels, identified by an stk topology type.
template <typename ReturnType, typename RegistryIdentifier = GlobalIdentifier>
using MetaTopologyKernelFactory = MetaKernelFactory<stk::topology::topology_t, RegistryIdentifier>;
//@}

/// \name Type specializations for generating \c MetaKWayKernels.
//@{

/// \brief Partial specialization for \c MetaKWayKernel.
template <std::size_t K, typename ReturnType, typename RegistrationType = std::string,
          typename RegistryIdentifier = GlobalIdentifier>
using MetaKWayKernelFactory =
    MetaFactory<MetaKWayKernelBase<K, ReturnType, RegistryIdentifier>, RegistrationType, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaKWayKernels, identified by a mundy multibody type.
template <std::size_t K, typename ReturnType, typename RegistryIdentifier = GlobalIdentifier>
using MetaKWayMultibodyKernelFactory =
    MetaKWayKernelFactory<K, std::array<mundy::multibody::multibody_t, K>, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaKWayKernels, identified by an stk topology type.
template <std::size_t K, typename ReturnType, typename RegistryIdentifier = GlobalIdentifier>
using MetaKWayTopologyKernelFactory =
    MetaKWayKernelFactory<K, std::array<stk::topology::topology_t, K>, RegistryIdentifier>;
//@}

/// \name Type specializations for generating \c MetaTwoWayKernels.
//@{

/// \brief Partial specialization for \c MetaTwoWayKernels.
template <typename ReturnType, typename RegistrationType = std::string, typename RegistryIdentifier = GlobalIdentifier>
using MetaTwoWayKernelFactory =
    MetaFactory<MetaTwoWayKernelBase<ReturnType, RegistryIdentifier>, RegistrationType, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaTwoWayKernels, identified by two mundy multibody types.
template <typename ReturnType, typename RegistryIdentifier = GlobalIdentifier>
using MetaMultibodyTwoWayKernelFactory =
    MetaTwoWayKernelFactory<std::array<mundy::multibody::multibody_t, 2>, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaTwoWayKernels, identified by a two stk topology types.
template <typename ReturnType, typename RegistryIdentifier = GlobalIdentifier>
using MetaTopologyTwoWayKernelFactory =
    MetaTwoWayKernelFactory<std::array<mundy::multibody::multibody_t, 2>, RegistryIdentifier>;
//@}

/// \name Type specializations for generating \c MetaThreeWayKernels.
//@{

/// \brief Partial specialization for \c MetaThreeWayKernels.
template <typename ReturnType, typename RegistrationType = std::string, typename RegistryIdentifier = GlobalIdentifier>
using MetaThreeWayKernelFactory =
    MetaFactory<MetaThreeWayKernelBase<ReturnType, RegistryIdentifier>, RegistrationType, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaThreeWayKernels, identified by three mundy multibody types.
template <typename ReturnType, typename RegistryIdentifier = GlobalIdentifier>
using MetaMultibodyThreeWayKernelFactory =
    MetaThreeWayKernelFactory<std::array<mundy::multibody::multibody_t, 3>, RegistryIdentifier>;

/// \brief Partial specialization for \c MetaThreeWayKernels, identified by a three stk topology types.
template <typename ReturnType, typename RegistryIdentifier = GlobalIdentifier>
using MetaTopologyThreeWayKernelFactory =
    MetaThreeWayKernelFactory<std::array<mundy::multibody::multibody_t, 3>, RegistryIdentifier>;
//@}
}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAFACTORY_HPP_
