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

#ifndef MUNDY_META_METAREGISTRY_HPP_
#define MUNDY_META_METAREGISTRY_HPP_

/// \file MetaRegistry.hpp
/// \brief Declaration of the MetaRegistry class

// C++ core libs
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <utility>      // for std::pair

// Mundy libs
#include <mundy_meta/MetaFactory.hpp>  // for mundy::meta::MetaMethodFactory
#include <mundy_meta/MetaKernel.hpp>   // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethod.hpp>   // for mundy::meta::MetaMethod

namespace mundy {

namespace meta {

/// \class MetaRegistry
/// \brief A class for registering \c MetaMethods within \c MetaMethodFactory.
///
/// All classes derived from \c MetaMethod, which wish to be registered within the \c MetaMethodFactory should inherit
/// from this class where the template parameter is the derived type itself (follows the Curiously Recurring Template
/// Pattern).
///
/// \tparam BaseType A polymorphic base type shared by each registered class.
/// \tparam ClassToRegister A class derived from \c MetaMethod.
/// \tparam RegistrationType The type of each class's identifier.
/// \tparam RegistryIdentifier A template type used to create different independent instances of MetaMethodFactory.
template <typename BaseType, class ClassToRegister, typename RegistryIdentifier,
          typename RegistrationType = std::string, bool overwrite_existing = false>
struct MetaRegistry {
  //! \name Actions
  //@{

  /// @brief Register \c ClassToRegister with the \c MetaMethodFactory.
  ///
  /// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
  /// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
  static inline bool register_type() {
    MetaMethodFactory<BaseType, RegistryIdentifier, RegistrationType>::template register_new_method<ClassToRegister>(
        overwrite_existing);
    return true;
  }
  //@}

  //! \name Member variables
  //@{

  /// @brief A flag for if the given type has been registered with the \c MetaMethodFactory or not.
  static const bool is_registered;
  //@}
};  // MetaRegistry

/// @brief Perform the static registration.
///
/// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
/// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
template <typename BaseType, class ClassToRegister, typename RegistryIdentifier,
          typename RegistrationType, bool overwrite_existing>
const bool MetaRegistry<BaseType, ClassToRegister, RegistryIdentifier, RegistrationType,
                        overwrite_existing>::is_registered =
    MetaRegistry<BaseType, ClassToRegister, RegistryIdentifier, RegistrationType, overwrite_existing>::register_type();

//! \name Partial Specializations
//@{

/// \brief Partial specialization for global classes.
template <typename BaseType, class ClassToRegister, typename RegistrationType = std::string,
          bool overwrite_existing = false>
using GlobalMetaRegistry =
    MetaRegistry<BaseType, ClassToRegister, GlobalIdentifier, RegistrationType, overwrite_existing>;

/// \brief Partial specialization for MetaMethods.
template <typename ReturnType, class ClassToRegister, typename RegistryIdentifier,
          typename RegistrationType = std::string, bool overwrite_existing = false>
using MetaMethodRegistry =
    MetaRegistry<MetaMethodBase<ReturnType>, ClassToRegister, RegistryIdentifier, RegistrationType, overwrite_existing>;

/// \brief Partial specialization for MetaMethods.
template <typename ReturnType, class ClassToRegister, typename RegistrationType = std::string,
          bool overwrite_existing = false>
using GlobalMetaMethodRegistry =
    GlobalMetaRegistry<MetaMethodBase<ReturnType>, ClassToRegister, RegistrationType, overwrite_existing>;

/// \brief Partial specialization for MetaKernels.
template <typename ReturnType, class ClassToRegister, typename RegistryIdentifier,
          typename RegistrationType = std::string, bool overwrite_existing = false>
using MetaKernelRegistry =
    MetaRegistry<MetaKernelBase<ReturnType>, ClassToRegister, RegistryIdentifier, RegistrationType, overwrite_existing>;

/// \brief Partial specialization for MetaKernels.
template <typename ReturnType, class ClassToRegister, typename RegistrationType = std::string,
          bool overwrite_existing = false>
using GlobalMetaKernelRegistry =
    GlobalMetaRegistry<MetaKernelBase<ReturnType>, ClassToRegister, RegistrationType, overwrite_existing>;

/// \brief Partial specialization for MetaTwoWayKernels.
template <typename ReturnType, class ClassToRegister, typename RegistryIdentifier,
          typename RegistrationType = std::string, bool overwrite_existing = false>
using MetaTwoWayKernelRegistry = MetaRegistry<MetaTwoWayKernelBase<ReturnType>, ClassToRegister, RegistryIdentifier,
                                              RegistrationType, overwrite_existing>;

/// \brief Partial specialization for MetaTwoWayKernels.
template <typename ReturnType, class ClassToRegister, typename RegistrationType = std::string,
          bool overwrite_existing = false>
using GlobalMetaTwoWayKernelRegistry =
    GlobalMetaRegistry<MetaTwoWayKernelBase<ReturnType>, ClassToRegister, RegistrationType, overwrite_existing>;

/// \brief Partial specialization for MetaKernels, identified by a mundy multibody type.
template <typename ReturnType, class ClassToRegister, typename RegistryIdentifier, bool overwrite_existing = false>
using MetaMultibodyKernelRegistry = MetaKernelRegistry<ReturnType, mundy::multibody::multibody_t, ClassToRegister,
                                                       RegistryIdentifier, overwrite_existing>;

/// \brief Partial specialization for MetaKernels, identified by a mundy multibody type.
template <typename ReturnType, class ClassToRegister, bool overwrite_existing = false>
using GlobalMetaMultibodyKernelRegistry =
    GlobalMetaKernelRegistry<ReturnType, mundy::multibody::multibody_t, ClassToRegister, overwrite_existing>;

/// \brief Partial specialization for MetaTwoWayKernels, identified by a mundy multibody type.
/// To make a new key use:
///     auto key = std::make_pair(multibody_t1, multibody_t2)
/// This key can then be used like any other key.
template <typename ReturnType, class ClassToRegister, typename RegistryIdentifier, bool overwrite_existing = false>
using MetaMultibodyTwoWayKernelRegistry =
    MetaTwoWayKernelRegistry<ReturnType, std::pair<mundy::multibody::multibody_t, mundy::multibody::multibody_t>,
                             ClassToRegister, RegistryIdentifier, overwrite_existing>;

/// \brief Partial specialization for MetaTwoWayKernels, identified by a mundy multibody type.
/// To make a new key use:
///     auto key = std::make_pair(multibody_t1, multibody_t2)
/// This key can then be used like any other key.
template <typename ReturnType, class ClassToRegister, bool overwrite_existing = false>
using GlobalMetaMultibodyTwoWayKernelRegistry =
    GlobalMetaTwoWayKernelRegistry<ReturnType, std::pair<mundy::multibody::multibody_t, mundy::multibody::multibody_t>,
                                   ClassToRegister, overwrite_existing>;

/// \brief Partial specialization for MetaKernels, identified by an stk topology type.
template <typename ReturnType, class ClassToRegister, typename RegistryIdentifier, bool overwrite_existing = false>
using MetaTopologyKernelRegistry =
    MetaKernelRegistry<ReturnType, stk::topology::topology_t, ClassToRegister, RegistryIdentifier, overwrite_existing>;

/// \brief Partial specialization for MetaKernels, identified by an stk topology type.
template <typename ReturnType, class ClassToRegister, bool overwrite_existing = false>
using GlobalMetaTopologyKernelRegistry =
    GlobalMetaKernelRegistry<ReturnType, stk::topology::topology_t, ClassToRegister, overwrite_existing>;

/// \brief Partial specialization for MetaTwoWayKernels, identified by a pair of stk topology types.
/// To make a new key use:
///     auto key = std::make_pair(topology_t1, topology_t2)
/// This key can then be used like any other key.
template <typename ReturnType, class ClassToRegister, typename RegistryIdentifier, bool overwrite_existing = false>
using MetaTopologyTwoWayKernelRegistry =
    MetaTwoWayKernelRegistry<ReturnType, std::pair<stk::topology::topology_t, stk::topology::topology_t>,
                             ClassToRegister, RegistryIdentifier, overwrite_existing>;

/// \brief Partial specialization for MetaTwoWayKernels, identified by a pair of stk topology types.
/// To make a new key use:
///     auto key = std::make_pair(topology_t1, topology_t2)
/// This key can then be used like any other key.
template <typename ReturnType, class ClassToRegister, bool overwrite_existing = false>
using GlobalMetaTopologyTwoWayKernelRegistry =
    GlobalMetaTwoWayKernelRegistry<ReturnType, std::pair<stk::topology::topology_t, stk::topology::topology_t>,
                                   ClassToRegister, overwrite_existing>;
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAREGISTRY_HPP_
