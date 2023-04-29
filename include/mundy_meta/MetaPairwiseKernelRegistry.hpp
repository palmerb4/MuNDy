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

#ifndef MUNDY_META_METAPAIRWISEKERNELREGISTRY_HPP_
#define MUNDY_META_METAPAIRWISEKERNELREGISTRY_HPP_

/// \file MetaPairwiseKernelRegistry.hpp
/// \brief Declaration of the MetaPairwiseKernelRegistry class

// C++ core libs
#include <type_traits>  // for std::enable_if, std::is_base_of

// Mundy libs
#include <mundy_meta/MetaPairwiseKernel.hpp>         // for mundy::meta::MetaPairwiseKernel
#include <mundy_meta/MetaPairwiseKernelFactory.hpp>  // for mundy::meta::MetaPairwiseKernelFactory

namespace mundy {

namespace meta {

/// \class MetaPairwiseKernelRegistry
/// \brief A class for registering \c MetaPairwiseKernels within \c MetaPairwiseKernelFactory.
///
/// All classes derived from \c MetaPairwiseKernel, which wish to be registered within the \c MetaPairwiseKernelFactory should inherit
/// from this class where the template parameter is the derived type itself (follows the Curiously Recurring Template
/// Pattern).
///
/// \tparam DerivedMetaPairwiseKernel A class derived from \c MetaPairwiseKernel.
/// \tparam RegistryIdentifier A template type used to create different independent instances of MetaPairwiseKernelFactory.
template <typename ReturnType, class DerivedMetaPairwiseKernel, typename RegistryIdentifier = DefaultKernelIdentifier>
struct MetaPairwiseKernelRegistry {
  //! \name Actions
  //@{

  /// @brief Register \c DerivedMetaPairwiseKernel with the \c MetaPairwiseKernelFactory.
  ///
  /// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
  /// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
  static inline bool register_type() {
    MetaPairwiseKernelFactory<ReturnType, RegistryIdentifier>::register_new_method<DerivedMetaPairwiseKernel>();
    return true;
  }
  //@}

  //! \name Member variables
  //@{

  /// @brief A flag for if the given type has been registered with the \c MetaPairwiseKernelFactory or not.
  static const bool is_registered;
  //@}
};  // MetaPairwiseKernelRegistry

/// @brief Perform the static registration.
///
/// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
/// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
///
/// \tparam DerivedMetaPairwiseKernel A class derived from \c MetaPairwiseKernel.
template <typename ReturnType, class DerivedMetaPairwiseKernel, typename RegistryIdentifier>
const bool MetaPairwiseKernelRegistry<ReturnType, DerivedMetaPairwiseKernel, RegistryIdentifier>::is_registered =
    MetaPairwiseKernelRegistry<ReturnType, DerivedMetaPairwiseKernel, RegistryIdentifier>::register_type();

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAPAIRWISEKERNELREGISTRY_HPP_
