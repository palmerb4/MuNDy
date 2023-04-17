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

#ifndef MUNDY_META_METAMETHODREGISTRY_HPP_
#define MUNDY_META_METAMETHODREGISTRY_HPP_

/// \file MetaMethodRegistry.hpp
/// \brief Declaration of the MetaMethodRegistry class

// clang-format off
#include <gtest/gtest.h>                             // for AssertHelper, etc
#include <mpi.h>                                     // for MPI_COMM_WORLD, etc
#include <stddef.h>                                  // for size_t
#include <vector>                                    // for vector, etc
#include <random>                                    // for rand
#include <memory>                                    // for shared_ptr
#include <string>                                    // for string
#include <type_traits>                               // for static_assert
#include <stk_math/StkVector.hpp>                    // for Vec
#include <stk_mesh/base/BulkData.hpp>                // for BulkData
#include <stk_mesh/base/MetaData.hpp>                // for MetaData
#include <stk_mesh/base/GetEntities.hpp>             // for count_selected_entities
#include <stk_mesh/base/Types.hpp>                   // for EntityVector, etc
#include <stk_mesh/base/Ghosting.hpp>                // for create_ghosting
#include <stk_io/WriteMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_search/SearchMethod.hpp>               // for KDTREE
#include <stk_search/CoarseSearch.hpp>               // for coarse_search
#include <stk_search/BoundingBox.hpp>                // for Sphere, Box, tec.
#include <stk_balance/balance.hpp>                   // for balanceStkMesh
#include <stk_util/parallel/Parallel.hpp>            // for ParallelMachine
#include <stk_util/environment/WallTime.hpp>         // for wall_time
#include <stk_util/environment/perf_util.hpp>
#include <stk_unit_test_utils/BuildMesh.hpp>
#include "stk_util/util/ReportHandler.hpp"           // for ThrowAssert, etc
// clang-format on

namespace mundy {

namespace meta {

/// \class MetaMethodRegistry
/// \brief A class for registering \c MetaMethods within \c MetaMethodFactory.
///
/// All classes derived from \c MetaMethod, which wish to be registered within the \c MetaMethodFactory should inherit
/// from this class where the template parameter is the derived type itself (follows the Curiously Recurring Template
/// Pattern).
///
/// \tparam DerivedMetaMethod A class derived from \c MetaMethod.
/// \tparam RegistryIdentifier A template type used to create different independent instances of MetaMethodFactory.
template <class DerivedMetaMethod, typename RegistryIdentifier = DefaultMethodIdentifier,
          typename std::enable_if<std::is_base_of<MetaMethodBase, DerivedMetaMethod>::value, void>::type>
struct MetaMethodRegistry {
  //! \name Actions
  //@{

  /// @brief Register DerivedMetaMethod with the MetaMethodFactory.
  ///
  /// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
  /// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
  static inline bool register_type() {
    MetaMethodFactory::register_new_method<DerivedMetaMethod>();
    return true;
  }
  //@}

  //! \name Member variables
  //@{

  /// @brief A flag for if the given type has been registered with the \c MetaMethodFactory or not.
  static const bool is_registered;
  //@}
};  // MetaMethodRegistry

/// @brief Perform the registration.
///
/// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
/// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
template <class T>
const bool MetaMethodRegistry<T>::is_registered = ShapeInterface<T>::register_type();

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_METAMETHODREGISTRY_HPP_
