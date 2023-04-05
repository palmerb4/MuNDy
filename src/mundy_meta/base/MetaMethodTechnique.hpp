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

#ifndef MUNDY_META_METAMETHODTECHNIQUE_HPP_
#define MUNDY_META_METAMETHODTECHNIQUE_HPP_

/// \file MetaMethodTechnique.hpp
/// \brief Declaration of the MetaMethodTechnique class

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

/// \class MetaMethodTechnique
/// \brief \c MetaMethod is to a task, what \c MetaMethodTechnique is to a style of carrying out that task.
///
/// \note \c MetaMethodTechnique is identical to \c MetaMethod in form, but we chose to separate the names to emphasize
/// their distinct uses. Expected usage for creating a technique class and registering it with its parent
/// meta method's factory registry:
/// \c SomeTechnique : \c MetaMethodTechnique<SomeTechnique, ParentMetaMethod>.
///
/// The goal of \c MetaMethodTechnique is to wrap a function that acts on Mundy's multibody hierarchy with a class that
/// can output the assumptions that function with respect to the fields and structure of the hierarchy.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c MetaMethodTechnique
/// must implement the following static member functions
///   - \c details_get_part_requirements implementation of the \c get_part_requirements interface.
///   class.
///   - \c details_get_default_params implementation of the \c get_default_params interface.
///   - \c details_get_class_identifier implementation of the \c get_class_identifier interface.
///   - \c details_create_new_instance implementation of the \c create_new_instance interface.
///
/// \tparam A class derived from \c MetaMethodTechnique that implements the desired interface.
template <class DerivedMetaMethodTechnique>
using MetaMethodTechnique = MetaMethod<DerivedMetaMethodTechnique>;

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_METAMETHODTECHNIQUE_HPP_
