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

#ifndef MUNDY_METHODS_COMPUTEAABB_HPP_
#define MUNDY_METHODS_COMPUTEAABB_HPP_

/// \file ComputeAABB.hpp
/// \brief Declaration of the ComputeAABB class

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

namespace methods {

/// \class ComputeAABB
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeAABB : public MetaMethod<ComputeAABB>, public MetaMethodRegistry<ComputeAABB> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  ComputeAABB();
  //@}

  run(const stk::mesh::BulkData *bulk_data_ptr, const stk::mesh::Part &part, const std::string &multibody_name,
      const stk::util::ParameterList &parameter_list) {
    // create and run a ComputeAABB variant corresponding to the provided multibody type name
    MetaMethodFactory<ComputeAABB>::create_new_instance(multibody_name, parameter_list).run(bulk_data_ptr, part);
  }

  static std::unique_ptr<PartParams> get_part_requirements(const std::string &multibody_name,
                                                           const stk::util::ParameterList &parameter_list) {
    return MetaMethodFactory<ComputeAABB>::get_part_requirements(multibody_name, parameter_list);
  }
}

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTEAABB_HPP_
