// @HEADER
// **********************************************************************************************************************
//
//                                          MuNDy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
//                                                 Author: Bryce Palmer
//
// MuNDy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// MuNDy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with MuNDy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_METHODS_GHOSTNEIGHBORS_HPP_
#define MUNDY_METHODS_GHOSTNEIGHBORS_HPP_

/// \file GhostNeighbors.hpp
/// \brief Declaration of the GhostNeighbors class

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

/// \class GhostNeighbors
/// \brief A collection of entities, their sub-groups, and their associated fields.
class GhostNeighbors : MetaMethod {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor using a parameter list.
  ///
  /// \param bulk_data_ptr [in] A pointer to a larger <tt>BulkData</tt> with (potentially) multiple groups
  /// \param parameter_list [in] The input parameters. See the discription below for parameter options
  ///
  GhostNeighbors(const std::shared_ptr<stk::mesh::BulkData> &bulk_data_ptr,
              const stk::util::ParameterList &parameter_list);

  //@}
  //! \name Attributes
  //@{

  /// \brief Get the requirements that this MetaMethod imposes upon each particle and/or constraint.
  ///
  /// \note It is important to note that these requirements encode the assumptions made by this method with respect to
  /// the topology and fields of each multibody object. As such, assumptions may vary based on values passed to the
  /// MetaMethod's constructor.
  std::map<mundy::multibody, std::unique_ptr<PartParams>> get_multibody_part_requirements();
  //@}

  //! \name Actions
  //@{

  /// \brief Run the wrapped functioon
  virtual void run() = 0;
  //@}
}

//! \name template implementations
//@{

// Constructors and destructor
//{
template <stk::topology GroupTopology, typename Scalar>
GhostNeighbors<GroupTopology, Scalar>::GhostNeighbors(const std::shared_ptr<stk::mesh::BulkData> &bulk_data_ptr,
                                                const stk::util::ParameterList &parameter_list) {
  static_assert(std::std::is_floating_point_v<Scalar>, "Scalar must be a floating point type");

  // enable io for the group part
  stk::io::put_io_part_attribute(group_part_);

  // put the default fields on the group
  stk::mesh::put_field_on_mesh(new_entity_flag_field_, group_part_, 1, nullptr);
}
//}

// Attributes
//{
std::map<mundy::multibody, std::unique_ptr<PartParams>>
GhostNeighbors<GroupTopology, Scalar>::get_multibody_part_requirements() const {
  return group_part_;
}
//}
//@}

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_GHOSTNEIGHBORS_HPP_
