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

/// \file GroupOfConstraints.tpp
/// \brief Implementation of the methods of the GroupOfConstraints class

// clang-format off
#include <mpi.h>                                     // for MPI_COMM_WORLD, etc
#include <stddef.h>                                  // for size_t
#include <vector>                                    // for vector, etc
#include <random>                                    // for rand
#include <memory>                                    // for shared_ptr
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

namespace core {

// Constructors and destructor
//{
template <stk::topology GroupTopology, typename Scalar>
GroupOfConstraints<GroupTopology, Scalar>::GroupOfConstraints(const std::shared_ptr<stk::mesh::BulkData> &bulk_data_ptr,
                                                              const std::string &group_name)
    : GroupOfEntities(bulk_data_ptr, group_name),
      node_coord_field_(bulk_data_ptr_->mesh_meta_data().declare_field<Scalar>(stk::topology::NODE_RANK, "node_coord")),
      node_orientation_field_(
          bulk_data_ptr_->mesh_meta_data().declare_field<Scalar>(stk::topology::NODE_RANK, "node_orientation")),
      node_force_field_(bulk_data_ptr_->mesh_meta_data().declare_field<Scalar>(stk::topology::NODE_RANK, "node_force")),
      node_torque_field_(
          bulk_data_ptr_->mesh_meta_data().declare_field<Scalar>(stk::topology::NODE_RANK, "node_torque")),
      node_translational_velocity_field_(bulk_data_ptr_->mesh_meta_data().declare_field<Scalar>(
          stk::topology::NODE_RANK, "node_translational_velocity")),
      node_rotational_velocity_field_(
          bulk_data_ptr_->mesh_meta_data().declare_field<Scalar>(stk::topology::NODE_RANK, "node_rotational_velocity")),
{
  static_assert(std::std::is_floating_point_v<Scalar>, "Scalar must be a floating point type");

  // put the default fields on the group
  stk::mesh::put_field_on_mesh(node_coord_field_, group_part_, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_orientation_field_, group_part_, 4, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field_, group_part_, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_torque_field_, group_part_, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_translational_velocity_field_, group_part_, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_rotational_velocity_field_, group_part_, 3, nullptr);
}
//}

// Attributes
//{
template <stk::topology GroupTopology, typename Scalar>
FlagFieldType &GroupOfConstraints<GroupTopology, Scalar>::get_node_coord_field() {
  return node_coord_field_;
}

template <stk::topology GroupTopology, typename Scalar>
FlagFieldType &GroupOfConstraints<GroupTopology, Scalar>::get_node_orientation_field() {
  return node_orientation_field_;
}

template <stk::topology GroupTopology, typename Scalar>
FlagFieldType &GroupOfConstraints<GroupTopology, Scalar>::get_node_force_field() {
  return node_force_field_;
}

template <stk::topology GroupTopology, typename Scalar>
FlagFieldType &GroupOfConstraints<GroupTopology, Scalar>::get_node_torque_field() {
  return node_torque_field_;
}

template <stk::topology GroupTopology, typename Scalar>
FlagFieldType &GroupOfConstraints<GroupTopology, Scalar>::get_node_translational_velocity_field() {
  return node_translational_velocity_field_;
}

template <stk::topology GroupTopology, typename Scalar>
FlagFieldType &GroupOfConstraints<GroupTopology, Scalar>::get_node_rotational_velocity_field() {
  return node_rotational_velocity_field_;
}
//}
}  // namespace core

}  // namespace mundy