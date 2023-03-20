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

/// \file GroupOfEntities.tpp
/// \brief Implementation of the methods of the GroupOfEntities class

// clang-format off
#include <gtest/gtest.h>                             // for AssertHelper, etc
#include <mpi.h>                                     // for MPI_COMM_WORLD, etc
#include <stddef.h>                                  // for size_t
#include <vector>                                    // for vector, etc
#include <random>                                    // for rand
#include <memory>                                    // for shared_ptr
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
GroupOfEntities<GroupTopology, Scalar>::GroupOfEntities(const std::shared_ptr<stk::mesh::BulkData> &bulk_data_ptr,
                                                        const std::string &group_name)
    : bulk_data_ptr_(bulk_data_ptr),
      group_part_(bulk_data_ptr_->mesh_meta_data().declare_part_with_topology(group_name, GroupTopology)),
      new_entity_flag_field_(
          bulk_data_ptr_->mesh_meta_data().declare_field<bool>(stk::topology::NODE_RANK, "new_entity_flag")) {
  // put the default fields on the group
  stk::mesh::put_field_on_mesh(new_entity_flag_field_, group_part_, 1, nullptr);
}
//}

// Attributes
//{
template <stk::topology GroupTopology, typename Scalar>
FlagFieldType &GroupOfEntities<GroupTopology, Scalar>::get_group_part() {
  return group_part_;
}

template <stk::topology GroupTopology, typename Scalar>
FlagFieldType &GroupOfEntities<GroupTopology, Scalar>::get_new_entity_flag_field() {
  return new_entity_flag_field_;
}
//}

// Pre-commit setup routines
//{
template <stk::topology GroupTopology, typename Scalar>
template <stk::topology SubGroupTopology, typename SubGroupScalar>
void GroupOfEntities<GroupTopology, Scalar>::declare_subgroup(
    const GroupOfEntities<SubGroupTopology, SubGroupScalar> &subgroup) {
  // declare the subgroup's part a subset of our part
  // declare_part_subset enforces topology agreement and field compatability
  stk::mesh::declare_part_subset(group_part_, subgroup.get_group_part());
}

template <class field_type>
field_type &GroupOfEntities<GroupTopology, Scalar>::put_field_on_entire_group(
    const field_type &field, const unsigned int field_dimension,
    const typename stk::mesh::FieldTraits<field_type>::data_type *init_value) {
  stk::mesh::put_field_on_mesh(field, group_part_, field_dimension, init_value);
}
//}

// Post-commit modification routines
//{
template <stk::topology GroupTopology, typename Scalar>
stk::mesh::Selector GroupOfEntities<GroupTopology, Scalar>::generate_new_entities_in_group(
    const size_t num_new_entities, const bool generate_and_attach_nodes) {
  // count the number of entities of each rank that need requested
  std::vector<size_t> num_requests_per_rank(bulk_data_ptr_->mesh_meta_data().entity_rank_count(), 0);
  num_requests_per_rank[GroupTopology.rank()] += num_new_entities;

  const unsigned int num_nodes_per_entity = GroupOfEntities.num_nodes();
  const size_t num_nodes_requested = generate_and_attach_nodes ? num_new_entities * num_nodes_per_entity : 0;
  num_requests_per_rank[stk::topology::NODE_RANK] += num_nodes_requested;

  // generate the new entities
  // For example, if num_requests_per_rank = { 0, 4,  8} then this will requests 0 entites of rank 0, 4 entites of rank
  // 1, and 8 entites of rank 2. In this case, the result is requested_entities = {4 entites of rank 1, 8 entites of
  // rank 2}
  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data_ptr_->generate_new_entities(num_requests_per_rank, requested_entities);

  // associate each entity with a single part
  // change_entity_parts expects a vector of pointers to parts
  std::vector<stk::mesh::Part *> part_vector{&group_part_};

  // set topologies and downward relations of new entities
  for (int i = 0; i < num_particles_local; i++) {
    // the elements should be associated with a topology before they are connected to their nodes/edges
    stk::mesh::Entity entity_i = requested_entities[num_nodes_requested + i];
    bulk_data_ptr_->change_entity_parts(entity_i, part_vector);

    if (generate_and_attach_nodes) {
      // attach each node
      for (int j = 0; j < GroupOfEntities.num_nodes(); j++) {
        bulk_data_ptr_->declare_relation(entity_i, requested_entities[i * num_nodes_per_entity + j], j);
      }
    }
  }
}

//}

}  // namespace core

}  // namespace mundy