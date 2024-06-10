// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
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

//! \file RepeatedEntityCreationAndDestruction.cpp
/// \brief Performance test the repeated creation and destruction of entities.

// C++ core
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Kokkos_Core.hpp>                  // for Kokkos::initialize, Kokkos::finalize
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>       // for stk::mesh::BulkData
#include <stk_mesh/base/Comm.hpp>           // for comm_mesh_counts
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/MeshBuilder.hpp>    // for stk::mesh::MeshBuilder
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>          // for stk::mesh::EntityProc, EntityVector, etc
#include <stk_util/parallel/Parallel.hpp>   // for stk::parallel_machine_init, stk::parallel_machine_finalize

void verify_global_entity_count(std::vector<size_t> expected_total_num_entities, const stk::mesh::BulkData &bulk_data) {
  std::vector<size_t> entity_counts;
  stk::mesh::comm_mesh_counts(bulk_data, entity_counts);
  for (int i = 0; i < expected_total_num_entities.size(); i++) {
    assert(expected_total_num_entities[i] == entity_counts[i]);
  }
}

int main(int argc, char **argv) {
  // Initialize MPI and Kokkos
  // Note, we mitigate our interaction with MPI through STK's stk::ParallelMachine.
  // If STK is MPI enabled, then we're MPI enabled. As such, Mundy doesn't directly depend on or interact with MPI.
  // However, if tests are to be run in parallel, then TPL_ENABLE_MPI must be set to ON in the TriBITS configuration.

  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  /*
  The schematic for this test:
                        CONSTRAINT1
                       /     |     \
       SPRING1        /   SPHERE1   \       SPRING2
      /       \      /       |       \     /       \
  NODE1         NODE2      NODE3      NODE4         NODE5

  We divide this schematic over two processes as follows:
                      RANK0                            |                  RANK1
                                      CONSTRAINT1(G)   |              CONSTRAINT1(LO)
                                     /                 |             /      |        \
            SPRING1(LO)             /                  |            /   SPHERE1(LO)   \           SPRING2(LO)
           /           \           /                   |           /        |          \         /           \
  NODE1(LO)             NODE2(LO,S)                    |   NODE2(S)      NODE3(LO)      NODE4(LO)             NODE5(LO)
  LO: Locally owned
  G: Ghosted
  S: Shared
  */

  // Construct metaData and bulk data.
  const size_t spatial_dimension = 3;
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(spatial_dimension);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create();

  if (bulk_data_ptr->parallel_size() == 2) {
    // Setup the meta mesh.
    stk::mesh::MetaData &metaData = bulk_data_ptr->mesh_meta_data();
    metaData.use_simple_fields();

    stk::mesh::Part &linker_part = metaData.declare_part("linker", stk::topology::CONSTRAINT_RANK);
    stk::mesh::Part &particle_part = metaData.declare_part_with_topology("particle", stk::topology::PARTICLE);
    stk::mesh::Part &spring_part = metaData.declare_part_with_topology("spring", stk::topology::BEAM_2);
    metaData.commit();

    bulk_data_ptr->modification_begin();
    const int my_rank = bulk_data_ptr->parallel_rank();
    if (my_rank == 0) {
      stk::mesh::Entity spring1 = bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{&spring_part});
      stk::mesh::Entity node1 = bulk_data_ptr->declare_node(1);
      stk::mesh::Entity node2 = bulk_data_ptr->declare_node(2);
      bulk_data_ptr->declare_relation(spring1, node1, 0);
      bulk_data_ptr->declare_relation(spring1, node2, 1);
      bulk_data_ptr->add_node_sharing(node2, 1);

      std::vector<size_t> expected_total_num_entities_prior_to_end = {6 /*nodes*/, 0 /*edges*/, 0 /*faces*/,
                                                                      3 /*elements*/, 1 /*constraint*/};
      verify_global_entity_count(expected_total_num_entities_prior_to_end, *bulk_data_ptr);
      bulk_data_ptr->modification_end();

      std::vector<size_t> expected_total_num_entities_after_end = {5 /*nodes*/, 0 /*edges*/, 0 /*faces*/,
                                                                   3 /*elements*/, 1 /*constraint*/};
      verify_global_entity_count(expected_total_num_entities_after_end, *bulk_data_ptr);

      assert(bulk_data_ptr->num_nodes(spring1) == 2u);
      assert(node1 == bulk_data_ptr->begin_nodes(spring1)[0]);
      assert(node2 == bulk_data_ptr->begin_nodes(spring1)[1]);
    } else {
      stk::mesh::Entity spring2 = bulk_data_ptr->declare_element(2, stk::mesh::ConstPartVector{&spring_part});
      stk::mesh::Entity particle1 = bulk_data_ptr->declare_element(3, stk::mesh::ConstPartVector{&particle_part});
      stk::mesh::Entity linker1 = bulk_data_ptr->declare_constraint(1, stk::mesh::ConstPartVector{&linker_part});
      stk::mesh::Entity node2 = bulk_data_ptr->declare_node(2);
      stk::mesh::Entity node3 = bulk_data_ptr->declare_node(3);
      stk::mesh::Entity node4 = bulk_data_ptr->declare_node(4);
      stk::mesh::Entity node5 = bulk_data_ptr->declare_node(5);
      bulk_data_ptr->declare_relation(particle1, node3, 0);
      bulk_data_ptr->declare_relation(linker1, particle1, 0);
      bulk_data_ptr->declare_relation(linker1, node2, 0);
      bulk_data_ptr->declare_relation(linker1, node4, 1);
      bulk_data_ptr->declare_relation(spring2, node4, 0);
      bulk_data_ptr->declare_relation(spring2, node5, 1);

      bulk_data_ptr->add_node_sharing(node2, 0);

      std::vector<size_t> expected_total_num_entities_prior_to_end = {6 /*nodes*/, 0 /*edges*/, 0 /*faces*/,
                                                                      3 /*elements*/, 1 /*constraint*/};
      verify_global_entity_count(expected_total_num_entities_prior_to_end, *bulk_data_ptr);
      bulk_data_ptr->modification_end();

      std::vector<size_t> expected_total_num_entities_after_end = {5 /*nodes*/, 0 /*edges*/, 0 /*faces*/,
                                                                   3 /*elements*/, 1 /*constraint*/};
      verify_global_entity_count(expected_total_num_entities_after_end, *bulk_data_ptr);

      // Check the node connectivity.
      assert(bulk_data_ptr->num_nodes(spring2) == 2u);
      assert(node4 == bulk_data_ptr->begin_nodes(spring2)[0]);
      assert(node5 == bulk_data_ptr->begin_nodes(spring2)[1]);
      assert(bulk_data_ptr->num_nodes(particle1) == 1u);
      assert(node3 == bulk_data_ptr->begin_nodes(particle1)[0]);
      assert(bulk_data_ptr->num_nodes(linker1) == 2u);
      assert(node2 == bulk_data_ptr->begin_nodes(linker1)[0]);
      assert(node4 == bulk_data_ptr->begin_nodes(linker1)[1]);

      // Check the constraint connectivity.
      assert(bulk_data_ptr->num_elements(linker1) == 1u);
      assert(particle1 == bulk_data_ptr->begin_elements(linker1)[0]);
      assert(linker1 == bulk_data_ptr->begin(particle1, stk::topology::CONSTRAINT_RANK)[0]);
    }
  }

  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
