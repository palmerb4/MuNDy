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

// External libs
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>       // for stk::mesh::BulkData
#include <stk_mesh/base/Comm.hpp>           // for comm_mesh_counts
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/GetEntities.hpp>    // for stk::mesh::get_selected_entities
#include <stk_mesh/base/MeshBuilder.hpp>    // for stk::mesh::MeshBuilder
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>          // for stk::mesh::EntityProc, EntityVector, etc

// Mundy libs
#include <mundy_linkers/Linkers.hpp>   // for mundy::linkers::Linker and  mundy::linkers::declare_family_tree_relation
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData

namespace mundy {

namespace linkers {

namespace {

void verify_global_entity_count(std::vector<size_t> expected_total_num_entities, const stk::mesh::BulkData &bulk_data) {
  std::vector<size_t> entity_counts;
  stk::mesh::comm_mesh_counts(bulk_data, entity_counts);
  for (size_t i = 0; i < expected_total_num_entities.size(); i++) {
    EXPECT_EQ(expected_total_num_entities[i], entity_counts[i]);
  }
}

//! \name Linkers functionality unit tests
//@{

TEST(Linkers, STKProperlyHandlesConstraintRankSharingGhostingAndAura) {
  /* This test doesn't depend on MundyLinker. Instead, it tests the core functionality of STK that is necessary for
    MundyLinker to work properly.

  The schematic for this test:
                        CONSTRAINT1
                       /     |   |  \
       SPRING1        / SPHERE1  |   \       SPRING2
      /       \      /       |  /     \     /       \
  NODE1         NODE2       NODE3      NODE4         NODE5

  We divide this schematic over two processes as follows:
                      RANK0                            |                  RANK1
                                      CONSTRAINT1(G)   |              CONSTRAINT1(LO)
                                     /                 |             /     |      |  \
            SPRING1(LO)             /                  |            / SPHERE1(LO) |   \           SPRING2(LO)
           /           \           /                   |           /      |       /    \         /           \
  NODE1(LO)             NODE2(LO,S)                    |   NODE2(S)       NODE3(LO)     NODE4(LO)             NODE5(LO)
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

      // Check the node connectivity.
      EXPECT_EQ(bulk_data_ptr->num_nodes(spring1), 2u);
      EXPECT_EQ(node1, bulk_data_ptr->begin_nodes(spring1)[0]);
      EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(spring1)[1]);

      // Check ownership.
      EXPECT_EQ(bulk_data_ptr->parallel_size(), 2);
      EXPECT_EQ(bulk_data_ptr->parallel_rank(), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node2), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node1), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(spring1), 0);

      // Check sharing.
      stk::mesh::Part &shared_part = metaData.globally_shared_part();
      stk::mesh::EntityVector shared_nodes;
      stk::mesh::Selector shared_selector = shared_part;
      stk::mesh::get_selected_entities(shared_selector, bulk_data_ptr->buckets(stk::topology::NODE_RANK), shared_nodes);
      EXPECT_EQ(shared_nodes.size(), 1u);
      EXPECT_EQ(node2, shared_nodes[0]);

      // Check aura generation.
      stk::mesh::Part &aura_part = metaData.aura_part();
      stk::mesh::EntityVector aura_constraints;
      stk::mesh::Selector aura_selector = aura_part;
      stk::mesh::get_selected_entities(aura_selector, bulk_data_ptr->buckets(stk::topology::CONSTRAINT_RANK),
                                       aura_constraints);
      EXPECT_EQ(aura_constraints.size(), 1u);
      EXPECT_EQ(1, bulk_data_ptr->num_connectivity(node2, stk::topology::CONSTRAINT_RANK));
      EXPECT_EQ(aura_constraints[0], bulk_data_ptr->begin(node2, stk::topology::CONSTRAINT_RANK)[0]);
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
      bulk_data_ptr->declare_relation(linker1, node3, 1);
      bulk_data_ptr->declare_relation(linker1, node4, 2);
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
      EXPECT_EQ(bulk_data_ptr->num_nodes(spring2), 2u);
      EXPECT_EQ(node4, bulk_data_ptr->begin_nodes(spring2)[0]);
      EXPECT_EQ(node5, bulk_data_ptr->begin_nodes(spring2)[1]);
      EXPECT_EQ(bulk_data_ptr->num_nodes(particle1), 1u);
      EXPECT_EQ(node3, bulk_data_ptr->begin_nodes(particle1)[0]);
      EXPECT_EQ(bulk_data_ptr->num_nodes(linker1), 3u);
      EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(linker1)[0]);
      EXPECT_EQ(node3, bulk_data_ptr->begin_nodes(linker1)[1]);
      EXPECT_EQ(node4, bulk_data_ptr->begin_nodes(linker1)[2]);

      // Check the constraint connectivity.
      EXPECT_EQ(bulk_data_ptr->num_elements(linker1), 1u);
      EXPECT_EQ(particle1, bulk_data_ptr->begin_elements(linker1)[0]);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(particle1, stk::topology::CONSTRAINT_RANK)[0]);

      // Check ownership.
      EXPECT_EQ(bulk_data_ptr->parallel_size(), 2);
      EXPECT_EQ(bulk_data_ptr->parallel_rank(), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node2), 0);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node3), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node4), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node5), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(spring2), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(particle1), 1);
      EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(linker1), 1);

      // Check sharing by intersecting with shared part.
      stk::mesh::Part &shared_part = metaData.globally_shared_part();
      stk::mesh::EntityVector shared_nodes;
      stk::mesh::Selector shared_selector = shared_part;
      stk::mesh::get_selected_entities(shared_selector, bulk_data_ptr->buckets(stk::topology::NODE_RANK), shared_nodes);
      EXPECT_EQ(shared_nodes.size(), 1u);
      EXPECT_EQ(node2, shared_nodes[0]);
    }
  }
}

TEST(Linkers, FamilyTreeRelationGeneration) {
  /*
  The schematic for this test:
        CONSTRAINT1
      /    |   |    \
     |SPHERE1 SPHERE2|
      \    |   |    /
       NODE1   NODE2

  We divide this schematic over two processes as follows:
                RANK0            |      RANK1
            CONSTRAINT1(LO)      |  CONSTRAINT1(G)
      /         |   |         \  |       |        \
     | SPHERE1(LO)  SPHERE2(G) | |    SPHERE2(LO)  |
      \    |          |       /  |       |        /
        NODE1(LO)    NODE2(S)    |    NODE1(LO,S)
  LO: Locally owned
  G: Ghosted
  S: Shared
  */

  // Construct metaData and bulk data.
  const size_t spatial_dimension = 3;
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(spatial_dimension);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  // Setup the meta mesh.
  mundy::mesh::MetaData &metaData = bulk_data_ptr->mesh_meta_data();
  metaData.use_simple_fields();
  stk::mesh::Part &linker_part = metaData.declare_part("linker", stk::topology::CONSTRAINT_RANK);
  stk::mesh::Part &particle_part = metaData.declare_part_with_topology("particle", stk::topology::PARTICLE);

  metaData.commit();

  // Construct the mesh.
  bulk_data_ptr->modification_begin();
  const int my_rank = bulk_data_ptr->parallel_rank();

  // Both ranks will declare the linker. This might not work!
  if (my_rank == 0) {
    stk::mesh::Entity sphere1 = bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{&particle_part});
    stk::mesh::Entity node1 = bulk_data_ptr->declare_node(1);
    bulk_data_ptr->declare_relation(sphere1, node1, 0);

    stk::mesh::Ghosting &custom_ghosting = bulk_data_ptr->create_ghosting("custom_ghosting");
    bulk_data_ptr->change_ghosting(custom_ghosting, {std::make_pair(sphere1, 1)});
    bulk_data_ptr->modification_end();

    // Fetch the sphere and node from the other rank. If the ghosting worked, they should be valid.
    stk::mesh::Entity sphere2 = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 2);
    stk::mesh::Entity node2 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 2);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sphere2));
    ASSERT_TRUE(bulk_data_ptr->is_valid(node2));

    // Connect the linker to the spheres
    bulk_data_ptr->modification_begin();
    stk::mesh::Entity linker1 = bulk_data_ptr->declare_constraint(1, stk::mesh::ConstPartVector{&linker_part});
    ASSERT_NO_THROW(mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing(
        bulk_data_ptr.get(), linker1, sphere1, sphere2));
    bulk_data_ptr->modification_end();

    // Check the connectivity.
    EXPECT_EQ(bulk_data_ptr->num_nodes(sphere1), 1u);
    EXPECT_EQ(bulk_data_ptr->num_nodes(sphere2), 1u);
    EXPECT_EQ(node1, bulk_data_ptr->begin_nodes(sphere1)[0]);
    EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(sphere2)[0]);
    EXPECT_EQ(bulk_data_ptr->num_elements(linker1), 2u);
    EXPECT_EQ(sphere1, bulk_data_ptr->begin_elements(linker1)[0]);
    EXPECT_EQ(sphere2, bulk_data_ptr->begin_elements(linker1)[1]);
    EXPECT_EQ(linker1, bulk_data_ptr->begin(sphere1, stk::topology::CONSTRAINT_RANK)[0]);
    EXPECT_EQ(linker1, bulk_data_ptr->begin(sphere2, stk::topology::CONSTRAINT_RANK)[0]);
    EXPECT_EQ(bulk_data_ptr->num_nodes(linker1), 2u);
    EXPECT_EQ(node1, bulk_data_ptr->begin_nodes(linker1)[0]);
    EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(linker1)[1]);

    // Check ownership.
    EXPECT_EQ(bulk_data_ptr->parallel_size(), 2);
    EXPECT_EQ(bulk_data_ptr->parallel_rank(), 0);
    EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node1), 0);
    EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node2), 1);
    EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(sphere1), 0);
    EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(sphere2), 1);
    if (bulk_data_ptr->bucket(linker1).owned()) {
      std::cout << "Linker is owned on rank " << my_rank << std::endl;
    }

    // Check sharing.
    stk::mesh::Part &shared_part = metaData.globally_shared_part();
    stk::mesh::EntityVector shared_nodes;
    stk::mesh::Selector shared_selector = shared_part;
    stk::mesh::get_selected_entities(shared_selector, bulk_data_ptr->buckets(stk::topology::NODE_RANK), shared_nodes);
    EXPECT_EQ(shared_nodes.size(), 2u);
    EXPECT_EQ(node1, shared_nodes[0]);
    EXPECT_EQ(node2, shared_nodes[1]);

    // Check aura generation.
    stk::mesh::Part &aura_part = metaData.aura_part();
    stk::mesh::EntityVector aura_constraints;
    stk::mesh::Selector aura_selector = aura_part;
    stk::mesh::get_selected_entities(aura_selector, bulk_data_ptr->buckets(stk::topology::CONSTRAINT_RANK),
                                     aura_constraints);
    if (aura_constraints.size() == 1u) {
      std::cout << "Linker is in the aura on rank " << my_rank << std::endl;
    }
  } else {
    stk::mesh::Entity sphere2 = bulk_data_ptr->declare_element(2, stk::mesh::ConstPartVector{&particle_part});
    stk::mesh::Entity node2 = bulk_data_ptr->declare_node(2);
    bulk_data_ptr->declare_relation(sphere2, node2, 0);

    stk::mesh::Ghosting &custom_ghosting = bulk_data_ptr->create_ghosting("custom_ghosting");
    bulk_data_ptr->change_ghosting(custom_ghosting, {std::make_pair(sphere2, 0)});
    bulk_data_ptr->modification_end();

    // Fetch the sphere and node from the other rank. If the ghosting worked, they should be valid.
    stk::mesh::Entity sphere1 = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);
    stk::mesh::Entity node1 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 1);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sphere1));
    ASSERT_TRUE(bulk_data_ptr->is_valid(node2));

    // Connect the linker to the spheres
    bulk_data_ptr->modification_begin();
    stk::mesh::Entity linker1 = bulk_data_ptr->declare_constraint(1, stk::mesh::ConstPartVector{&linker_part});
    ASSERT_NO_THROW(mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing(
        bulk_data_ptr.get(), linker1, sphere1, sphere2));
    bulk_data_ptr->modification_end();

    // Check the connectivity.
    EXPECT_EQ(bulk_data_ptr->num_nodes(sphere1), 1u);
    EXPECT_EQ(bulk_data_ptr->num_nodes(sphere2), 1u);
    EXPECT_EQ(node1, bulk_data_ptr->begin_nodes(sphere1)[0]);
    EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(sphere2)[0]);
    EXPECT_EQ(bulk_data_ptr->num_elements(linker1), 2u);
    EXPECT_EQ(sphere1, bulk_data_ptr->begin_elements(linker1)[0]);
    EXPECT_EQ(sphere2, bulk_data_ptr->begin_elements(linker1)[1]);
    EXPECT_EQ(linker1, bulk_data_ptr->begin(sphere1, stk::topology::CONSTRAINT_RANK)[0]);
    EXPECT_EQ(linker1, bulk_data_ptr->begin(sphere2, stk::topology::CONSTRAINT_RANK)[0]);
    EXPECT_EQ(bulk_data_ptr->num_nodes(linker1), 2u);
    EXPECT_EQ(node1, bulk_data_ptr->begin_nodes(linker1)[0]);
    EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(linker1)[1]);

    // Check ownership.
    EXPECT_EQ(bulk_data_ptr->parallel_size(), 2);
    EXPECT_EQ(bulk_data_ptr->parallel_rank(), 1);
    EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node1), 0);
    EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(node2), 1);
    EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(sphere1), 0);
    EXPECT_EQ(bulk_data_ptr->parallel_owner_rank(sphere2), 1);
    if (bulk_data_ptr->bucket(linker1).owned()) {
      std::cout << "Linker is owned on rank " << my_rank << std::endl;
    }

    // Check sharing.
    stk::mesh::Part &shared_part = metaData.globally_shared_part();
    stk::mesh::EntityVector shared_nodes;
    stk::mesh::Selector shared_selector = shared_part;
    stk::mesh::get_selected_entities(shared_selector, bulk_data_ptr->buckets(stk::topology::NODE_RANK), shared_nodes);
    EXPECT_EQ(shared_nodes.size(), 2u);
    EXPECT_EQ(node1, shared_nodes[0]);
    EXPECT_EQ(node2, shared_nodes[1]);

    // Check aura generation.
    stk::mesh::Part &aura_part = metaData.aura_part();
    stk::mesh::EntityVector aura_constraints;
    stk::mesh::Selector aura_selector = aura_part;
    stk::mesh::get_selected_entities(aura_selector, bulk_data_ptr->buckets(stk::topology::CONSTRAINT_RANK),
                                     aura_constraints);
    if (aura_constraints.size() == 1u) {
      std::cout << "Linker is in the aura on rank " << my_rank << std::endl;
    }
  }
}

//@}

}  // namespace

}  // namespace linkers

}  // namespace mundy
