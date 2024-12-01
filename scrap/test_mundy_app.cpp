// C++ core libs
#include <gtest/gtest.h>  // for AssertHelper, EXPECT_EQ, etc

#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs

#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>       // for stk::mesh::BulkData
#include <stk_mesh/base/Comm.hpp>           // for comm_mesh_counts
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/MeshBuilder.hpp>    // for stk::mesh::MeshBuilder
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>          // for stk::mesh::EntityProc, EntityVector, etc

void verify_global_entity_count(std::vector<size_t> expected_total_num_entities, const stk::mesh::BulkData &bulk_data) {
  std::vector<size_t> entity_counts;
  stk::mesh::comm_mesh_counts(bulk_data, entity_counts);
  for (int i = 0; i < expected_total_num_entities.size(); i++) {
    EXPECT_EQ(expected_total_num_entities[i], entity_counts[i]);
  }
}

int main(int argc, char **argv) {
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "MPI_Init failed." << std::endl;
    return -1;
  }

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

      EXPECT_EQ(bulk_data_ptr->num_nodes(spring1), 2u);
      EXPECT_EQ(node1, bulk_data_ptr->begin_nodes(spring1)[0]);
      EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(spring1)[1]);
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
      EXPECT_EQ(bulk_data_ptr->num_nodes(spring2), 2u);
      EXPECT_EQ(node4, bulk_data_ptr->begin_nodes(spring2)[0]);
      EXPECT_EQ(node5, bulk_data_ptr->begin_nodes(spring2)[1]);
      EXPECT_EQ(bulk_data_ptr->num_nodes(particle1), 1u);
      EXPECT_EQ(node3, bulk_data_ptr->begin_nodes(particle1)[0]);
      EXPECT_EQ(bulk_data_ptr->num_nodes(linker1), 2u);
      EXPECT_EQ(node2, bulk_data_ptr->begin_nodes(linker1)[0]);
      EXPECT_EQ(node4, bulk_data_ptr->begin_nodes(linker1)[1]);

      // Check the constraint connectivity.
      EXPECT_EQ(bulk_data_ptr->num_elements(linker1), 1u);
      EXPECT_EQ(particle1, bulk_data_ptr->begin_elements(linker1)[0]);
      EXPECT_EQ(linker1, bulk_data_ptr->begin(particle1, stk::topology::CONSTRAINT_RANK)[0]);
    }
  }

  MPI_Finalize();
  return 0;
}