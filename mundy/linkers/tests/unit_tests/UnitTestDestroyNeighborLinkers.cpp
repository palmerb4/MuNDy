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

// External libs
#include <gtest/gtest.h>      // for TEST, ASSERT_NO_THROW, etc
#include <openrand/philox.h>  // for openrand::Philox

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
#include <stk_mesh/base/FieldParallel.hpp>  // for stk:::mesh::communicate_field_data
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/GetEntities.hpp>    // for stk::mesh::get_selected_entities
#include <stk_mesh/base/MeshBuilder.hpp>    // for stk::mesh::MeshBuilder
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>          // for stk::mesh::EntityProc, EntityVector, etc

// Mundy libs
#include <MundyLinker_config.hpp>                    // for HAVE_MUNDYLINKER_MUNDYSHAPES
#include <mundy_linkers/DestroyNeighborLinkers.hpp>   // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/GenerateNeighborLinkers.hpp>  // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/Linkers.hpp>  // for mundy::linkers::Linker and  mundy::linkers::declare_family_tree_relation
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>            // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>      // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaFactory.hpp>  // for mundy::meta::MetaMethodFactory and mundy::meta::HasMeshRequirementsAndIsRegisterable
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements

namespace mundy {

namespace linkers {

namespace {

//! \name GenerateNeighborLinkers functionality unit tests
//@{

#ifdef HAVE_MUNDYLINKER_MUNDYSHAPES

TEST(GenerateNeighborLinkers, PerformsNeighborLinkerDestructionCorrectlyForSpheresSimple) {
  /* Outline
    For this test, we generate a 1 sphere per process of equal radii along a line with overlap, compute their AABBs, and
    connect them with linkers. For example, for 3 processes, we would have 3 spheres along the x-axis. Process 0
    would own sphere 0, process 1 would own sphere 1, and process 2 would own sphere 2. Process 0 would generate a
    linker between spheres 0 and 1 and process 1 would generate a linker between spheres 1 and 2.

    We will run the destroy distant neighbor linkers routine on the linkers and validate that none are deleted since
    they all overlap. We'll then move the sphere on process 0 far from the others and rerun the routine. We'll validate
    that the linker between spheres 0 and 1 is deleted.
  */

  // Free variables
  const double overlap = 0.1;

  // Create an instance of DestroyNeighborLinkers based on committed mesh that meets both of their
  // default requirements.
  Teuchos::ParameterList generate_neighbor_linkers_fixed_params = Teuchos::ParameterList();

  Teuchos::ParameterList destroy_neighbor_linkers_fixed_params = Teuchos::ParameterList();
  destroy_neighbor_linkers_fixed_params.set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS")
      .sublist("DESTROY_DISTANT_NEIGHBORS")
      .set<Teuchos::Array<std::string>>("valid_entity_part_names",
                                        Teuchos::tuple<std::string>(std::string("NEIGHBOR_LINKERS")))
      .set<Teuchos::Array<std::string>>("valid_connected_source_and_target_part_names",
                                        Teuchos::tuple<std::string>(std::string("SPHERES")))
      .set("linker_destroy_flag_field_name", "LINKER_DESTROY_FLAG")
      .set("element_aabb_field_name", "ELEMENT_AABB");

  auto [generate_neighbor_linkers_ptr, destroy_neighbor_linkers_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<GenerateNeighborLinkers,
                                                                                        DestroyNeighborLinkers>(
          std::array{generate_neighbor_linkers_fixed_params, destroy_neighbor_linkers_fixed_params});
  ASSERT_TRUE(bulk_data_ptr != nullptr) << "bulk_data_ptr cannot be null";
  ASSERT_TRUE(generate_neighbor_linkers_ptr != nullptr) << "generate_neighbor_linkers_ptr cannot be null";
  ASSERT_TRUE(destroy_neighbor_linkers_ptr != nullptr) << "destroy_neighbor_linkers_ptr cannot be null";
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr) << "meta_data_ptr cannot be null";

  // Fetch the required fields.
  stk::mesh::Field<double> *element_aabb_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_AABB");
  ASSERT_TRUE(element_aabb_field_ptr != nullptr) << "element_aabb_field_ptr cannot be null";

  // Fetch the requested parts.
  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
  stk::mesh::Part *neighbor_linkers_part_ptr = meta_data_ptr->get_part("NEIGHBOR_LINKERS");
  ASSERT_TRUE(spheres_part_ptr != nullptr) << "spheres_part_ptr cannot be null";
  ASSERT_TRUE(neighbor_linkers_part_ptr != nullptr) << "neighbor_linkers_part_ptr cannot be null";

  // Add 1 sphere and 1 node to the mesh per process using the process rank + 1 as the sphere and node's ID.
  bulk_data_ptr->modification_begin();
  const int process_rank = bulk_data_ptr->parallel_rank();
  stk::mesh::Entity node = bulk_data_ptr->declare_node(process_rank + 1);
  stk::mesh::Entity sphere =
      bulk_data_ptr->declare_element(process_rank + 1, stk::mesh::ConstPartVector{spheres_part_ptr});
  bulk_data_ptr->declare_relation(sphere, node, 0);
  bulk_data_ptr->modification_end();

  // Set the sphere's AABB
  {
    double *element_aabb = stk::mesh::field_data(*element_aabb_field_ptr, sphere);
    element_aabb[0] = process_rank - 1 - overlap;
    element_aabb[1] = -1.0;
    element_aabb[2] = -1.0;
    element_aabb[3] = process_rank + 1 + overlap;
    element_aabb[4] = 1.0;
    element_aabb[5] = 1.0;
  }

  // At this point we have 1 sphere per process. We will now generate the neighbor linkers between the spheres.
  generate_neighbor_linkers_ptr->execute(*spheres_part_ptr, *spheres_part_ptr);

  // Get the total number of spheres and linkers. Must be called parallel synchronously.
  {
    std::vector<size_t> entity_counts;
    stk::mesh::comm_mesh_counts(*bulk_data_ptr, entity_counts);
    const size_t total_num_spheres = entity_counts[stk::topology::ELEMENT_RANK];
    const size_t total_num_linkers = entity_counts[stk::topology::CONSTRAINT_RANK];
    ASSERT_EQ(total_num_spheres, bulk_data_ptr->parallel_size());
    EXPECT_EQ(total_num_linkers, bulk_data_ptr->parallel_size() - 1);
  }

  // Attempt to destroy the distant neighbor linkers. None should be destroyed since they all overlap.
  destroy_neighbor_linkers_ptr->execute(*neighbor_linkers_part_ptr);
  {
    std::vector<size_t> entity_counts;
    stk::mesh::comm_mesh_counts(*bulk_data_ptr, entity_counts);
    const size_t total_num_linkers = entity_counts[stk::topology::CONSTRAINT_RANK];
    EXPECT_EQ(total_num_linkers, bulk_data_ptr->parallel_size() - 1) << "No linkers should have been destroyed.";
  }

  // Move the sphere on process 0 far from the others and rerun the routine. The linker between spheres 0 and 1 should
  // be deleted.
  if (process_rank == 0) {
    double *element_aabb = stk::mesh::field_data(*element_aabb_field_ptr, sphere);
    element_aabb[0] = -100.0;
    element_aabb[1] = -1.0;
    element_aabb[2] = -1.0;
    element_aabb[3] = -99.0;
    element_aabb[4] = 1.0;
    element_aabb[5] = 1.0;
  }

  destroy_neighbor_linkers_ptr->execute(*neighbor_linkers_part_ptr);
  {
    std::vector<size_t> entity_counts;
    stk::mesh::comm_mesh_counts(*bulk_data_ptr, entity_counts);
    const size_t total_num_linkers = entity_counts[stk::topology::CONSTRAINT_RANK];
    EXPECT_EQ(total_num_linkers, bulk_data_ptr->parallel_size() - 2) << "One linker should have been destroyed.";

    // As a sanity check, loop over all linkers and validate that they are valid.
    stk::mesh::for_each_entity_run(
        *static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()), stk::topology::CONSTRAINT_RANK, *neighbor_linkers_part_ptr,
        []([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &linker) {
          EXPECT_TRUE(bulk_data.is_valid(linker));
        });
  }
}

#endif  // HAVE_MUNDYLINKER_MUNDYSHAPES

}  // namespace

}  // namespace linkers

}  // namespace mundy
