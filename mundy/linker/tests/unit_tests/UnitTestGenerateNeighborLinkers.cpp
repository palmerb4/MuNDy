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
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/GetEntities.hpp>    // for stk::mesh::get_selected_entities
#include <stk_mesh/base/MeshBuilder.hpp>    // for stk::mesh::MeshBuilder
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>          // for stk::mesh::EntityProc, EntityVector, etc

// Mundy libs
#include <MundyLinker_config.hpp>                    // for HAVE_MUNDYLINKER_MUNDYSHAPE
#include <mundy_linker/GenerateNeighborLinkers.hpp>  // for mundy::linker::GenerateNeighborLinkers
#include <mundy_linker/Linkers.hpp>  // for mundy::linker::Linker and  mundy::linker::declare_family_tree_relation
#include <mundy_linker/PerformRegistration.hpp>  // for mundy::linker::perform_registration
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>            // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>      // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaFactory.hpp>  // for mundy::meta::MetaMethodFactory and mundy::meta::HasMeshRequirementsAndIsRegisterable
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shape/ComputeAABB.hpp>  // for mundy::shape::ComputeAABB

namespace mundy {

namespace linker {

namespace {

/* What tests should we run?

GenerateNeighborLinkers is the first of our MetaMethodPairwiseSubsetExecutionInterface classes and one of our first
technique dispatchers, so we'll want to explicitly test every piece of GenerateNeighborLinkers.


Following the pattern of the ComputeAABB unit tests, we'll want to test the following:
IsRegisterable, FixedParameterDefaults, MutableParameterDefaults, FixedParameterValidation, MutableParameterValidation,
GetMeshRequirementsFromDefaultParameters, CreateNewInstanceFromDefaultParameters,
PerformsNeighborLinkerGenerationCorrectlyForSpheres
*/

//! \name GenerateNeighborLinkers static interface implementations unit tests
//@{

TEST(GenerateNeighborLinkersStaticInterface, IsRegisterable) {
  perform_registration();

  // Check if GenerateNeighborLinkers has the correct static interface to be compatible with MetaFactory.
  ASSERT_TRUE(mundy::meta::HasMeshRequirementsAndIsRegisterable<GenerateNeighborLinkers>::value);
}

TEST(GenerateNeighborLinkersStaticInterface, FixedParameterDefaults) {
  perform_registration();

  // Check the expected default values.
  Teuchos::ParameterList fixed_params;
  fixed_params.validateParametersAndSetDefaults(GenerateNeighborLinkers::get_valid_fixed_params());

  // Check that all the enabled technique is in the list of registered techniques.
  ASSERT_TRUE(fixed_params.isParameter("enabled_technique_name"));
  ASSERT_TRUE(GenerateNeighborLinkers::OurTechniqueFactory::num_registered_classes() > 0);

  std::string enabled_technique_name = fixed_params.get<std::string>("enabled_technique_name");
  const auto valid_technique_names = GenerateNeighborLinkers::OurTechniqueFactory::get_keys();
  ASSERT_TRUE(std::find(valid_technique_names.begin(), valid_technique_names.end(), enabled_technique_name) !=
              valid_technique_names.end());

  // Check that the fixed parameters for the technique are present.
  for (const std::string &valid_technique_name : valid_technique_names) {
    ASSERT_TRUE(fixed_params.isSublist(valid_technique_name));
    fixed_params.sublist(valid_technique_name, true);
  }

  // TODO(palmerb4): Check that the parameters are forwarded correctly.
}

TEST(GenerateNeighborLinkersStaticInterface, MutableParameterDefaults) {
  perform_registration();

  // Check the expected default values.
  Teuchos::ParameterList mutable_params;
  mutable_params.validateParametersAndSetDefaults(GenerateNeighborLinkers::get_valid_mutable_params());

  // Check that all the enabled technique is in the list of registered techniques.
  ASSERT_TRUE(mutable_params.isParameter("enabled_technique_name"));
  ASSERT_TRUE(GenerateNeighborLinkers::OurTechniqueFactory::num_registered_classes() > 0);

  std::string enabled_technique_name = mutable_params.get<std::string>("enabled_technique_name");
  const auto valid_technique_names = GenerateNeighborLinkers::OurTechniqueFactory::get_keys();
  ASSERT_TRUE(std::find(valid_technique_names.begin(), valid_technique_names.end(), enabled_technique_name) !=
              valid_technique_names.end());

  // Check that the mutable parameters for the technique are present.
  for (const std::string &valid_technique_name : valid_technique_names) {
    ASSERT_TRUE(mutable_params.isSublist(valid_technique_name));
    mutable_params.sublist(valid_technique_name, true);
  }

  // TODO(palmerb4): Check that the parameters are forwarded correctly.
}

TEST(GenerateNeighborLinkersStaticInterface, FixedParameterValidation) {
  // TODO(palmerb4): How should we perform validation tests?
  perform_registration();
}

TEST(GenerateNeighborLinkersStaticInterface, MutableParameterValidation) {
  // TODO(palmerb4): How should we perform validation tests?
  perform_registration();
}

TEST(GenerateNeighborLinkersStaticInterface, GetMeshRequirementsFromDefaultParameters) {
  perform_registration();

  // Attempt to get the mesh requirements using the default parameters.
  Teuchos::ParameterList fixed_params;
  fixed_params.validateParametersAndSetDefaults(GenerateNeighborLinkers::get_valid_fixed_params());
  ASSERT_NO_THROW(GenerateNeighborLinkers::get_mesh_requirements(fixed_params));
}

TEST(GenerateNeighborLinkersStaticInterface, CreateNewInstanceFromDefaultParameters) {
  perform_registration();

  // Attempt to get the mesh requirements using the default parameters.
  auto mesh_reqs_ptr = std::make_shared<meta::MeshRequirements>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  Teuchos::ParameterList fixed_params;
  fixed_params.validateParametersAndSetDefaults(GenerateNeighborLinkers::get_valid_fixed_params());
  mesh_reqs_ptr->merge(GenerateNeighborLinkers::get_mesh_requirements(fixed_params));

  // Create a new instance of GenerateNeighborLinkers using the default parameters and the mesh generated from the mesh
  // requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  EXPECT_NO_THROW(GenerateNeighborLinkers::create_new_instance(bulk_data_ptr.get(), fixed_params));
}
//@}

//! \name GenerateNeighborLinkers functionality unit tests
//@{

#ifdef HAVE_MUNDYLINKER_MUNDYSHAPE
TEST(GenerateNeighborLinkers, PerformsNeighborLinkerGenerationCorrectlyForSpheres) {
  /* Outline
    For this test, we generate a N spheres of equal radii isotropically distributed within a box and split across all
    processes, compute their AABBs, and generate their neighbor linkers. We then communicate all spheres to process 0
    and use a direct N^2 neighbor search to create a matrix of booleans (adjacency list) stating if particle (i,j) are
    neighbors or not. We use this matrix to validate the neighbor linkers. The sum of the matrix should be equal to the
    number of neighbor linkers and for each linker connecting particles i and j, the matrix should be true at (i,j) and
    (j,i).

    We let the density of spheres be large enough to guarantee that some neighbor linkers span multiple processes.
  */
  perform_registration();

  // Free variables
  const int num_spheres_per_process = 10;
  const double volume_fraction = 0.4;
  const double sphere_radius = 1.0;
  const double sphere_volume = (4.0 / 3.0) * M_PI * std::pow(sphere_radius, 3);
  const double length_of_domain = std::cbrt(num_spheres_per_process * sphere_volume / volume_fraction);

  // Create an instance of GenerateNeighborLinkers and ComputeAABB based on committed mesh that meets both of their
  // default requirements.
  auto [compute_aabb_ptr, generate_neighbor_linkers_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<mundy::shape::ComputeAABB,
                                                                                        GenerateNeighborLinkers>();
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();

  GenerateNeighborLinkers::get_mesh_requirements(Teuchos::ParameterList())->print_reqs();
  mundy::shape::ComputeAABB::get_mesh_requirements(Teuchos::ParameterList())->print_reqs();

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDINATES");
  stk::mesh::Field<double> *element_radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  stk::mesh::Field<double> *element_aabb_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_AABB");
  ASSERT_TRUE(node_coord_field_ptr != nullptr) << "node_coord_field_ptr cannot be null";
  ASSERT_TRUE(element_aabb_field_ptr != nullptr) << "element_aabb_field_ptr cannot be null";
  ASSERT_TRUE(element_radius_field_ptr != nullptr) << "element_radius_field_ptr cannot be null";

  // Fetch the requested parts.
  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
  stk::mesh::Part *neighbor_linkers_part_ptr = meta_data_ptr->get_part("NEIGHBOR_LINKERS");
  ASSERT_TRUE(spheres_part_ptr != nullptr) << "spheres_part_ptr cannot be null";
  ASSERT_TRUE(neighbor_linkers_part_ptr != nullptr) << "neighbor_linkers_part_ptr cannot be null";

  // Add n spheres to the mesh per process.
  bulk_data_ptr->modification_begin();
  std::vector<size_t> requests(meta_data_ptr->entity_rank_count(), 0);
  requests[stk::topology::NODE_RANK] = num_spheres_per_process;
  requests[stk::topology::ELEMENT_RANK] = num_spheres_per_process;

  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data_ptr->generate_new_entities(requests, requested_entities);

  // Associate each segments with the sphere part and connect them to their nodes.
  std::vector<stk::mesh::Part *> add_spheres_part = {spheres_part_ptr};
  for (int i = 0; i < num_spheres_per_process; i++) {
    stk::mesh::Entity sphere_i = requested_entities[num_spheres_per_process + i];
    stk::mesh::Entity node_i = requested_entities[i];
    bulk_data_ptr->change_entity_parts(sphere_i, add_spheres_part);
    bulk_data_ptr->declare_relation(sphere_i, node_i, 0);
  }
  bulk_data_ptr->modification_end();

  // Set the sphere's position and radius
  openrand::Philox rng(bulk_data_ptr->parallel_rank(), 0);
  for (int i = 0; i < num_spheres_per_process; i++) {
    stk::mesh::Entity node_i = requested_entities[i];
    stk::mesh::Entity sphere_i = requested_entities[num_spheres_per_process + i];
    double *node_coords = stk::mesh::field_data(*node_coord_field_ptr, node_i);
    double *element_radius = stk::mesh::field_data(*element_radius_field_ptr, sphere_i);

    node_coords[0] = length_of_domain * rng.rand<double>();
    node_coords[1] = length_of_domain * rng.rand<double>();
    node_coords[2] = length_of_domain * rng.rand<double>();

    element_radius[0] = sphere_radius;
  }

  // Compute the AABB for all the spheres. By default this writes to the ELEMENT_AABB field.
  compute_aabb_ptr->execute(*spheres_part_ptr);

  // Compute the neighbor linkers. Between neighboring spheres.
  generate_neighbor_linkers_ptr->execute(*spheres_part_ptr, *spheres_part_ptr);

  // Ghost all particles with process rank 0.
  bulk_data_ptr->modification_begin();
  std::vector<stk::mesh::EntityProc> send_all_entities;
  if (bulk_data_ptr->parallel_rank() != 0) {
    for (int i = 0; i < num_spheres_per_process; i++) {
      stk::mesh::Entity sphere_i = requested_entities[num_spheres_per_process + i];
      send_all_entities.push_back(std::make_pair(sphere_i, 0));
    }
  }
  stk::mesh::Ghosting &ghosting = bulk_data_ptr->create_ghosting("GHOST_SPHERES_TO_RANK0");
  bulk_data_ptr->change_ghosting(ghosting, send_all_entities);
  bulk_data_ptr->modification_end();

  // Get the total number of spheres. Must be called parallel synchronously.
  std::vector<size_t> entity_counts;
  stk::mesh::comm_mesh_counts(*bulk_data_ptr, entity_counts);
  const size_t total_num_spheres = entity_counts[stk::topology::ELEMENT_RANK];
  ASSERT_EQ(total_num_spheres, bulk_data_ptr->parallel_size() * num_spheres_per_process);

  // Perform the direct N^2 neighbor comparison on process 0.
  if (bulk_data_ptr->parallel_rank() == 0) {
    // Helper function for checking if two AABBs overlap
    // Note the aabbs are stored as a 6-tuple (min_x, min_y, min_z, max_x, max_y, max_z).
    auto aabbs_overlap = [](const double *aabb_i, const double *aabb_j) -> bool {
      for (int k = 0; k < 3; k++) {
        if (aabb_i[k] > aabb_j[k + 3] || aabb_i[k + 3] < aabb_j[k]) {
          return false;
        }
      }
      return true;
    };  // aabbs_overlap

    // Construct the adjacency matrix for each sphere in the sphere part.
    // NOTE: GIDs are 1-indexed.
    std::vector<std::vector<bool>> adjacency_matrix(total_num_spheres, std::vector<bool>(total_num_spheres, false));
    const stk::mesh::BucketVector &sphere_buckets =
        bulk_data_ptr->get_buckets(stk::topology::ELEMENT_RANK, *spheres_part_ptr);
    for (size_t bucket_idx_i = 0; bucket_idx_i < sphere_buckets.size(); ++bucket_idx_i) {
      stk::mesh::Bucket &sphere_bucket_i = *sphere_buckets[bucket_idx_i];
      for (size_t sphere_idx_i = 0; sphere_idx_i < sphere_bucket_i.size(); ++sphere_idx_i) {
        stk::mesh::Entity const &sphere_i = sphere_bucket_i[sphere_idx_i];
        const stk::mesh::EntityId sphere_i_gid = bulk_data_ptr->identifier(sphere_i);
        std::cout << "sphere_i_gid: " << sphere_i_gid << std::endl;
        ASSERT_LT(sphere_i_gid - 1, total_num_spheres);

        double *aabb_i = stk::mesh::field_data(*element_aabb_field_ptr, sphere_i);
        for (size_t bucket_idx_j = 0; bucket_idx_j < sphere_buckets.size(); ++bucket_idx_j) {
          stk::mesh::Bucket &sphere_bucket_j = *sphere_buckets[bucket_idx_j];
          for (size_t sphere_idx_j = 0; sphere_idx_j < sphere_bucket_j.size(); ++sphere_idx_j) {
            stk::mesh::Entity const &sphere_j = sphere_bucket_j[sphere_idx_j];
            const stk::mesh::EntityId sphere_j_gid = bulk_data_ptr->identifier(sphere_j);
            std::cout << "sphere_j_gid: " << sphere_j_gid << std::endl;
            ASSERT_LT(sphere_j_gid - 1, total_num_spheres);

            double *aabb_j = stk::mesh::field_data(*element_aabb_field_ptr, sphere_j);

            // Check if the AABBs overlap.
            adjacency_matrix[sphere_i_gid - 1][sphere_j_gid - 1] = aabbs_overlap(aabb_i, aabb_j);
          }
        }
      }
    }

    // Loop over each neighbor linker and check if its connected spheres are neighbors.
    const stk::mesh::BucketVector &neighbor_linker_buckets =
        bulk_data_ptr->get_buckets(stk::topology::CONSTRAINT_RANK, *neighbor_linkers_part_ptr);
    for (size_t bucket_idx = 0; bucket_idx < neighbor_linker_buckets.size(); ++bucket_idx) {
      stk::mesh::Bucket &neighbor_linker_bucket = *neighbor_linker_buckets[bucket_idx];
      for (size_t neighbor_linker_idx = 0; neighbor_linker_idx < neighbor_linker_bucket.size(); ++neighbor_linker_idx) {
        stk::mesh::Entity const &neighbor_linker = neighbor_linker_bucket[neighbor_linker_idx];
        stk::mesh::Entity const *connected_spheres = bulk_data_ptr->begin_elements(neighbor_linker);
        const int num_connected_spheres = bulk_data_ptr->num_elements(neighbor_linker);
        ASSERT_EQ(num_connected_spheres, 2);

        const stk::mesh::EntityId sphere_i_gid = bulk_data_ptr->identifier(connected_spheres[0]);
        const stk::mesh::EntityId sphere_j_gid = bulk_data_ptr->identifier(connected_spheres[1]);
        ASSERT_TRUE(adjacency_matrix[sphere_i_gid - 1][sphere_j_gid - 1]);
      }
    }
  }

  // Have all processes wait for process 0 to finish.
  stk::parallel_machine_barrier(bulk_data_ptr->parallel());
}
#endif  // HAVE_MUNDYLINKER_MUNDYSHAPE

}  // namespace

}  // namespace linker

}  // namespace mundy
