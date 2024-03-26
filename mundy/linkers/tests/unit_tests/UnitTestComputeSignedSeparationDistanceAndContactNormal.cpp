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
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Field.hpp>         // for stk::mesh::Field
#include <stk_mesh/base/Types.hpp>         // for stk::mesh::ConstPartVector
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/Linkers.hpp>  // for mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>            // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>      // for mundy::meta::FieldRequirements
#include <mundy_meta/FieldRequirementsBase.hpp>  // for mundy::meta::FieldRequirementsBase
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements

namespace mundy {

namespace linkers {

namespace {

//! \name ComputeSignedSeparationDistanceAndContactNormal functionality unit tests
//@{

TEST(ComputeSignedSeparationDistanceAndContactNormal, PerformsCalculationCorrectlyForSphere) {

  /* Check that ComputeSignedSeparationDistanceAndContactNormal works correctly for spheres.
  The signed separation distance between two spheres is the distance between their centers minus the sum of their radii.
  The contact normal is the normalized vector pointing from the first sphere to the second sphere. In our case, that
  vector points from the left sphere to the right sphere.
  */

  // Create an instance of ComputeSignedSeparationDistanceAndContactNormal based on committed mesh that meets the
  // default requirements for ComputeSignedSeparationDistanceAndContactNormal.
  auto [compute_ssd_and_cn_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<
          ComputeSignedSeparationDistanceAndContactNormal>();
  ASSERT_TRUE(compute_ssd_and_cn_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);

  // Fetch the parts.
  stk::mesh::Part *sphere_part_ptr = meta_data_ptr->get_part("SPHERES");
  stk::mesh::Part *sphere_sphere_linker_part_ptr = meta_data_ptr->get_part("SPHERE_SPHERE_LINKERS");
  ASSERT_TRUE(sphere_part_ptr != nullptr);
  ASSERT_TRUE(sphere_sphere_linker_part_ptr != nullptr);

  // Declare two spheres and connect them with a linker.
  // Typically linker generation would be done via a generator class, but we'll do it by hand here to keep the test
  // simple.
  bulk_data_ptr->modification_begin();
  stk::mesh::Entity sphere_element1 = bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{sphere_part_ptr});
  stk::mesh::Entity sphere_element2 = bulk_data_ptr->declare_element(2, stk::mesh::ConstPartVector{sphere_part_ptr});
  stk::mesh::Entity sphere_node1 = bulk_data_ptr->declare_node(1);
  stk::mesh::Entity sphere_node2 = bulk_data_ptr->declare_node(2);
  bulk_data_ptr->declare_relation(sphere_element1, sphere_node1, 0);
  bulk_data_ptr->declare_relation(sphere_element2, sphere_node2, 0);

  stk::mesh::Entity linker_constraint =
      bulk_data_ptr->declare_constraint(1, stk::mesh::ConstPartVector{sphere_sphere_linker_part_ptr});
  mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr.get(), linker_constraint,
                                                                          sphere_element1, sphere_element2);
  bulk_data_ptr->modification_end();

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDINATES");
  stk::mesh::Field<double> *radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  stk::mesh::Field<double> *linker_signed_separation_distance_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_SIGNED_SEPARATION_DISTANCE");
  stk::mesh::Field<double> *linker_contact_normal_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_CONTACT_NORMAL");

  ASSERT_TRUE(node_coord_field_ptr != nullptr);
  ASSERT_TRUE(radius_field_ptr != nullptr);
  ASSERT_TRUE(linker_signed_separation_distance_field_ptr != nullptr);
  ASSERT_TRUE(linker_contact_normal_field_ptr != nullptr);

  // Set the sphere's position.
  double sphere_positions1[3] = {0.0, 0.0, 0.0};
  double *node_coords1 = stk::mesh::field_data(*node_coord_field_ptr, sphere_node1);
  node_coords1[0] = sphere_positions1[0];
  node_coords1[1] = sphere_positions1[1];
  node_coords1[2] = sphere_positions1[2];

  double sphere_positions2[3] = {1.0, 2.0, 2.0};
  double *node_coords2 = stk::mesh::field_data(*node_coord_field_ptr, sphere_node2);
  node_coords2[0] = sphere_positions2[0];
  node_coords2[1] = sphere_positions2[1];
  node_coords2[2] = sphere_positions2[2];

  // Set the sphere's radius.
  double sphere_radius1 = 1.5;
  double *radius1 = stk::mesh::field_data(*radius_field_ptr, sphere_element1);
  radius1[0] = sphere_radius1;

  double sphere_radius2 = 2.0;
  double *radius2 = stk::mesh::field_data(*radius_field_ptr, sphere_element2);
  radius2[0] = sphere_radius2;

  // Compute the signed separation distance and contact normal.
  compute_ssd_and_cn_ptr->execute(*sphere_sphere_linker_part_ptr);

  // Check that the result is as expected.
  double *ssd = stk::mesh::field_data(*linker_signed_separation_distance_field_ptr, linker_constraint);
  double expected_ssd = -0.5;
  EXPECT_DOUBLE_EQ(ssd[0], expected_ssd);

  double *cn = stk::mesh::field_data(*linker_contact_normal_field_ptr, linker_constraint);
  double expected_cn[3] = {1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0};
  for (int i = 0; i < 3; i++) {
    EXPECT_DOUBLE_EQ(cn[i], expected_cn[i]);
  }
}
//@}

}  // namespace

}  // namespace linkers

}  // namespace mundy
