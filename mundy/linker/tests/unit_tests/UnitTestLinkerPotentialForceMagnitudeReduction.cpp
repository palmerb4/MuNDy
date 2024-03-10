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
#include <mundy_linker/LinkerPotentialForceMagnitudeReduction.hpp>  // for mundy::linker::LinkerPotentialForceMagnitudeReduction
#include <mundy_linker/Linkers.hpp>  // for mundy::linker::declare_constraint_relations_to_family_tree_with_sharing
#include <mundy_linker/PerformRegistration.hpp>  // for mundy::linker::perform_registration
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>            // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>      // for mundy::meta::FieldRequirements
#include <mundy_meta/FieldRequirementsBase.hpp>  // for mundy::meta::FieldRequirementsBase

// Mundy test libs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements

namespace mundy {

namespace linker {

namespace {

//! \name LinkerPotentialForceMagnitudeReduction functionality unit tests
//@{

TEST(LinkerPotentialForceMagnitudeReduction, PerformsHertzianContactCalculationCorrectlyForSpheresSimple) {
  /* Check that LinkerPotentialForceMagnitudeReduction properly performs a reduction over linker potential forces.
  The only edge case to consider is where multiple linkers connect to a single entity and yet we only wish to reduce
  over a subset of them.

  For this test, we'll create one sphere and 3 linkers. Two of the linkers will be in the specified linker part and one
  will simply be a neighbor linker.
  */
  perform_registration();

  // Create an instance of LinkerPotentialForceMagnitudeReduction based on committed mesh that meets the
  // default requirements for LinkerPotentialForceMagnitudeReduction.
  Teuchos::ParameterList fixed_params;
  fixed_params.set<std::string>("name_of_linker_part_to_reduce_over", "CUSTOM_LINKERS");
  auto [potential_force_magnitude_reduction_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<
          LinkerPotentialForceMagnitudeReduction>({fixed_params});
  ASSERT_TRUE(potential_force_magnitude_reduction_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);

  // Fetch the parts.
  stk::mesh::Part *sphere_part_ptr = meta_data_ptr->get_part("SPHERES");
  stk::mesh::Part *neighbor_linkers_part_ptr = meta_data_ptr->get_part("NEIGHBOR_LINKERS");
  stk::mesh::Part *custom_linkers_part_ptr = meta_data_ptr->get_part("CUSTOM_LINKERS");
  ASSERT_TRUE(sphere_part_ptr != nullptr);
  ASSERT_TRUE(neighbor_linkers_part_ptr != nullptr);
  ASSERT_TRUE(custom_linkers_part_ptr != nullptr);

  // Declare a sphere, three linkers and each linker to the sphere.
  // Typically linker generation would be done via a generator class, but we'll do it by hand here to keep the test
  // simple.
  bulk_data_ptr->modification_begin();
  stk::mesh::Entity sphere_element = bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{sphere_part_ptr});
  stk::mesh::Entity sphere_node = bulk_data_ptr->declare_node(1);
  bulk_data_ptr->declare_relation(sphere_element, sphere_node, 0);

  stk::mesh::Entity linker1 = bulk_data_ptr->declare_constraint(1, stk::mesh::ConstPartVector{custom_linkers_part_ptr});
  stk::mesh::Entity linker2 = bulk_data_ptr->declare_constraint(2, stk::mesh::ConstPartVector{custom_linkers_part_ptr});
  stk::mesh::Entity linker3 =
      bulk_data_ptr->declare_constraint(3, stk::mesh::ConstPartVector{neighbor_linkers_part_ptr});
  mundy::linker::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr.get(), linker1, sphere_element);
  mundy::linker::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr.get(), linker2, sphere_element);
  mundy::linker::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr.get(), linker3, sphere_element);
  bulk_data_ptr->modification_end();

  // Double check that the sphere has three linkers connected to it
  const unsigned num_constraint_rank_conn =
      bulk_data_ptr->num_connectivity(sphere_element, stk::topology::CONSTRAINT_RANK);
  ASSERT_EQ(num_constraint_rank_conn, 3);

  // Fetch the required fields.
  stk::mesh::Field<double> *node_force_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_FORCE");
  stk::mesh::Field<double> *linker_potential_force_magnitude_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_POTENTIAL_FORCE_MAGNITUDE");
  stk::mesh::Field<double> *linker_contact_normal_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_CONTACT_NORMAL");

  ASSERT_TRUE(node_force_field_ptr != nullptr);
  ASSERT_TRUE(linker_potential_force_magnitude_field_ptr != nullptr);
  ASSERT_TRUE(linker_contact_normal_field_ptr != nullptr);

  // Set the linker force magnitudes
  // Note, at this point, only the custom linkers have the linker potential force magnitude.
  double *linker1_potential_force_magnitude =
      stk::mesh::field_data(*linker_potential_force_magnitude_field_ptr, linker1);
  linker1_potential_force_magnitude[0] = 1.0;
  double *linker2_potential_force_magnitude =
      stk::mesh::field_data(*linker_potential_force_magnitude_field_ptr, linker2);
  linker2_potential_force_magnitude[0] = 2.0;

  // Set the linker contact normals
  double *linker_contact_normal1 = stk::mesh::field_data(*linker_contact_normal_field_ptr, linker1);
  linker_contact_normal1[0] = 0.0;
  linker_contact_normal1[1] = 0.0;
  linker_contact_normal1[2] = 1.0;

  double *linker_contact_normal2 = stk::mesh::field_data(*linker_contact_normal_field_ptr, linker2);
  linker_contact_normal2[0] = 1.0 / std::sqrt(2);
  linker_contact_normal2[1] = 0.0;
  linker_contact_normal2[2] = 1.0 / std::sqrt(2);

  // Zero out the node force field.
  double *node_force = stk::mesh::field_data(*node_force_field_ptr, sphere_node);
  for (unsigned i = 0; i < 3; ++i) {
      node_force[i] = 0.0;
  }

  // Reduce the potential force.
  potential_force_magnitude_reduction_ptr->execute(*sphere_part_ptr);

  // Check that the result is as expected.
  // The reduction should have been over the custom linkers, but not the more general neighbor linkers.
  const double *new_node_force = stk::mesh::field_data(*node_force_field_ptr, sphere_node);
  for (unsigned i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(new_node_force[i], -linker_contact_normal1[i] * linker1_potential_force_magnitude[0] -
                                        linker_contact_normal2[i] * linker2_potential_force_magnitude[0]);
  }
}
//@}

}  // namespace

}  // namespace linker

}  // namespace mundy
