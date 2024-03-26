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
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>  // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/Linkers.hpp>   // for mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>      // for mundy::meta::FieldRequirements
#include <mundy_meta/FieldRequirementsBase.hpp>  // for mundy::meta::FieldRequirementsBase

// Mundy test libs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements

namespace mundy {

namespace linkers {

namespace {

//! \name EvaluateLinkerPotentials functionality unit tests
//@{

TEST(EvaluateLinkerPotentials, PerformsHertzianContactCalculationCorrectlyForSpheresSimple) {
  /* Check that EvaluateLinkerPotentials evaluates the Hertzian contact correctly for spheres.
    For this test, we generate a 2 spheres of different radii and generate sphere-sphere linkers between neighboring
    spheres. For simplicity, we directly compute the signed separation distances and contact normals. We then pass these
    fields to the EvaluateLinkerPotentials kernel, compute the potential force, and check it against the expected value.
  */

  // Create an instance of EvaluateLinkerPotentials based on committed mesh that meets the
  // default requirements for EvaluateLinkerPotentials.
  auto [evaluate_linker_potentials_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<EvaluateLinkerPotentials>();
  ASSERT_TRUE(evaluate_linker_potentials_ptr != nullptr);
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
  stk::mesh::Field<double> *element_radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  stk::mesh::Field<double> *element_youngs_modulus_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_YOUNGS_MODULUS");
  stk::mesh::Field<double> *element_poissons_ratio_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_POISSONS_RATIO");
  stk::mesh::Field<double> *linker_signed_separation_distance_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_SIGNED_SEPARATION_DISTANCE");
  stk::mesh::Field<double> *linker_potential_force_magnitude_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_POTENTIAL_FORCE_MAGNITUDE");

  ASSERT_TRUE(node_coord_field_ptr != nullptr);
  ASSERT_TRUE(element_radius_field_ptr != nullptr);
  ASSERT_TRUE(element_youngs_modulus_field_ptr != nullptr);
  ASSERT_TRUE(element_poissons_ratio_field_ptr != nullptr);
  ASSERT_TRUE(linker_signed_separation_distance_field_ptr != nullptr);
  ASSERT_TRUE(linker_potential_force_magnitude_field_ptr != nullptr);

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
  double *radius1 = stk::mesh::field_data(*element_radius_field_ptr, sphere_element1);
  radius1[0] = sphere_radius1;

  double sphere_radius2 = 2.0;
  double *radius2 = stk::mesh::field_data(*element_radius_field_ptr, sphere_element2);
  radius2[0] = sphere_radius2;

  // Set the sphere's Young's modulus
  double sphere_youngs_modulus1 = 1.0e6;
  double *youngs_modulus1 = stk::mesh::field_data(*element_youngs_modulus_field_ptr, sphere_element1);
  youngs_modulus1[0] = sphere_youngs_modulus1;

  double sphere_youngs_modulus2 = 1.0e6;
  double *youngs_modulus2 = stk::mesh::field_data(*element_youngs_modulus_field_ptr, sphere_element2);
  youngs_modulus2[0] = sphere_youngs_modulus2;

  // Set the sphere's Poisson's ratio
  double sphere_poissons_ratio1 = 0.3;
  double *poissons_ratio1 = stk::mesh::field_data(*element_poissons_ratio_field_ptr, sphere_element1);
  poissons_ratio1[0] = sphere_poissons_ratio1;

  double sphere_poissons_ratio2 = 0.3;
  double *poissons_ratio2 = stk::mesh::field_data(*element_poissons_ratio_field_ptr, sphere_element2);
  poissons_ratio2[0] = sphere_poissons_ratio2;

  // Set the linker's signed separation distance.
  double *ssd = stk::mesh::field_data(*linker_signed_separation_distance_field_ptr, linker_constraint);
  ssd[0] = -0.5;

  // Compute the potential force.
  evaluate_linker_potentials_ptr->execute(*sphere_sphere_linker_part_ptr);

  // Check that the result is as expected.
  double *potential_force_magnitude =
      stk::mesh::field_data(*linker_potential_force_magnitude_field_ptr, linker_constraint);

  // The expected potential force magnitude is computed using the Hertzian contact model.
  // F = \frac{4}{3} E \sqrt{R} \delta^{3/2}
  // where:
  // - F is the contact force,
  // - E is the effective modulus of elasticity, calculated as
  //   E = \left( \frac{1 - \nu_1^2}{E_1} + \frac{1 - \nu_2^2}{E_2} \right)^{-1},
  // - R is the effective radius of contact, defined as
  //   \frac{1}{R} = \frac{1}{R_1} + \frac{1}{R_2},
  // - \delta is the deformation at the contact point,
  // - R_1 and R_2 are the radii of the two spheres,
  // - E_1 and E_2 are the Young's moduli of the materials,
  // - \nu_1 and \nu_2 are the Poisson's ratios of the materials.
  const double E = 1.0 / ((1.0 - sphere_poissons_ratio1 * sphere_poissons_ratio1) / sphere_youngs_modulus1 +
                          (1.0 - sphere_poissons_ratio2 * sphere_poissons_ratio2) / sphere_youngs_modulus2);
  const double R = 1.0 / (1.0 / sphere_radius1 + 1.0 / sphere_radius2);
  const double delta = 0.5;
  const double expected_potential_force_magnitude = 4.0 / 3.0 * E * std::sqrt(R) * std::pow(delta, 1.5);
  EXPECT_DOUBLE_EQ(potential_force_magnitude[0], expected_potential_force_magnitude);
}
//@}

}  // namespace

}  // namespace linkers

}  // namespace mundy
