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
#include <mundy_linkers/Linkers.hpp>  // for mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing
#include <mundy_linkers/neighbor_linkers/SphereSphereLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSphereLinkers
#include <mundy_linkers/neighbor_linkers/SphereSpherocylinderLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSpherocylinderLinkers
#include <mundy_linkers/neighbor_linkers/SphereSpherocylinderSegmentLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSpherocylinderSegmentLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSegmentSpherocylinderSegmentLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSpherocylinderLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SpherocylinderSpherocylinderLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSpherocylinderSegmentLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SpherocylinderSpherocylinderSegmentLinkers
#include <mundy_mesh/BulkData.hpp>                                                        // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>     // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MeshBuilder.hpp>    // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>      // for mundy::meta::FieldReqs
#include <mundy_meta/FieldReqsBase.hpp>  // for mundy::meta::FieldReqsBase
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/Spheres.hpp>                 // for mundy::shapes::Spheres
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments
#include <mundy_shapes/Spherocylinders.hpp>         // for mundy::shapes::Spherocylinders

namespace mundy {

namespace linkers {

namespace {

//! \name EvaluateLinkerPotentials functionality unit tests
//@{

TEST(EvaluateLinkerPotentials, PerformsHertzianContactCalculationCorrectlyForSpheresSimple) {
  /* Check that EvaluateLinkerPotentials evaluates the Hertzian contact correctly for each type of pair of spheres,
  spherocylinders, and spherocylinder segments.

  For this test, we generate a 2 spheres, 2 spherocylinders, and 2 spherocylinder segments. generate neighbor linkers
  between unique pairs of these objects. For simplicity, we directly assign the signed separation distances and contact
  normals for these pairs. We then pass these fields to the EvaluateLinkerPotentials kernel, compute the potential
  force, and check it against the expected value.

  This test is designed for either 1 or 2 ranks. For the 1 rank case, rank 0 will own all the entities. For the 2 rank
  case, rank 0 will own the first half of the entities and rank 1 will own the second half of the entities and linkers
  will span the two ranks.

  Our configuration is as follows:
    Sp1-Sp2:   r1 = 1.5, r2 = 2.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    Sp1-Sy1:   r1 = 1.5, r2 = 2.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    Sp1-Seg1:  r1 = 1.5, r2 = 3.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    Sy1-Seg1:  r1 = 2.0, r2 = 3.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    Sy1-Sy2:   r1 = 2.0, r2 = 3.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    Seg1-Seg2: r1 = 3.5, r2 = 4.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
  */


// MUNDY_REGISTER_METACLASS("SPHERE_SPHERE_HERTZIAN_CONTACT",
//                          mundy::linkers::evaluate_linker_potentials::kernels::SphereSphereHertzianContact,
//                          mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory)
// MUNDY_REGISTER_METACLASS("SPHERE_SPHEROCYLINDER_HERTZIAN_CONTACT",
//                          mundy::linkers::evaluate_linker_potentials::kernels::SphereSpherocylinderHertzianContact,
//                          mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory)
// MUNDY_REGISTER_METACLASS(
//     "SPHERE_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT",
//     mundy::linkers::evaluate_linker_potentials::kernels::SphereSpherocylinderSegmentHertzianContact,
//     mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory)
// MUNDY_REGISTER_METACLASS(
//     "SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT",
//     mundy::linkers::evaluate_linker_potentials::kernels::SpherocylinderSegmentSpherocylinderSegmentHertzianContact,
//     mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory)
// MUNDY_REGISTER_METACLASS(
//     "SPHEROCYLINDER_SPHEROCYLINDER_HERTZIAN_CONTACT",
//     mundy::linkers::evaluate_linker_potentials::kernels::SpherocylinderSpherocylinderHertzianContact,
//     mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory)
// MUNDY_REGISTER_METACLASS(
//     "SPHEROCYLINDER_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT",
//     mundy::linkers::evaluate_linker_potentials::kernels::SpherocylinderSpherocylinderSegmentHertzianContact,
//     mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory)

// MUNDY_REGISTER_METACLASS("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_FRICTIONAL_HERTZIAN_CONTACT",
//                          mundy::linkers::evaluate_linker_potentials::kernels::
//                              SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact,
//                          mundy::linkers::EvaluateLinkerPotentials::OurKernelFactory)



  // Create an instance of EvaluateLinkerPotentials based on committed mesh that meets the
  // default requirements for EvaluateLinkerPotentials.
  auto hertzian_contact_fixed_params = Teuchos::ParameterList().set("enabled_kernel_names", 
        mundy::core::make_string_array("SPHERE_SPHERE_HERTZIAN_CONTACT", 
        "SPHERE_SPHEROCYLINDER_HERTZIAN_CONTACT", "SPHERE_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT", 
        "SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT",
        "SPHEROCYLINDER_SPHEROCYLINDER_HERTZIAN_CONTACT", "SPHEROCYLINDER_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT"));
  auto [evaluate_linker_potentials_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<EvaluateLinkerPotentials>({hertzian_contact_fixed_params});
  ASSERT_TRUE(evaluate_linker_potentials_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);

  // This test is designed for either 1 or 2 ranks.
  const int num_ranks = bulk_data_ptr->parallel_size();
  const int rank = bulk_data_ptr->parallel_rank();
  MUNDY_THROW_ASSERT(num_ranks == 1 || num_ranks == 2, std::logic_error, "This test is designed for 1 or 2 ranks.");

  // Fetch the parts.
  stk::mesh::Part *sphere_part_ptr = meta_data_ptr->get_part(mundy::shapes::Spheres::get_name());
  stk::mesh::Part *spherocylinder_part_ptr = meta_data_ptr->get_part(mundy::shapes::Spherocylinders::get_name());
  stk::mesh::Part *spherocylinder_segment_part_ptr =
      meta_data_ptr->get_part(mundy::shapes::SpherocylinderSegments::get_name());

  stk::mesh::Part *neighbor_linker_part_ptr = meta_data_ptr->get_part(NeighborLinkers::get_name());
  stk::mesh::Part *sphere_sphere_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SphereSphereLinkers::get_name());
  stk::mesh::Part *sphere_spherocylinder_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SphereSpherocylinderLinkers::get_name());
  stk::mesh::Part *sphere_spherocylinder_segment_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SphereSpherocylinderSegmentLinkers::get_name());
  stk::mesh::Part *spherocylinder_segment_spherocylinder_segment_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers::get_name());
  stk::mesh::Part *spherocylinder_spherocylinder_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SpherocylinderSpherocylinderLinkers::get_name());
  stk::mesh::Part *spherocylinder_spherocylinder_segment_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SpherocylinderSpherocylinderSegmentLinkers::get_name());

  ASSERT_TRUE(sphere_part_ptr != nullptr);
  ASSERT_TRUE(spherocylinder_part_ptr != nullptr);
  ASSERT_TRUE(spherocylinder_segment_part_ptr != nullptr);
  ASSERT_TRUE(neighbor_linker_part_ptr != nullptr);
  ASSERT_TRUE(sphere_sphere_linker_part_ptr != nullptr);
  ASSERT_TRUE(sphere_spherocylinder_linker_part_ptr != nullptr);
  ASSERT_TRUE(sphere_spherocylinder_segment_linker_part_ptr != nullptr);
  ASSERT_TRUE(spherocylinder_segment_spherocylinder_segment_linker_part_ptr != nullptr);
  ASSERT_TRUE(spherocylinder_spherocylinder_linker_part_ptr != nullptr);
  ASSERT_TRUE(spherocylinder_spherocylinder_segment_linker_part_ptr != nullptr);

  // Declare two of each type of shape for 3 * 2 total shapes and generate 6 linkers between the unqiue shape pairings.
  // Typically linker generation would be done via a generator class, but we'll do it by hand here to keep the test
  // simple.
  //
  // Each rank declares one of each shape and Rank 0 declares the linkers between the shapes. Rank 1 will ghost its
  // shapes to Rank 0.
  bulk_data_ptr->modification_begin();
  std::vector<stk::mesh::EntityProc> send_shapes_to_rank0;

  // Declare the spheres.
  if (rank == 0) {
    stk::mesh::Entity sp1_element = bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{sphere_part_ptr});
    stk::mesh::Entity sp1_node = bulk_data_ptr->declare_node(1);
    bulk_data_ptr->declare_relation(sp1_element, sp1_node, 0);
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity sp2_element = bulk_data_ptr->declare_element(2, stk::mesh::ConstPartVector{sphere_part_ptr});
    stk::mesh::Entity sp2_node = bulk_data_ptr->declare_node(2);
    bulk_data_ptr->declare_relation(sp2_element, sp2_node, 0);
    send_shapes_to_rank0.push_back(std::make_pair(sp2_element, 0));
  }

  // Declare the spherocylinders
  if (rank == 0) {
    stk::mesh::Entity sy1_element =
        bulk_data_ptr->declare_element(3, stk::mesh::ConstPartVector{spherocylinder_part_ptr});
    stk::mesh::Entity sy1_node = bulk_data_ptr->declare_node(3);
    bulk_data_ptr->declare_relation(sy1_element, sy1_node, 0);
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity sy2_element =
        bulk_data_ptr->declare_element(4, stk::mesh::ConstPartVector{spherocylinder_part_ptr});
    stk::mesh::Entity sy2_node = bulk_data_ptr->declare_node(4);
    bulk_data_ptr->declare_relation(sy2_element, sy2_node, 0);
    send_shapes_to_rank0.push_back(std::make_pair(sy2_element, 0));
  }

  // Declare the spherocylinder segments
  if (rank == 0) {
    stk::mesh::Entity seg1_element =
        bulk_data_ptr->declare_element(5, stk::mesh::ConstPartVector{spherocylinder_segment_part_ptr});
    stk::mesh::Entity seg1_node1 = bulk_data_ptr->declare_node(5);
    stk::mesh::Entity seg1_node2 = bulk_data_ptr->declare_node(6);
    bulk_data_ptr->declare_relation(seg1_element, seg1_node1, 0);
    bulk_data_ptr->declare_relation(seg1_element, seg1_node2, 1);
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity seg2_element =
        bulk_data_ptr->declare_element(6, stk::mesh::ConstPartVector{spherocylinder_segment_part_ptr});
    stk::mesh::Entity seg2_node1 = bulk_data_ptr->declare_node(7);
    stk::mesh::Entity seg2_node2 = bulk_data_ptr->declare_node(8);
    bulk_data_ptr->declare_relation(seg2_element, seg2_node1, 0);
    bulk_data_ptr->declare_relation(seg2_element, seg2_node2, 1);
    send_shapes_to_rank0.push_back(std::make_pair(seg2_element, 0));
  }
  stk::mesh::Ghosting &ghosting = bulk_data_ptr->create_ghosting("GHOST_SHAPES_TO_RANK0");
  bulk_data_ptr->change_ghosting(ghosting, send_shapes_to_rank0);
  bulk_data_ptr->modification_end();

  bulk_data_ptr->modification_begin();
  if (rank == 0) {
    // Fatch the declared entities
    stk::mesh::Entity sp1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);
    stk::mesh::Entity sp2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 2);
    stk::mesh::Entity sy1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 3);
    stk::mesh::Entity sy2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 4);
    stk::mesh::Entity seg1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 5);
    stk::mesh::Entity seg2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 6);

    for (auto &entity : {sp1_element, sp2_element, sy1_element, sy2_element, seg1_element, seg2_element}) {
      ASSERT_TRUE(bulk_data_ptr->is_valid(entity));
    }

    // Declare the linkers
    stk::mesh::Entity sp1_sp2_linker =
        bulk_data_ptr->declare_constraint(1, stk::mesh::ConstPartVector{sphere_sphere_linker_part_ptr});
    mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr.get(), sp1_sp2_linker,
                                                                             sp1_element, sp2_element);

    stk::mesh::Entity sp1_sy1_linker =
        bulk_data_ptr->declare_constraint(2, stk::mesh::ConstPartVector{sphere_spherocylinder_linker_part_ptr});
    mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr.get(), sp1_sy1_linker,
                                                                             sp1_element, sy1_element);

    stk::mesh::Entity sp1_seg1_linker =
        bulk_data_ptr->declare_constraint(3, stk::mesh::ConstPartVector{sphere_spherocylinder_segment_linker_part_ptr});
    mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr.get(), sp1_seg1_linker,
                                                                             sp1_element, seg1_element);

    stk::mesh::Entity seg1_seg2_linker = bulk_data_ptr->declare_constraint(
        4, stk::mesh::ConstPartVector{spherocylinder_segment_spherocylinder_segment_linker_part_ptr});
    mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr.get(), seg1_seg2_linker,
                                                                             seg1_element, seg2_element);

    stk::mesh::Entity sy1_sy2_linker =
        bulk_data_ptr->declare_constraint(5, stk::mesh::ConstPartVector{spherocylinder_spherocylinder_linker_part_ptr});
    mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr.get(), sy1_sy2_linker,
                                                                             sy1_element, sy2_element);

    stk::mesh::Entity sy1_seg1_linker = bulk_data_ptr->declare_constraint(
        6, stk::mesh::ConstPartVector{spherocylinder_spherocylinder_segment_linker_part_ptr});
    mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr.get(), sy1_seg1_linker,
                                                                             sy1_element, seg1_element);

    // Fill the linker contact normal with an arbitrary unit vector, we'll use [1, 2, 3] / sqrt(14).
    stk::mesh::Field<double> *linker_contact_normal_field_ptr =
        meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_CONTACT_NORMAL");
    ASSERT_TRUE(linker_contact_normal_field_ptr != nullptr);
    for (auto &linker :
         {sp1_sp2_linker, sp1_sy1_linker, sp1_seg1_linker, seg1_seg2_linker, sy1_sy2_linker, sy1_seg1_linker}) {
      ASSERT_TRUE(bulk_data_ptr->is_valid(linker));
      mundy::mesh::vector3_field_data(*linker_contact_normal_field_ptr, linker)
          .set(1.0 / std::sqrt(14.0), 2.0 / std::sqrt(14.0), 3.0 / std::sqrt(14.0));
    }
  }
  bulk_data_ptr->modification_end();

  // Fetch the required fields.
  stk::mesh::Field<double> *element_radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  stk::mesh::Field<double> *element_youngs_modulus_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_YOUNGS_MODULUS");
  stk::mesh::Field<double> *element_poissons_ratio_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_POISSONS_RATIO");
  stk::mesh::Field<double> *linker_signed_separation_distance_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_SIGNED_SEPARATION_DISTANCE");
  stk::mesh::Field<double> *linker_potential_force_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_POTENTIAL_FORCE");

  ASSERT_TRUE(element_radius_field_ptr != nullptr);
  ASSERT_TRUE(element_youngs_modulus_field_ptr != nullptr);
  ASSERT_TRUE(element_poissons_ratio_field_ptr != nullptr);
  ASSERT_TRUE(linker_signed_separation_distance_field_ptr != nullptr);
  ASSERT_TRUE(linker_potential_force_field_ptr != nullptr);

  // Initialize the spheres. Only performed for local entities.
  // sp1 radius: 1.5, sp2 radius: 2.0, sy1 radius: 2.5, sy2 radius: 3.0, seg1 radius: 3.5, seg2 radius: 4.0
  if (rank == 0) {
    stk::mesh::Entity sp1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sp1_element));
    stk::mesh::field_data(*element_radius_field_ptr, sp1_element)[0] = 1.5;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, sp1_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, sp1_element)[0] = 0.3;
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity sp2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 2);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sp2_element));
    stk::mesh::field_data(*element_radius_field_ptr, sp2_element)[0] = 2.0;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, sp2_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, sp2_element)[0] = 0.3;
  }

  // Initialize the spherocylinders. Only performed form local entities.
  // As a reminder, the quaternion orientation of the sylinder represents the rotation from the local frame (in which
  // the rode is aligned with the x-axis) to the global frame.
  if (rank == 0) {
    stk::mesh::Entity sy1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 3);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sy1_element));
    stk::mesh::field_data(*element_radius_field_ptr, sy1_element)[0] = 2.5;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, sy1_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, sy1_element)[0] = 0.3;
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity sy2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 4);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sy2_element));
    stk::mesh::field_data(*element_radius_field_ptr, sy2_element)[0] = 3.0;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, sy2_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, sy2_element)[0] = 0.3;
  }

  // Initialize the spherocylinder segments. Only performed form local entities.
  if (rank == 0) {
    stk::mesh::Entity seg1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 5);
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg1_element));
    stk::mesh::field_data(*element_radius_field_ptr, seg1_element)[0] = 3.5;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, seg1_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, seg1_element)[0] = 0.3;
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity seg2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 6);
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg2_element));
    stk::mesh::field_data(*element_radius_field_ptr, seg2_element)[0] = 4.0;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, seg2_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, seg2_element)[0] = 0.3;
  }

  // Initialize the linker signed separation distances. Only performed for local entities.
  if (rank == 0) {
    // Fetch the linkers
    stk::mesh::Entity sp1_sp2_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 1);
    stk::mesh::Entity sp1_sy1_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 2);
    stk::mesh::Entity sp1_seg1_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 3);
    stk::mesh::Entity seg1_seg2_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 4);
    stk::mesh::Entity sy1_sy2_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 5);
    stk::mesh::Entity sy1_seg1_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 6);
    for (auto &linker :
         {sp1_sp2_linker, sp1_sy1_linker, sp1_seg1_linker, seg1_seg2_linker, sy1_sy2_linker, sy1_seg1_linker}) {
      ASSERT_TRUE(bulk_data_ptr->is_valid(linker));
      stk::mesh::field_data(*linker_signed_separation_distance_field_ptr, linker)[0] = -0.5;
    }
  }

  // Execute the EvaluateLinkerPotentials kernel for all neighbor linkers.
  ASSERT_NO_THROW(evaluate_linker_potentials_ptr->execute(*neighbor_linker_part_ptr))
      << "Failed to evaluate linker potentials.";

  auto check_potential_force_magnitude = [&linker_potential_force_field_ptr](
                                             stk::mesh::Entity linker, const double &radius1, const double &radius2,
                                             const double &ssd, const double &poissons_ratio1,
                                             const double &youngs_modulus1, const double &poissons_ratio2,
                                             const double &youngs_modulus2, const std::string &message) {
    // Check that the result is as expected.
    const auto potential_force = mundy::mesh::vector3_field_data(*linker_potential_force_field_ptr, linker);
    const double potential_force_magnitude = mundy::math::norm(potential_force);

    // The expected potential force is computed using the Hertzian contact model.
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
    // Delta is the amount of overlap between the two spheres.
    const double delta = -std::min(ssd, 0.0);
    const double E = 1.0 / ((1.0 - poissons_ratio1 * poissons_ratio1) / youngs_modulus1 +
                            (1.0 - poissons_ratio2 * poissons_ratio2) / youngs_modulus2);
    const double R = 1.0 / (1.0 / radius1 + 1.0 / radius2);
    const double expected_potential_force_magnitude = 4.0 / 3.0 * E * std::sqrt(R) * std::pow(delta, 1.5);
    EXPECT_DOUBLE_EQ(potential_force_magnitude, expected_potential_force_magnitude) << message;
  };

  // Only rank 0 will check the results since it owns the linkers.
  if (rank == 0) {
    // Fetch the linkers
    stk::mesh::Entity sp1_sp2_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 1);
    stk::mesh::Entity sp1_sy1_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 2);
    stk::mesh::Entity sp1_seg1_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 3);
    stk::mesh::Entity seg1_seg2_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 4);
    stk::mesh::Entity sy1_sy2_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 5);
    stk::mesh::Entity sy1_seg1_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 6);
    for (auto &linker :
         {sp1_sp2_linker, sp1_sy1_linker, sp1_seg1_linker, seg1_seg2_linker, sy1_sy2_linker, sy1_seg1_linker}) {
      ASSERT_TRUE(bulk_data_ptr->is_valid(linker));
    }

    // Sp1-Sp2:   r1 = 1.5, r2 = 2.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    // Sp1-Sy1:   r1 = 1.5, r2 = 2.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    // Sp1-Seg1:  r1 = 1.5, r2 = 3.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    // Sy1-Sy2:   r1 = 2.5, r2 = 3.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    // Sy1-Seg1:  r1 = 2.5, r2 = 3.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    // Seg1-Seg2: r1 = 3.5, r2 = 4.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    check_potential_force_magnitude(sp1_sp2_linker, 1.5, 2.0, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "sp1_sp2_linker");
    check_potential_force_magnitude(sp1_sy1_linker, 1.5, 2.5, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "sp1_sy1_linker");
    check_potential_force_magnitude(sp1_seg1_linker, 1.5, 3.5, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "sp1_seg1_linker");
    check_potential_force_magnitude(sy1_sy2_linker, 2.5, 3.0, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "sy1_sy2_linker");
    check_potential_force_magnitude(sy1_seg1_linker, 2.5, 3.5, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "sy1_seg1_linker");
    check_potential_force_magnitude(seg1_seg2_linker, 3.5, 4.0, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "seg1_seg2_linker");
  }
}
//@}

}  // namespace

}  // namespace linkers

}  // namespace mundy
