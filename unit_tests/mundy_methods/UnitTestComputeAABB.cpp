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
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>            // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>      // for mundy::meta::FieldRequirements
#include <mundy_meta/FieldRequirementsBase.hpp>  // for mundy::meta::FieldRequirementsBase
#include <mundy_methods/ComputeAABB.hpp>         // for mundy::methods::ComputeAABB

// Mundy test libs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements

namespace mundy {

namespace methods {

namespace {

//! \name ComputeAABB static interface implementations unit tests
//@{

TEST(ComputeAABBStaticInterface, IsRegisterable) {
  // Check if ComputeAABB has the correct static interface to be compatible with MetaFactory.
  ASSERT_TRUE(meta::HasMeshRequirementsAndIsRegisterable<ComputeAABB>::value);
}

TEST(ComputeAABBStaticInterface, FixedParameterDefaults) {
  // Check the expected default values.
  Teuchos::ParameterList fixed_params;
  ComputeAABB::validate_fixed_parameters_and_set_defaults(&fixed_params);
  ASSERT_TRUE(fixed_params.isSublist("kernels"));
  Teuchos::ParameterList &kernels_sublist = fixed_params.sublist("kernels", true);
  ASSERT_TRUE(kernels_sublist.isParameter("count"));
  ASSERT_EQ(kernels_sublist.get<unsigned>("count"), ComputeAABB::OurKernelFactory::num_registered_classes());
  ASSERT_TRUE(kernels_sublist.get<unsigned>("count") > 0);
  int i = 0;
  for (auto &key : ComputeAABB::OurKernelFactory::get_keys()) {
    ASSERT_TRUE(kernels_sublist.isSublist("kernel_" + std::to_string(i)));
    Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i), true);
    ASSERT_TRUE(kernel_params.isParameter("name"));
    ASSERT_EQ(kernel_params.get<std::string>("name"), key);
    i++;
  }
}

TEST(ComputeAABBStaticInterface, MutableParameterDefaults) {
  // Check the expected default values.
  Teuchos::ParameterList mutable_params;
  ComputeAABB::validate_mutable_parameters_and_set_defaults(&mutable_params);
  ASSERT_TRUE(mutable_params.isSublist("kernels"));
  Teuchos::ParameterList &kernels_sublist = mutable_params.sublist("kernels", true);
  ASSERT_TRUE(kernels_sublist.isParameter("count"));
  ASSERT_EQ(kernels_sublist.get<unsigned>("count"), ComputeAABB::OurKernelFactory::num_registered_classes());
  ASSERT_TRUE(kernels_sublist.get<unsigned>("count") > 0);
  int i = 0;
  for (auto &key : ComputeAABB::OurKernelFactory::get_keys()) {
    ASSERT_TRUE(kernels_sublist.isSublist("kernel_" + std::to_string(i)));
    Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i), true);
    ASSERT_TRUE(kernel_params.isParameter("name"));
    ASSERT_EQ(kernel_params.get<std::string>("name"), key);
    i++;
  }
}

TEST(ComputeAABBStaticInterface, FixedParameterValidation) {
  // TODO(palmerb4): How should we perform validation tests?
}

TEST(ComputeAABBStaticInterface, MutableParameterValidation) {
  // TODO(palmerb4): How should we perform validation tests?
}

TEST(ComputeAABBStaticInterface, GetMeshRequirementsFromDefaultParameters) {
  // Attempt to get the mesh requirements using the default parameters.
  Teuchos::ParameterList fixed_params;
  ComputeAABB::validate_fixed_parameters_and_set_defaults(&fixed_params);
  ASSERT_NO_THROW(ComputeAABB::get_mesh_requirements(fixed_params));
}

TEST(ComputeAABBStaticInterface, CreateNewInstanceFromDefaultParameters) {
  // Attempt to get the mesh requirements using the default parameters.
  auto mesh_reqs_ptr = std::make_shared<meta::MeshRequirements>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  Teuchos::ParameterList fixed_params;
  ComputeAABB::validate_fixed_parameters_and_set_defaults(&fixed_params);
  EXPECT_NO_THROW(ComputeAABB::get_mesh_requirements(fixed_params));
  mesh_reqs_ptr->merge(ComputeAABB::get_mesh_requirements(fixed_params));

  // Create a new instance of ComputeAABB using the default parameters and the mesh generated from the mesh
  // requirements.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  EXPECT_NO_THROW(ComputeAABB::create_new_instance(bulk_data_ptr.get(), fixed_params));
}
//@}

//! \name ComputeAABB functionality unit tests
//@{

TEST(ComputeAABB, PerformsAABBCalculationCorrectlyForSphere) {
  /* Check that ComputeAABB works correctly for spheres.
  For a sphere at any arbitrary position, the AABB should be a cube with side length equal to the diameter of the sphere
  and center at the sphere's position.
  */

  // Create an instance of ComputeAABB based on committed mesh that meets the requirements for ComputeAABB.
  auto [compute_aabb_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<ComputeAABB>();
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();

  // Fetch the multibody sphere part and add a single sphere to it.
  stk::mesh::Part *sphere_part_ptr = meta_data_ptr->get_part("SPHERE");
  ASSERT_TRUE(sphere_part_ptr != nullptr);

  bulk_data_ptr->modification_begin();
  stk::mesh::EntityId sphere_id = 1;
  stk::mesh::Entity sphere_element =
      bulk_data_ptr->declare_element(sphere_id, stk::mesh::ConstPartVector{sphere_part_ptr});
  stk::mesh::Entity sphere_node = bulk_data_ptr->declare_node(sphere_id);
  bulk_data_ptr->declare_relation(sphere_element, sphere_node, 0);
  bulk_data_ptr->modification_end();

  // Set the sphere's position.
  double sphere_position[3] = {0.0, 0.0, 0.0};
  stk::mesh::Field<double> &node_coord_field =
      *meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORD");
  double *node_coords = stk::mesh::field_data(node_coord_field, sphere_node);
  node_coords[0] = sphere_position[0];
  node_coords[1] = sphere_position[1];
  node_coords[2] = sphere_position[2];

  // Set the sphere's radius.
  double sphere_radius = 1.0;
  stk::mesh::Field<double> &radius_field = *meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "RADIUS");
  double *radius = stk::mesh::field_data(radius_field, sphere_element);
  radius[0] = sphere_radius;

  // Compute the AABB.
  compute_aabb_ptr->execute(*sphere_part_ptr);

  // Check that the computed aabb is as expected.
  stk::mesh::Field<double> &aabb_field = *meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "AABB");
  double *aabb = stk::mesh::field_data(aabb_field, sphere_element);
  double expected_aabb[6] = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
  for (int i = 0; i < 6; i++) {
    EXPECT_DOUBLE_EQ(aabb[i], expected_aabb[i]);
  }
}
//@}

}  // namespace

}  // namespace methods

}  // namespace mundy
