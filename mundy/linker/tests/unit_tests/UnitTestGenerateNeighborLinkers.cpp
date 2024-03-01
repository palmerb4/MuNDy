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
#include <mundy_linker/GenerateNeighborLinkers.hpp>  // for mundy::linker::GenerateNeighborLinkers
#include <mundy_linker/Linkers.hpp>  // for mundy::linker::Linker and  mundy::linker::declare_family_tree_relation
#include <mundy_linker/PerformRegistration.hpp>  // for mundy::linker::perform_registration
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>            // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>      // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaFactory.hpp>  // for mundy::meta::MetaMethodFactory and mundy::meta::HasMeshRequirementsAndIsRegisterable
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
  ASSERT_TRUE(std::find(valid_technique_names.begin(), valid_technique_names.end(), enabled_technique_name) != valid_technique_names.end());

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
  ASSERT_TRUE(std::find(valid_technique_names.begin(), valid_technique_names.end(), enabled_technique_name) != valid_technique_names.end());

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

TEST(GenerateNeighborLinkers, PerformsNeighborLinkerGenerationCorrectlyForSpheres) {
  // TODO(palmerb4): Implement this test.
}

}  // namespace

}  // namespace linker

}  // namespace mundy
