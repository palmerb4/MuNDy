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
#include <gtest/gtest.h>      // for TEST, ASSERT_NO_THROW, etc
#include <openrand/philox.h>  // for openrand::Philox

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Ioss_ElementBlock.h>  // for Ioss::ElementBlock
#include <Ioss_IOFactory.h>     // for Ioss::IOFactory
#include <Ioss_NodeBlock.h>     // for Ioss::NodeBlock

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
#include <MundyIo_config.hpp>                               // for HAVE_MUNDYIO_MUNDYSHAPES
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_io/IOBroker.hpp>                            // for mundy::io::IoBroker
#include <mundy_mesh/BulkData.hpp>                          // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>                       // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>                          // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>                         // for mundy::meta::FieldReqs
#include <mundy_meta/MetaFactory.hpp>  // for mundy::meta::MetaMethodFactory and mundy::meta::HasMeshReqsAndIsRegisterable
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB

namespace mundy {

namespace io {

namespace {

/* What tests should we run?

IoBroker is the main IO class for mundy, and we want to make sure it works in parallel environments.
*/

// Check if the field role is what we expect
bool check_field_role(std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr, const std::string &field_name,
                      const stk::topology::rank_t &field_rank, const Ioss::Field::RoleType &expected_role) {
  auto field_ptr = meta_data_ptr->get_field(field_rank, field_name);
  const Ioss::Field::RoleType *field_role = stk::io::get_field_role(*field_ptr);
  return (*field_role == expected_role);
}

//! \name IOBroker functional tests
//@{

//! \brief Test the IOBroker class in a parallel environment
TEST(IOBrokerMPI, WriteParallelChainOfSpringsFile) {
  const int mrank = stk::parallel_machine_rank(MPI_COMM_WORLD);
  const int msize = stk::parallel_machine_size(MPI_COMM_WORLD);

  // Set the number of objects in the system directly
  const int num_spheres = 4;

  // Get the other parallel rank
  const int other_mrank = mrank == 0 ? 1 : 0;

  std::cout << "Sanity check on MPI rank: " << mrank << " and size: " << msize << std::endl;

  ////////////////
  // Configure MuNDy directly
  ////////////////

  // Setup the mesh directly
  auto mesh_reqs_ptr = std::make_shared<meta::MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  // Crate some custom sphere variables and load them (like velocity and the transient coordinates)
  auto custom_sphere_part_reqs = std::make_shared<mundy::meta::PartReqs>();
  custom_sphere_part_reqs->add_field_reqs<double>("NODE_VELOCITY", stk::topology::NODE_RANK, 3, 1)
      .add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", stk::topology::NODE_RANK, 3, 1);
  mundy::shapes::Spheres::add_and_sync_part_reqs(custom_sphere_part_reqs);
  mesh_reqs_ptr->sync(mundy::shapes::Spheres::get_mesh_requirements());

  // Construct a chain of springs and investigate it
  auto declare_and_init_constraints_fixed_params =
      Teuchos::ParameterList().set("enabled_technique_name", "CHAIN_OF_SPRINGS");
  declare_and_init_constraints_fixed_params.sublist("CHAIN_OF_SPRINGS")
      .set("hookean_springs_part_names", mundy::core::make_string_array("BACKBONE_SPRINGS"))
      .set("sphere_part_names", mundy::core::make_string_array("SPHERES"))
      .set<bool>("generate_hookean_springs", true)
      .set<bool>("generate_spheres_at_nodes", true);

  mesh_reqs_ptr->sync(
      mundy::constraints::DeclareAndInitConstraints::get_mesh_requirements(declare_and_init_constraints_fixed_params));

  ////////////////
  // Build the mesh
  ////////////////

  // Create the mesh
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();

  // Set the simple fields
  bulk_data_ptr->mesh_meta_data_ptr()->use_simple_fields();

  // Create the metadata pointer
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();

  // Set the coordinate field name
  meta_data_ptr->set_coordinate_field_name("NODE_COORDS");

  // Commit the mesh
  meta_data_ptr->commit();

  // Print the mesh (Delay on each task to print in sequence)
  std::this_thread::sleep_for(std::chrono::milliseconds(mrank));
  std::cout << "[Rank " << mrank << "] Mesh after commit:" << std::endl;
  stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr, std::cout);

  ////////////////
  // Populate the mesh
  ////////////////

  // Instantiate the metamethod we will call for the springs
  auto declare_and_init_constraints_ptr = mundy::constraints::DeclareAndInitConstraints::create_new_instance(
      bulk_data_ptr.get(), declare_and_init_constraints_fixed_params);

  // Set the mutable parameters
  Teuchos::ParameterList declare_and_init_constraints_mutable_params;
  using CoordinateMappingType =
      mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping;
  using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::StraightLine;

  double center_x = 0.0;
  double center_y = 0.0;
  double center_z = 0.0;
  double orientation_x = 1.0;
  double orientation_y = 0.0;
  double orientation_z = 0.0;
  auto coord_mapping_ptr = std::make_shared<OurCoordinateMappingType>(num_spheres, center_x, center_y, center_z,
                                                                      (static_cast<double>(num_spheres) - 1) * 1.0,
                                                                      orientation_x, orientation_y, orientation_z);
  declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
      .set<size_t>("num_nodes", num_spheres)
      .set<size_t>("node_id_start", 1u)
      .set<size_t>("element_id_start", 1u)
      .set("hookean_spring_constant", 10.0)
      .set("hookean_spring_rest_length", 1.0)
      .set("sphere_radius", 1.0)
      .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", coord_mapping_ptr);
  declare_and_init_constraints_ptr->set_mutable_params(declare_and_init_constraints_mutable_params);

  // Run the initial setup
  declare_and_init_constraints_ptr->execute();

  // Dump the mesh to a file to inspect later (Figure out a way to automate this or store the information)
  stk::mesh::impl::dump_mesh_per_proc(*bulk_data_ptr, "AfterInitialSetup");

  ////////////////
  // Use MundyIO to dump the mesh into two different per-processor files
  ////////////////
  // Create a mundy io broker via it's fixed parameters
  auto fixed_params_iobroker =
      Teuchos::ParameterList()
          .set("enabled_io_parts", mundy::core::make_string_array("SPHERES", "BACKBONE_SPRINGS"))
          .set("enabled_io_fields_node_rank", mundy::core::make_string_array("NODE_VELOCITY"))
          .set("enabled_io_fields_element_rank",
               mundy::core::make_string_array("ELEMENT_RADIUS", "ELEMENT_HOOKEAN_SPRING_CONSTANT",
                                              "ELEMENT_HOOKEAN_SPRING_REST_LENGTH"))
          .set("coordinate_field_name", "NODE_COORDS")
          .set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES")
          .set("exodus_database_output_filename", "IOBrokerChainOfSprings.exo")
          .set("enable_ioss_logging", "ON")
          .set("database_purpose", "results");
  // Create the IO broker
  auto io_broker_ptr = mundy::io::IOBroker::create_new_instance(bulk_data_ptr.get(), fixed_params_iobroker);

  // Write a single 'timestep' out from the iobroker
  io_broker_ptr->write_io_broker_timestep(1, 1.0);
}

//@}

}  // namespace

}  // namespace io

}  // namespace mundy
