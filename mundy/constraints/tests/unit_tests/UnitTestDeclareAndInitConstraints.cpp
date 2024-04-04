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
#include <stk_io/StkMeshIoBroker.hpp>      // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Field.hpp>         // for stk::mesh::Field
#include <stk_mesh/base/Types.hpp>         // for stk::mesh::ConstPartVector
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_constraints/AngularSprings.hpp>             // for mundy::constraints::AngularSprings
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>             // for mundy::constraints::HookeanSprings
#include <mundy_constraints/declare_and_initialize_constraints/techniques/ArchlengthCoordinateMapping.hpp>  // for mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>            // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>      // for mundy::meta::FieldRequirements
#include <mundy_meta/FieldRequirementsBase.hpp>  // for mundy::meta::FieldRequirementsBase
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements

namespace mundy {

namespace constraints {

namespace {

//! \name DeclareAndInitConstraints functionality unit tests
//@{

void declare_linear_and_angular_springs_and_dump_mesh(const Teuchos::ParameterList &fixed_params,
                                                      const Teuchos::ParameterList &mutable_params,
                                                      const std::string &output_file_name) {
  
  std::cout << "#######################################################" << std::endl;
  std::cout << "Generating " << output_file_name << "..." << std::endl;
  std::cout << "#######################################################" << std::endl;
  
  // Create an instance of DeclareAndInitConstraints based on committed mesh that meets the requirements for
  // DeclareAndInitConstraints.
  auto [declare_and_init_constraints_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<DeclareAndInitConstraints>(
          {fixed_params});
  ASSERT_TRUE(declare_and_init_constraints_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);
  meta_data_ptr->set_coordinate_field_name("NODE_COORDINATES");

  // Fetch the spring parts
  stk::mesh::Part *angular_spring_part_ptr = meta_data_ptr->get_part(mundy::constraints::AngularSprings::get_name());
  stk::mesh::Part *linear_spring_part_ptr = meta_data_ptr->get_part(mundy::constraints::HookeanSprings::get_name());
  ASSERT_TRUE(angular_spring_part_ptr != nullptr);
  ASSERT_TRUE(linear_spring_part_ptr != nullptr);
  stk::io::put_io_part_attribute(*angular_spring_part_ptr);
  stk::io::put_io_part_attribute(*linear_spring_part_ptr);

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDINATES");
  ASSERT_TRUE(node_coord_field_ptr != nullptr);

  // Execute and then dumping the mesh to file given a set of mutable params.
  declare_and_init_constraints_ptr->set_mutable_params(mutable_params);
  declare_and_init_constraints_ptr->execute();

  // setup stk io
  stk::io::StkMeshIoBroker stk_io_broker;
  stk_io_broker.use_simple_fields();
  stk_io_broker.set_bulk_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));

  size_t output_file_index = stk_io_broker.create_output_mesh(output_file_name, stk::io::WRITE_RESULTS);
  stk_io_broker.add_field(output_file_index, *node_coord_field_ptr);

  // Write the mesh to file.
  stk_io_broker.begin_output_step(output_file_index, 0.0);
  stk_io_broker.write_defined_output_fields(output_file_index);
  stk_io_broker.end_output_step(output_file_index);
}

void declare_linear_springs_and_dump_mesh(const Teuchos::ParameterList &fixed_params,
                                          const Teuchos::ParameterList &mutable_params,
                                          const std::string &output_file_name) {

  std::cout << "#######################################################" << std::endl;
  std::cout << "Generating " << output_file_name << "..." << std::endl;
  std::cout << "#######################################################" << std::endl;
  
  // Create an instance of DeclareAndInitConstraints based on committed mesh that meets the requirements for
  // DeclareAndInitConstraints.
  auto [declare_and_init_constraints_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<DeclareAndInitConstraints>(
          {fixed_params});
  ASSERT_TRUE(declare_and_init_constraints_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);
  meta_data_ptr->set_coordinate_field_name("NODE_COORDINATES");

  // Fetch the spring parts
  stk::mesh::Part *linear_spring_part_ptr = meta_data_ptr->get_part(mundy::constraints::HookeanSprings::get_name());
  ASSERT_TRUE(linear_spring_part_ptr != nullptr);
  stk::io::put_io_part_attribute(*linear_spring_part_ptr);

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDINATES");
  ASSERT_TRUE(node_coord_field_ptr != nullptr);

  // Execute and then dumping the mesh to file given a set of mutable params.
  declare_and_init_constraints_ptr->set_mutable_params(mutable_params);
  declare_and_init_constraints_ptr->execute();

  // setup stk io
  stk::io::StkMeshIoBroker stk_io_broker;
  stk_io_broker.use_simple_fields();
  stk_io_broker.set_bulk_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));

  size_t output_file_index = stk_io_broker.create_output_mesh(output_file_name, stk::io::WRITE_RESULTS);
  stk_io_broker.add_field(output_file_index, *node_coord_field_ptr);

  // Write the mesh to file.
  stk_io_broker.begin_output_step(output_file_index, 0.0);
  stk_io_broker.write_defined_output_fields(output_file_index);
  stk_io_broker.end_output_step(output_file_index);
}

void declare_angular_springs_and_dump_mesh(const Teuchos::ParameterList &fixed_params,
                                           const Teuchos::ParameterList &mutable_params,
                                           const std::string &output_file_name) {
  
  std::cout << "#######################################################" << std::endl;
  std::cout << "Generating " << output_file_name << "..." << std::endl;
  std::cout << "#######################################################" << std::endl;
  
  // Create an instance of DeclareAndInitConstraints based on committed mesh that meets the requirements for
  // DeclareAndInitConstraints.
  auto [declare_and_init_constraints_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<DeclareAndInitConstraints>(
          {fixed_params});
  ASSERT_TRUE(declare_and_init_constraints_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);
  meta_data_ptr->set_coordinate_field_name("NODE_COORDINATES");

  // Fetch the spring parts
  stk::mesh::Part *angular_spring_part_ptr = meta_data_ptr->get_part(mundy::constraints::AngularSprings::get_name());
  ASSERT_TRUE(angular_spring_part_ptr != nullptr);
  stk::io::put_io_part_attribute(*angular_spring_part_ptr);

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDINATES");
  ASSERT_TRUE(node_coord_field_ptr != nullptr);

  // Execute and then dumping the mesh to file given a set of mutable params.
  declare_and_init_constraints_ptr->set_mutable_params(mutable_params);
  declare_and_init_constraints_ptr->execute();

  // setup stk io
  stk::io::StkMeshIoBroker stk_io_broker;
  stk_io_broker.use_simple_fields();
  stk_io_broker.set_bulk_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));

  size_t output_file_index = stk_io_broker.create_output_mesh(output_file_name, stk::io::WRITE_RESULTS);
  stk_io_broker.add_field(output_file_index, *node_coord_field_ptr);

  // Write the mesh to file.
  stk_io_broker.begin_output_step(output_file_index, 0.0);
  stk_io_broker.write_defined_output_fields(output_file_index);
  stk_io_broker.end_output_step(output_file_index);
}

TEST(DeclareAndInitConstraints, ChainOfLinearAndAngularSpringsVisualInspection) {
  /* Check that DeclareAndInitConstraints works correctly for angular and linear springs. It's difficult to check the
   * correctness of the spring placement algorithm, so we will visually inspect the output.
   */

  // Setup fixed params for linear and angular springs
  Teuchos::ParameterList declare_and_init_constraints_fixed_params;
  declare_and_init_constraints_fixed_params.set("enabled_technique_name", "CHAIN_OF_SPRINGS")
      .sublist("CHAIN_OF_SPRINGS")
      .set<bool>("generate_hookean_springs", true)
      .set<bool>("generate_angular_springs", true);

  // Setup mutable params for a default chain of springs
  Teuchos::ParameterList declare_and_init_constraints_mutable_params;
  declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS").set<size_t>("num_nodes", 13);
  declare_linear_and_angular_springs_and_dump_mesh(declare_and_init_constraints_fixed_params,
                                                   declare_and_init_constraints_mutable_params,
                                                   "test_chain_of_springs_n13_default.exo");

  // Setup mutable params for a chain of linear springs with a helix coordinate mapping
  using CoordinateMappingType =
      mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping;
  using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::Helix;
  auto levis_function_mapping_ptr =
      std::make_shared<OurCoordinateMappingType>(100, 0.1, 1.0, 10, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
  declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
      .set<size_t>("num_nodes", 100)
      .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", levis_function_mapping_ptr);
  declare_linear_and_angular_springs_and_dump_mesh(declare_and_init_constraints_fixed_params,
                                                   declare_and_init_constraints_mutable_params,
                                                   "test_chain_of_springs_n13_default.exo");
}

TEST(DeclareAndInitConstraints, ChainOfLinearSpringsVisualInspection) {
  /* Check that DeclareAndInitConstraints works correctly for angular and linear springs. It's difficult to check the
   * correctness of the spring placement algorithm, so we will visually inspect the output.
   */

  // Setup fixed params for only linear springs
  Teuchos::ParameterList declare_and_init_constraints_fixed_params;
  declare_and_init_constraints_fixed_params.set("enabled_technique_name", "CHAIN_OF_SPRINGS")
      .sublist("CHAIN_OF_SPRINGS")
      .set<bool>("generate_hookean_springs", true)
      .set<bool>("generate_angular_springs", false);

  // Setup our mutable params for a default chain of springs
  Teuchos::ParameterList declare_and_init_constraints_mutable_params;
  declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS").set<size_t>("num_nodes", 13);
  declare_linear_springs_and_dump_mesh(declare_and_init_constraints_fixed_params,
                                       declare_and_init_constraints_mutable_params,
                                       "test_chain_of_hookean_springs_n13_default.exo");

  // Setup our mutable params for a chain of linear springs with a helix coordinate mapping
  using CoordinateMappingType =
      mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping;
  using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::Helix;
  auto levis_function_mapping_ptr =
      std::make_shared<OurCoordinateMappingType>(100, 0.1, 1.0, 10, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
  declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
      .set<size_t>("num_nodes", 100)
      .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", levis_function_mapping_ptr);
  declare_linear_springs_and_dump_mesh(declare_and_init_constraints_fixed_params,
                                       declare_and_init_constraints_mutable_params,
                                       "test_chain_of_hookean_springs_n100_helix.exo");
}

TEST(DeclareAndInitConstraints, ChainOfAngularSpringsVisualInspection) {
  /* Check that DeclareAndInitConstraints works correctly for angular and linear springs. It's difficult to check the
   * correctness of the spring placement algorithm, so we will visually inspect the output.
   */

  // Setup fixed params for only angular springs
  Teuchos::ParameterList declare_and_init_constraints_fixed_params;
  declare_and_init_constraints_fixed_params.set("enabled_technique_name", "CHAIN_OF_SPRINGS")
      .sublist("CHAIN_OF_SPRINGS")
      .set<bool>("generate_hookean_springs", false)
      .set<bool>("generate_angular_springs", true);

  // Setup our mutable params for a default chain of springs
  Teuchos::ParameterList declare_and_init_constraints_mutable_params;
  declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS").set<size_t>("num_nodes", 13);
  declare_angular_springs_and_dump_mesh(declare_and_init_constraints_fixed_params,
                                        declare_and_init_constraints_mutable_params,
                                        "test_chain_of_angular_springs_n13_default.exo");

  // Setup our mutable params for a chain of linear springs with a helix coordinate mapping
  using CoordinateMappingType =
      mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping;
  using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::Helix;
  auto levis_function_mapping_ptr =
      std::make_shared<OurCoordinateMappingType>(100, 0.1, 1.0, 10, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
  declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
      .set<size_t>("num_nodes", 100)
      .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", levis_function_mapping_ptr);
  declare_angular_springs_and_dump_mesh(declare_and_init_constraints_fixed_params,
                                        declare_and_init_constraints_mutable_params,
                                        "test_chain_of_angular_springs_n100_helix.exo");
}
//@}

}  // namespace

}  // namespace constraints

}  // namespace mundy
