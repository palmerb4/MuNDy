
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

// C++

// Kokkos
#include <Kokkos_Core.hpp>
#include <stk_balance/balance.hpp>  // for stk::balance::balanceStkMesh
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_io/WriteMesh.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>         // stk::mesh::EntityRank
#include <stk_topology/topology.hpp>       // stk::topology
#include <stk_util/ngp/NgpSpaces.hpp>      // stk::ngp::ExecSpace, stk::ngp::RangePolicy
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_ASSERT
#include <mundy_mesh/Aggregate.hpp>        // for mundy::mesh::Aggregate
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper
#include <mundy_mesh/LinkData.hpp>         // for mundy::mesh::LinkData
#include <mundy_mesh/MeshBuilder.hpp>      // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData

namespace mundy {

void run_main() {
  // STK usings
  using stk::mesh::Field;
  using stk::mesh::Part;
  using stk::mesh::Selector;
  using stk::topology::ELEM_RANK;
  using stk::topology::NODE_RANK;

  // Mundy things
  using mesh::BulkData;
  using mesh::DeclareEntitiesHelper;
  using mesh::LinkData;
  using mesh::LinkMetaData;
  using mesh::MeshBuilder;
  using mesh::MetaData;

  // Setup the STK mesh (boiler plate)
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder
      .set_spatial_dimension(3)  //
      .set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data_ptr = mesh_builder.create_meta_data();
  meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition
                                       // to all fields are simple.
  meta_data_ptr->set_coordinate_field_name("NODE_COORDS");
  std::shared_ptr<BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
  MetaData& meta_data = *meta_data_ptr;
  BulkData& bulk_data = *bulk_data_ptr;

  // Setup the link data (boilerplate)
  LinkMetaData link_meta_data = declare_link_meta_data(meta_data, "ALL_LINKS", NODE_RANK);
  LinkData link_data = declare_link_data(bulk_data, link_meta_data);

  // Declare parts
  //   Rod parts: PARTICLE_TOPOLOGY
  //   Spring part: BEAM_2
  //   Link part: SURFACE_LINKS
  Part& rod_parts = meta_data.declare_part_with_topology("RODS", stk::topology::PARTICLE);
  Part& prc1_part = meta_data.declare_part_with_topology("PRC1", stk::topology::BEAM_2);
  Part& slink_part = link_meta_data.declare_link_part("SURFACE_LINKS", 2 /* our dimensionality */);

  // Declare all fields

  // Microtubules PARTICLE top
  //   Node fields:
  //   - NODE_COORDS
  //   - NODE_QUATERNION
  //   - NODE_VELOCITY
  //   - NODE_OMEGA
  //   - NODE_FORCE
  //   - NODE_TORQUE
  //   - NODE_LENGTH
  //   - NODE_RADIUS
  //
  //   ELEM_RANK:
  //   - NOT needed (rods do not share node data due to PARTICLE top)
  //
  // PRC1: BEAM_2 top
  //  Node fields:
  //  - NODE_COORDS
  //  - NODE_TANGENT
  //  - NODE_FORCE
  //  - NODE_TORQUE
  //  - NODE_IS_BOUND
  //
  //  Elem fields:
  //  - ELEM_REST_LENGTH
  //  - ELEM_SPRING_CONSTANT
  //  - ELEM_ANG_SPRING_CONSTANT
  //  - ELEM_REST_ANGLE
  //
  // Surface Links: NODE rank but no fields for now
  Field<double>& node_coords_field = meta_data.declare_field<double>(NODE_RANK, "NODE_COORDS");
  Field<double>& node_quaternion_field = meta_data.declare_field<double>(NODE_RANK, "NODE_QUATERNION");
  Field<double>& node_velocity_field = meta_data.declare_field<double>(NODE_RANK, "NODE_VELOCITY");
  Field<double>& node_omega_field = meta_data.declare_field<double>(NODE_RANK, "NODE_OMEGA");
  Field<double>& node_force_field = meta_data.declare_field<double>(NODE_RANK, "NODE_FORCE");
  Field<double>& node_torque_field = meta_data.declare_field<double>(NODE_RANK, "NODE_TORQUE");
  Field<double>& node_length_field = meta_data.declare_field<double>(NODE_RANK, "NODE_LENGTH");
  Field<double>& node_radius_field = meta_data.declare_field<double>(NODE_RANK, "NODE_RADIUS");
  Field<double>& node_tangent_field = meta_data.declare_field<double>(NODE_RANK, "NODE_TANGENT");
  Field<unsigned>& node_is_bound_field = meta_data.declare_field<unsigned>(NODE_RANK, "NODE_IS_BOUND");

  Field<double>& elem_rest_length_field = meta_data.declare_field<double>(ELEM_RANK, "ELEM_REST_LENGTH");
  Field<double>& elem_spring_constant_field = meta_data.declare_field<double>(ELEM_RANK, "ELEM_SPRING_CONSTANT");
  Field<double>& elem_ang_spring_constant_field =
      meta_data.declare_field<double>(ELEM_RANK, "ELEM_ANG_SPRING_CONSTANT");
  Field<double>& elem_rest_angle_field = meta_data.declare_field<double>(ELEM_RANK, "ELEM_REST_ANGLE");

  // All parts store the node coords
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);

  // Assemble the microtubule parts
  stk::mesh::put_field_on_mesh(node_quaternion_field, rod_parts, 4, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, rod_parts, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_omega_field, rod_parts, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, rod_parts, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_torque_field, rod_parts, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_length_field, rod_parts, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_radius_field, rod_parts, 1, nullptr);

  // Assemble the PRC1 parts
  stk::mesh::put_field_on_mesh(node_tangent_field, prc1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, prc1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_torque_field, prc1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_is_bound_field, prc1_part, 1, nullptr);

  stk::mesh::put_field_on_mesh(elem_rest_length_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_spring_constant_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rest_angle_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_ang_spring_constant_field, prc1_part, 1, nullptr);

  // Decide IO stuff (boilerplate)
  stk::io::put_io_part_attribute(rod_parts);
  stk::io::put_io_part_attribute(prc1_part);
  stk::io::put_io_part_attribute(slink_part);

  stk::io::set_field_role(node_coords_field, Ioss::Field::MESH);
  stk::io::set_field_role(node_quaternion_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_velocity_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_omega_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_force_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_torque_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_length_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_radius_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_tangent_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_is_bound_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rest_length_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_spring_constant_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_ang_spring_constant_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rest_angle_field, Ioss::Field::TRANSIENT);

  stk::io::set_field_output_type(node_coords_field, stk::io::FieldOutputType::VECTOR_3D);
  // stk::io::set_field_output_type(node_quaternion_field, stk::io::FieldOutputType::VECTOR_4D);  // No special quat
  // format :(
  stk::io::set_field_output_type(node_velocity_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_omega_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_force_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_torque_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_length_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_radius_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_tangent_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_is_bound_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rest_length_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_spring_constant_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_ang_spring_constant_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rest_angle_field, stk::io::FieldOutputType::SCALAR);

  // Commit the meta data
  meta_data.commit();

  // Hard code system params
  double rod_radius = 0.5;
  double rod_length = 20;
  math::Quaternion<double> rod_orient = math::Quaternion<double>::identity();
  double init_rod_sep = 2.0;

  double spring_constant = 1.0;
  double ang_spring_constant = 1.0;
  double rest_length = 2.0;
  double rest_cos_angle = 0.0;

  // Fill the declare entities helper
  DeclareEntitiesHelper dec_helper;

  // Create the rods
  auto center_node1 = dec_helper.create_node();
  center_node1
      .owning_proc(0)  //
      .id(1)           // 1 indexed
      .add_field_data<double>(&node_coords_field, {0.0, 0.0, 0.0});
  auto center_node2 = dec_helper.create_node();
  center_node2
      .owning_proc(0)  //
      .id(2)           // 1 indexed
      .add_field_data<double>(&node_coords_field, {init_rod_sep, 0.0, 0.0});

  auto mt1 = dec_helper.create_element();
  mt1.owning_proc(0)  //
      .id(1)          //
      .topology(stk::topology::PARTICLE)
      .add_part(&rod_parts)  //
      .nodes({1});

  auto mt2 = dec_helper.create_element();
  mt2.owning_proc(0)  //
      .id(2)          //
      .topology(stk::topology::PARTICLE)
      .add_part(&rod_parts)  //
      .nodes({2});

  // Declare the PRC1
  auto prc1_node1 = dec_helper.create_node();
  prc1_node1
      .owning_proc(0)  //
      .id(3)           // 1 indexed
      .add_field_data<double>(&node_coords_field, {0.0, 0.0, 0.0});
  
  auto prc1_node2 = dec_helper.create_node();
  prc1_node2
      .owning_proc(0)  //
      .id(4)           // 1 indexed
      .add_field_data<double>(&node_coords_field, {init_rod_sep, 0.0, 0.0});

  auto prc1_elem = dec_helper.create_element();
  prc1_elem.owning_proc(0)  //
      .id(3)                 //
      .topology(stk::topology::BEAM_2)
      .add_part(&prc1_part)  //
      .nodes({3, 4})
      .add_field_data<double>(&elem_rest_length_field, {rest_length})
      .add_field_data<double>(&elem_spring_constant_field, {spring_constant})
      .add_field_data<double>(&elem_ang_spring_constant_field, {ang_spring_constant})
      .add_field_data<double>(&elem_rest_angle_field, {rest_cos_angle});

  // Declare the entities
  dec_helper.check_consistency(bulk_data);
  bulk_data.modification_begin();
  dec_helper.declare_entities(bulk_data);
  bulk_data.modification_end();
  
  // Write the mesh to file
  size_t step = 1;  // Step = 0 doesn't write out fields...
  stk::io::StkMeshIoBroker stk_io_broker;
  stk_io_broker.property_add(Ioss::Property("MAXIMUM_NAME_LENGTH", 180));
  stk_io_broker.set_bulk_data(bulk_data);
  stk::io::write_mesh_with_fields("will.exo", stk_io_broker, step);
}

}  // namespace mundy

int main(int argc, char** argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  mundy::run_main();

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}