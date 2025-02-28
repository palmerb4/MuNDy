
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

struct COORDS {};
struct VELOCITY {};
struct FORCE {};
struct TORQUE {};
struct RADIUS {};
struct LENGTH {};
struct OMEGA {};
struct IS_BOUND {};
struct LINKED_ENTITIES {};
struct QUAT {};
struct TANGENT {};
struct REST_LENGTH {};
struct SPRING_CONSTANT {};
struct ANG_SPRING_CONSTANT {};
struct REST_ANGLE {};

namespace mundy {

class eval_prc1_force {
 public:
  eval_prc1_force() {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& prc1_view) const {
    auto x0 = get<COORDS>(prc1_view, 0);
    auto x1 = get<COORDS>(prc1_view, 1);
    double k_lin = get<SPRING_CONSTANT>(prc1_view)[0];
    double r0_lin = get<REST_LENGTH>(prc1_view)[0];
    double k_ang = get<ANG_SPRING_CONSTANT>(prc1_view)[0];
    double r0_ang = get<REST_ANGLE>(prc1_view)[0];

    // x10hat = (x1 - x0) / |x1 - x0|
    // f_lin = -k_lin * (x1 - x0) / |x1 - x0|
    auto x10 = x1 - x0;
    auto x10_norm = math::norm(x10);
    auto x10hat = x10 / x10_norm;
    auto f_lin = -k_lin * (x10_norm - r0_lin) * x10hat;

    // We know that the downward connected nodes are only connected to us, therefore we do not need atomics
    get<FORCE>(prc1_view, 0) += f_lin;
    get<FORCE>(prc1_view, 1) -= f_lin;
    std::cout << get<FORCE>(prc1_view, 0) << std::endl;
  }

  void apply_to(auto& prc1_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(prc1_agg.topology() == stk::topology::BEAM_2, "PRC1 must be beam 2 top.");

    auto ngp_prc1_agg = mesh::get_updated_ngp_aggregate(prc1_agg);
    ngp_prc1_agg
        .template sync_to_device<COORDS, FORCE, SPRING_CONSTANT, REST_LENGTH, ANG_SPRING_CONSTANT, REST_ANGLE>();
    ngp_prc1_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_prc1_agg.template modify_on_device<FORCE>();
  }

  void apply_to(auto& prc1_agg) {
    static_assert(prc1_agg.topology() == stk::topology::BEAM_2, "PRC1 must be beam 2 top.");

    auto ngp_prc1_agg = mesh::get_updated_ngp_aggregate(prc1_agg);
    ngp_prc1_agg
        .template sync_to_device<COORDS, FORCE, SPRING_CONSTANT, REST_LENGTH, ANG_SPRING_CONSTANT, REST_ANGLE>();
    ngp_prc1_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_prc1_agg.template modify_on_device<FORCE>();
  }
};

template <typename RodAgg, typename SurfaceNodeAgg>
class map_surface_force_to_com_force {
 public:
  map_surface_force_to_com_force(mesh::LinkData& link_data, RodAgg& rod_agg, SurfaceNodeAgg& surfacenode_agg)
      : link_data_(link_data), rod_agg_(rod_agg), surfacenode_agg_(surfacenode_agg) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& slink_view) const {
    // Get the two entities from the slink_view, this is already inside of a parallel loop
    auto rod_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[0]);
    auto surfacenode_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[1]);

    // Go from entity view to the aggregate
    auto rod_view = ngp_rod_agg_.get_view(slink_view.ngp_mesh().fast_mesh_index(rod_entity));
    auto surfacenode_view = ngp_surfacenode_agg_.get_view(slink_view.ngp_mesh().fast_mesh_index(surfacenode_entity));

    auto rod_force = get<FORCE>(rod_view, 0 /* center node */);
    auto rod_torque = get<TORQUE>(rod_view, 0);
    auto rod_center = get<COORDS>(rod_view, 0);
    auto surfacenode_force = get<FORCE>(surfacenode_view);
    auto surfacenode_torque = get<TORQUE>(surfacenode_view);
    auto surfacenode_pos = get<COORDS>(surfacenode_view);

    // Add the COM force
    Kokkos::atomic_add(&rod_force[0], surfacenode_force[0]);
    Kokkos::atomic_add(&rod_force[1], surfacenode_force[1]);
    Kokkos::atomic_add(&rod_force[2], surfacenode_force[2]);

    // Add the torques along with the off-center force contribution
    auto torque = surfacenode_torque + math::cross(surfacenode_pos - rod_center, surfacenode_force);
    Kokkos::atomic_add(&rod_torque[0], torque[0]);
    Kokkos::atomic_add(&rod_torque[1], torque[1]);
    Kokkos::atomic_add(&rod_torque[2], torque[2]);

    std::cout << "rod_force: " << rod_force << std::endl;
    std::cout << "rod_torque: " << rod_torque << std::endl;
  }

  void apply_to(auto& slink_agg, const stk::mesh::Selector& subset_selector) {
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);
    // Explicitly construct a local instance of the NGP rod_agg and surfacenode_agg
    ngp_rod_agg_ = mesh::get_updated_ngp_aggregate(rod_agg_);
    ngp_surfacenode_agg_ = mesh::get_updated_ngp_aggregate(surfacenode_agg_);

    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_rod_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();
    ngp_surfacenode_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();

    ngp_slink_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);

    ngp_rod_agg_.template modify_on_device<FORCE, TORQUE>();
  }

  void apply_to(auto& slink_agg) {
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);
    // Explicitly construct a local instance of the NGP rod_agg and surfacenode_agg
    ngp_rod_agg_ = mesh::get_updated_ngp_aggregate(rod_agg_);
    ngp_surfacenode_agg_ = mesh::get_updated_ngp_aggregate(surfacenode_agg_);

    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_rod_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();
    ngp_surfacenode_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();

    ngp_slink_agg.template for_each((*this) /*use my operator as a lambda*/);

    ngp_rod_agg_.template modify_on_device<FORCE, TORQUE>();
  }

 private:
  mesh::LinkData& link_data_;
  RodAgg& rod_agg_;
  SurfaceNodeAgg& surfacenode_agg_;
  using rod_agg_ngp_t = decltype(mesh::get_updated_ngp_aggregate(std::declval<RodAgg>()));
  using surfacenode_agg_ngp_t = decltype(mesh::get_updated_ngp_aggregate(std::declval<SurfaceNodeAgg>()));
  rod_agg_ngp_t ngp_rod_agg_;
  surfacenode_agg_ngp_t ngp_surfacenode_agg_;
};

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
  using mesh::FieldComponent;
  using mesh::LinkData;
  using mesh::LinkMetaData;
  using mesh::MeshBuilder;
  using mesh::MetaData;
  using mesh::QuaternionFieldComponent;
  using mesh::ScalarFieldComponent;
  using mesh::Vector3FieldComponent;

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
  Part& rod_part = meta_data.declare_part_with_topology("RODS", stk::topology::PARTICLE);
  Part& prc1_part = meta_data.declare_part_with_topology("PRC1", stk::topology::BEAM_2);
  Part& slink_part = link_meta_data.declare_link_part("SURFACE_LINKS", 2 /* our dimensionality */);

  // Declare all fields

  // Microtubules PARTICLE top
  //   Node fields:
  //   - NODE_COORDS
  //   - NODE_QUATERNION
  //   - NODE_TANGENT
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
  Field<double>& node_tangent_field = meta_data.declare_field<double>(NODE_RANK, "NODE_TANGENT");
  Field<double>& node_velocity_field = meta_data.declare_field<double>(NODE_RANK, "NODE_VELOCITY");
  Field<double>& node_omega_field = meta_data.declare_field<double>(NODE_RANK, "NODE_OMEGA");
  Field<double>& node_force_field = meta_data.declare_field<double>(NODE_RANK, "NODE_FORCE");
  Field<double>& node_torque_field = meta_data.declare_field<double>(NODE_RANK, "NODE_TORQUE");
  Field<double>& node_length_field = meta_data.declare_field<double>(NODE_RANK, "NODE_LENGTH");
  Field<double>& node_radius_field = meta_data.declare_field<double>(NODE_RANK, "NODE_RADIUS");
  Field<unsigned>& node_is_bound_field = meta_data.declare_field<unsigned>(NODE_RANK, "NODE_IS_BOUND");

  Field<double>& elem_rest_length_field = meta_data.declare_field<double>(ELEM_RANK, "ELEM_REST_LENGTH");
  Field<double>& elem_spring_constant_field = meta_data.declare_field<double>(ELEM_RANK, "ELEM_SPRING_CONSTANT");
  Field<double>& elem_ang_spring_constant_field =
      meta_data.declare_field<double>(ELEM_RANK, "ELEM_ANG_SPRING_CONSTANT");
  Field<double>& elem_rest_angle_field = meta_data.declare_field<double>(ELEM_RANK, "ELEM_REST_ANGLE");

  // All parts store the node coords
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);

  // Assemble the microtubule parts
  stk::mesh::put_field_on_mesh(node_quaternion_field, rod_part, 4, nullptr);
  stk::mesh::put_field_on_mesh(node_tangent_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_omega_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_torque_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_length_field, rod_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_radius_field, rod_part, 1, nullptr);

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
  stk::io::put_io_part_attribute(rod_part);
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

  // Create our accessors and aggregates
  auto coord_accessor = Vector3FieldComponent(node_coords_field);
  auto quaternion_accessor = QuaternionFieldComponent(node_quaternion_field);
  auto tangent_accessor = Vector3FieldComponent(node_tangent_field);
  auto velocity_accessor = Vector3FieldComponent(node_velocity_field);
  auto omega_accessor = Vector3FieldComponent(node_omega_field);
  auto force_accessor = Vector3FieldComponent(node_force_field);
  auto torque_accessor = Vector3FieldComponent(node_torque_field);
  auto length_accessor = ScalarFieldComponent(node_length_field);
  auto radius_accessor = ScalarFieldComponent(node_radius_field);
  auto is_bound_accessor = ScalarFieldComponent(node_is_bound_field);
  auto rest_length_accessor = ScalarFieldComponent(elem_rest_length_field);
  auto spring_constant_accessor = ScalarFieldComponent(elem_spring_constant_field);
  auto ang_spring_constant_accessor = ScalarFieldComponent(elem_ang_spring_constant_field);
  auto rest_angle_accessor = ScalarFieldComponent(elem_rest_angle_field);
  auto linked_entities_accessor = FieldComponent(link_meta_data.linked_entities_field());

  // Create the aggregates
  auto rod_agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, rod_part)
                     .add_component<COORDS, NODE_RANK>(coord_accessor)
                     .add_component<QUAT, NODE_RANK>(quaternion_accessor)
                     .add_component<TANGENT, NODE_RANK>(tangent_accessor)
                     .add_component<VELOCITY, NODE_RANK>(velocity_accessor)
                     .add_component<OMEGA, NODE_RANK>(omega_accessor)
                     .add_component<FORCE, NODE_RANK>(force_accessor)
                     .add_component<TORQUE, NODE_RANK>(torque_accessor)
                     .add_component<LENGTH, ELEM_RANK>(length_accessor)
                     .add_component<RADIUS, ELEM_RANK>(radius_accessor);

  auto prc1_agg = make_aggregate<stk::topology::BEAM_2>(bulk_data, prc1_part)
                      .add_component<COORDS, NODE_RANK>(coord_accessor)
                      .add_component<TANGENT, NODE_RANK>(tangent_accessor)
                      .add_component<FORCE, NODE_RANK>(force_accessor)
                      .add_component<TORQUE, NODE_RANK>(torque_accessor)
                      .add_component<IS_BOUND, NODE_RANK>(is_bound_accessor)
                      .add_component<REST_LENGTH, ELEM_RANK>(rest_length_accessor)
                      .add_component<SPRING_CONSTANT, ELEM_RANK>(spring_constant_accessor)
                      .add_component<ANG_SPRING_CONSTANT, ELEM_RANK>(ang_spring_constant_accessor)
                      .add_component<REST_ANGLE, ELEM_RANK>(rest_angle_accessor);

  auto prc1_head_agg = make_ranked_aggregate<NODE_RANK>(bulk_data, prc1_part)
                           .add_component<COORDS, NODE_RANK>(coord_accessor)
                           .add_component<TANGENT, NODE_RANK>(tangent_accessor)
                           .add_component<FORCE, NODE_RANK>(force_accessor)
                           .add_component<TORQUE, NODE_RANK>(torque_accessor)
                           .add_component<IS_BOUND, NODE_RANK>(is_bound_accessor);

  auto slink_agg = make_ranked_aggregate<NODE_RANK>(bulk_data, slink_part)
                       .add_component<LINKED_ENTITIES, NODE_RANK>(linked_entities_accessor);

  // Hard code system params
  size_t num_timesteps = 1;
  double dt = 0.01;

  double rod_radius = 0.5;
  double rod_length = 20;
  math::Quaternion<double> rod_orient = math::Quaternion<double>::identity();
  math::Vector3<double> rod_tangent = rod_orient * math::Vector3<double>(1.0, 0.0, 0.0);
  double init_rod_sep = 2.1;

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
      .add_field_data<double>(&node_coords_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_quaternion_field, {rod_orient[0], rod_orient[1], rod_orient[2], rod_orient[3]})
      .add_field_data<double>(&node_tangent_field, {rod_tangent[0], rod_tangent[1], rod_tangent[2]})
      .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_omega_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_length_field, {rod_length})
      .add_field_data<double>(&node_radius_field, {rod_radius});
  auto center_node2 = dec_helper.create_node();
  center_node2
      .owning_proc(0)  //
      .id(2)           // 1 indexed
      .add_field_data<double>(&node_coords_field, {init_rod_sep, 0.0, 0.0})
      .add_field_data<double>(&node_quaternion_field, {rod_orient[0], rod_orient[1], rod_orient[2], rod_orient[3]})
      .add_field_data<double>(&node_tangent_field, {rod_tangent[0], rod_tangent[1], rod_tangent[2]})
      .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_omega_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_length_field, {rod_length})
      .add_field_data<double>(&node_radius_field, {rod_radius});

  auto mt1 = dec_helper.create_element();
  mt1.owning_proc(0)  //
      .id(1)          //
      .topology(stk::topology::PARTICLE)
      .add_part(&rod_part)  //
      .nodes({1});

  auto mt2 = dec_helper.create_element();
  mt2.owning_proc(0)  //
      .id(2)          //
      .topology(stk::topology::PARTICLE)
      .add_part(&rod_part)  //
      .nodes({2});

  // Declare the PRC1
  auto prc1_node1 = dec_helper.create_node();
  prc1_node1
      .owning_proc(0)  //
      .id(3)           // 1 indexed
      .add_field_data<double>(&node_coords_field, {0.0, 0.0, 0.1})
      .add_field_data<double>(&node_tangent_field, {rod_tangent[0], rod_tangent[1], rod_tangent[2]})
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0})
      .add_field_data<unsigned>(&node_is_bound_field, {1});

  auto prc1_node2 = dec_helper.create_node();
  prc1_node2
      .owning_proc(0)  //
      .id(4)           // 1 indexed
      .add_field_data<double>(&node_coords_field, {init_rod_sep, 0.0, 0.1})
      .add_field_data<double>(&node_tangent_field, {rod_tangent[0], rod_tangent[1], rod_tangent[2]})
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0})
      .add_field_data<unsigned>(&node_is_bound_field, {1});

  auto prc1_elem = dec_helper.create_element();
  prc1_elem
      .owning_proc(0)  //
      .id(3)           //
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

  // Add the links
  stk::mesh::PartVector locally_owned_link_parts = {&slink_part, &meta_data.locally_owned_part()};
  stk::mesh::Entity e_mt1 = bulk_data.get_entity(ELEM_RANK, 1);
  stk::mesh::Entity e_mt2 = bulk_data.get_entity(ELEM_RANK, 2);
  stk::mesh::Entity e_prc1_head1 = bulk_data.get_entity(NODE_RANK, 3);
  stk::mesh::Entity e_prc1_head2 = bulk_data.get_entity(NODE_RANK, 4);
  MUNDY_THROW_REQUIRE(bulk_data.is_valid(e_mt1) && bulk_data.is_valid(e_mt2) && bulk_data.is_valid(e_prc1_head1) &&
                          bulk_data.is_valid(e_prc1_head2),
                      std::runtime_error, "Invalid entities for link creation");
  auto& partition = link_data.get_partition(link_data.get_partition_key(locally_owned_link_parts));
  partition.increase_request_link_capacity(2);
  partition.request_link(e_mt1, e_prc1_head1);
  partition.request_link(e_mt2, e_prc1_head2);
  link_data.process_requests();

  // Check the links
  stk::mesh::EntityVector e_links;
  stk::mesh::get_entities(bulk_data, NODE_RANK, slink_part, e_links);
  MUNDY_THROW_ASSERT(e_links.size() == 2, std::runtime_error, "Expected 2 links");
  MUNDY_THROW_ASSERT(bulk_data.is_valid(e_links[0]), std::runtime_error, "Link 1 is not valid");
  MUNDY_THROW_ASSERT(bulk_data.is_valid(e_links[1]), std::runtime_error, "Link 2 is not valid");
  MUNDY_THROW_ASSERT(bulk_data.bucket(e_links[0]).member(slink_part), std::runtime_error,
                     "Link 1 is not in slink_part");
  MUNDY_THROW_ASSERT(bulk_data.bucket(e_links[1]).member(slink_part), std::runtime_error,
                     "Link 2 is not in slink_part");
  MUNDY_THROW_ASSERT(bulk_data.bucket(e_links[0]).member(meta_data.locally_owned_part()), std::runtime_error,
                     "Link 1 is not in locally owned part");

  // Write the mesh to file
  size_t step = 1;  // Step = 0 doesn't write out fields...
  stk::io::StkMeshIoBroker stk_io_broker;
  stk_io_broker.property_add(Ioss::Property("MAXIMUM_NAME_LENGTH", 180));
  stk_io_broker.set_bulk_data(bulk_data);
  stk::io::write_mesh_with_fields("will.exo", stk_io_broker, step);

  for (size_t timestep_index = 0; timestep_index < num_timesteps; timestep_index++) {
    // One timestep

    // Zero the velocities, forces, and torques
    mesh::field_fill<double>(0.0, node_velocity_field, stk::ngp::ExecSpace{});
    mesh::field_fill<double>(0.0, node_force_field, stk::ngp::ExecSpace{});
    mesh::field_fill<double>(0.0, node_torque_field, stk::ngp::ExecSpace{});

    // Compute spring forces
    std::cout << "Computing spring forces" << std::endl;
    eval_prc1_force().apply_to(prc1_agg);

    // Compute map surface forces to center forces
    std::cout << "Mapping spring surface forces to center forces" << std::endl;
    map_surface_force_to_com_force(link_data, rod_agg, prc1_head_agg).apply_to(slink_agg);

    // Map center forces to center velocity

    // Write to file

    // Update center positions and orientations (quat and tangent)

    // Reconcile surface node positions and tangents
  }
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