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

// External
#include <openrand/philox.h>  // for openrand::Philox

// C++ core
#include <fstream>   // for std::ofstream
#include <iostream>  // for std::cout, std::endl
#include <numeric>   // for std::accumulate
#include <vector>    // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

// #if TRILINOS_MAJOR_MINOR_VERSION >= 160000

// Kokkos and Kokkos-Kernels
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

// STK IO
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_io/WriteMesh.hpp>

// STK Mesh
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/ForEachEntity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>

// STK Search
#include <stk_search/BoxIdent.hpp>
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_search/Point.hpp>
#include <stk_search/SearchMethod.hpp>
#include <stk_search/Sphere.hpp>

// STK Topology
#include <stk_topology/topology.hpp>

// STK Util
#include <stk_util/ngp/NgpSpaces.hpp>
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// STK Balance
#include <stk_balance/balance.hpp>  // for balanceStkMesh

using ExecSpace = stk::ngp::ExecSpace;
using IdentProc = stk::search::IdentProc<stk::mesh::EntityId, int>;
using BoxIdentProc = stk::search::BoxIdentProc<stk::search::AABB<double>, IdentProc>;
using Intersection = stk::search::IdentProcIntersection<IdentProc, IdentProc>;
using SearchBoxesViewType = Kokkos::View<BoxIdentProc *, ExecSpace>;
using ResultViewType = Kokkos::View<Intersection *, ExecSpace>;
using FastMeshIndicesViewType = Kokkos::View<stk::mesh::FastMeshIndex *, ExecSpace>;

using LocalIdentProc = stk::search::IdentProc<stk::mesh::FastMeshIndex, int>;
using LocalIntersection = stk::search::IdentProcIntersection<LocalIdentProc, LocalIdentProc>;
using LocalResultViewType = Kokkos::View<LocalIntersection *, ExecSpace>;

//! \name Setup
//@{

void generate_particles(stk::mesh::BulkData &bulk_data, const size_t num_particles_global,
                        stk::mesh::Part &particle_part) {
  // get the avenge number of particles per process
  size_t num_particles_local = num_particles_global / bulk_data.parallel_size();

  // num_particles_local isn't guaranteed to divide perfectly
  // add the extra workload to the first r ranks
  size_t remaining_particles = num_particles_global - num_particles_local * bulk_data.parallel_size();
  if (bulk_data.parallel_rank() < remaining_particles) {
    num_particles_local += 1;
  }

  bulk_data.modification_begin();

  std::vector<size_t> requests(bulk_data.mesh_meta_data().entity_rank_count(), 0);
  const size_t num_nodes_requested = num_particles_local;
  const size_t num_elems_requested = num_particles_local;
  requests[stk::topology::NODE_RANK] = num_nodes_requested;
  requests[stk::topology::ELEM_RANK] = num_elems_requested;

  // ex.
  //  requests = { 0, 4,  8}
  //  requests 0 entites of rank 0, 4 entites of rank 1, and 8 entites of rank 2
  //  requested_entities = {0 entites of rank 0, 4 entites of rank 1, 8 entites of rank 2}
  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data.generate_new_entities(requests, requested_entities);

  // associate each particle with a single part
  std::vector<stk::mesh::Part *> add_particle_part(1);
  add_particle_part[0] = &particle_part;

  // set topologies of new entities
  for (int i = 0; i < num_particles_local; i++) {
    stk::mesh::Entity particle_i = requested_entities[num_nodes_requested + i];
    bulk_data.change_entity_parts(particle_i, add_particle_part);
  }

  // the elements should be associated with a topology before they are connected to their nodes/edges
  // set downward relations of entities
  for (int i = 0; i < num_particles_local; i++) {
    stk::mesh::Entity particle_i = requested_entities[num_nodes_requested + i];
    bulk_data.declare_relation(particle_i, requested_entities[i], 0);
  }

  bulk_data.modification_end();
}

void randomize_position_and_orientation(const stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &spherocylinders,
                                        const stk::mesh::NgpField<double> &node_coordinates,
                                        const stk::mesh::NgpField<double> &element_orientation,
                                        const mundy::math::Vector3<double> &domain_low,
                                        const mundy::math::Vector3<double> &domain_high) {
  node_coordinates.sync_to_device();
  element_orientation.sync_to_device();

  // Note, we use z as the reference axis. The quaternion will take the z axis to the tangent of the spherocylinder.
  const some_counter = 1234;
  constexpr mundy::math::Vector3<double> z_hat(0.0, 0.0, 1.0);
  constexpr double two_pi = 2.0 * Kokkos::numbers::pi_v<double>;

  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEM_RANK, spherocylinders, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &elem_index) {
        const stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, elem_index);
        const stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(nodes[0]);

        // Initialize the random number generator
        // Use the ID of the element as the seed and choose an arbitrary counter.
        stk::mesh::Entity elem = ngp_mesh.get_entity(stk::topology::ELEM_RANK, elem_index);
        stk::mesh::EntityId elem_id = npg_mesh.identifier(elem);
        openrand::Philox rng(elem_id, some_counter);

        // Randomize the position of the node
        auto node_position = mundy::mesh::vector3_field_data<double>(node_coordinates, node_index);
        node_position[0] = openrand::uniform<double>(domain_low[0], domain_high[0]);
        node_position[1] = openrand::uniform<double>(domain_low[1], domain_high[1]);
        node_position[2] = openrand::uniform<double>(domain_low[2], domain_high[2]);

        // Randomize the orientation of the element
        auto orientation = mundy::mesh::quaternion_field_data<double>(element_orientation, elem_index);
        const double zrand = rng.rand<double>() - 1.0;
        const double wrand = std::sqrt(1.0 - zrand * zrand);
        const double trand = two_pi * rng.rand<double>();
        mundy::math::Vector3<double> u_hat{wrand * Kokkos::cos(trand), wrand * Kokkos::sin(trand), zrand};
        orientation = mundy::math::quat_from_parallel_transport(z_hat, u_hat);
      });

  node_coordinates.modify_on_device();
  element_orientation.modify_on_device();
}
//@}

//! \name Search
//@{

// Create local entities on host and copy to device
FastMeshIndicesViewType get_local_entity_indices(const stk::mesh::BulkData &bulk_data, stk::mesh::EntityRank rank,
                                                 stk::mesh::Selector selector) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(bulk_data, rank, selector, local_entities);

  FastMeshIndicesViewType mesh_indices("mesh_indices", local_entities.size());
  FastMeshIndicesViewType::HostMirror host_mesh_indices =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, mesh_indices);

  for (size_t i = 0; i < local_entities.size(); ++i) {
    const stk::mesh::MeshIndex &mesh_index = bulk_data.mesh_index(local_entities[i]);
    host_mesh_indices(i) = stk::mesh::FastMeshIndex{mesh_index.bucket->bucket_id(), mesh_index.bucket_ordinal};
  }

  Kokkos::deep_copy(mesh_indices, host_mesh_indices);
  return mesh_indices;
}

LocalResultViewType get_local_neighbor_indices(const stk::mesh::BulkData &bulk_data, stk::mesh::EntityRank rank,
                                               const ResultViewType &search_results) {
  auto host_search_results = Kokkos::create_mirror_view_and_copy(stk::ngp::HostExecSpace(), search_results);

  // For each search result, get the local indices and store them in a view
  LocalResultViewType local_neighbor_indices("local_neighbor_indices", search_results.size());
  LocalResultViewType::HostMirror host_local_neighbor_indices =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, local_neighbor_indices);

  for (size_t i = 0; i < search_results.size(); ++i) {
    const auto search_result = host_search_results(i);

    stk::mesh::Entity source_entity = bulk_data.get_entity(rank, search_result.domainIdentProc.id());
    stk::mesh::Entity target_entity = bulk_data.get_entity(rank, search_result.rangeIdentProc.id());

    const stk::mesh::MeshIndex &source_mesh_index = bulk_data.mesh_index(source_entity);
    const stk::mesh::MeshIndex &target_mesh_index = bulk_data.mesh_index(target_entity);

    const stk::mesh::FastMeshIndex source_fast_mesh_index = {source_mesh_index.bucket->bucket_id(),
                                                             source_mesh_index.bucket_ordinal};
    const stk::mesh::FastMeshIndex target_fast_mesh_index = {target_mesh_index.bucket->bucket_id(),
                                                             target_mesh_index.bucket_ordinal};

    host_local_neighbor_indices(i) =
        LocalIntersection{LocalIdentProc{source_fast_mesh_index, search_result.domainIdentProc.proc()},
                          LocalIdentProc{target_fast_mesh_index, search_result.rangeIdentProc.proc()}};
  }

  Kokkos::deep_copy(local_neighbor_indices, host_local_neighbor_indices);
  return local_neighbor_indices;
}

SearchBoxesViewType create_search_aabbs(const stk::mesh::BulkData &bulk_data, const stk::mesh::NgpMesh &ngp_mesh,
                                        const stk::mesh::Selector &spherocylinders,
                                        const stk::mesh::NgpField<double> &node_coordinates,
                                        const stk::mesh::NgpField<double> &element_orientation,
                                        const stk::mesh::NgpField<double> &element_radius,
                                        const stk::mesh::NgpField<double> &element_length) {
  auto locally_owned_spherocylinders = spherocylinders & bulk_data.mesh_meta_data().locally_owned_part();
  const unsigned num_local_spherocylinders =
      stk::mesh::count_entities(bulk_data, stk::topology::ELEM_RANK, locally_owned_spherocylinders);
  SearchBoxesViewType search_aabbs("search_aabbs", num_local_spherocylinders);

  // Slow host operation that is needed to get an index. There is plans to add this to the stk::mesh::NgpMesh.
  FastMeshIndicesViewType spherocylinder_indices =
      get_local_entity_indices(bulk_data, stk::topology::ELEM_RANK, locally_owned_spherocylinders);
  const int my_rank = bulk_data.parallel_rank();

  Kokkos::parallel_for(
      stk::ngp::DeviceRangePolicy(0, num_local_spherocylinders), KOKKOS_LAMBDA(const unsigned &i) {
        stk::mesh::Entity spherocylinder = ngp_mesh.get_entity(stk::topology::ELEM_RANK, spherocylinder_indices(i));
        stk::mesh::FastMeshIndex spherocylinder_index = ngp_mesh.fast_mesh_index(spherocylinder);

        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, spherocylinder_index);
        stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(nodes[0]);

        const auto center = mundy::mesh::vector3_field_data<double>(node_coords, node_index);
        const auto orientation = mundy::mesh::quaternion_field_data<double>(element_orientation, spherocylinder_index);
        double radius = element_radius(spherocylinder_index, 0);
        double length = element_length(spherocylinder_index, 0);

        // The AABB for a spherocylinder
        // TODO(palmerb4): Validate that the quaternion maps from the ref config to the current config
        const double half_length = 0.5 * length;
        const auto unit_tangent = orientation * mundy::math::Vector3<double>{0.0, 0.0, 1.0};
        const auto left_point = center - half_length * unit_tangent;
        const auto right_point = center + half_length * unit_tangent;

        search_aabbs(i) = BoxIdentProc{Kokkos::min(left_point[0], right_point[0]),  //
                                       Kokkos::min(left_point[1], right_point[1]),  //
                                       Kokkos::min(left_point[2], right_point[2]),  //
                                       Kokkos::max(left_point[0], right_point[0]),  //
                                       Kokkos::max(left_point[1], right_point[1]),  //
                                       Kokkos::max(left_point[2], right_point[2]),  //
                                       IdentProc(ngp_mesh.identifier(spherocylinder), my_rank)};
      });

  return search_aabbs;
}
//@}

//! \name Physics
//@{

void apply_hertzian_contact_between_spherocylinders(
    stk::mesh::NgpMesh &ngp_mesh, const LocalResultViewType &local_search_results, const double youngs_modulus,
    const double poisson_ratio, stk::mesh::NgpField<double> &node_coordinates,
    stk::mesh::NgpField<double> &element_orientation, stk::mesh::NgpField<double> &element_radius,
    stk::mesh::NgpField<double> &element_length, stk::mesh::NgpField<double> &node_forces,
    stk::mesh::NgpField<double> &node_torques) {
  node_coordinates.sync_to_device();
  element_orientation.sync_to_device();
  element_radius.sync_to_device();
  element_length.sync_to_device();
  node_forces.sync_to_device();

  const double effective_youngs_modulus =
      (youngs_modulus * youngs_modulus) / (youngs_modulus - youngs_modulus * poisson_ratio * poisson_ratio +
                                           youngs_modulus - youngs_modulus * poisson_ratio * poisson_ratio);
  constexpr double four_thirds = 4.0 / 3.0;

  Kokkos::parallel_for(
      stk::ngp::DeviceRangePolicy(0, local_search_results.size()), KOKKOS_LAMBDA(const unsigned &i) {
        const auto search_result = local_search_results(i);

        stk::mesh::FastMeshIndex source_entity_index = search_result.domainIdentProc.id();
        stk::mesh::FastMeshIndex target_entity_index = search_result.rangeIdentProc.id();
        stk::mesh::Entity source_node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_entity_index)[0];
        stk::mesh::Entity target_node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, target_entity_index)[0];

        stk::mesh::FastMeshIndex source_node_index = ngp_mesh.fast_mesh_index(source_node);
        stk::mesh::FastMeshIndex target_node_index = ngp_mesh.fast_mesh_index(target_node);

        const auto source_coords = mundy::mesh::vector3_field_data<double>(node_coordinates, source_node_index);
        const auto target_coords = mundy::mesh::vector3_field_data<double>(node_coordinates, target_node_index);
        const auto source_orientation =
            mundy::mesh::quaternion_field_data<double>(element_orientation, source_entity_index);
        const auto target_orientation =
            mundy::mesh::quaternion_field_data<double>(element_orientation, target_entity_index);
        const double source_radius = element_radius(source_entity_index, 0);
        const double target_radius = element_radius(target_entity_index, 0);
        const double source_length = element_length(source_entity_index, 0);
        const double target_length = element_length(target_entity_index, 0);

        mundy::geom::Spherocylinder<double> source_spherocylinder{source_coords, source_orientation, source_radius,
                                                                  source_length};
        mundy::geom::Spherocylinder<double> target_spherocylinder{target_coords, target_orientation, target_radius,
                                                                  target_length};

        mundy::geom::Point<double> source_contact_point;
        mundy::geom::Point<double> target_contact_point;
        double source_contact_point_arch_length;
        double target_contact_point_arch_length;
        mundy::math::Vector3<double> source_to_target_centerline_sep;
        const double signed_sep_dist =
            mundy::geom::distance(source_spherocylinder, target_spherocylinder,                        //
                                  source_contact_point, target_contact_point,                          //
                                  source_contact_point_arch_length, target_contact_point_arch_length,  //
                                  source_to_target_centerline_sep);

        if (signed_sep_dist < 0.0) {
          auto source_forces = mundy::mesh::vector3_field_data<double>(node_forces, source_node_index);
          auto target_forces = mundy::mesh::vector3_field_data<double>(node_forces, target_node_index);
          auto source_torques = mundy::mesh::vector3_field_data<double>(node_torques, source_node_index);
          auto target_torques = mundy::mesh::vector3_field_data<double>(node_torques, target_node_index);

          const double effective_radius = (source_radius * target_radius) / (source_radius + target_radius);
          const double normal_force_magnitude_scaled =
              four_thirds * effective_youngs_modulus * Kokkos::sqrt(effective_radius) *
              Kokkos::pow(-signed_separation_distance, 1.5) / (-signed_sep_dist);

          source_forces += normal_force_magnitude_scaled * source_to_target_centerline_sep;
          target_forces -= normal_force_magnitude_scaled * source_to_target_centerline_sep;
          source_torque = mundy::math::cross(source_contact_point - source_coords,  //
                                             normal_force_magnitude_scaled * source_to_target_centerline_sep);
          target_torque = mundy::math::cross(target_contact_point - target_coords,  //
                                             -normal_force_magnitude_scaled * source_to_target_centerline_sep);
        }
      });

  node_forces.modify_on_device();
}

/// \brief Rotate a quaternion by omega dt
///
/// Delong, JCP, 2015, Appendix A eq1, not linearized
///
/// \param q The quaternion to rotate
/// \param omega The angular velocity
/// \param dt The time
template <mundy::math::ValidQuaternionType QuaternionType, mundy::math::ValidVectorType VectorType>
void rotate_quaternion(QuaternionType &quat, const VectorType &omega, const double &dt) {
  const double w = mundy::math::norm(omega);
  if (w < mundy::math::get_zero_tolerance<double>()) {
    // Omega is zero, no rotation
    return;
  }
  const double winv = 1 / w;
  const double sw = Kokkos::sin(0.5 * w * dt);
  const double cw = Kokkos::cos(0.5 * w * dt);
  const double s = quat.w();
  const auto p = quat.vector();
  const auto xyz = s * sw * omega * winv + cw * p + sw * winv * mundy::math::cross(omega, p);
  quat.w() = s * cw - mundy::math::dot(omega, p) * sw * winv;
  quat.vector() = xyz;
  quat.normalize();
}

void move_spherocylinders(const stk::mesh::NgpMesh &ngp_mesh, const double dt,
                          stk::mesh::NgpField<double> &node_coordinates, stk::mesh::NgpField<double> &elem_orientation,
                          stk::mesh::NgpField<double> &node_velocity,
                          stk::mesh::NgpField<double> &elem_angular_velocity) {
  node_coordinates.sync_to_device();
  elem_orientation.sync_to_device();
  node_velocity.sync_to_device();
  elem_angular_velocity.sync_to_device();

  auto selector = stk::mesh::Selector(*node_coordinates.get_field_base());
  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEM_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sp_index) {
        stk::mesh::Entity node = npg_mesh.get_nodes(stk::topology::ELEM_RANK, sp_index)[0];
        stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);
        stk::mesh::EntityFieldData<double> node_coords = node_coordinates(node_index);
        stk::mesh::EntityFieldData<double> node_vel = node_velocity(node_index);

        auto center = mundy::mesh::vector3_field_data<double>(node_coordinates, node_index);
        auto vel = mundy::mesh::vector3_field_data<double>(node_velocity, node_index);
        auto orientation = mundy::mesh::quaternion_field_data<double>(elem_orientation, sp_index);
        auto omega = mundy::mesh::vector3_field_data<double>(elem_angular_velocity, sp_index);

        center += dt * vel;
        rotate_quaternion(orientation, omega, dt);
      });

  node_coordinates.modify_on_device();
  elem_orientation.modify_on_device();
}
//@}

//! \name Load Balance
//@{

class RcbSettings : public stk::balance::BalanceSettings {
 public:
  RcbSettings() {
  }
  virtual ~RcbSettings() {
  }

  virtual bool isIncrementalRebalance() const {
    return false;
  }
  virtual std::string getDecompMethod() const {
    return std::string("rcb");
  }
  virtual std::string getCoordinateFieldName() const {
    return std::string("COORDS");
  }
  virtual bool shouldPrintMetrics() const {
    return false;
  }
};  // RcbSettings
//@}

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  {
    // Simulation of N spherocylinders in a cube
    const double viscosity = 1.0;
    const double youngs_modulus = 10000.0;
    const double poisson_ratio = 0.3;
    const double radius = 0.1;
    const double length = 1.0;
    const double num_spherocyliners = 1000;
    const mundy::Point<double, 3> domain_low{0.0, 0.0, 0.0};
    const mundy::Point<double, 3> domain_high{100.0, 100.0, 100.0};
    const double time_step = 0.00001;
    const size_t num_time_steps = 1000;
    const size_t io_frequency = 1;

    const double single_particle_volume = 4.0 / 3.0 * M_PI * radius * radius * radius + M_PI * radius * radius * length;
    const double domain_volume = (domain_high[0] - domain_low[0]) *
                                 (domain_high[1] - domain_low[1]) *
                                 (domain_high[2] - domain_low[2]);
    const double volume_fraction = num_spherocyliners * single_particle_volume / domain_volume;
    std::cout << "Setup: " << std::endl;
    std::cout << "  Number of spherocyliners: " << num_spherocyliners << std::endl;
    std::cout << "  Radius: " << radius << std::endl;
    std::cout << "  Length: " << length << std::endl;
    std::cout << "  Volume fraction: " << volume_fraction << std::endl;

    // Setup the STK mesh
    stk::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD);
    mesh_builder.set_spatial_dimension(3);
    mesh_builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = mesh_builder.create_meta_data();
    meta_data_ptr->use_simple_fields();
    meta_data_ptr->set_coordinate_field_name("COORDS");
    std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = mesh_builder.create(meta_data_ptr);
    stk::mesh::MetaData &meta_data = *meta_data_ptr;
    stk::mesh::BulkData &bulk_data = *bulk_data_ptr;

    // Create the spherocyliners
    auto &spherocyliners_part = meta_data.declare_part_with_topology("RODS", stk::topology::PARTICLE);
    stk::io::put_io_part_attribute(spherocyliners_part);
    auto &node_coordinates = meta_data.declare_field<double>(stk::topology::NODE_RANK, "COORDS");
    auto &node_force = meta_data.declare_field<double>(stk::topology::NODE_RANK, "FORCE");
    auto &node_velocity = meta_data.declare_field<double>(stk::topology::NODE_RANK, "VELOCITY");
    auto &element_orientation = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "ORIENTATION");
    auto &element_radius = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "RADIUS");
    auto &element_length = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "LENGTH");
    auto &element_angular_velocity = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "ANGULAR_VELOCITY");

    // Assign fields to parts
    stk::mesh::put_field_on_mesh(node_coordinates, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(node_force, spherocyliners_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(node_velocity, spherocyliners_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(element_orientation, spherocyliners_part, 4, nullptr);
    stk::mesh::put_field_on_mesh(element_radius, spherocyliners_part, 1, nullptr);
    stk::mesh::put_field_on_mesh(element_length, spherocyliners_part, 1, nullptr);
    stk::mesh::put_field_on_mesh(element_angular_velocity, spherocyliners_part, 3, nullptr);
    meta_data.commit();

    // Generate the particles
    generate_particles(bulk_data, num_spherocyliners, spherocyliners_part);

     // Get the NGP stuff
    stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
    auto &ngp_node_coordinates = stk::mesh::get_updated_ngp_field<double>(node_coordinates);
    auto &ngp_node_force = stk::mesh::get_updated_ngp_field<double>(node_force);
    auto &ngp_node_velocity = stk::mesh::get_updated_ngp_field<double>(node_velocity);
    auto &ngp_element_orientation = stk::mesh::get_updated_ngp_field<double>(element_orientation);
    auto &ngp_element_radius = stk::mesh::get_updated_ngp_field<double>(element_radius);
    auto &ngp_element_length = stk::mesh::get_updated_ngp_field<double>(element_length);
    auto &ngp_element_angular_velocity = stk::mesh::get_updated_ngp_field<double>(element_angular_velocity);

    // Randomize the positions and radii
    randomize_position_and_orientation(ngp_mesh, spherocyliners_part, ngp_node_coordinates, ngp_element_orientation,
                                       domain_low, domain_high);
    // Balance the mesh
    stk::balance::balanceStkMesh(RcbSettings{}, bulk_data);

    // Timeloop
    bool rebuild_neighbors = true;
    ResultViewType search_results;
    LocalResultViewType local_search_results;
    SearchSpheresViewType search_spherocyliners;
    Kokkos::Timer tps_timer;
    for (size_t time_step = 0; time_step < num_time_steps; ++time_step) {
      if (time_step % io_frequency == 0) {
        std::cout << "Time step: " << time_step << " running at "
                  << static_cast<double>(io_frequency) / tps_timer.seconds() << " tps." << std::endl;
        tps_timer.reset();
        // Comm fields to host
        // ngp_node_coordinates.sync_to_host();
        // ngp_node_force.sync_to_host();
        // ngp_node_velocity.sync_to_host();
        // ngp_element_radius.sync_to_host();

        // Write to file
        // stk::io::write_mesh("spherocyliners_" + std::to_string(time_step) + ".e", bulk_data, stk::io::WRITE_RESULTS);
      }

      if (time_step % io_frequency == 0) {
        Kokkos::Timer create_search_timer;
        search_spherocyliners = create_search_spherocyliners(bulk_data, ngp_mesh, spherocyliners_part,
                                                             ngp_node_coordinates, ngp_element_radius);
        std::cout << "Create search spherocyliners time: " << create_search_timer.seconds() << std::endl;

        Kokkos::Timer search_timer;
        stk::search::SearchMethod search_method = stk::search::MORTON_LBVH;

        stk::ngp::ExecSpace exec_space = Kokkos::DefaultExecutionSpace{};
        const bool results_parallel_symmetry = true;   // create source -> target and target -> source pairs
        const bool auto_swap_domain_and_range = true;  // swap source and target if target is owned and source is not
        const bool sort_search_results = true;         // sort the search results by source id
        stk::search::coarse_search(search_spherocyliners, search_spherocyliners, search_method, bulk_data.parallel(),
                                   search_results, exec_space, results_parallel_symmetry);
        std::cout << "Search time: " << search_timer.seconds() << " with " << search_results.size() << " results."
                  << std::endl;

        // Ghost the non-owned spherocyliners
        Kokkos::Timer ghost_timer;
        ghost_neighbors(bulk_data, search_results);
        std::cout << "Ghost time: " << ghost_timer.seconds() << std::endl;

        // Create local neighbor indices
        Kokkos::Timer local_index_conversion_timer;
        local_search_results = get_local_neighbor_indices(bulk_data, stk::topology::ELEM_RANK, search_results);
        std::cout << "Local index conversion time: " << local_index_conversion_timer.seconds() << std::endl;
      }

      Kokkos::Timer contact_timer;
      apply_hertzian_contact_between_spherocyliners(ngp_mesh, local_search_results, youngs_modulus, poisson_ratio,
                                                    ngp_node_coordinates, ngp_node_force, ngp_element_radius);
      std::cout << "Contact time: " << contact_timer.seconds() << std::endl;

      Kokkos::Timer update_timer;
      move_spherocylinders(ngp_mesh, time_step, ngp_node_coordinates, ngp_node_velocity);
      std::cout << "Update time: " << update_timer.seconds() << std::endl;
    }
  }

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}

// #else

// int main() {
//   std::cout << "TEST DISABLED. Trilinos version must be at least 16.0.0." << std::endl;
//   return 0;
// }

// #endif  // TRILINOS_MAJOR_MINOR_VERSION
