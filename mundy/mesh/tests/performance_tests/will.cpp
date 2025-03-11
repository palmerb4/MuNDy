
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

// Openrand
#include <openrand/philox.h>  // for openrand::Philox

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
#include <mundy_geom/distance.hpp>         // for mundy::geom::distance
#include <mundy_geom/primitives.hpp>       // for mundy::geom::Spherocylinder
#include <mundy_math/Quaternion.hpp>       // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>          // for mundy::math::Vector3
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
struct ARCLENGTH {};
struct OMEGA {};
struct IS_BOUND {};
struct LINKED_ENTITIES {};
struct QUAT {};
struct TANGENT {};
struct REST_LENGTH {};
struct SPRING_CONSTANT {};
struct ANG_SPRING_CONSTANT {};
struct REST_COS_ANGLE {};
struct RNG_COUNTER {};

namespace mundy {

class apply_brownian_motion {
 public:
  apply_brownian_motion(double dt, double kbt)
      : dt_(dt), kbt_(kbt), kbt_coeff_(Kokkos::sqrt(2.0 * kbt_ / dt_)), a_small_number_(0.1 * dt_) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& rod_view) const {
    // Setup the rng. Seed by rod id and use the rod's counter
    stk::mesh::EntityId rod_id = rod_view.entity_id();
    auto rod_counter = get<RNG_COUNTER>(rod_view);
    openrand::Philox rng(rod_id, rod_counter[0]);
    rod_counter[0]++;

    // Node data
    auto velocity = get<VELOCITY>(rod_view, 0);
    auto omega = get<OMEGA>(rod_view, 0);
    auto quat = get<QUAT>(rod_view, 0);

    // Rod data
    double radius = get<RADIUS>(rod_view)[0];
    double length = get<LENGTH>(rod_view)[0];

    // RFD from Delong, JCP, 2015
    // Slender fiber has 0 rot drag about the long axis, regularize with identity rot mobility
    const double inv_drag_para = 1.0;
    const double inv_drag_perp = 1.0;
    const double inv_drag_rot = 1.0;

    auto tangent = quat * math::Vector3<double>(0.0, 0.0, 1.0);

    auto n_mat = (inv_drag_para - inv_drag_perp) * math::outer_product(tangent, tangent) +
                 inv_drag_perp * math::Matrix3<double>::identity();
    auto n_mat_sqrt = math::cholesky(n_mat);  // This is just L in N = LL^T

    // Drift and diffusion
    math::Vector3<double> w_rot(rng.randn<double>(), rng.randn<double>(), rng.randn<double>());
    math::Vector3<double> w_pos(rng.randn<double>(), rng.randn<double>(), rng.randn<double>());
    math::Vector3<double> w_rfd_rot(rng.randn<double>(), rng.randn<double>(), rng.randn<double>());

    math::rotate_quaternion(quat, w_rfd_rot, dt_);          // Update the quaternion for the rod
    tangent = quat * math::Vector3<double>(0.0, 0.0, 1.0);  // Propagate to the tangent
    auto n_mat_rfd = (inv_drag_para - inv_drag_perp) * math::outer_product(tangent, tangent) +
                     inv_drag_perp * math::Matrix3<double>::identity();  // Propagate to the mobility matrix

    auto vel_brown = kbt_coeff_ * (n_mat_sqrt * w_pos);                     // Gaussian noise
    vel_brown += (kbt_ / a_small_number_) * ((n_mat_rfd - n_mat) * w_pos);  // RFD drift
    auto omega_brown = Kokkos::sqrt(inv_drag_rot) * kbt_coeff_ * w_rot;     // Regularized identity rotation drag

    // Update the velocity and omega
    velocity += vel_brown;
    omega += omega_brown;
  }

  void apply_to(auto& rod_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Rod must be particle top.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<VELOCITY, OMEGA, QUAT, RNG_COUNTER, RADIUS, LENGTH>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_rod_agg.template modify_on_device<VELOCITY, OMEGA>();
  }

  void apply_to(auto& rod_agg) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Rod must be particle top.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<VELOCITY, OMEGA, QUAT, RNG_COUNTER, RADIUS, LENGTH>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_rod_agg.template modify_on_device<VELOCITY, OMEGA>();
  }

 private:
  double dt_;
  double kbt_;
  double kbt_coeff_;
  double a_small_number_;  // In Delong 2015, this is delta
};

class eval_prc1_force {
 public:
  eval_prc1_force() {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& prc1_view) const {
    auto rA = get<COORDS>(prc1_view, 0);
    auto rB = get<COORDS>(prc1_view, 1);
    auto a1 = get<TANGENT>(prc1_view, 0);
    auto b1 = get<TANGENT>(prc1_view, 1);

    double k_lin = get<SPRING_CONSTANT>(prc1_view)[0];
    double r0_lin = get<REST_LENGTH>(prc1_view)[0];
    double k_ang = get<ANG_SPRING_CONSTANT>(prc1_view)[0];
    double cos_theta0 = get<REST_COS_ANGLE>(prc1_view)[0];

    auto rAB_vec = rA - rB; /* Note the backwards convention from Allen and Germano */
    auto rAB_norm = math::norm(rAB_vec);
    auto rAB_hat = rAB_vec / rAB_norm;

    double delta_cosA = math::dot(a1, rAB_hat) - cos_theta0;
    double delta_cosB = math::dot(b1, rAB_hat) - cos_theta0;
    auto triple_crossA = math::cross(rAB_hat, math::cross(rAB_hat, a1));
    auto triple_crossB = math::cross(rAB_hat, math::cross(rAB_hat, b1));

    auto fA = -k_lin * (rAB_norm - r0_lin) * rAB_hat  //
              + k_ang / rAB_norm * (delta_cosA * triple_crossA + delta_cosB * triple_crossB);
    auto torqueA = k_ang * delta_cosA * math::cross(rAB_hat, a1);
    auto torqueB = k_ang * delta_cosB * math::cross(rAB_hat, b1);

    // We know that the downward connected nodes are only connected to us, therefore we do not need atomics,
    // note that we are also not summing in the forces
    get<FORCE>(prc1_view, 0) = fA;
    get<FORCE>(prc1_view, 1) = -fA;
    get<TORQUE>(prc1_view, 0) = torqueA;
    get<TORQUE>(prc1_view, 1) = torqueB;
  }

  void apply_to(auto& prc1_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(prc1_agg.topology() == stk::topology::BEAM_2, "PRC1 must be beam 2 top.");

    auto ngp_prc1_agg = mesh::get_updated_ngp_aggregate(prc1_agg);
    ngp_prc1_agg.template sync_to_device<COORDS, FORCE, TORQUE, SPRING_CONSTANT, REST_LENGTH, ANG_SPRING_CONSTANT,
                                         REST_COS_ANGLE>();
    ngp_prc1_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_prc1_agg.template modify_on_device<FORCE, TORQUE>();
  }

  void apply_to(auto& prc1_agg) {
    static_assert(prc1_agg.topology() == stk::topology::BEAM_2, "PRC1 must be beam 2 top.");

    auto ngp_prc1_agg = mesh::get_updated_ngp_aggregate(prc1_agg);
    ngp_prc1_agg.template sync_to_device<COORDS, FORCE, TORQUE, SPRING_CONSTANT, REST_LENGTH, ANG_SPRING_CONSTANT,
                                         REST_COS_ANGLE>();
    ngp_prc1_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_prc1_agg.template modify_on_device<FORCE, TORQUE>();
  }
};

template <typename RodAgg, typename SurfaceNodeAgg>
class map_surface_force_to_com_force {
 public:
  map_surface_force_to_com_force(mesh::LinkData& link_data, RodAgg& rod_agg, SurfaceNodeAgg& surface_node_agg)
      : link_data_(link_data), rod_agg_(rod_agg), surface_node_agg_(surface_node_agg) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& slink_view) const {
    // Get the two entities from the slink_view, this is already inside of a parallel loop
    auto rod_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[0]);
    auto surface_node_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[1]);

    // Go from entity view to the aggregate
    auto rod_view = ngp_rod_agg_.get_view(slink_view.ngp_mesh().fast_mesh_index(rod_entity));
    auto surface_node_view = ngp_surface_node_agg_.get_view(slink_view.ngp_mesh().fast_mesh_index(surface_node_entity));

    auto rod_force = get<FORCE>(rod_view, 0 /* center node */);
    auto rod_torque = get<TORQUE>(rod_view, 0);
    auto rod_center = get<COORDS>(rod_view, 0);
    auto surface_node_force = get<FORCE>(surface_node_view);
    auto surface_node_torque = get<TORQUE>(surface_node_view);
    auto surface_node_pos = get<COORDS>(surface_node_view);

    // Add the COM force
    Kokkos::atomic_add(&rod_force[0], surface_node_force[0]);
    Kokkos::atomic_add(&rod_force[1], surface_node_force[1]);
    Kokkos::atomic_add(&rod_force[2], surface_node_force[2]);

    // Add the torques along with the off-center force contribution
    auto torque = surface_node_torque + math::cross(surface_node_pos - rod_center, surface_node_force);
    Kokkos::atomic_add(&rod_torque[0], torque[0]);
    Kokkos::atomic_add(&rod_torque[1], torque[1]);
    Kokkos::atomic_add(&rod_torque[2], torque[2]);
  }

  void apply_to(auto& slink_agg, const stk::mesh::Selector& subset_selector) {
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);
    // Explicitly construct a local instance of the NGP rod_agg and surface_node_agg
    ngp_rod_agg_ = mesh::get_updated_ngp_aggregate(rod_agg_);
    ngp_surface_node_agg_ = mesh::get_updated_ngp_aggregate(surface_node_agg_);

    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_rod_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();
    ngp_surface_node_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();

    ngp_slink_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);

    ngp_rod_agg_.template modify_on_device<FORCE, TORQUE>();
  }

  void apply_to(auto& slink_agg) {
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);
    // Explicitly construct a local instance of the NGP rod_agg and surface_node_agg
    ngp_rod_agg_ = mesh::get_updated_ngp_aggregate(rod_agg_);
    ngp_surface_node_agg_ = mesh::get_updated_ngp_aggregate(surface_node_agg_);

    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_rod_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();
    ngp_surface_node_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();

    ngp_slink_agg.template for_each((*this) /*use my operator as a lambda*/);

    ngp_rod_agg_.template modify_on_device<FORCE, TORQUE>();
  }

 private:
  mesh::LinkData& link_data_;
  RodAgg& rod_agg_;
  SurfaceNodeAgg& surface_node_agg_;
  using rod_agg_ngp_t = decltype(mesh::get_updated_ngp_aggregate(std::declval<RodAgg>()));
  using surface_node_agg_ngp_t = decltype(mesh::get_updated_ngp_aggregate(std::declval<SurfaceNodeAgg>()));
  rod_agg_ngp_t ngp_rod_agg_;
  surface_node_agg_ngp_t ngp_surface_node_agg_;
};

class eval_mobility {
 public:
  eval_mobility() {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& rod_view) const {
    // TODO(palmerb4): Update with real drag coefficients
    double drag_para = 1.0;
    double drag_perp = 1.0;
    double drag_rot = 1.0;

    // Center node data:
    auto force = get<FORCE>(rod_view, 0);
    auto torque = get<TORQUE>(rod_view, 0);
    auto velocity = get<VELOCITY>(rod_view, 0);
    auto omega = get<OMEGA>(rod_view, 0);
    auto tangent = get<TANGENT>(rod_view, 0);

    // Rod data:
    double length = get<LENGTH>(rod_view)[0];
    double radius = get<RADIUS>(rod_view)[0];

    // Note, += is safe here without atomic under the assumption that no two particles share a node
    auto force_para = math::dot(force, tangent) * tangent;
    auto force_perp = force - force_para;
    velocity += 1.0 / drag_para * force_para + 1.0 / drag_perp * force_perp;
    omega += 1.0 / drag_rot * torque;
  }

  void apply_to(auto& rod_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Expected a particle topology.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<FORCE, TORQUE, VELOCITY, OMEGA, LENGTH, RADIUS, TANGENT>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_rod_agg.template modify_on_device<VELOCITY, OMEGA>();
  }

  void apply_to(auto& rod_agg) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Expected a particle topology.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<FORCE, TORQUE, VELOCITY, OMEGA, LENGTH, RADIUS, TANGENT>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_rod_agg.template modify_on_device<VELOCITY, OMEGA>();
  }
};

class update_configuration {
 public:
  update_configuration(double dt) : dt_(dt) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& rod_node_view) const {
    // First order Euler
    auto velocity = get<VELOCITY>(rod_node_view);
    auto omega = get<OMEGA>(rod_node_view);
    auto position = get<COORDS>(rod_node_view);
    auto quat = get<QUAT>(rod_node_view);
    auto tangent = get<TANGENT>(rod_node_view);

    // Update position
    position += dt_ * velocity;
    math::rotate_quaternion(quat, omega, dt_);
    tangent = quat * math::Vector3<double>(0.0, 0.0, 1.0);
  }

  void apply_to(auto& rod_node_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(rod_node_agg.rank() == stk::topology::NODE_RANK, "Expected the rod centers.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_node_agg);
    ngp_rod_agg.template sync_to_device<VELOCITY, OMEGA, COORDS, QUAT, TANGENT>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_rod_agg.template modify_on_device<COORDS, QUAT, TANGENT>();
  }

  void apply_to(auto& rod_node_agg) {
    static_assert(rod_node_agg.rank() == stk::topology::NODE_RANK, "Expected the rod centers.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_node_agg);
    ngp_rod_agg.template sync_to_device<VELOCITY, OMEGA, COORDS, QUAT, TANGENT>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_rod_agg.template modify_on_device<COORDS, QUAT, TANGENT>();
  }

  double dt_;
};

template <typename RodAgg, typename SurfaceNodeAgg>
class reconcile_surface_nodes {
 public:
  reconcile_surface_nodes(mesh::LinkData& link_data, RodAgg& rod_agg, SurfaceNodeAgg& surface_node_agg)
      : link_data_(link_data), rod_agg_(rod_agg), surface_node_agg_(surface_node_agg) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& slink_view) const {
    // Take in the surface link view and update the surface node positions/tangents using their arclength
    // and the center/tangent of the rod. Arclength is measured from the left endpoint of the rod

    // Get the two entities from the slink_view, this is already inside of a parallel loop
    auto rod_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[0]);
    auto surface_node_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[1]);

    // Go from entity view to the aggregate
    auto rod_view = ngp_rod_agg_.get_view(slink_view.ngp_mesh().fast_mesh_index(rod_entity));
    auto surface_node_view = ngp_surface_node_agg_.get_view(slink_view.ngp_mesh().fast_mesh_index(surface_node_entity));

    // Node data:
    auto rod_center = get<COORDS>(rod_view, 0);
    auto rod_tangent = get<TANGENT>(rod_view, 0);

    // Rod data:
    double rod_length = get<LENGTH>(rod_view)[0];

    // Surface node data:
    auto surface_node_pos = get<COORDS>(surface_node_view);
    auto surface_node_tangent = get<TANGENT>(surface_node_view);
    double surface_node_arclength = get<ARCLENGTH>(surface_node_view)[0];

    // Update the position of the surface node
    surface_node_pos = rod_center + (surface_node_arclength - 0.5 * rod_length) * rod_tangent;
    surface_node_tangent = rod_tangent;
  }

  void apply_to(auto& slink_agg, const stk::mesh::Selector& subset_selector) {
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);
    ngp_rod_agg_ = mesh::get_updated_ngp_aggregate(rod_agg_);
    ngp_surface_node_agg_ = mesh::get_updated_ngp_aggregate(surface_node_agg_);

    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_rod_agg_.template sync_to_device<COORDS, TANGENT, LENGTH>();
    ngp_surface_node_agg_.template sync_to_device<COORDS, TANGENT, ARCLENGTH>();

    ngp_slink_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);

    ngp_surface_node_agg_.template modify_on_device<COORDS, TANGENT>();
  }

  void apply_to(auto& slink_agg) {
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);
    ngp_rod_agg_ = mesh::get_updated_ngp_aggregate(rod_agg_);
    ngp_surface_node_agg_ = mesh::get_updated_ngp_aggregate(surface_node_agg_);

    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_rod_agg_.template sync_to_device<COORDS, TANGENT, LENGTH>();
    ngp_surface_node_agg_.template sync_to_device<COORDS, TANGENT, ARCLENGTH>();

    ngp_slink_agg.template for_each((*this) /*use my operator as a lambda*/);

    ngp_surface_node_agg_.template modify_on_device<COORDS, TANGENT>();
  }

 private:
  mesh::LinkData& link_data_;
  RodAgg& rod_agg_;
  SurfaceNodeAgg& surface_node_agg_;
  using rod_agg_ngp_t = decltype(mesh::get_updated_ngp_aggregate(std::declval<RodAgg>()));
  using surface_node_agg_ngp_t = decltype(mesh::get_updated_ngp_aggregate(std::declval<SurfaceNodeAgg>()));
  rod_agg_ngp_t ngp_rod_agg_;
  surface_node_agg_ngp_t ngp_surface_node_agg_;
};

//! \name Temporary N^2 contact
//@{

Kokkos::View<stk::mesh::FastMeshIndex*, stk::ngp::ExecSpace> get_local_entity_indices(
    const stk::mesh::BulkData& bulk_data, stk::mesh::EntityRank rank, stk::mesh::Selector selector) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(bulk_data, rank, selector, local_entities);

  Kokkos::View<stk::mesh::FastMeshIndex*, stk::ngp::ExecSpace> mesh_indices("mesh_indices", local_entities.size());
  Kokkos::View<stk::mesh::FastMeshIndex*, stk::ngp::ExecSpace>::HostMirror host_mesh_indices =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, mesh_indices);

  for (size_t i = 0; i < local_entities.size(); ++i) {
    const stk::mesh::MeshIndex& mesh_index = bulk_data.mesh_index(local_entities[i]);
    host_mesh_indices(i) = stk::mesh::FastMeshIndex{mesh_index.bucket->bucket_id(), mesh_index.bucket_ordinal};
  }

  Kokkos::deep_copy(mesh_indices, host_mesh_indices);
  return mesh_indices;
}

class apply_hertzian_contact {
 public:
  apply_hertzian_contact(const double youngs_modulus, const double poisson_ratio)
      : youngs_modulus_(youngs_modulus), poisson_ratio_(poisson_ratio) {
  }

  void apply_to(auto& rod_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Expected the rods to be particles.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<COORDS, TANGENT, RADIUS, LENGTH, FORCE, TORQUE>();

    auto ngp_rod_entities = get_local_entity_indices(rod_agg.bulk_data(), rod_agg.rank(), subset_selector);
    impl_run(ngp_rod_entities, ngp_rod_agg);

    ngp_rod_agg.template modify_on_device<FORCE, TORQUE>();
  }

  void apply_to(auto& rod_agg) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Expected the rods to be particles.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<COORDS, TANGENT, RADIUS, LENGTH, FORCE, TORQUE>();

    auto ngp_rod_entities = get_local_entity_indices(rod_agg.bulk_data(), rod_agg.rank(), rod_agg.selector());
    impl_run(ngp_rod_entities, ngp_rod_agg);

    ngp_rod_agg.template modify_on_device<FORCE, TORQUE>();
  }

 private:
  void impl_run(Kokkos::View<stk::mesh::FastMeshIndex*, stk::ngp::ExecSpace>& ngp_rod_entities, auto& ngp_rod_agg) {
    // Get the entities we want to act on
    const size_t num_rods = ngp_rod_entities.extent(0);
    const double effective_youngs_modulus =
        (youngs_modulus_ * youngs_modulus_) / (youngs_modulus_ - youngs_modulus_ * poisson_ratio_ * poisson_ratio_ +
                                               youngs_modulus_ - youngs_modulus_ * poisson_ratio_ * poisson_ratio_);
    Kokkos::parallel_for(
        "apply_hertzian_contact", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_rods, num_rods}),
        KOKKOS_LAMBDA(const size_t t, const size_t s) {
          if (s >= t) {
            return;  // skip self interaction and duplication
          }
          auto rod1_fast_mesh_index = ngp_rod_entities(t);
          auto rod2_fast_mesh_index = ngp_rod_entities(s);

          auto rod1_view = ngp_rod_agg.get_view(rod1_fast_mesh_index);
          auto rod2_view = ngp_rod_agg.get_view(rod2_fast_mesh_index);

          // The signed separation distance is the distance between the centerline of the two spherocylinders minus the
          // sum of their radii
          auto rod1_coords = get<COORDS>(rod1_view, 0 /* node data*/);
          auto rod2_coords = get<COORDS>(rod2_view, 0);
          auto rod1_tangent = get<TANGENT>(rod1_view, 0);
          auto rod2_tangent = get<TANGENT>(rod2_view, 0);
          double rod1_radius = get<RADIUS>(rod1_view)[0];
          double rod2_radius = get<RADIUS>(rod2_view)[0];
          double rod1_length = get<LENGTH>(rod1_view)[0];
          double rod2_length = get<LENGTH>(rod2_view)[0];
          geom::LineSegment<double> rod1_centerline{rod1_coords - 0.5 * rod1_length * rod1_tangent,
                                                    rod1_coords + 0.5 * rod1_length * rod1_tangent};
          geom::LineSegment<double> rod2_centerline{rod2_coords - 0.5 * rod2_length * rod2_tangent,
                                                    rod2_coords + 0.5 * rod2_length * rod2_tangent};

          geom::Point<double> rod1_centerline_contact_point, rod2_centerline_contact_point;
          double rod1_contact_point_arch_length, rod2_contact_point_arch_length;
          math::Vector3<double> rod1_to_rod2_centerline_sep;
          const double signed_sep_dist =
              geom::distance(rod1_centerline, rod2_centerline,                                //
                             rod1_centerline_contact_point, rod2_centerline_contact_point,    //
                             rod1_contact_point_arch_length, rod2_contact_point_arch_length,  //
                             rod1_to_rod2_centerline_sep) -
              rod1_radius - rod2_radius;

          if (signed_sep_dist < 0.0) {
            const double effective_radius = (rod1_radius * rod2_radius) / (rod1_radius + rod2_radius);
            constexpr double four_thirds = 4.0 / 3.0;
            const double force_mag = four_thirds * effective_youngs_modulus * Kokkos::sqrt(effective_radius) *
                                     Kokkos::pow(-signed_sep_dist, 1.5);

            // Mind the signs. rod1 to rod2 is normal to the rod1 and contact forces act along the negative normal
            auto rod1_force = get<FORCE>(rod1_view, 0);
            auto rod2_force = get<FORCE>(rod2_view, 0);
            auto rod1_torque = get<TORQUE>(rod1_view, 0);
            auto rod2_torque = get<TORQUE>(rod2_view, 0);

            auto rod1_to_rod2_normal = rod1_to_rod2_centerline_sep / math::norm(rod1_to_rod2_centerline_sep);

            auto local_rod1_force = -force_mag * rod1_to_rod2_normal;
            auto local_rod2_force = force_mag * rod1_to_rod2_normal;
            auto local_rod1_torque = math::cross(rod1_centerline_contact_point - rod1_coords,  //
                                                 -force_mag * rod1_to_rod2_normal);
            auto local_rod2_torque = math::cross(rod2_centerline_contact_point - rod2_coords,  //
                                                 force_mag * rod1_to_rod2_normal);
            Kokkos::atomic_add(&rod1_force[0], local_rod1_force[0]);
            Kokkos::atomic_add(&rod1_force[1], local_rod1_force[1]);
            Kokkos::atomic_add(&rod1_force[2], local_rod1_force[2]);
            Kokkos::atomic_add(&rod2_force[0], local_rod2_force[0]);
            Kokkos::atomic_add(&rod2_force[1], local_rod2_force[1]);
            Kokkos::atomic_add(&rod2_force[2], local_rod2_force[2]);
           
            Kokkos::atomic_add(&rod1_torque[0], local_rod1_torque[0]);
            Kokkos::atomic_add(&rod1_torque[1], local_rod1_torque[1]);
            Kokkos::atomic_add(&rod1_torque[2], local_rod1_torque[2]);
            Kokkos::atomic_add(&rod2_torque[0], local_rod2_torque[0]);
            Kokkos::atomic_add(&rod2_torque[1], local_rod2_torque[1]);
            Kokkos::atomic_add(&rod2_torque[2], local_rod2_torque[2]);
          }
        });
  }

  double youngs_modulus_;
  double poisson_ratio_;
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
  meta_data_ptr->set_coordinate_field_name("COORDS");
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
  //  Node fields: (Only what is logical to potentially share with other entities that connect to the node)
  //   - COORDS
  //   - QUATERNION
  //   - TANGENT
  //   - VELOCITY
  //   - OMEGA
  //   - FORCE
  //   - TORQUE
  //
  //  Elem fields:
  //   - LENGTH
  //   - RADIUS
  //
  // PRC1: BEAM_2 top
  //  Node fields:
  //  - ARCLENGTH
  //  - COORDS
  //  - TANGENT
  //  - FORCE
  //  - TORQUE
  //  - IS_BOUND
  //
  //  Elem fields:
  //  - REST_LENGTH
  //  - SPRING_CONSTANT
  //  - ANG_SPRING_CONSTANT
  //  - REST_COS_ANGLE
  //
  // Surface Links: NODE rank but no fields for now
  auto& node_coords_field = meta_data.declare_field<double>(NODE_RANK, "COORDS");
  auto& node_quaternion_field = meta_data.declare_field<double>(NODE_RANK, "QUATERNION");
  auto& node_arclength_field = meta_data.declare_field<double>(NODE_RANK, "ARCLENGTH");
  auto& node_tangent_field = meta_data.declare_field<double>(NODE_RANK, "TANGENT");
  auto& node_velocity_field = meta_data.declare_field<double>(NODE_RANK, "VELOCITY");
  auto& node_omega_field = meta_data.declare_field<double>(NODE_RANK, "OMEGA");
  auto& node_force_field = meta_data.declare_field<double>(NODE_RANK, "FORCE");
  auto& node_torque_field = meta_data.declare_field<double>(NODE_RANK, "TORQUE");
  auto& node_is_bound_field = meta_data.declare_field<unsigned>(NODE_RANK, "IS_BOUND");

  auto& elem_length_field = meta_data.declare_field<double>(ELEM_RANK, "LENGTH");
  auto& elem_radius_field = meta_data.declare_field<double>(ELEM_RANK, "RADIUS");
  auto& elem_rng_counter_field = meta_data.declare_field<double>(ELEM_RANK, "RNG_COUNTER");
  auto& elem_rest_length_field = meta_data.declare_field<double>(ELEM_RANK, "REST_LENGTH");
  auto& elem_spring_constant_field = meta_data.declare_field<double>(ELEM_RANK, "SPRING_CONSTANT");
  auto& elem_ang_spring_constant_field = meta_data.declare_field<double>(ELEM_RANK, "ANG_SPRING_CONSTANT");
  auto& elem_rest_cos_angle_field = meta_data.declare_field<double>(ELEM_RANK, "REST_COS_ANGLE");

  // All parts store the node coords
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);

  // Assemble the microtubule parts
  stk::mesh::put_field_on_mesh(node_quaternion_field, rod_part, 4, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_omega_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_torque_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_tangent_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_length_field, rod_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, rod_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rng_counter_field, rod_part, 1, nullptr);

  // Assemble the PRC1 parts
  stk::mesh::put_field_on_mesh(node_arclength_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_tangent_field, prc1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, prc1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_torque_field, prc1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_is_bound_field, prc1_part, 1, nullptr);

  stk::mesh::put_field_on_mesh(elem_rest_length_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_spring_constant_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rest_cos_angle_field, prc1_part, 1, nullptr);
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
  stk::io::set_field_role(node_arclength_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_tangent_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_is_bound_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rng_counter_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_length_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_radius_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rest_length_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_spring_constant_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_ang_spring_constant_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rest_cos_angle_field, Ioss::Field::TRANSIENT);

  stk::io::set_field_output_type(node_coords_field, stk::io::FieldOutputType::VECTOR_3D);
  // stk::io::set_field_output_type(node_quaternion_field, stk::io::FieldOutputType::VECTOR_4D);  // No special quat
  // format :(
  stk::io::set_field_output_type(node_velocity_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_omega_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_force_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_torque_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(elem_radius_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_arclength_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_tangent_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_is_bound_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rng_counter_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_length_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rest_length_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_spring_constant_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_ang_spring_constant_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rest_cos_angle_field, stk::io::FieldOutputType::SCALAR);

  // Commit the meta data
  meta_data.commit();

  // Create our accessors and aggregates
  auto coord_accessor = Vector3FieldComponent(node_coords_field);
  auto quaternion_accessor = QuaternionFieldComponent(node_quaternion_field);
  auto tangent_accessor = Vector3FieldComponent(node_tangent_field);
  auto arclength_accessor = ScalarFieldComponent(node_arclength_field);
  auto velocity_accessor = Vector3FieldComponent(node_velocity_field);
  auto omega_accessor = Vector3FieldComponent(node_omega_field);
  auto force_accessor = Vector3FieldComponent(node_force_field);
  auto torque_accessor = Vector3FieldComponent(node_torque_field);
  auto is_bound_accessor = ScalarFieldComponent(node_is_bound_field);
  auto length_accessor = ScalarFieldComponent(elem_length_field);
  auto radius_accessor = ScalarFieldComponent(elem_radius_field);
  auto rng_counter_accessor = ScalarFieldComponent(elem_rng_counter_field);
  auto rest_length_accessor = ScalarFieldComponent(elem_rest_length_field);
  auto spring_constant_accessor = ScalarFieldComponent(elem_spring_constant_field);
  auto ang_spring_constant_accessor = ScalarFieldComponent(elem_ang_spring_constant_field);
  auto rest_cos_angle_accessor = ScalarFieldComponent(elem_rest_cos_angle_field);
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
                     .add_component<RADIUS, ELEM_RANK>(radius_accessor)
                     .add_component<RNG_COUNTER, ELEM_RANK>(rng_counter_accessor);

  auto rod_node_agg = make_ranked_aggregate<stk::topology::NODE_RANK>(bulk_data, rod_part)
                          .add_component<COORDS, NODE_RANK>(coord_accessor)
                          .add_component<QUAT, NODE_RANK>(quaternion_accessor)
                          .add_component<TANGENT, NODE_RANK>(tangent_accessor)
                          .add_component<VELOCITY, NODE_RANK>(velocity_accessor)
                          .add_component<OMEGA, NODE_RANK>(omega_accessor)
                          .add_component<FORCE, NODE_RANK>(force_accessor)
                          .add_component<TORQUE, NODE_RANK>(torque_accessor);

  auto prc1_agg = make_aggregate<stk::topology::BEAM_2>(bulk_data, prc1_part)
                      .add_component<ARCLENGTH, NODE_RANK>(arclength_accessor)
                      .add_component<COORDS, NODE_RANK>(coord_accessor)
                      .add_component<TANGENT, NODE_RANK>(tangent_accessor)
                      .add_component<FORCE, NODE_RANK>(force_accessor)
                      .add_component<TORQUE, NODE_RANK>(torque_accessor)
                      .add_component<IS_BOUND, NODE_RANK>(is_bound_accessor)
                      .add_component<REST_LENGTH, ELEM_RANK>(rest_length_accessor)
                      .add_component<SPRING_CONSTANT, ELEM_RANK>(spring_constant_accessor)
                      .add_component<ANG_SPRING_CONSTANT, ELEM_RANK>(ang_spring_constant_accessor)
                      .add_component<REST_COS_ANGLE, ELEM_RANK>(rest_cos_angle_accessor);

  auto prc1_head_agg = make_ranked_aggregate<NODE_RANK>(bulk_data, prc1_part)
                           .add_component<ARCLENGTH, NODE_RANK>(arclength_accessor)
                           .add_component<COORDS, NODE_RANK>(coord_accessor)
                           .add_component<TANGENT, NODE_RANK>(tangent_accessor)
                           .add_component<FORCE, NODE_RANK>(force_accessor)
                           .add_component<TORQUE, NODE_RANK>(torque_accessor)
                           .add_component<IS_BOUND, NODE_RANK>(is_bound_accessor);

  auto slink_agg = make_ranked_aggregate<NODE_RANK>(bulk_data, slink_part)
                       .add_component<LINKED_ENTITIES, NODE_RANK>(linked_entities_accessor);

  // Hard code system params
  size_t num_timesteps = 100;
  size_t io_frequency = 10;
  double dt = 0.1;
  double brownian_kbt = 0.0;
  double hertz_youngs_modulus = 1.0;
  double hertz_poisson_ratio = 0.5;

  double rod_radius = 0.5;
  double rod_length = 20;
  math::Quaternion<double> rod_orient1 = math::euler_to_quat(0.0, 2.5 * M_PI / 180, 0.0);
  math::Quaternion<double> rod_orient2 = math::euler_to_quat(0.0, -2.5 * M_PI / 180, 0.0);
  math::Vector3<double> rod_tangent1 = rod_orient1 * math::Vector3<double>(0.0, 0.0, 1.0);
  math::Vector3<double> rod_tangent2 = rod_orient2 * math::Vector3<double>(0.0, 0.0, 1.0);
  double init_rod_sep = 1.0 * (2.0 * rod_radius);

  double spring_constant = 0.0;
  double ang_spring_constant = 0.0;
  double rest_length = 2.0;
  double rest_cos_angle = 0.0;

  // Fill the declare entities helper
  DeclareEntitiesHelper dec_helper;

  // Create the rods
  auto center_node1 = dec_helper.create_node();
  center_node1
      .owning_proc(0)                                               //
      .id(1)                                                        // 1 indexed
      .add_field_data<double>(&node_coords_field, {0.0, 0.0, 0.0})  //
      .add_field_data<double>(&node_quaternion_field,
                              {rod_orient1[0], rod_orient1[1], rod_orient1[2], rod_orient1[3]})          //
      .add_field_data<double>(&node_tangent_field, {rod_tangent1[0], rod_tangent1[1], rod_tangent1[2]})  //
      .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})                                     //
      .add_field_data<double>(&node_omega_field, {0.0, 0.0, 0.0})                                        //
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})                                        //
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0});
  auto center_node2 = dec_helper.create_node();
  center_node2
      .owning_proc(0)                                                        //
      .id(2)                                                                 // 1 indexed
      .add_field_data<double>(&node_coords_field, {init_rod_sep, 0.0, 0.0})  //
      .add_field_data<double>(&node_quaternion_field,
                              {rod_orient2[0], rod_orient2[1], rod_orient2[2], rod_orient2[3]})          //
      .add_field_data<double>(&node_tangent_field, {rod_tangent2[0], rod_tangent2[1], rod_tangent2[2]})  //
      .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})                                     //
      .add_field_data<double>(&node_omega_field, {0.0, 0.0, 0.0})                                        //
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})                                        //
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0});

  auto mt1 = dec_helper.create_element();
  mt1.owning_proc(0)  //
      .id(1)          //
      .topology(stk::topology::PARTICLE)
      .add_part(&rod_part)                                       //
      .nodes({1})                                                //
      .add_field_data<double>(&elem_length_field, {rod_length})  //
      .add_field_data<double>(&elem_radius_field, {rod_radius});

  auto mt2 = dec_helper.create_element();
  mt2.owning_proc(0)  //
      .id(2)          //
      .topology(stk::topology::PARTICLE)
      .add_part(&rod_part)                                       //
      .nodes({2})                                                //
      .add_field_data<double>(&elem_length_field, {rod_length})  //
      .add_field_data<double>(&elem_radius_field, {rod_radius});

  // Declare the PRC1
  auto prc1_node1 = dec_helper.create_node();
  prc1_node1
      .owning_proc(0)                                                     //
      .id(3)                                                              // 1 indexed
      .add_field_data<double>(&node_arclength_field, {0.5 * rod_length})  //
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})         //
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0})        //
      .add_field_data<unsigned>(&node_is_bound_field, {1});

  auto prc1_node2 = dec_helper.create_node();
  prc1_node2
      .owning_proc(0)                                                     //
      .id(4)                                                              // 1 indexed
      .add_field_data<double>(&node_arclength_field, {0.5 * rod_length})  //
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})         //
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0})        //
      .add_field_data<unsigned>(&node_is_bound_field, {1});

  auto prc1_elem = dec_helper.create_element();
  prc1_elem
      .owning_proc(0)  //
      .id(3)           //
      .topology(stk::topology::BEAM_2)
      .add_part(&prc1_part)                                                            //
      .nodes({3, 4})                                                                   //
      .add_field_data<double>(&elem_rest_length_field, {rest_length})                  //
      .add_field_data<double>(&elem_spring_constant_field, {spring_constant})          //
      .add_field_data<double>(&elem_ang_spring_constant_field, {ang_spring_constant})  //
      .add_field_data<double>(&elem_rest_cos_angle_field, {rest_cos_angle});

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

  // Reconcile the surface node positions/tangents using their arclengths
  reconcile_surface_nodes(link_data, rod_agg, prc1_head_agg).apply_to(slink_agg);

  for (size_t timestep_index = 0; timestep_index < num_timesteps; timestep_index++) {
    // One timestep

    // Zero the velocities, forces, and torques
    mesh::field_fill<double>(0.0, node_velocity_field, stk::ngp::ExecSpace{});
    mesh::field_fill<double>(0.0, node_omega_field, stk::ngp::ExecSpace{});
    mesh::field_fill<double>(0.0, node_force_field, stk::ngp::ExecSpace{});
    mesh::field_fill<double>(0.0, node_torque_field, stk::ngp::ExecSpace{});

    // Compute Brownian motions
    apply_brownian_motion(dt, brownian_kbt).apply_to(rod_agg);

    // Compute spring forces
    std::cout << "Computing spring forces" << std::endl;
    eval_prc1_force().apply_to(prc1_agg);

    // Compute Hertzian contact forces
    apply_hertzian_contact(hertz_youngs_modulus, hertz_poisson_ratio).apply_to(rod_agg);

    // Compute map surface forces to center forces
    std::cout << "Mapping spring surface forces to center forces" << std::endl;
    map_surface_force_to_com_force(link_data, rod_agg, prc1_head_agg).apply_to(slink_agg);

    // Map center forces to center velocity
    eval_mobility().apply_to(rod_agg);

    // Write to file
    if (timestep_index % io_frequency == 0) {
      std::cout << "Writing to file" << std::endl;
      stk::io::StkMeshIoBroker stk_io_broker;
      stk_io_broker.property_add(Ioss::Property("MAXIMUM_NAME_LENGTH", 180));
      stk_io_broker.set_bulk_data(bulk_data);
      size_t step = timestep_index + 1;
      double time = timestep_index * dt;

      // Left pad the timestep index with zeros to as many digits as the number of timesteps
      auto pad_on_left_with_zeros = [](int number, int width) {
        std::stringstream ss;
        ss << std::setw(width) << std::setfill('0') << number;
        return ss.str();
      };
      std::string file_name =
          "will.e-s." + pad_on_left_with_zeros(timestep_index, std::to_string(num_timesteps).size());
      stk::io::write_mesh_with_fields(file_name, stk_io_broker, step, time, stk::io::WRITE_RESULTS);
    }

    // Update center positions and orientations (quat and tangent)
    update_configuration(dt).apply_to(rod_node_agg);

    // Reconcile surface node positions and tangents
    reconcile_surface_nodes(link_data, rod_agg, prc1_head_agg).apply_to(slink_agg);
  }
}
//@}

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