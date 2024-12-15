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
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <openrand/philox.h>

// C++ core
#include <algorithm>   // for std::transform
#include <filesystem>  // for std::filesystem::path
#include <fstream>     // for std::ofstream
#include <iostream>    // for std::cout, std::endl
#include <memory>      // for std::shared_ptr, std::unique_ptr
#include <numeric>     // for std::accumulate
#include <regex>       // for std::regex
#include <string>      // for std::string
#include <vector>      // for std::vector

// Kokkos and Kokkos-Kernels
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// Teuchos
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList

// STK Mesh
#include <stk_mesh/base/Comm.hpp>              // for stk::mesh::comm_mesh_counts
#include <stk_mesh/base/DumpMeshInfo.hpp>      // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>            // for stk::mesh::Entity
#include <stk_mesh/base/FieldParallel.hpp>     // for stk::parallel_sum
#include <stk_mesh/base/ForEachEntity.hpp>     // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/GetNgpField.hpp>       // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/GetNgpMesh.hpp>        // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/NgpField.hpp>          // for stk::mesh::NgpField
#include <stk_mesh/base/NgpForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/NgpMesh.hpp>           // for stk::mesh::NgpMesh
#include <stk_mesh/base/NgpReductions.hpp>
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector

// STK Search
#include <stk_search/BoxIdent.hpp>
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_search/Point.hpp>
#include <stk_search/SearchMethod.hpp>
#include <stk_search/Sphere.hpp>

// STK Topology
#include <stk_topology/topology.hpp>  // for stk::topology

// STK Util
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// STK Balance
#include <stk_balance/balance.hpp>  // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings

// STK IO
#include <stk_io/IossBridge.hpp>  // for stk::io::set_field_role and stk::io::put_io_part_attribute
#include <stk_io/WriteMesh.hpp>   // for stk::io::write_mesh

// Mundy core
#include <mundy_core/OurAnyNumberParameterEntryValidator.hpp>  // for mundy::core::OurAnyNumberParameterEntryValidator
#include <mundy_core/throw_assert.hpp>                         // for MUNDY_THROW_ASSERT

// Mundy math
#include <mundy_math/Hilbert.hpp>                      // for mundy::math::create_hilbert_positions_and_directors
#include <mundy_math/Vector3.hpp>                      // for mundy::math::Vector3
#include <mundy_math/distance/EllipsoidEllipsoid.hpp>  // for mundy::math::distance::ellipsoid_ellipsoid

// Mundy geom
#include <mundy_geom/distance.hpp>    // for mundy::geom::distance(primA, primB)
#include <mundy_geom/primitives.hpp>  // for all geometric primitives mundy::geom::Point, Line, Sphere, Ellipsoid...

// Mundy mesh
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MeshBuilder.hpp>      // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_mesh/fmt_stk_types.hpp>    // adds fmt::format for stk types
#include <mundy_mesh/utils/FillFieldWithValue.hpp>  // for mundy::mesh::utils::fill_field_with_value

// aLENS
#include <mundy_alens/actions_crosslinkers.hpp>  // for mundy::alens::crosslinkers...
#include <mundy_alens/periphery/Periphery.hpp>   // for gen_sphere_quadrature

using DoubleVecDeviceView = Kokkos::View<double *, Kokkos::LayoutLeft, stk::ngp::MemSpace>;
using DoubleMatDeviceView = Kokkos::View<double **, Kokkos::LayoutLeft, stk::ngp::MemSpace>;

// namespace mundy {

// namespace mesh {

// void get_selected_entities(const Selector &selector, const BucketVector &input_buckets,
//                            stk::NgpVector<stk::mesh::Entity> &ngp_entities, bool sort_by_global_id = true) {
//   if (input_buckets.empty()) {
//     return;
//   }

//   // Fetch the entities on the host
//   stk::mesh::EntityVector entity_vector,
//       stk::mesh::get_selected_entities(selector, input_buckets, entity_vector, sort_by_global_id);

//   // Fill the ngp vector on the host
//   const size_t num_entities = entity_vector.size();
//   ngp_entities.resize(num_entities);
//   for (unsigned i = 0; i < num_entities; i++) {
//     ngp_entities[i] = entity_vector[i];
//   }
//   ngp_entities.copy_host_to_device();
// }

// template <typename FieldDataType>
// void print_field(const stk::mesh::Field<FieldDataType> &field) {
//   mundy::mesh::BulkData &bulk_data = field.get_mesh();
//   stk::mesh::Selector selector = stk::mesh::Selector(field);

//   stk::mesh::EntityVector entities;
//   stk::mesh::get_selected_entities(selector, bulk_data_ptr_->buckets(field.entity_rank()), entities);

//   for (const stk::mesh::Entity &entity : entities) {
//     const FieldDataType *field_data = stk::mesh::field_data(field, entity);
//     std::cout << "Entity " << entity << " field data: ";
//     const unsigned field_num_components =
//         stk::mesh::field_scalars_per_entity(field, bulk_data.bucket(entity).bucket_id());
//     for (size_t i = 0; i < field_num_components; ++i) {
//       std::cout << field_data[i] << " ";
//     }
//     std::cout << std::endl;
//   }
// }

// }  // namespace mesh

// namespace geom {

// void compute_aabb_spheres(stk::mesh::NgpMesh &ngp_mesh, const double &skin_distance,
//                           stk::mesh::NgpField<double> &node_coords, stk::mesh::NgpField<double> &elem_radius,
//                           stk::mesh::NgpField<double> &elem_aabb_field, const stk::mesh::Selector &selector) {
//   node_coords.sync_to_device();
//   elem_radius.sync_to_device();
//   elem_aabb_field.sync_to_device();

//   stk::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &elem_index) {
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, elem_index);
//         stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(nodes[0]);

//         const auto coords = mundy::mesh::vector3_field_data(node_coords, node_index);
//         const double radius = elem_radius(elem, 0);
//         double min_x = coords[0] - radius;
//         double min_y = coords[1] - radius;
//         double min_z = coords[2] - radius;
//         double max_x = coords[0] + radius;
//         double max_y = coords[1] + radius;
//         double max_z = coords[2] + radius;
//         elem_aabb_field(elem, 0) = min_x - skin_distance;
//         elem_aabb_field(elem, 1) = min_y - skin_distance;
//         elem_aabb_field(elem, 2) = min_z - skin_distance;
//         elem_aabb_field(elem, 3) = max_x + skin_distance;
//         elem_aabb_field(elem, 4) = max_y + skin_distance;
//         elem_aabb_field(elem, 5) = max_z + skin_distance;
//       });

//   elem_aabb_field.modify_on_device();
// }

// void compute_aabb_segs(stk::mesh::NgpMesh &ngp_mesh, const double &skin_distance,
//                        stk::mesh::NgpField<double> &node_coords, stk::mesh::NgpField<double> &elem_radius,
//                        stk::mesh::NgpField<double> &elem_aabb_field, const stk::mesh::Selector &selector) {
//   node_coords.sync_to_device();
//   elem_radius.sync_to_device();
//   elem_aabb_field.sync_to_device();

//   stk::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &elem_index) {
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, elem_index);
//         stk::mesh::FastMeshIndex node0_index = ngp_mesh.fast_mesh_index(nodes[0]);
//         stk::mesh::FastMeshIndex node1_index = ngp_mesh.fast_mesh_index(nodes[1]);

//         const auto coord0 = mundy::mesh::vector3_field_data(node_coords, node0_index);
//         const auto coord1 = mundy::mesh::vector3_field_data(node_coords, node1_index);

//         const double radius = elem_radius(elem, 0);
//         double min_x = Kokkos::min(coord0[0], coord1[0]) - radius;
//         double min_y = Kokkos::min(coord0[1], coord1[1]) - radius;
//         double min_z = Kokkos::min(coord0[2], coord1[2]) - radius;
//         double max_x = Kokkos::max(coord0[0], coord1[0]) + radius;
//         double max_y = Kokkos::max(coord0[1], coord1[1]) + radius;
//         double max_z = Kokkos::max(coord0[2], coord1[2]) + radius;
//         elem_aabb_field(elem, 0) = min_x - skin_distance;
//         elem_aabb_field(elem, 1) = min_y - skin_distance;
//         elem_aabb_field(elem, 2) = min_z - skin_distance;
//         elem_aabb_field(elem, 3) = max_x + skin_distance;
//         elem_aabb_field(elem, 4) = max_y + skin_distance;
//         elem_aabb_field(elem, 5) = max_z + skin_distance;
//       });

//   elem_aabb_field.modify_on_device();
// }

// void accumulate_aabb_displacements(stk::mesh::NgpMesh &ngp_mesh, stk::mesh::NgpField<double> &old_elem_aabb_field,
//                                    stk::mesh::NgpField<double> &elem_aabb_field,
//                                    stk::mesh::NgpField<double> &elem_displacement_field,
//                                    const stk::mesh::Selector &selector) {
//   old_elem_aabb_field.sync_to_device();
//   elem_aabb_field.sync_to_device();
//   elem_displacement_field.sync_to_device();

//   stk::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &elem_index) {
//         for (int i = 0; i < 6; ++i) {
//           elem_displacement_field(elem, i) += elem_aabb_field(elem, i) - old_elem_aabb_field(elem, i);
//         }
//       });

//   elem_displacement_field.modify_on_device();
// }

// }  // namespace geom

// namespace mech {

// //! \name Spring forces
// //@{

// void compute_hookean_spring_forces(stk::mesh::NgpMesh &ngp_mesh, stk::mesh::NgpField<double> &node_coords_field,
//                                    stk::mesh::NgpField<double> &node_force_field,
//                                    stk::mesh::NgpField<double> &spring_constant_field,
//                                    stk::mesh::NgpField<double> &spring_rest_length_field,
//                                    const stk::mesh::Selector &selector) {
//   Kokkos::Profiling::pushRegion("HP1::compute_hookean_spring_forces");

//   node_coords_field.sync_to_device();
//   node_force_field.sync_to_device();
//   spring_constant_field.sync_to_device();
//   spring_rest_length_field.sync_to_device();

//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::mesh::ELEM_RANG, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &spring_index) {
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, spring_index);
//         const stk::mesh::Entity node1 = nodes[0];
//         const stk::mesh::Entity node2 = nodes[1];
//         const auto node1_coords = node_coords_field(node1);
//         const auto node2_coords = node_coords_field(node2);
//         const double spring_constant = spring_constant_field(spring_index);
//         const double spring_rest_length = spring_rest_length_field(spring_index);
//         const double inv_spring_length = 1.0 / spring_length;

//         // Compute the spring force
//         const double dx = node2_coords[0] - node1_coords[0];
//         const double dy = node2_coords[1] - node1_coords[1];
//         const double dz = node2_coords[2] - node1_coords[2];
//         const double length = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
//         const double force_magnitude = spring_constant * (length - spring_rest_length) * inv_spring_length;
//         const double fx = force_magnitude * dx;
//         const double fy = force_magnitude * dy;
//         const double fz = force_magnitude * dz;

//         // Apply the force to the nodes. Use atomic add
//         Kokkos::atomic_add(&node_force_field(node1, 0), fx);
//         Kokkos::atomic_add(&node_force_field(node1, 1), fy);
//         Kokkos::atomic_add(&node_force_field(node1, 2), fz);
//         Kokkos::atomic_add(&node_force_field(node2, 0), -fx);
//         Kokkos::atomic_add(&node_force_field(node2, 1), -fy);
//         Kokkos::atomic_add(&node_force_field(node2, 2), -fz);
//       });

//   node_force_field.modify_on_device();
//   Kokkos::Profiling::popRegion();
// }

// void compute_fene_spring_forces(stk::mesh::NgpMesh &ngp_mesh, stk::mesh::NgpField<double> &node_coords_field,
//                                 stk::mesh::NgpField<double> &node_force_field,
//                                 stk::mesh::NgpField<double> &spring_constant_field,
//                                 stk::mesh::NgpField<double> &spring_max_length_field,
//                                 const stk::mesh::Selector &selector) {
//   Kokkos::Profiling::pushRegion("HP1::compute_fene_spring_forces");

//   node_coords_field.sync_to_device();
//   node_force_field.sync_to_device();
//   spring_constant_field.sync_to_device();
//   spring_max_length_field.sync_to_device();

//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::mesh::ELEM_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &spring_index) {
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, spring_index);
//         const stk::mesh::Entity node1 = nodes[0];
//         const stk::mesh::Entity node2 = nodes[1];
//         const auto node1_coords = node_coords_field(node1);
//         const auto node2_coords = node_coords_field(node2);
//         const double spring_constant = spring_constant_field(spring_index);
//         const double spring_max_length = spring_max_length_field(spring_index);
//         const double inv_spring_max_length = 1.0 / spring_max_length;

//         // Compute the spring force
//         // The fene spring force is F = -k * r / (1 - (r / r_max)^2)
//         const double dx = node2_coords[0] - node1_coords[0];
//         const double dy = node2_coords[1] - node1_coords[1];
//         const double dz = node2_coords[2] - node1_coords[2];
//         const double length = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
//         const double inv_length = 1.0 / length;
//         const double force_magnitude =
//             spring_constant / (1.0 - (length * inv_spring_max_length) * (length * inv_spring_max_length));
//         const double fx = force_magnitude * dx;
//         const double fy = force_magnitude * dy;
//         const double fz = force_magnitude * dz;

//         // Apply the force to the nodes. Use atomic add
//         Kokkos::atomic_add(&node_force_field(node1, 0), -fx);
//         Kokkos::atomic_add(&node_force_field(node1, 1), -fy);
//         Kokkos::atomic_add(&node_force_field(node1, 2), -fz);
//         Kokkos::atomic_add(&node_force_field(node2, 0), fx);
//         Kokkos::atomic_add(&node_force_field(node2, 1), fy);
//         Kokkos::atomic_add(&node_force_field(node2, 2), fz);
//       });

//   node_force_field.modify_on_device();
//   Kokkos::Profiling::popRegion();
// }
// //@}

// }  // namespace mech

// namespace alens {

// //! \name Dynamic spring bind/unbind via Kinetic Monte Carlo
// //@{

// struct HarmonicCrosslinkerBindRateHeterochromatin {
//   HarmonicCrosslinkerBindRateHeterochromatin(const mundy::mesh::BulkData &bulk_data, const double &kt,
//                                              const double &crosslinker_binding_rate,
//                                              const double &crosslinker_spring_constant,
//                                              const double &crosslinker_rest_length,
//                                              const stk::mesh::Field<double> &node_coords_field,
//                                              const stk::mesh::Selector &heterochromatin_selector)
//       : bulk_data_(bulk_data),
//         inv_kt_(1.0 / kt),
//         crosslinker_binding_rate_(crosslinker_binding_rate),
//         crosslinker_spring_constant_(crosslinker_spring_constant),
//         crosslinker_rest_length_(crosslinker_rest_length),
//         node_coord_field_(node_coords_field) {
//     heterochromatin_selector.get_parts(heterochromatin_parts_);
//   }

//   double operator()(const stk::mesh::Entity &crosslinker, const stk::mesh::Entity &bind_site) const {
//     // Get the distance from the crosslinker's left node and the bind site
//     const stk::mesh::Entity left_node = bulk_data_.begin_nodes(crosslinker)[0];
//     const double *left_node_coords = stk::mesh::field_data(node_coord_field_, left_node);
//     const double *bind_site_coords = stk::mesh::field_data(node_coord_field_, bind_site);

//     const double dx = bind_site_coords[0] - left_node_coords[0];
//     const double dy = bind_site_coords[1] - left_node_coords[1];
//     const double dz = bind_site_coords[2] - left_node_coords[2];
//     const double dr_mag = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

//     // Z = A * exp(-0.5 * 1/kt * k * (dr - r0)^2)
//     // A = crosslinker_binding_rates
//     // k = crosslinker_spring_constant
//     // r0 = crosslinker_spring_rest_length
//     const double A = crosslinker_binding_rate_;
//     const double k = crosslinker_spring_constant_;
//     const double r0 = crosslinker_rest_length_;
//     double Z = A * Kokkos::exp(-0.5 * inv_kt_ * k * (dr_mag - r0) * (dr_mag - r0));
//     return Z;
//   }

//   const mundy::mesh::BulkData &bulk_data_;
//   const double inv_kt_;
//   const double crosslinker_binding_rate_;
//   const double crosslinker_spring_constant_;
//   const double crosslinker_rest_length_;
//   const stk::mesh::Field<double> &node_coord_field_;
//   stk::mesh::PartVector heterochromatin_parts_;
// };

// struct FeneCrosslinkerBindRateHeterochromatin {
//   FeneCrosslinkerBindRateHeterochromatin(const mundy::mesh::BulkData &bulk_data, const double &kt,
//                                          const double &crosslinker_binding_rate,
//                                          const double &crosslinker_spring_constant,
//                                          const double &crosslinker_rest_length,
//                                          const stk::mesh::Field<double> &node_coords_field,
//                                          const stk::mesh::Selector &heterochromatin_selector)
//       : bulk_data_(bulk_data),
//         inv_kt_(1.0 / kt),
//         crosslinker_binding_rate_(crosslinker_binding_rate),
//         crosslinker_spring_constant_(crosslinker_spring_constant),
//         crosslinker_rest_length_(crosslinker_rest_length),
//         node_coord_field_(node_coords_field) {
//     heterochromatin_selector.get_parts(heterochromatin_parts_);
//   }

//   double operator()(const stk::mesh::Entity &crosslinker, const stk::mesh::Entity &bind_site) const {
//     // Get the distance from the crosslinker's left node and the bind site
//     const stk::mesh::Entity left_node = bulk_data_.begin_nodes(crosslinker)[0];
//     const double *left_node_coords = stk::mesh::field_data(node_coord_field_, left_node);
//     const double *bind_site_coords = stk::mesh::field_data(node_coord_field_, bind_site);

//     const double dx = bind_site_coords[0] - left_node_coords[0];
//     const double dy = bind_site_coords[1] - left_node_coords[1];
//     const double dz = bind_site_coords[2] - left_node_coords[2];
//     const double dr_mag = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

//     // Z = A * (1 - (r/r0)^2)^(0.5 * 1/kt * k * r0^2)
//     // A = crosslinker_binding_rates
//     // k = crosslinker_spring_constant
//     // r0 = crosslinker_spring_max_length (FENE)
//     // R = crosslinker_fene_max_distance
//     const double A = crosslinker_right_binding_rate;
//     const double k = stk::mesh::field_data(crosslinker_spring_constant, crosslinker)[0];
//     const double r0 = stk::mesh::field_data(crosslinker_spring_r0, crosslinker)[0];
//     double Z = A * std::pow(1.0 - (dr_mag / r0) * (dr_mag / r0), 0.5 * inv_kt * k * r0 * r0);
//     return Z;
//   }

//   const mundy::mesh::BulkData &bulk_data_;
//   const double inv_kt_;
//   const double crosslinker_binding_rate_;
//   const double crosslinker_spring_constant_;
//   const double crosslinker_rest_length_;
//   const stk::mesh::Field<double> &node_coord_field_;
//   stk::mesh::PartVector heterochromatin_parts_;
// };

// template <typename CrosslinkerBindRateHeterochromatin, typename CrosslinkerBindRatePeriphery>
// struct CrosslinkerBindRateHeterochromatinOrPeriphery {
//   CrosslinkerBindRateHeterochromatinOrPeriphery(
//       const CrosslinkerBindRateHeterochromatin &crosslinker_bind_rate_heterochromatin,
//       const CrosslinkerBindRatePeriphery &crosslinker_bind_rate_periphery,
//       const stk::mesh::Selector &heterochromatin_selector, const stk::mesh::Selector &periphery_selector)
//       : crosslinker_bind_rate_heterochromatin_(crosslinker_bind_rate_heterochromatin),
//         crosslinker_bind_rate_periphery_(crosslinker_bind_rate_periphery) {
//     heterochromatin_selector.get_parts(heterochromatin_parts_);
//     periphery_selector.get_parts(periphery_parts_);
//   }

//   double operator()(const stk::mesh::Entity &crosslinker, const stk::mesh::Entity &bind_site) const {
//     // Determine if the bind site is in the heterochromatin or periphery
//     bool is_heterochromatin = false;
//     for (const stk::mesh::Part *part : heterochromatin_parts_) {
//       if (bind_site.bucket().member(*part)) {
//         is_heterochromatin = true;
//         break;
//       }
//     }
//     bool is_periphery = false;
//     for (const stk::mesh::Part *part : periphery_parts_) {
//       if (bind_site.bucket().member(*part)) {
//         is_periphery = true;
//         break;
//       }
//     }

//     if (is_heterochromatin) {
//       return crosslinker_bind_rate_heterochromatin_(crosslinker, bind_site);
//     } else if (is_periphery) {
//       return crosslinker_bind_rate_periphery_(crosslinker, bind_site);
//     } else {
//       MUNDY_THROW_ASSERT(false, std::logic_error, "Bind site is not in heterochromatin or periphery.");
//     }
//   }

//   const CrosslinkerBindRateHeterochromatin &crosslinker_bind_rate_heterochromatin_;
//   const CrosslinkerBindRatePeriphery &crosslinker_bind_rate_periphery_;
//   stk::mesh::PartVector heterochromatin_parts_;
//   stk::mesh::PartVector periphery_parts_;
// };

// template <typename LeftToDoublyStateChangeRate>
// void kmc_perform_state_change_left_bound(mundy::mesh::BulkData &bulk_data, const double timestep_size,
//                                          const LeftToDoublyStateChangeRate &left_to_doubly_state_change_rate_getter,
//                                          const stk::mesh::Field<double> &neighboring_bind_sites_field,
//                                          const stk::mesh::Field<double> &el_rng_field,
//                                          const stk::mesh::Selector &left_bound_springs_selector,
//                                          const stk::mesh::Selector &doubly_bound_springs_selector) {
//   MUNDY_THROW_REQUIRE(bulk_data.in_modifiable_state(), std::logic_error, "Bulk data is not in a modification
//   cycle.");

//   // Get the vector of left/right bound parts in the selector
//   stk::mesh::PartVector left_bound_spring_parts;
//   stk::mesh::PartVector doubly_bound_spring_parts;
//   left_bound_springs_selector.get_parts(left_bound_spring_parts);
//   doubly_bound_springs_selector.get_parts(doubly_bound_spring_parts);

//   // Get the vector of entities to modify
//   stk::mesh::EntityVector left_bound_springs;
//   stk::mesh::get_selected_entities(left_bound_springs_selector, bulk_data.buckets(stk::mesh::ELEMENT_RANK),
//                                    left_bound_springs);

//   for (const stk::mesh::Entity &left_bound_spring : left_bound_springs) {
//     const double *neighboring_bind_sites = stk::mesh::field_data(neighboring_bind_sites_field, left_bound_spring);
//     const int num_neighboring_bind_sites = neighboring_bind_sites[0];

//     double z_tot = 0.0;
//     for (int s = 0; s < num_neighboring_bind_sites; ++s) {
//       const auto &neighboring_bind_site_index = static_cast<stk::mesh::FastMeshIndex>(neighboring_bind_sites[s + 1]);
//       const stk::mesh::Entity &bind_site = bulk_data.get_entity(neighboring_bind_site_index);
//       const double z_i = timestep_size * left_to_doubly_state_change_rate_getter(left_bound_spring, bind_site);
//       z_tot += z_i;
//     }

//     // Fetch the RNG state, get a random number out of it, and increment
//     unsigned *rng_counter = stk::mesh::field_data(el_rng_field, left_bound_spring);
//     const stk::mesh::EntityId spring_gid = bulk_data.identifier(left_bound_spring);
//     openrand::Philox rng(spring_gid, rng_counter[0]);
//     const double randu01 = rng.rand<double>();
//     rng_counter[0]++;

//     // Notice that the sum of all probabilities is 1.
//     // The probability of nothing happening is
//     //   std::exp(-z_tot)
//     // The probability of an individual event happening is
//     //   z_i / z_tot * (1 - std::exp(-z_tot))
//     //
//     // This is (by construction) true since
//     //  std::exp(-z_tot) + sum_i(z_i / z_tot * (1 - std::exp(-z_tot)))
//     //    = std::exp(-z_tot) + (1 - std::exp(-z_tot)) = 1
//     //
//     // This means that binding only happens if randu01 < (1 - std::exp(-z_tot))
//     const double probability_of_no_state_change = 1.0 - std::exp(-z_tot);
//     if (randu01 < probability_of_no_state_change) {
//       // Binding occurs. Loop back over the neighbor bind sites to see which one we bind to.
//       const double scale_factor = probability_of_no_state_change * timestep_size / z_tot;
//       double cumsum = 0.0;
//       for (int s = 0; s < num_neighboring_bind_sites; ++s) {
//         const auto &neighboring_bind_site_index = static_cast<stk::mesh::FastMeshIndex>(neighboring_bind_sites[s +
//         1]); const stk::mesh::Entity &bind_site =
//             bulk_data.get_entity(neighboring_bind_site_index, stk::topology::NODE_RANK);
//         const double binding_probability =
//             scale_factor * left_to_doubly_state_change_rate_getter(left_bound_spring, bind_site);
//         cumsum += binding_probability;
//         if (randu01 < cumsum) {
//           // Bind to the given site
//           const int right_node_index = 1;
//           const bool bind_worked =
//               bind_crosslinker_to_node_unbind_existing(bulk_data, left_bound_spring, bind_site, right_node_index);
//           MUNDY_THROW_ASSERT(bind_worked, std::logic_error, "Failed to bind crosslinker to node.");

//           std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Binding crosslinker "
//                     << bulk_data.identifier(spring) << " to node " << bulk_data.identifier(target_sphere_node)
//                     << std::endl;

//           // Now change the part from left to doubly bound. Add to doubly bound, remove
//           // from left bound
//           const bool is_spring_locally_owned =
//               bulk_data.parallel_owner_rank(left_bound_spring) == bulk_data.parallel_rank();
//           if (is_spring_locally_owned) {
//             bulk_data.change_entity_parts(left_bound_spring, doubly_bound_spring_parts, left_bound_spring_parts);
//           }
//         }
//       }
//     }
//   }
// }

// template <typename LeftToDoublyStateChangeRate>
// void kmc_perform_state_change_doubly_bound(mundy::mesh::BulkData &bulk_data, const double timestep_size,
//                                            const LeftToDoublyStateChangeRate
//                                            &doubly_to_left_state_change_rate_getter, const stk::mesh::Field<double>
//                                            &neighboring_bind_sites_field, const stk::mesh::Field<double>
//                                            &el_rng_field, const stk::mesh::Selector &left_bound_springs_selector,
//                                            const stk::mesh::Selector &doubly_bound_springs_selector) {
//   MUNDY_THROW_REQUIRE(bulk_data.in_modifiable_state(), std::logic_error, "Bulk data is not in a modification
//   cycle.");

//   // Get the vector of left/right bound parts in the selector
//   stk::mesh::PartVector left_bound_spring_parts;
//   stk::mesh::PartVector doubly_bound_spring_parts;
//   left_bound_springs_selector.get_parts(left_bound_spring_parts);
//   doubly_bound_springs_selector.get_parts(doubly_bound_spring_parts);

//   // Get the vector of entities to modify
//   stk::mesh::EntityVector doubly_bound_springs;
//   stk::mesh::get_selected_entities(doubly_bound_springs_selector, bulk_data.buckets(stk::mesh::ELEMENT_RANK),
//                                    doubly_bound_springs);

//   for (const stk::mesh::Entity &doubly_bound_spring : doubly_bound_springs) {
//     double z_tot - timestep_size *doubly_to_left_state_change_rate_getter(doubly_bound_spring);

//     // Fetch the RNG state, get a random number out of it, and increment
//     unsigned *rng_counter = stk::mesh::field_data(el_rng_field, doubly_bound_spring);
//     const stk::mesh::EntityId spring_gid = bulk_data.identifier(doubly_bound_spring);
//     openrand::Philox rng(spring_gid, rng_counter[0]);
//     const double randu01 = rng.rand<double>();
//     rng_counter[0]++;

//     // Notice that the sum of all probabilities is 1.
//     // The probability of nothing happening is
//     //   std::exp(-z_tot)
//     // The probability of an individual event happening is
//     //   z_i / z_tot * (1 - std::exp(-z_tot))
//     //
//     // This is (by construction) true since
//     //  std::exp(-z_tot) + sum_i(z_i / z_tot * (1 - std::exp(-z_tot)))
//     //    = std::exp(-z_tot) + (1 - std::exp(-z_tot)) = 1
//     //
//     // This means that unbinding only happens if randu01 < (1 - std::exp(-z_tot))
//     // For now, its either transition to right bound or nothing
//     const double probability_of_no_state_change = 1.0 - std::exp(-z_tot);
//     if (randu01 < probability_of_no_state_change) {
//       // Unbind the right side of the crosslinker from the current node and bind it to
//       // the left crosslinker node
//       const stk::mesh::Entity &left_node = bulk_data.begin_nodes(doubly_bound_spring)[0];
//       const int right_node_index = 1;
//       const bool unbind_worked =
//           bind_crosslinker_to_node_unbind_existing(bulk_data, doubly_bound_spring, left_node, right_node_index);
//       MUNDY_THROW_ASSERT(unbind_worked, std::logic_error, "Failed to unbind crosslinker from node.");

//       std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Unbinding crosslinker "
//                 << bulk_data.identifier(spring) << " from node "
//                 << bulk_data.identifier(bulk_data.begin_nodes(spring)[1]) << std::endl;

//       // Now change the part from doubly to left bound. Add to left bound, remove from
//       // doubly bound
//       const bool is_spring_locally_owned =
//           bulk_data.parallel_owner_rank(doubly_bound_spring) == bulk_data.parallel_rank();
//       if (is_spring_locally_owned) {
//         bulk_data.change_entity_parts(doubly_bound_spring, left_bound_spring_parts, doubly_bound_spring_parts);
//       }
//     }
//   }
// }
// //@}

// //! \name Misc domain-specific physics
// //@{

// void compute_brownian_velocity(stk::mesh::NgpMesh &ngp_mesh, const double &timestep_size, const double &kt,
//                                const double &viscosity, stk::mesh::NgpField<double> &node_velocity_field,
//                                stk::mesh::NgpField<unsigned> &elem_rng_field,
//                                stk::mesh::NgpField<double> &element_radius_field, stk::mesh::Selector &selector) {
//   Kokkos::Profiling::pushRegion("HP1::compute_brownian_velocity");

//   node_velocity_field.sync_to_device();
//   elem_rng_field.sync_to_device();
//   element_radius_field.sync_to_device();

//   constexpr double pi = Kokkos::numbers::pi_v<double>;
//   const double six_pi_mu = 6.0 * pi * viscosity;
//   const double sqrt_6_pi_mu = Kokkos::sqrt(six_pi_mu);
//   const double inv_six_pi_mu = 1.0 / six_pi_mu;
//   const double inv_dt = 1.0 / timestep_size;
//   const double sqrt_2_kt = Kokkos::sqrt(2.0 * kt);

//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::ELEM_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
//         const stk::mesh::Entity node = nodes[0];

//         // Setup the rng
//         const stk::mesh::EntityId sphere_gid = ngp_mesh.identifier(sphere_index);
//         auto rng_counter = elem_rng_field(sphere_index);
//         openrand::Philox rng(sphere_gid, rng_counter[0]);

//         // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
//         // for drag coeff gamma = 6 * pi * mu * r
//         const double sphere_radius = element_radius_field(node, 0);
//         auto node_velocity = node_velocity_field(node);
//         const double coeff =
//             sqrt_2_kt * Kokkos::sqrt(sqrt_6_pi_mu * sphere_radius * inv_dt) * inv_six_pi_mu / sphere_radius;
//         node_velocity[0] += coeff * rng.randn<double>();
//         node_velocity[1] += coeff * rng.randn<double>();
//         node_velocity[2] += coeff * rng.randn<double>();
//         rng_counter[0]++;
//       });

//   Kokkos::Profiling::popRegion();
// }

// void node_euler_position_update(stk::mesh::NgpMesh &ngp_mesh, const double &timestep_size,
//                                 stk::mesh::NgpField<double> &node_coords_field,
//                                 stk::mesh::NgpField<double> &node_velocity_field, const stk::mesh::Selector
//                                 &selector) {
//   Kokkos::Profiling::pushRegion("HP1::update_positions");

//   node_coords_field.sync_to_device();
//   node_velocity_field.sync_to_device();

//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::NODE_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
//         const stk::mesh::Entity node = ngp_mesh.get_entity(stk::topology::NODE_RANK, node_index);
//         auto node_coords = node_coords_field(node);
//         auto node_velocity = node_velocity_field(node);

//         node_coords[0] += timestep_size * node_velocity[0];
//         node_coords[1] += timestep_size * node_velocity[1];
//         node_coords[2] += timestep_size * node_velocity[2];
//       });

//   node_coords_field.modify_on_device();

//   Kokkos::Profiling::popRegion();
// }
// //@}

// }  // namespace alens

// }  // namespace mundy

namespace mundy {

namespace chromalens {

// using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
// using DeviceMemorySpace = typename DeviceExecutionSpace::memory_space;

// void print_rank0(auto thing_to_print, int indent_level = 0) {
//   if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
//     std::string indent(indent_level * 2, ' ');
//     std::cout << indent << thing_to_print << std::endl;
//   }
// }

// class RcbSettings : public stk::balance::BalanceSettings {
//  public:
//   RcbSettings() {
//   }
//   virtual ~RcbSettings() {
//   }

//   virtual bool isIncrementalRebalance() const {
//     return false;
//   }
//   virtual std::string getDecompMethod() const {
//     return std::string("rcb");
//   }
//   virtual std::string getCoordinateFieldName() const {
//     return std::string("NODE_COORDS");
//   }
//   virtual bool shouldPrintMetrics() const {
//     return false;
//   }
// };  // RcbSettings

//! \name Chromatin position generators
//@{

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_from_file(
    const std::string &file_name, const unsigned num_chromosomes) {
  // The file should be formatted as follows:
  // chromosome_id x y z
  // 0 x1 y1 z1
  // 0 x2 y2 z2
  // ...
  // 1 x1 y1 z1
  // 1 x2 y2 z2
  //
  // chromosome_id should start at 1 and increase by 1 for each new chromosome.
  //
  // And so on for each chromosome. The total number of chromosomes should match the expected total, lest we throw an
  // exception.
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  std::ifstream infile(file_name);
  MUNDY_THROW_REQUIRE(infile.is_open(), std::invalid_argument, fmt::format("Could not open file {}", file_name));

  // Read each line. While the chromosome_id is the same, keep adding nodes to the chromosome.
  size_t current_chromosome_id = 1;
  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    int chromosome_id;
    double x, y, z;
    if (!(iss >> chromosome_id >> x >> y >> z)) {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument, fmt::format("Could not parse line {}", line));
    }
    if (chromosome_id != current_chromosome_id) {
      // We are starting a new chromosome
      MUNDY_THROW_REQUIRE(chromosome_id == current_chromosome_id + 1, std::invalid_argument,
                          "Chromosome IDs should be sequential.");
      MUNDY_THROW_REQUIRE(chromosome_id <= num_chromosomes, std::invalid_argument,
                          fmt::format("Chromosome ID {} is greater than the number of chromosomes.", chromosome_id));
      current_chromosome_id = chromosome_id;
    }
    // Add the node to the chromosome
    all_chromosome_positions[current_chromosome_id - 1].emplace_back(x, y, z);
  }

  return all_chromosome_positions;
}

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_grid(
    const unsigned num_chromosomes, const unsigned num_nodes_per_chromosome, const double segment_length) {
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  const mundy::math::Vector3<double> alignment_dir{0.0, 0.0, 1.0};
  for (size_t j = 0; j < num_chromosomes; j++) {
    all_chromosome_positions[j].reserve(num_nodes_per_chromosome);
    openrand::Philox rng(j, 0);
    mundy::math::Vector3<double> start_pos(2.0 * static_cast<double>(j), 0.0, 0.0);
    for (size_t i = 0; i < num_nodes_per_chromosome; ++i) {
      const auto pos = start_pos + static_cast<double>(i) * segment_length * alignment_dir;
      all_chromosome_positions[j].emplace_back(pos);
    }
  }

  return all_chromosome_positions;
}

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_random_unit_cell(
    const unsigned num_chromosomes, const unsigned num_nodes_per_chromosome, const double segment_length,
    const double domain_low[3], const double domain_high[3]) {
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  for (size_t j = 0; j < num_chromosomes; j++) {
    all_chromosome_positions[j].reserve(num_nodes_per_chromosome);

    // Find a random place within the unit cell with a random orientation for the chain.
    openrand::Philox rng(j, 0);
    mundy::math::Vector3<double> pos_start{rng.uniform<double>(domain_low[0], domain_high[0]),
                                           rng.uniform<double>(domain_low[1], domain_high[1]),
                                           rng.uniform<double>(domain_low[2], domain_high[2])};

    // Find a random unit vector direction
    const double zrand = rng.rand<double>() - 1.0;
    const double wrand = std::sqrt(1.0 - zrand * zrand);
    const double trand = 2.0 * M_PI * rng.rand<double>();
    mundy::math::Vector3<double> u_hat{wrand * std::cos(trand), wrand * std::sin(trand), zrand};

    for (size_t i = 0; i < num_nodes_per_chromosome; ++i) {
      auto pos = pos_start + static_cast<double>(i) * segment_length * u_hat;
      all_chromosome_positions[j].emplace_back(pos);
    }
  }

  return all_chromosome_positions;
}

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_hilbert_random_unit_cell(
    const unsigned num_chromosomes, const unsigned num_nodes_per_chromosome, const double segment_length,
    const double domain_low[3], const double domain_high[3]) {
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  std::vector<mundy::geom::Point<double>> chromosome_centers_array(num_chromosomes);
  std::vector<double> chromosome_radii_array(num_chromosomes);
  for (size_t ichromosome = 0; ichromosome < num_chromosomes; ichromosome++) {
    // Generate a random unit vector (will be used for creating the location of the nodes, the random position in
    // the unit cell will be handled later).
    openrand::Philox rng(ichromosome, 0);
    const double zrand = rng.rand<double>() - 1.0;
    const double wrand = std::sqrt(1.0 - zrand * zrand);
    const double trand = 2.0 * M_PI * rng.rand<double>();
    mundy::math::Vector3<double> u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);

    // Once we have the number of chromosome spheres we can get the hilbert curve set up. This will be at some
    // orientation and then have sides with a length of initial_chromosome_separation.
    auto [hilbert_position_array, hilbert_directors] =
        mundy::math::create_hilbert_positions_and_directors(num_nodes_per_chromosome, u_hat, segment_length);

    // Create the local positions of the spheres
    std::vector<mundy::math::Vector3<double>> sphere_position_array;
    for (size_t isphere = 0; isphere < num_nodes_per_chromosome; isphere++) {
      sphere_position_array.push_back(hilbert_position_array[isphere]);
    }

    // Figure out where the center of the chromosome is, and its radius, in its own local space
    mundy::math::Vector3<double> r_chromosome_center_local(0.0, 0.0, 0.0);
    double r_max = 0.0;
    for (size_t i = 0; i < sphere_position_array.size(); i++) {
      r_chromosome_center_local += sphere_position_array[i];
    }
    r_chromosome_center_local /= static_cast<double>(sphere_position_array.size());
    for (size_t i = 0; i < sphere_position_array.size(); i++) {
      r_max = std::max(r_max, mundy::math::two_norm(r_chromosome_center_local - sphere_position_array[i]));
    }

    // Do max_trials number of insertion attempts to get a random position and orientation within the unit cell that
    // doesn't overlap with exiting chromosomes.
    const size_t max_trials = 1000;
    size_t itrial = 0;
    bool chromosome_inserted = false;
    while (itrial <= max_trials) {
      // Generate a random position within the unit cell.
      mundy::math::Vector3<double> r_start(rng.uniform<double>(domain_low[0], domain_high[0]),
                                           rng.uniform<double>(domain_low[1], domain_high[1]),
                                           rng.uniform<double>(domain_low[2], domain_high[2]));

      // Check for overlaps with existing chromosomes
      bool found_overlap = false;
      for (size_t jchromosome = 0; jchromosome < chromosome_centers_array.size(); ++jchromosome) {
        double r_chromosome_distance = mundy::math::two_norm(chromosome_centers_array[jchromosome] - r_start);
        if (r_chromosome_distance < (r_max + chromosome_radii_array[jchromosome])) {
          found_overlap = true;
          break;
        }
      }
      if (found_overlap) {
        itrial++;
      } else {
        chromosome_inserted = true;
        chromosome_centers_array[ichromosome] = r_start;
        chromosome_radii_array[ichromosome] = r_max;
        break;
      }
    }
    MUNDY_THROW_REQUIRE(chromosome_inserted, std::runtime_error,
                        fmt::format("Failed to insert chromosome after {} trials.", max_trials));

    // Generate all the positions along the curve due to the placement in the global space
    const size_t num_nodes_per_chromosome = sphere_position_array.size();
    for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
      all_chromosome_positions[ichromosome].emplace_back(chromosome_centers_array.back() + r_chromosome_center_local -
                                                         sphere_position_array[i]);
    }
  }

  return all_chromosome_positions;
}
//@}

// //! \name Mobility
// //@{

// void check_maximum_overlap_with_periphery(stk::mesh::NgpMesh ngp_mesh, const double &maximum_allowed_overlap,
//                                           const mundy::geom::Sphere &periphery_shape,
//                                           stk::mesh::NgpField<double> &node_coords_field,
//                                           stk::mesh::NgpField<double> &element_radius_field,
//                                           const stk::mesh::Selector &selector) {
//   node_coords_field.sync_to_device();
//   element_radius_field.sync_to_device();

//   double shifted_periphery_hydro_radius = periphery_shape.radius() + maximum_allowed_overlap;

//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
//         const stk::mesh::Entity node = nodes[0];
//         const auto node_coords = mundy::mesh::vector3_field_data(node_coords_field, node);
//         const double sphere_radius = element_radius_field(node, 0);
//         const bool overlap_exceeds_threshold =
//             mundy::math::norm(node_coords) + sphere_radius > shifted_periphery_hydro_radius;
//         MUNDY_THROW_REQUIRE(!overlap_exceeds_threshold, std::runtime_error,
//                             "Sphere overlaps with peruphery beyond maximum extent allowed.");
//       });
// }

// void check_maximum_overlap_with_periphery(stk::mesh::NgpMesh &ngp_mesh, const double &maximum_allowed_overlap,
//                                           const mundy::geom::Ellipsoid &periphery_shape,
//                                           stk::mesh::NgpField<double> &node_coords_field,
//                                           stk::mesh::NgpField<double> &element_radius_field,
//                                           const stk::mesh::Selector &selector) {
//   node_coords_field.sync_to_device();
//   element_radius_field.sync_to_device();

//   double shifted_periphery_hydro_radius1 = 0.5 * periphery_shape.axis_length1() + maximum_allowed_overlap;
//   double shifted_periphery_hydro_radius2 = 0.5 * periphery_shape.axis_length2() + maximum_allowed_overlap;
//   double shifted_periphery_hydro_radius3 = 0.5 * periphery_shape.axis_length3() + maximum_allowed_overlap;

//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
//         // The following is an in-exact but cheap check.
//         // If shrinks the periphery's level set by the maximum allowed overlap and the sphere radius and then checks
//         if the
//         // sphere's center is inside the shrunk periphery. Level sets don't follow the same rules as Euclidean
//         geometry, so
//         // this is a rough check and is not even guarenteed to be conservative.
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
//         const stk::mesh::Entity node = nodes[0];
//         const auto node_coords = mundy::mesh::vector3_field_data(node_coords_field, node);
//         const double sphere_radius = element_radius_field(node, 0);
//         const double x = node_coords[0];
//         const double y = node_coords[1];
//         const double z = node_coords[2];
//         const double x2 = x * x;
//         const double y2 = y * y;
//         const double z2 = z * z;
//         const double a2 =
//             (shifted_periphery_hydro_radius1 - sphere_radius) * (shifted_periphery_hydro_radius1 - sphere_radius);
//         const double b2 =
//             (shifted_periphery_hydro_radius2 - sphere_radius) * (shifted_periphery_hydro_radius2 - sphere_radius);
//         const double c2 =
//             (shifted_periphery_hydro_radius3 - sphere_radius) * (shifted_periphery_hydro_radius3 - sphere_radius);
//         const double value = x2 / a2 + y2 / b2 + z2 / c2;
//         MUNDY_THROW_REQUIRE(value <= 1.0, std::runtime_error,
//                             "Sphere overlaps with periphery beyond maximum extent allowed.");
//       });
// }

// void copy_spheres_to_view(stk::NgpVector<stk::mesh::Entity> &ngp_sphere_entities,
//                           stk::mesh::NgpField<double> &node_coords_field,
//                           stk::mesh::NgpField<double> &element_radius_field,
//                           stk::mesh::NgpField<double> &node_force_field,
//                           stk::mesh::NgpField<double> &node_velocity_field, DoubleVecDeviceView &sphere_positions,
//                           DoubleVecDeviceView &sphere_radii, DoubleVecDeviceView &sphere_forces,
//                           DoubleVecDeviceView &sphere_velocities) {
//   node_coords_field.sync_to_device();
//   element_radius_field.sync_to_device();
//   node_force_field.sync_to_device();
//   node_velocity_field.sync_to_device();

//   const size_t num_spheres = ngp_sphere_entities.size();
//   Kokkos::parallel_for(
//       stk::ngp::RangePolicy<ExecSpace>(0, num_spheres), KOKKOS_LAMBDA(const int &vector_index) {
//         stk::mesh::Entity sphere = ngp_sphere_entities.device_get(vector_index);
//         auto sphere_index = ngp_mesh.fast_mesh_index(sphere);

//         sphere_positions(vector_index * 3 + 0) = node_coords_field(sphere_index, 0);
//         sphere_positions(vector_index * 3 + 1) = node_coords_field(sphere_index, 1);
//         sphere_positions(vector_index * 3 + 2) = node_coords_field(sphere_index, 2);

//         sphere_radii(vector_index) = element_radius_field(sphere_index, 0);

//         sphere_forces(vector_index * 3 + 0) = node_force_field(sphere_index, 0);
//         sphere_forces(vector_index * 3 + 1) = node_force_field(sphere_index, 1);
//         sphere_forces(vector_index * 3 + 2) = node_force_field(sphere_index, 2);

//         sphere_velocities(vector_index * 3 + 0) = node_velocity_field(sphere_index, 0);
//         sphere_velocities(vector_index * 3 + 1) = node_velocity_field(sphere_index, 1);
//         sphere_velocities(vector_index * 3 + 2) = node_velocity_field(sphere_index, 2);
//       });
// }

// void compute_rpy_hydro(const double &viscosity, DoubleVecDeviceView &sphere_positions,
//                        DoubleVecDeviceView &sphere_radii, DoubleVecDeviceView &sphere_forces,
//                        DoubleVecDeviceView &sphere_velocities) {
//   Kokkos::Profiling::pushRegion("HP1::compute_rpy_hydro");
//   const size_t num_spheres = sphere_radii.extent(0);
//   MUNDY_THROW_ASSERT(sphere_positions.extent(0) == 3 * num_spheres, std::runtime_error,
//                      "Sphere positions has the wrong size.");
//   MUNDY_THROW_ASSERT(sphere_forces.extent(0) == 3 * num_spheres, std::runtime_error,
//                      "Sphere forces has the wrong size.");
//   MUNDY_THROW_ASSERT(sphere_velocities.extent(0) == 3 * num_spheres, std::runtime_error,
//                      "Sphere velocities has the wrong size.");

//   // Apply the RPY kernel from spheres to spheres
//   mundy::alens::periphery::apply_rpy_kernel(stk::ngp::ExecSpace(), viscosity, sphere_positions, sphere_positions,
//                                             sphere_radii, sphere_radii, sphere_forces, sphere_velocities);

//   // The RPY kernel is only long-range, it doesn't add on self-interaction for the spheres
//   mundy::alens::periphery::apply_local_drag(stk::ngp::ExecSpace(), viscosity, sphere_velocities, sphere_forces,
//                                             sphere_radii);

//   Kokkos::Profiling::popRegion();
// }

// void compute_confined_rpy_hydro(const double &viscosity, DoubleVecDeviceView &sphere_positions,
//                                 DoubleVecDeviceView &sphere_radii, DoubleVecDeviceView &sphere_forces,
//                                 DoubleVecDeviceView &sphere_velocities, DoubleVecDeviceView &surface_positions,
//                                 DoubleVecDeviceView &surface_normals, DoubleVecDeviceView &surface_weights,
//                                 DoubleVecDeviceView &surface_radii, DoubleVecDeviceView &surface_velocities,
//                                 DoubleVecDeviceView &surface_forces, DoubleMatDeviceView
//                                 &inv_self_interaction_matrix) {
//   Kokkos::Profiling::pushRegion("HP1::compute_rpy_hydro");
//   const size_t num_spheres = sphere_radii.extent(0);
//   const size_t num_surface_nodes = surface_weights.extent(0);
//   MUNDY_THROW_ASSERT(sphere_positions.extent(0) == 3 * num_spheres, std::runtime_error,
//                      "Sphere positions has the wrong size.");
//   MUNDY_THROW_ASSERT(sphere_forces.extent(0) == 3 * num_spheres, std::runtime_error,
//                      "Sphere forces has the wrong size.");
//   MUNDY_THROW_ASSERT(sphere_velocities.extent(0) == 3 * num_spheres, std::runtime_error,
//                      "Sphere velocities has the wrong size.");
//   MUNDY_THROW_ASSERT(surface_positions.extent(0) == 3 * num_surface_nodes, std::runtime_error,
//                      "Surface positions has the wrong size.");
//   MUNDY_THROW_ASSERT(surface_normals.extent(0) == 3 * num_surface_nodes, std::runtime_error,
//                      "Surface normals has the wrong size.");
//   MUNDY_THROW_ASSERT(surface_weights.extent(0) == num_surface_nodes, std::runtime_error,
//                      "Surface weights has the wrong size.");
//   MUNDY_THROW_ASSERT(surface_radii.extent(0) == num_surface_nodes, std::runtime_error,
//                      "Surface radii has the wrong size.");
//   MUNDY_THROW_ASSERT(surface_forces.extent(0) == 3 * num_surface_nodes, std::runtime_error,
//                      "Surface forces has the wrong size.");
//   MUNDY_THROW_ASSERT(surface_velocities.extent(0) == 3 * num_surface_nodes, std::runtime_error,
//                      "Surface velocities has the wrong size.");
//   MUNDY_THROW_ASSERT((inv_self_interaction_matrix.extent(0) == 3 * num_spheres &&
//                       inv_self_interaction_matrix.extent(1) == 3 * num_spheres),
//                      std::runtime_error, "Self interaction matrix has the wrong size.");

//   // Apply the RPY kernel from spheres to spheres
//   mundy::alens::periphery::apply_rpy_kernel(stk::ngp::ExecSpace(), viscosity, sphere_positions, sphere_positions,
//                                             sphere_radii, sphere_radii, sphere_forces, sphere_velocities);

//   /////////////////////////////////////////////////////////////
//   // Apply the correction for the no-slip boundary condition //
//   /////////////////////////////////////////////////////////////
//   // Apply the RPY kernel from spheres to periphery
//   mundy::alens::periphery::apply_rpy_kernel(stk::ngp::ExecSpace(), viscosity, sphere_positions, surface_positions,
//                                             sphere_radii, surface_radii, sphere_forces, surface_velocities);

//   // Map the slip velocities to the surface forces
//   // The negative one in the gemv call accounts for the fact that our force should balance the u_slip
//   KokkosBlas::gemv(stk::ngp::ExecSpace(), "N", -1.0, inv_self_interaction_matrix, surface_velocities, 1.0,
//                    surface_forces);
//   mundy::alens::periphery::apply_stokes_double_layer_kernel(
//       stk::ngp::ExecSpace(), viscosity, num_surface_nodes, num_spheres, surface_positions, sphere_positions,
//       surface_normals, surface_weights, surface_forces, sphere_velocities);

//   //////////////////////
//   // Self-interaction //
//   //////////////////////
//   // The RPY kernel is only long-range, it doesn't add on self-interaction for the spheres
//   mundy::alens::periphery::apply_local_drag(stk::ngp::ExecSpace(), viscosity, sphere_velocities, sphere_forces,
//                                             sphere_radii);

//   Kokkos::Profiling::popRegion();
// }

// void compute_dry_velocity(stk::mesh::NgpMesh &ngp_mesh, const double &viscosity,
//                           stk::mesh::NgpField<double> &node_velocity_field,
//                           stk::mesh::NgpField<double> &node_force_field,
//                           stk::mesh::NgpField<double> &element_radius_field, const stk::mesh::Selector &selector) {
//   Kokkos::Profiling::pushRegion("HP1::compute_dry_velocity");

//   node_velocity_field.sync_to_device();
//   node_force_field.sync_to_device();
//   element_radius_field.sync_to_device();

//   constexpr double pi = Kokkos::numbers::pi_v<double>;
//   const double six_pi_mu = 6.0 * pi * viscosity;
//   const double inv_six_pi_mu = 1.0 / six_pi_mu;

//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
//         const stk::mesh::Entity node = nodes[0];
//         const auto node_velocity = mundy::mesh::vector3_field_data(node_velocity_field, node);
//         const auto node_force = mundy::mesh::vector3_field_data(node_force_field, node);
//         const double sphere_radius = element_radius_field(node, 0);

//         const double inv_drag_coeff = inv_six_pi_mu / sphere_radius;
//         node_velocity[0] += node_force[0] * inv_drag_coeff;
//         node_velocity[1] += node_force[1] * inv_drag_coeff;
//         node_velocity[2] += node_force[2] * inv_drag_coeff;
//       });

//   node_velocity_field.modify_on_device();

//   Kokkos::Profiling::popRegion();
// }
// //@}

// //! \name Collision resolution
// //@{

// void compute_spherical_periphery_collision_forces(stk::mesh::NgpMesh &ngp_mesh,
//                                                   const double &periphery_collision_spring_constant,
//                                                   const mundy::geom::Sphere &periphery_shape,
//                                                   stk::mesh::NgpField<double> &node_coords_field,
//                                                   stk::mesh::NgpField<double> &node_force_field,
//                                                   stk::mesh::NgpField<double> &element_radius_field,
//                                                   const stk::mesh::Selector &selector) {
//   Kokkos::Profiling::pushRegion("HP1::compute_spherical_periphery_collision_forces");

//   node_coords_field.sync_to_device();
//   node_force_field.sync_to_device();
//   element_radius_field.sync_to_device();

//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
//         const stk::mesh::Entity node = nodes[0];
//         const auto node_coords = mundy::mesh::vector3_field_data(node_coords_field, node);
//         const double sphere_radius = element_radius_field(node, 0);
//         const double node_coords_norm = mundy::math::two_norm(node_coords);
//         const double shared_normal_ssd = periphery_shape.radius() - node_coords_norm - sphere_radius;
//         if (shared_normal_ssd < 0.0) {
//           auto node_force = mundy::mesh::vector3_field_data(node_force_field, node);
//           auto inward_normal = -node_coords / node_coords_norm;
//           node_force[0] -= periphery_collision_spring_constant * inward_normal[0] * shared_normal_ssd;
//           node_force[1] -= periphery_collision_spring_constant * inward_normal[1] * shared_normal_ssd;
//           node_force[2] -= periphery_collision_spring_constant * inward_normal[2] * shared_normal_ssd;
//         }
//       });

//   node_force_field.modify_on_device();
//   Kokkos::Profiling::popRegion();
// }

// void compute_ellipsoidal_periphery_collision_forces(
//     stk::mesh::NgpMesh &ngp_mesh, const double &periphery_collision_spring_constant,
//     const mundy::geom::Ellipsoid &periphery_shape, stk::mesh::NgpField<double> &node_coords_field,
//     stk::mesh::NgpField<double> &node_force_field, stk::mesh::NgpField<double> &element_radius_field,
//     stk::mesh::NgpField<double> &element_aabb_field, const stk::mesh::Selector &selector) {
//   Kokkos::Profiling::pushRegion("HP1::compute_ellipsoidal_periphery_collision_forces");

//   // Setup the ellipsoid level set function
//   const double a = 0.5 * periphery_shape.axis_length1();
//   const double b = 0.5 * periphery_shape.axis_length2();
//   const double c = 0.5 * periphery_shape.axis_length3();
//   const double inv_a2 = 1.0 / (a * a);
//   const double inv_b2 = 1.0 / (b * b);
//   const double inv_c2 = 1.0 / (c * c);
//   auto level_set = [&inv_a2, &inv_b2, &inv_c2, &periphery_shape](const mundy::math::Vector3<double> &point) -> double
//   {
//     const auto body_frame_point =
//         mundy::math::conjugate(periphery_shape.orientation()) * (point - periphery_shape.center());
//     return (body_frame_point[0] * body_frame_point[0] * inv_a2 + body_frame_point[1] * body_frame_point[1] * inv_b2 +
//             body_frame_point[2] * body_frame_point[2] * inv_c2) -
//            1;
//   };

//   // Evaluate the potential
//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
//         const stk::mesh::Entity node = nodes[0];

//         // For our coarse search, we check if the coners of the sphere's aabb lie inside the ellipsoidal periphery
//         // This can be done via the (body frame) inside outside unftion f(x, y, z) = 1 - (x^2/a^2 + y^2/b^2 +
//         z^2/c^2)
//         // This is possible due to the convexity of the ellipsoid
//         auto aabb = element_aabb_field(sphere_index);
//         const double &x0 = aabb[0];
//         const double &y0 = aabb[1];
//         const double &z0 = aabb[2];
//         const double &x1 = aabb[3];
//         const double &y1 = aabb[4];
//         const double &z1 = aabb[5];

//         // Compute all 8 corners of the AABB
//         const auto bottom_left_front = mundy::math::Vector3<double>(x0, y0, z0);
//         const auto bottom_right_front = mundy::math::Vector3<double>(x1, y0, z0);
//         const auto top_left_front = mundy::math::Vector3<double>(x0, y1, z0);
//         const auto top_right_front = mundy::math::Vector3<double>(x1, y1, z0);
//         const auto bottom_left_back = mundy::math::Vector3<double>(x0, y0, z1);
//         const auto bottom_right_back = mundy::math::Vector3<double>(x1, y0, z1);
//         const auto top_left_back = mundy::math::Vector3<double>(x0, y1, z1);
//         const auto top_right_back = mundy::math::Vector3<double>(x1, y1, z1);
//         const double all_points_inside_periphery =
//             level_set(bottom_left_front) < 0.0 && level_set(bottom_right_front) < 0.0 &&
//             level_set(top_left_front) < 0.0 && level_set(top_right_front) < 0.0 && level_set(bottom_left_back) < 0.0
//             && level_set(bottom_right_back) < 0.0 && level_set(top_left_back) < 0.0 && level_set(top_right_back) <
//             0.0;

//         if (!all_points_inside_periphery) {
//           // We might have a collision, perform the more expensive check
//           const auto node_coords = mundy::mesh::vector3_field_data(node_coords_field, node);
//           const double sphere_radius = element_radius_field(node, 0);

//           // Note, the ellipsoid for the ssd calc has outward normal, whereas the periphery has inward normal.
//           // Hence, the sign flip.
//           mundy::math::Vector3<double> contact_point;
//           mundy::math::Vector3<double> ellipsoid_nhat;
//           const double shared_normal_ssd = -mundy::math::distance::shared_normal_ssd_between_ellipsoid_and_point(
//                                                periphery_shape.center(), periphery_shape.orientation(), a, b, c,
//                                                node_coords, contact_point, ellipsoid_nhat) -
//                                            sphere_radius;

//           if (shared_normal_ssd < 0.0) {
//             // We have a collision, compute the force
//             auto node_force = mundy::mesh::vector3_field_data(node_force_field, node);
//             auto periphery_nhat = -ellipsoid_nhat;
//             node_force[0] -= periphery_collision_spring_constant * periphery_nhat[0] * shared_normal_ssd;
//             node_force[1] -= periphery_collision_spring_constant * periphery_nhat[1] * shared_normal_ssd;
//             node_force[2] -= periphery_collision_spring_constant * periphery_nhat[2] * shared_normal_ssd;
//           }
//         }
//       });

//   node_force_field.modify_on_device();
//   Kokkos::Profiling::popRegion();
// }
// //@}

// //! \name Misc problem-specific physics
// //@{

// void compute_brownian_velocity(stk::mesh::NgpMesh &ngp_mesh, const double &timestep_size, const double &kt,
//                                const double &viscosity, stk::mesh::NgpField<double> &node_velocity_field,
//                                stk::mesh::NgpField<unsigned> &elem_rng_field,
//                                stk::mesh::NgpField<double> &element_radius_field, stk::mesh::Selector &selector) {
//   Kokkos::Profiling::pushRegion("HP1::compute_brownian_velocity");

//   node_velocity_field.sync_to_device();
//   elem_rng_field.sync_to_device();
//   element_radius_field.sync_to_device();

//   constexpr double pi = Kokkos::numbers::pi_v<double>;
//   const double six_pi_mu = 6.0 * pi * viscosity;
//   const double sqrt_6_pi_mu = Kokkos::sqrt(six_pi_mu);
//   const double inv_six_pi_mu = 1.0 / six_pi_mu;
//   const double inv_dt = 1.0 / timestep_size;
//   const double sqrt_2_kt = Kokkos::sqrt(2.0 * kt);

//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::ELEM_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
//         stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
//         const stk::mesh::Entity node = nodes[0];

//         // Setup the rng
//         const stk::mesh::EntityId sphere_gid = ngp_mesh.identifier(sphere_index);
//         auto rng_counter = elem_rng_field(sphere_index);
//         openrand::Philox rng(sphere_gid, rng_counter[0]);

//         // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
//         // for drag coeff gamma = 6 * pi * mu * r
//         const double sphere_radius = element_radius_field(node, 0);
//         auto node_velocity = node_velocity_field(node);
//         const double coeff =
//             sqrt_2_kt * Kokkos::sqrt(sqrt_6_pi_mu * sphere_radius * inv_dt) * inv_six_pi_mu / sphere_radius;
//         node_velocity[0] += coeff * rng.randn<double>();
//         node_velocity[1] += coeff * rng.randn<double>();
//         node_velocity[2] += coeff * rng.randn<double>();
//         rng_counter[0]++;
//       });

//   Kokkos::Profiling::popRegion();
// }

// void node_euler_position_update(stk::mesh::NgpMesh &ngp_mesh, const double &timestep_size,
//                                 stk::mesh::NgpField<double> &node_coords_field,
//                                 stk::mesh::NgpField<double> &node_velocity_field, const stk::mesh::Selector
//                                 &selector) {
//   Kokkos::Profiling::pushRegion("HP1::update_positions");

//   node_coords_field.sync_to_device();
//   node_velocity_field.sync_to_device();

//   mundy::mesh::for_each_entity_run(
//       ngp_mesh, stk::topology::NODE_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
//         const stk::mesh::Entity node = ngp_mesh.get_entity(stk::topology::NODE_RANK, node_index);
//         auto node_coords = node_coords_field(node);
//         auto node_velocity = node_velocity_field(node);

//         node_coords[0] += timestep_size * node_velocity[0];
//         node_coords[1] += timestep_size * node_velocity[1];
//         node_coords[2] += timestep_size * node_velocity[2];
//       });

//   node_coords_field.modify_on_device();

//   Kokkos::Profiling::popRegion();
// }
// //@}

//! \name Simulation setupo/run
//@{

struct HP1ParamParser {
  void print_help_message() {
    std::cout << "#############################################################################################"
              << std::endl;
    std::cout << "To run this code, please pass in --params=<input.yaml> as a command line argument." << std::endl;
    std::cout << std::endl;
    std::cout << "Note, all parameters and sublists in input.yaml must be contained in a single top-level list."
              << std::endl;
    std::cout << "Such as:" << std::endl;
    std::cout << std::endl;
    std::cout << "HP1:" << std::endl;
    std::cout << "  num_time_steps: 1000" << std::endl;
    std::cout << "  timestep_size: 1e-6" << std::endl;
    std::cout << "#############################################################################################"
              << std::endl;
    std::cout << "The valid parameters that can be set in the input file are:" << std::endl;
    Teuchos::ParameterList valid_params = get_valid_params();

    auto print_options =
        Teuchos::ParameterList::PrintOptions().showTypes(false).showDoc(true).showDefault(true).showFlags(false).indent(
            1);
    valid_params.print(std::cout, print_options);
    std::cout << "#############################################################################################"
              << std::endl;
  }

  Teuchos::ParameterList parse(int argc, char **argv) {
    // Parse the command line options to find the input filename
    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("params", &input_parameter_filename_, "The name of the input file.");

    Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_result = cmdp.parse(argc, argv);
    if (parse_result == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
      print_help_message();

      // Safely exit the program
      // If we print the help message, we don't need to do anything else.
      exit(0);

    } else if (parse_result != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      throw std::invalid_argument("Failed to parse the command line arguments.");
    }

    // Read, validate, and parse in the parameters from the parameter list.
    try {
      Teuchos::ParameterList param_list = *Teuchos::getParametersFromYamlFile(input_parameter_filename_);
      return parse(param_list);
    } catch (const std::exception &e) {
      std::cerr << "ERROR: Failed to read the input parameter file." << std::endl;
      std::cerr << "During read, the following error occurred: " << e.what() << std::endl;
      std::cerr << "NOTE: This can happen for any number of reasons. Check that the file exists and contains the "
                   "expected parameters."
                << std::endl;
      throw e;
    }

    return Teuchos::ParameterList();
  }

  Teuchos::ParameterList parse(const Teuchos::ParameterList &param_list) {
    // Validate the parameters and set the defaults.
    Teuchos::ParameterList valid_param_list = param_list;
    valid_param_list.validateParametersAndSetDefaults(get_valid_params());
    check_invariants(valid_param_list);
    dump_parameters(valid_param_list);
    return valid_param_list;
  }

  void check_invariants(const Teuchos::ParameterList &valid_param_list) {
    // Check the sim params
    const auto &sim_params = valid_param_list.sublist("sim");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("timestep_size") > 0, std::invalid_argument,
                        "timestep_size must be greater than 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("viscosity") > 0, std::invalid_argument,
                        "viscosity must be greater than 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("initial_chromosome_separation") >= 0, std::invalid_argument,
                        "initial_chromosome_separation must be greater than or equal to 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<bool>("enable_periphery_hydrodynamics")
                            ? sim_params.get<bool>("enable_backbone_n_body_hydrodynamics")
                            : true,
                        std::invalid_argument,
                        "Periphery hydrodynamics requires backbone hydrodynamics to be enabled.");

    MUNDY_THROW_REQUIRE(sim_params.get<std::string>("initialization_type") == "GRID" ||
                            sim_params.get<std::string>("initialization_type") == "RANDOM_UNIT_CELL" ||
                            sim_params.get<std::string>("initialization_type") == "OVERLAP_TEST" ||
                            sim_params.get<std::string>("initialization_type") == "HILBERT_RANDOM_UNIT_CELL" ||
                            sim_params.get<std::string>("initialization_type") == "USHAPE_TEST" ||
                            sim_params.get<std::string>("initialization_type") == "FROM_EXO" ||
                            sim_params.get<std::string>("initialization_type") == "FROM_DAT",
                        std::invalid_argument,
                        fmt::format("Invalid initialization_type ({}). Valid options are GRID, RANDOM_UNIT_CELL, "
                                    "OVERLAP_TEST, HILBERT_RANDOM_UNIT_CELL, USHAPE_TEST, FROM_EXO, FROM_DAT.",
                                    sim_params.get<std::string>("initialization_type")));

    if (sim_params.get<bool>("enable_backbone_springs")) {
      const auto &backbone_spring_params = valid_param_list.sublist("backbone_springs");
      MUNDY_THROW_REQUIRE(backbone_spring_params.get<std::string>("spring_type") == "HARMONIC" ||
                              backbone_spring_params.get<std::string>("spring_type") == "FENE",
                          std::invalid_argument,
                          fmt::format("Invalid spring_type ({}). Valid options are HARMONIC and FENE.",
                                      backbone_spring_params.get<std::string>("spring_type")));
      MUNDY_THROW_REQUIRE(backbone_spring_params.get<double>("spring_constant") >= 0, std::invalid_argument,
                          "spring_constant must be non-negative.");
      MUNDY_THROW_REQUIRE(backbone_spring_params.get<double>("spring_r0") >= 0, std::invalid_argument,
                          "max_spring_length must be non-negative.");
    }

    // Check the periphery_hydro params
    if (sim_params.get<bool>("enable_periphery_hydrodynamics")) {
      const auto &periphery_hydro_params = valid_param_list.sublist("periphery_hydro");
      std::string periphery_hydro_shape = periphery_hydro_params.get<std::string>("shape");
      std::string periphery_hydro_quadrature = periphery_hydro_params.get<std::string>("quadrature");
      if (periphery_hydro_quadrature == "GAUSS_LEGENDRE") {
        double periphery_hydro_axis_radius1 = periphery_hydro_params.get<double>("axis_radius1");
        double periphery_hydro_axis_radius2 = periphery_hydro_params.get<double>("axis_radius2");
        double periphery_hydro_axis_radius3 = periphery_hydro_params.get<double>("axis_radius3");
        MUNDY_THROW_REQUIRE(
            (periphery_hydro_shape == "SPHERE") || ((periphery_hydro_shape == "ELLIPSOID") &&
                                                    (periphery_hydro_axis_radius1 == periphery_hydro_axis_radius2) &&
                                                    (periphery_hydro_axis_radius2 == periphery_hydro_axis_radius3) &&
                                                    (periphery_hydro_axis_radius3 == periphery_hydro_axis_radius1)),
            std::invalid_argument, "Gauss-Legendre quadrature is only valid for spherical peripheries.");
      }
    }
  }

  static Teuchos::ParameterList get_valid_params() {
    // Create a paramater entity validator for our large integers to allow for both int and long long.
    auto prefer_size_t = []() {
      if (std::is_same_v<size_t, unsigned short>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_SHORT;
      } else if (std::is_same_v<size_t, unsigned int>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
      } else if (std::is_same_v<size_t, unsigned long>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG;
      } else if (std::is_same_v<size_t, unsigned long long>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG_LONG;
      } else {
        throw std::runtime_error("Unknown size_t type.");
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
      }
    }();
    const bool allow_all_types_by_default = false;
    mundy::core::OurAnyNumberParameterEntryValidator::AcceptedTypes accept_int(allow_all_types_by_default);
    accept_int.allow_all_integer_types(true);
    auto make_new_validator = [](const auto &preferred_type, const auto &accepted_types) {
      return Teuchos::rcp(new mundy::core::OurAnyNumberParameterEntryValidator(preferred_type, accepted_types));
    };

    static Teuchos::ParameterList valid_parameter_list;

    valid_parameter_list.sublist("sim")
        .set("num_time_steps", 100, "Number of time steps.", make_new_validator(prefer_size_t, accept_int))
        .set("timestep_size", 0.001, "Time step size.")
        .set("viscosity", 1.0, "Viscosity.")
        .set("num_chromosomes", static_cast<size_t>(1), "Number of chromosomes.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_hetero_euchromatin_repeats", static_cast<size_t>(2), "Number of heterochromatin/euchromatin repeats per chain.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_euchromatin_per_repeat", static_cast<size_t>(1), "Number of euchromatin beads per repeat.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_heterochromatin_per_repeat", static_cast<size_t>(1), "Number of heterochromatin beads per repeat.",
             make_new_validator(prefer_size_t, accept_int))
        .set("backbone_sphere_hydrodynamic_radius", 0.05,
             "Backbone sphere hydrodynamic radius. Even if n-body hydrodynamics is disabled, we still have "
             "self-interaction.")
        .set("initial_chromosome_separation", 1.0, "Initial chromosome separation.")
        .set("initialization_type", std::string("GRID"),
             "Initialization_type. Valid options are GRID, RANDOM_UNIT_CELL, "
             "OVERLAP_TEST, HILBERT_RANDOM_UNIT_CELL, USHAPE_TEST, "
             "FROM_EXO, FROM_DAT.")
        .set("initialize_from_exo_filename", std::string("HP1"),
             "Exo file to initialize from if initialization_type is FROM_EXO.")
        .set("initialize_from_dat_filename", std::string("HP1_pos.dat"),
             "Dat file to initialize from if initialization_type is FROM_DAT.")
        .set<Teuchos::Array<double>>(
            "domain_low", Teuchos::tuple<double>(0.0, 0.0, 0.0),
            "Lower left corner of the unit cell. (Only used if initialization_type involves a 'UNIT_CELL').")
        .set<Teuchos::Array<double>>(
            "domain_high", Teuchos::tuple<double>(10.0, 10.0, 10.0),
            "Upper right corner of the unit cell. (Only used if initialization_type involves a 'UNIT_CELL').")
        .set("check_maximum_speed_pre_position_update", false, "Check maximum speed before updating positions.")
        .set("max_allowable_speed", std::numeric_limits<double>::max(),
             "Maximum allowable speed (only used if "
             "check_maximum_speed_pre_position_update is true).")
        .set("loadbalance_post_initialization", false, "If we should load balance post-initialization or not.")
        // IO
        .set("io_frequency", static_cast<size_t>(10), "Number of timesteps between writing output.",
             make_new_validator(prefer_size_t, accept_int))
        .set("log_frequency", static_cast<size_t>(10), "Number of timesteps between logging.",
             make_new_validator(prefer_size_t, accept_int))
        .set("output_filename", std::string("HP1"), "Output filename.")
        .set("enable_continuation_if_available", true,
             "Enable continuing a previous simulation if an output file already exists.")
        // Control flags
        .set("enable_chromatin_brownian_motion", true, "Enable chromatin Brownian motion.")
        .set("enable_backbone_springs", true, "Enable backbone springs.")
        .set("enable_backbone_collision", true, "Enable backbone collision.")
        .set("enable_backbone_n_body_hydrodynamics", true, "Enable backbone N-body hydrodynamics.")
        .set("enable_crosslinkers", true, "Enable crosslinkers.")
        .set("enable_periphery_collision", true, "Enable periphery collision.")
        .set("enable_periphery_hydrodynamics", true, "Enable periphery hydrodynamics.")
        .set("enable_periphery_binding", true, "Enable periphery binding.")
        .set("enable_active_euchromatin_forces", true, "Enable active euchromatin forces.");

    valid_parameter_list.sublist("brownian_motion").set("kt", 1.0, "Temperature kT for Brownian Motion.");

    valid_parameter_list.sublist("backbone_springs")
        .set("spring_type", std::string("HARMONIC"), "Chromatin spring type. Valid options are HARMONIC or FENE.")
        .set("spring_constant", 100.0, "Chromatin spring constant.")
        .set("spring_r0", 1.0, "Chromatin rest length (HARMONIC) or rmax (FENE).");

    valid_parameter_list.sublist("backbone_collision")
        .set("radius", 0.5, "Backbone excluded volume radius.")
        .set("youngs_modulus", 1000.0, "Backbone Young's modulus.")
        .set("poissons_ratio", 0.3, "Backbone Poisson's ratio.");

    valid_parameter_list.sublist("crosslinker")
        .set("spring_type", std::string("HARMONIC"), "Crosslinker spring type. Valid options are HARMONIC or FENE.")
        .set("kt", 1.0, "Temperature kT for crosslinkers.")
        .set("spring_constant", 10.0, "Crosslinker spring constant.")
        .set("spring_r0", 2.5, "Crosslinker rest length.")
        .set("left_binding_rate", 1.0, "Crosslinker left binding rate.")
        .set("right_binding_rate", 1.0, "Crosslinker right binding rate.")
        .set("left_unbinding_rate", 1.0, "Crosslinker left unbinding rate.")
        .set("right_unbinding_rate", 1.0, "Crosslinker right unbinding rate.");

    valid_parameter_list.sublist("periphery_hydro")
        .set("check_maximum_periphery_overlap", false, "Check maximum periphery overlap.")
        .set("maximum_allowed_periphery_overlap", 1e-6, "Maximum allowed periphery overlap.")
        .set("shape", std::string("SPHERE"), "Periphery hydrodynamic shape. Valid options are SPHERE or ELLIPSOID.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("quadrature", std::string("GAUSS_LEGENDRE"),
             "Periphery quadrature. Valid options are GAUSS_LEGENDRE or "
             "FROM_FILE.")
        .set("spectral_order", static_cast<size_t>(32),
             "Periphery spectral order (only used if periphery is spherical is Gauss-Legendre quadrature).",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_quadrature_points", static_cast<size_t>(1000),
             "Periphery number of quadrature points (only used if quadrature type is FROM_FILE). Number of points in "
             "the files must match this quantity.",
             make_new_validator(prefer_size_t, accept_int))
        .set("quadrature_points_filename", std::string("hp1_periphery_hydro_quadrature_points.dat"),
             "Periphery quadrature points filename (only used if quadrature type is FROM_FILE).")
        .set("quadrature_weights_filename", std::string("hp1_periphery_hydro_quadrature_weights.dat"),
             "Periphery quadrature weights filename (only used if quadrature type is FROM_FILE).")
        .set("quadrature_normals_filename", std::string("hp1_periphery_hydro_quadrature_normals.dat"),
             "Periphery quadrature normals filename (only used if quadrature type is FROM_FILE).");

    valid_parameter_list.sublist("periphery_collision")
        .set("shape", std::string("SPHERE"), "Periphery collision shape. Valid options are SPHERE or ELLIPSOID.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("collision_spring_constant", 1000.0, "Periphery collision spring constant.")
        .set("use_fast_approx", false, "Use fast periphery collision.")
        .set("shrink_periphery_over_time", false, "Shrink periphery over time.")
        .sublist("shrinkage")
        .set("num_shrinkage_steps", static_cast<size_t>(1000),
             "Number of steps over which to perform the shrinking process (should not exceed num_time_steps).",
             make_new_validator(prefer_size_t, accept_int))
        .set("scale_factor_before_shrinking", 1.0, "Scale factor before shrinking.");

    valid_parameter_list.sublist("periphery_binding")
        .set("binding_rate", 1.0, "Periphery binding rate.")
        .set("unbinding_rate", 1.0, "Periphery unbinding rate.")
        .set("spring_constant", 1000.0, "Periphery spring constant.")
        .set("spring_r0", 1.0, "Periphery spring rest length.")
        .set("bind_sites_type", std::string("RANDOM"),
             "Periphery bind sites type. Valid options are RANDOM or FROM_FILE.")       
        .set("shape", std::string("SPHERE"), "The shape of the binding site locations. Only used if bind_sites_type is RANDOM. Valid options are SPHERE or ELLIPSOID.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("num_bind_sites", static_cast<size_t>(1000),
             "Periphery number of binding sites (only used if periphery_binding_sites_type is RANDOM and periphery "
             "has spherical or ellipsoidal shape).",
             make_new_validator(prefer_size_t, accept_int))
        .set("bind_site_locations_filename", std::string("periphery_bind_sites.dat"),
             "Periphery binding sites filename (only used if periphery_binding_sites_type is FROM_FILE).");

    valid_parameter_list.sublist("active_euchromatin_forces")
        .set("force_sigma", 1.0, "Active euchromatin force sigma.")
        .set("kon", 1.0, "Active euchromatin force kon.")
        .set("koff", 1.0, "Active euchromatin force koff.");

    valid_parameter_list.sublist("neighbor_list")
        .set("skin_distance", 1.0, "Neighbor list skin distance.")
        .set("force_neighborlist_update", false, "Force update of the neighbor list.")
        .set("force_neighborlist_update_nsteps", static_cast<size_t>(10),
             "Number of timesteps between force update of the neighbor list.",
             make_new_validator(prefer_size_t, accept_int))
        .set("print_neighborlist_statistics", false, "Print neighbor list statistics.");

    return valid_parameter_list;
  }

  void dump_parameters(const Teuchos::ParameterList &valid_param_list) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;

      std::cout << std::endl;
      const auto &sim_params = valid_param_list.sublist("sim");
      std::cout << "SIMULATION:" << std::endl;
      std::cout << "  num_time_steps:  " << sim_params.get<size_t>("num_time_steps") << std::endl;
      std::cout << "  timestep_size:   " << sim_params.get<double>("timestep_size") << std::endl;
      std::cout << "  viscosity:       " << sim_params.get<double>("viscosity") << std::endl;
      std::cout << "  num_chromosomes: " << sim_params.get<size_t>("num_chromosomes") << std::endl;
      std::cout << "  num_hetero_euchromatin_repeats:      " << sim_params.get<size_t>("num_hetero_euchromatin_repeats") << std::endl;
      std::cout << "  num_euchromatin_per_repeat: " << sim_params.get<size_t>("num_euchromatin_per_repeat")
                << std::endl;
      std::cout << "  num_heterochromatin_per_repeat:  " << sim_params.get<size_t>("num_heterochromatin_per_repeat")
                << std::endl;
      std::cout << "  backbone_sphere_hydrodynamic_radius: "
                << sim_params.get<double>("backbone_sphere_hydrodynamic_radius") << std::endl;
      std::cout << "  initial_chromosome_separation:   " << sim_params.get<double>("initial_chromosome_separation")
                << std::endl;
      std::cout << "  initialization_type:             " << sim_params.get<std::string>("initialization_type")
                << std::endl;
      if (sim_params.get<std::string>("initialization_type") == "FROM_EXO") {
        std::cout << "  initialize_from_file_filename: " << sim_params.get<std::string>("initialize_from_exo_filename")
                  << std::endl;
      }

      if (sim_params.get<std::string>("initialization_type") == "FROM_DAT") {
        std::cout << "  initialize_from_file_filename: " << sim_params.get<std::string>("initialize_from_dat_filename")
                  << std::endl;
      }

      if ((sim_params.get<std::string>("initialization_type") == "RANDOM_UNIT_CELL") ||
          (sim_params.get<std::string>("initialization_type") == "HILBERT_RANDOM_UNIT_CELL")) {
        auto domain_low = sim_params.get<Teuchos::Array<double>>("domain_low");
        auto domain_high = sim_params.get<Teuchos::Array<double>>("domain_high");
        std::cout << "  domain_low: {" << domain_low[0] << ", " << domain_low[1] << ", " << domain_low[2] << "}"
                  << std::endl;
        std::cout << "  domain_high: {" << domain_high[0] << ", " << domain_high[1] << ", " << domain_high[2] << "}"
                  << std::endl;
      }

      std::cout << "  loadbalance_post_initialization: " << sim_params.get<bool>("loadbalance_post_initialization")
                << std::endl;
      std::cout << "  check_maximum_speed_pre_position_update: "
                << sim_params.get<bool>("check_maximum_speed_pre_position_update") << std::endl;
      if (sim_params.get<bool>("check_maximum_speed_pre_position_update")) {
        std::cout << "  max_allowable_speed: " << sim_params.get<double>("max_allowable_speed") << std::endl;
      }
      std::cout << std::endl;

      std::cout << "IO:" << std::endl;
      std::cout << "  io_frequency:    " << sim_params.get<size_t>("io_frequency") << std::endl;
      std::cout << "  log_frequency:   " << sim_params.get<size_t>("log_frequency") << std::endl;
      std::cout << "  output_filename: " << sim_params.get<std::string>("output_filename") << std::endl;
      std::cout << "  enable_continuation_if_available: " << sim_params.get<bool>("enable_continuation_if_available")
                << std::endl;
      std::cout << std::endl;

      std::cout << "CONTROL FLAGS:" << std::endl;
      std::cout << "  enable_chromatin_brownian_motion: " << sim_params.get<bool>("enable_chromatin_brownian_motion")
                << std::endl;
      std::cout << "  enable_backbone_springs:          " << sim_params.get<bool>("enable_backbone_springs")
                << std::endl;
      std::cout << "  enable_backbone_collision:        " << sim_params.get<bool>("enable_backbone_collision")
                << std::endl;
      std::cout << "  enable_backbone_n_body_hydrodynamics:    "
                << sim_params.get<bool>("enable_backbone_n_body_hydrodynamics") << std::endl;
      std::cout << "  enable_crosslinkers:              " << sim_params.get<bool>("enable_crosslinkers") << std::endl;
      std::cout << "  enable_periphery_hydrodynamics:   " << sim_params.get<bool>("enable_periphery_hydrodynamics")
                << std::endl;
      std::cout << "  enable_periphery_collision:       " << sim_params.get<bool>("enable_periphery_collision")
                << std::endl;
      std::cout << "  enable_periphery_binding:         " << sim_params.get<bool>("enable_periphery_binding")
                << std::endl;
      std::cout << "  enable_active_euchromatin_forces: " << sim_params.get<bool>("enable_active_euchromatin_forces")
                << std::endl;

      if (sim_params.get<bool>("enable_chromatin_brownian_motion")) {
        const auto &brownian_motion_params = valid_param_list.sublist("brownian_motion");

        std::cout << std::endl;
        std::cout << "BROWNIAN MOTION:" << std::endl;
        std::cout << "  kt: " << brownian_motion_params.get<double>("kt") << std::endl;
      }

      if (sim_params.get<bool>("enable_backbone_springs")) {
        const auto &backbone_springs_params = valid_param_list.sublist("backbone_springs");

        std::cout << std::endl;
        std::cout << "BACKBONE SPRINGS:" << std::endl;
        std::cout << "  spring_type:      " << backbone_springs_params.get<std::string>("spring_type") << std::endl;
        std::cout << "  spring_constant:  " << backbone_springs_params.get<double>("spring_constant") << std::endl;
        if (backbone_springs_params.get<std::string>("spring_type") == "HARMONIC") {
          std::cout << "  spring_rest_length: " << backbone_springs_params.get<double>("spring_r0") << std::endl;
        } else if (backbone_springs_params.get<std::string>("spring_type") == "FENE") {
          std::cout << "  spring_rmax:        " << backbone_springs_params.get<double>("spring_r0") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_backbone_collision")) {
        const auto &backbone_collision_params = valid_param_list.sublist("backbone_collision");

        std::cout << std::endl;
        std::cout << "BACKBONE COLLISION:" << std::endl;
        std::cout << "  excluded_volume_radius: " << backbone_collision_params.get<double>("radius") << std::endl;
        std::cout << "  youngs_modulus: " << backbone_collision_params.get<double>("youngs_modulus") << std::endl;
        std::cout << "  poissons_ratio: " << backbone_collision_params.get<double>("poissons_ratio") << std::endl;
      }

      if (sim_params.get<bool>("enable_crosslinkers")) {
        const auto &crosslinker_params = valid_param_list.sublist("crosslinker");

        std::cout << std::endl;
        std::cout << "CROSSLINKERS:" << std::endl;
        std::cout << "  spring_type: " << crosslinker_params.get<std::string>("spring_type") << std::endl;
        std::cout << "  kt: " << crosslinker_params.get<double>("kt") << std::endl;
        std::cout << "  spring_constant: " << crosslinker_params.get<double>("spring_constant") << std::endl;
        std::cout << "  spring_r0: " << crosslinker_params.get<double>("spring_r0") << std::endl;
        std::cout << "  left_binding_rate: " << crosslinker_params.get<double>("left_binding_rate") << std::endl;
        std::cout << "  right_binding_rate: " << crosslinker_params.get<double>("right_binding_rate") << std::endl;
        std::cout << "  left_unbinding_rate: " << crosslinker_params.get<double>("left_unbinding_rate") << std::endl;
        std::cout << "  right_unbinding_rate: " << crosslinker_params.get<double>("right_unbinding_rate") << std::endl;
      }

      if (sim_params.get<bool>("enable_periphery_hydrodynamics")) {
        const auto &periphery_hydro_params = valid_param_list.sublist("periphery_hydro");

        std::cout << std::endl;
        std::cout << "PERIPHERY HYDRODYNAMICS:" << std::endl;
        std::cout << "  check_maximum_periphery_overlap: "
                  << periphery_hydro_params.get<bool>("check_maximum_periphery_overlap") << std::endl;
        if (periphery_hydro_params.get<bool>("check_maximum_periphery_overlap")) {
          std::cout << "  maximum_allowed_periphery_overlap: "
                    << periphery_hydro_params.get<double>("maximum_allowed_periphery_overlap") << std::endl;
        }
        if (periphery_hydro_params.get<std::string>("shape") == "SPHERE") {
          std::cout << "  shape: SPHERE" << std::endl;
          std::cout << "  radius: " << periphery_hydro_params.get<double>("radius") << std::endl;
        } else if (periphery_hydro_params.get<std::string>("shape") == "ELLIPSOID") {
          std::cout << "  shape: ELLIPSOID" << std::endl;
          std::cout << "  axis_radius1: " << periphery_hydro_params.get<double>("axis_radius1") << std::endl;
          std::cout << "  axis_radius2: " << periphery_hydro_params.get<double>("axis_radius2") << std::endl;
          std::cout << "  axis_radius3: " << periphery_hydro_params.get<double>("axis_radius3") << std::endl;
        }
        if (periphery_hydro_params.get<std::string>("quadrature") == "GAUSS_LEGENDRE") {
          std::cout << "  quadrature: GAUSS_LEGENDRE" << std::endl;
          std::cout << "  spectral_order: " << periphery_hydro_params.get<size_t>("spectral_order") << std::endl;
        } else if (periphery_hydro_params.get<std::string>("quadrature") == "FROM_FILE") {
          std::cout << "  quadrature: FROM_FILE" << std::endl;
          std::cout << "  num_quadrature_points: " << periphery_hydro_params.get<size_t>("num_quadrature_points")
                    << std::endl;
          std::cout << "  quadrature_points_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_points_filename") << std::endl;
          std::cout << "  quadrature_weights_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_weights_filename") << std::endl;
          std::cout << "  quadrature_normals_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_normals_filename") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_periphery_collision")) {
        const auto &periphery_collision_params = valid_param_list.sublist("periphery_collision");

        std::cout << std::endl;
        std::cout << "PERIPHERY COLLISION:" << std::endl;
        if (periphery_collision_params.get<std::string>("shape") == "SPHERE") {
          std::cout << "  shape: SPHERE" << std::endl;
          std::cout << "  radius: " << periphery_collision_params.get<double>("radius") << std::endl;
        } else if (periphery_collision_params.get<std::string>("shape") == "ELLIPSOID") {
          std::cout << "  shape: ELLIPSOID" << std::endl;
          std::cout << "  axis_radius1: " << periphery_collision_params.get<double>("axis_radius1") << std::endl;
          std::cout << "  axis_radius2: " << periphery_collision_params.get<double>("axis_radius2") << std::endl;
          std::cout << "  axis_radius3: " << periphery_collision_params.get<double>("axis_radius3") << std::endl;
        }
        std::cout << "  collision_spring_constant: "
                  << periphery_collision_params.get<double>("collision_spring_constant") << std::endl;
        std::cout << "  periphery_collision_use_fast_approx: "
                  << periphery_collision_params.get<bool>("use_fast_approx") << std::endl;
        std::cout << "  shrink_periphery_over_time: "
                  << periphery_collision_params.get<bool>("shrink_periphery_over_time") << std::endl;
        if (periphery_collision_params.get<bool>("shrink_periphery_over_time")) {
          std::cout << "  SHRINKAGE:" << std::endl;
          std::cout << "    num_shrinkage_steps: "
                    << periphery_collision_params.sublist("shrinkage").get<size_t>("num_shrinkage_steps") << std::endl;
          std::cout << "    scale_factor_before_shrinking: "
                    << periphery_collision_params.sublist("shrinkage").get<double>("scale_factor_before_shrinking")
                    << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_periphery_binding")) {
        const auto &periphery_binding_params = valid_param_list.sublist("periphery_binding");

        std::cout << std::endl;
        std::cout << "PERIPHERY BINDING:" << std::endl;
        std::cout << "  binding_rate: " << periphery_binding_params.get<double>("binding_rate") << std::endl;
        std::cout << "  unbinding_rate: " << periphery_binding_params.get<double>("unbinding_rate") << std::endl;
        std::cout << "  spring_constant: " << periphery_binding_params.get<double>("spring_constant") << std::endl;
        std::cout << "  spring_r0: " << periphery_binding_params.get<double>("spring_r0") << std::endl;
        if (periphery_binding_params.get<std::string>("bind_sites_type") == "RANDOM") {
          std::cout << "  bind_sites_type: RANDOM" << std::endl;
          if (periphery_binding_params.get<std::string>("shape") == "SPHERE") {
            std::cout << "  shape: SPHERE" << std::endl;
            std::cout << "  radius: " << periphery_binding_params.get<double>("radius") << std::endl;
          } else if (periphery_binding_params.get<std::string>("shape") == "ELLIPSOID") {
            std::cout << "  shape: ELLIPSOID" << std::endl;
            std::cout << "  axis_radius1: " << periphery_binding_params.get<double>("axis_radius1") << std::endl;
            std::cout << "  axis_radius2: " << periphery_binding_params.get<double>("axis_radius2") << std::endl;
            std::cout << "  axis_radius3: " << periphery_binding_params.get<double>("axis_radius3") << std::endl;
          }

          std::cout << "  num_bind_sites: " << periphery_binding_params.get<size_t>("num_bind_sites") << std::endl;
        } else if (periphery_binding_params.get<std::string>("bind_sites_type") == "FROM_FILE") {
          std::cout << "  bind_sites_type: FROM_FILE" << std::endl;
          std::cout << "  bind_site_locations_filename: "
                    << periphery_binding_params.get<std::string>("bind_site_locations_filename") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_active_euchromatin_forces")) {
        const auto &active_euchromatin_forces_params = valid_param_list.sublist("active_euchromatin_forces");

        std::cout << std::endl;
        std::cout << "ACTIVE EUCHROMATIN FORCES:" << std::endl;
        std::cout << "  force_sigma: " << active_euchromatin_forces_params.get<double>("force_sigma") << std::endl;
        std::cout << "  kon: " << active_euchromatin_forces_params.get<double>("kon") << std::endl;
        std::cout << "  koff: " << active_euchromatin_forces_params.get<double>("koff") << std::endl;
      }

      std::cout << std::endl;

      std::cout << "NEIGHBOR LIST:" << std::endl;
      const auto &neighbor_list_params = valid_param_list.sublist("neighbor_list");
      std::cout << "  skin_distance: " << neighbor_list_params.get<double>("skin_distance") << std::endl;
      std::cout << "  force_neighborlist_update: " << neighbor_list_params.get<bool>("force_neighborlist_update")
                << std::endl;
      std::cout << "  force_neighborlist_update_nsteps: "
                << neighbor_list_params.get<size_t>("force_neighborlist_update_nsteps") << std::endl;
      std::cout << "  print_neighborlist_statistics: "
                << neighbor_list_params.get<bool>("print_neighborlist_statistics") << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

 private:
  /// \brief Default parameter filename if none is provided.
  std::string input_parameter_filename_ = "ngp_hp1.yaml";
};  // class HP1ParamParser

void run(int argc, char **argv) {
  // Preprocess
  Teuchos::ParameterList params = HP1ParamParser().parse(argc, argv);
  const auto &sim_params = params.sublist("sim");
  const auto &brownian_motion_params = params.sublist("brownian_motion");
  const auto &backbone_springs_params = params.sublist("backbone_springs");
  const auto &backbone_collision_params = params.sublist("backbone_collision");
  const auto &crosslinker_params = params.sublist("crosslinker");
  const auto &periphery_hydro_params = params.sublist("periphery_hydro");
  const auto &periphery_collision_params = params.sublist("periphery_collision");
  const auto &periphery_binding_params = params.sublist("periphery_binding");
  const auto &active_euchromatin_forces_params = params.sublist("active_euchromatin_forces");
  const auto &neighbor_list_params = params.sublist("neighbor_list");

  // Setup the STK mesh
  mundy::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder.set_spatial_dimension(3).set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = mesh_builder.create_meta_data();
  meta_data_ptr->use_simple_fields();
  meta_data_ptr->set_coordinate_field_name("COORDS");
  mundy::mesh::MetaData &meta_data = *meta_data_ptr;

  // Parts and their subsets
  auto particle_top = stk::topology::PARTICLE;
  auto beam2_top = stk::topology::BEAM_2;
  auto node_top = stk::topology::NODE;
  auto &spheres_part = meta_data.declare_part_with_topology("SPHERES", particle_top);
  auto &e_spheres_part = meta_data.declare_part_with_topology("EUCHROMATIN_SPHERES", particle_top);
  auto &h_spheres_part = meta_data.declare_part_with_topology("HETEROCHROMATIN_SPHERES", particle_top);
  meta_data.declare_part_subset(spheres_part, e_spheres_part);
  meta_data.declare_part_subset(spheres_part, h_spheres_part);
  // stk::io::put_io_part_attribute(spheres_part);  // This is an asstempy part. Do not write to exodus.
  stk::io::put_io_part_attribute(e_spheres_part);
  stk::io::put_io_part_attribute(h_spheres_part);

  auto &hp1_part = meta_data.declare_part_with_topology("HP1", beam2_top);
  auto &left_hp1_part = meta_data.declare_part_with_topology("LEFT_HP1", beam2_top);
  auto &doubly_hp1_h_part = meta_data.declare_part_with_topology("DOUBLY_HP1_H", beam2_top);
  auto &doubly_hp1_bs_part = meta_data.declare_part_with_topology("DOUBLY_HP1_BS", beam2_top);
  meta_data.declare_part_subset(hp1_part, left_hp1_part);
  meta_data.declare_part_subset(hp1_part, doubly_hp1_h_part);
  meta_data.declare_part_subset(hp1_part, doubly_hp1_bs_part);
  // stk::io::put_io_part_attribute(hp1_part);    // This is an asstempy part. Do not write to exodus.
  stk::io::put_io_part_attribute(left_hp1_part);
  stk::io::put_io_part_attribute(doubly_hp1_h_part);
  stk::io::put_io_part_attribute(doubly_hp1_bs_part);

  auto &binding_sites_part = meta_data.declare_part_with_topology("BIND_SITES", node_top);
  stk::io::put_io_part_attribute(binding_sites_part);

  auto &backbone_segs_part = meta_data.declare_part_with_topology("BACKBONE_SEGMENTS", beam2_top);
  auto &ee_segs_part = meta_data.declare_part_with_topology("EE_SEGMENTS", beam2_top);
  auto &eh_segs_part = meta_data.declare_part_with_topology("EH_SEGMENTS", beam2_top);
  auto &hh_segs_part = meta_data.declare_part_with_topology("HH_SEGMENTS", beam2_top);
  meta_data.declare_part_subset(backbone_segs_part, ee_segs_part);
  meta_data.declare_part_subset(backbone_segs_part, eh_segs_part);
  meta_data.declare_part_subset(backbone_segs_part, hh_segs_part);
  // stk::io::put_io_part_attribute(backbone_segs_part);    // This is an asstempy part. Do not write to exodus.
  stk::io::put_io_part_attribute(ee_segs_part);
  stk::io::put_io_part_attribute(eh_segs_part);
  stk::io::put_io_part_attribute(hh_segs_part);

  // Fields
  auto node_rank = stk::topology::NODE_RANK;
  auto element_rank = stk::topology::ELEMENT_RANK;
  auto &node_coords_field = meta_data.declare_field<double>(node_rank, "COORDS");
  auto &node_velocity_field = meta_data.declare_field<double>(node_rank, "VELOCITY");
  auto &node_force_field = meta_data.declare_field<double>(node_rank, "FORCE");
  auto &node_rng_field = meta_data.declare_field<unsigned>(node_rank, "RNG_COUNTER");

  auto &elem_hydrodynamic_radius_field = meta_data.declare_field<double>(element_rank, "HYDRODYNAMIC_RADIUS");
  auto &elem_collision_radius_field = meta_data.declare_field<double>(element_rank, "COLLISION_RADIUS");
  auto &elem_binding_radius_field = meta_data.declare_field<double>(element_rank, "BINDING_RADIUS");

  auto &elem_spring_constant_field = meta_data.declare_field<double>(element_rank, "SPRING_CONSTANT");
  auto &elem_spring_r0_field = meta_data.declare_field<double>(element_rank, "SPRING_R0");

  auto &elem_youngs_modulus_field = meta_data.declare_field<double>(element_rank, "YOUNGS_MODULUS");
  auto &elem_poissons_ratio_field = meta_data.declare_field<double>(element_rank, "POISSONS_RATIO");

  auto &elem_aabb_field = meta_data.declare_field<double>(element_rank, "AABB");
  auto &elem_aabb_displacement_field = meta_data.declare_field<double>(element_rank, "AABB_DISPLACEMENT");

  auto &elem_binding_rates_field = meta_data.declare_field<double>(element_rank, "BINDING_RATES");
  auto &elem_unbinding_rates_field = meta_data.declare_field<double>(element_rank, "UNBINDING_RATES");
  auto &elem_rng_field = meta_data.declare_field<unsigned>(element_rank, "RNG_COUNTER");
  auto &elem_chain_id_field = meta_data.declare_field<unsigned>(element_rank, "CHAIN_ID");

  auto &elem_e_state_field = meta_data.declare_field<unsigned>(element_rank, "EUCHROMATIN_STATE");
  auto &elem_e_state_change_next_time_field =
      meta_data.declare_field<unsigned>(element_rank, "EUCHROMATIN_STATE_CHANGE_NEXT_TIME");
  auto &elem_e_state_time_field = meta_data.declare_field<unsigned>(element_rank, "EUCHROMATIN_STATE_CHANGE_TIME");

  auto transient_role = Ioss::Field::TRANSIENT;
  stk::io::set_field_role(node_velocity_field, transient_role);
  stk::io::set_field_role(node_force_field, transient_role);
  stk::io::set_field_role(node_rng_field, transient_role);
  stk::io::set_field_role(elem_hydrodynamic_radius_field, transient_role);
  stk::io::set_field_role(elem_collision_radius_field, transient_role);
  stk::io::set_field_role(elem_binding_radius_field, transient_role);
  stk::io::set_field_role(elem_spring_constant_field, transient_role);
  stk::io::set_field_role(elem_spring_r0_field, transient_role);
  stk::io::set_field_role(elem_youngs_modulus_field, transient_role);
  stk::io::set_field_role(elem_poissons_ratio_field, transient_role);
  stk::io::set_field_role(elem_aabb_field, transient_role);
  stk::io::set_field_role(elem_aabb_displacement_field, transient_role);
  stk::io::set_field_role(elem_binding_rates_field, transient_role);
  stk::io::set_field_role(elem_unbinding_rates_field, transient_role);
  stk::io::set_field_role(elem_rng_field, transient_role);
  stk::io::set_field_role(elem_chain_id_field, transient_role);
  stk::io::set_field_role(elem_e_state_field, transient_role);
  stk::io::set_field_role(elem_e_state_change_next_time_field, transient_role);
  stk::io::set_field_role(elem_e_state_time_field, transient_role);

  auto scalar_io_type = stk::io::FieldOutputType::SCALAR;
  auto vector_3d_io_type = stk::io::FieldOutputType::VECTOR_3D;
  stk::io::set_field_output_type(node_velocity_field, vector_3d_io_type);
  stk::io::set_field_output_type(node_force_field, vector_3d_io_type);
  stk::io::set_field_output_type(node_rng_field, scalar_io_type);
  stk::io::set_field_output_type(elem_hydrodynamic_radius_field, scalar_io_type);
  stk::io::set_field_output_type(elem_collision_radius_field, scalar_io_type);
  stk::io::set_field_output_type(elem_binding_radius_field, scalar_io_type);
  stk::io::set_field_output_type(elem_spring_constant_field, scalar_io_type);
  stk::io::set_field_output_type(elem_spring_r0_field, scalar_io_type);
  stk::io::set_field_output_type(elem_youngs_modulus_field, scalar_io_type);
  stk::io::set_field_output_type(elem_poissons_ratio_field, scalar_io_type);
  // stk::io::set_field_output_type(elem_aabb_field, ...);
  // stk::io::set_field_output_type(elem_aabb_displacement_field, ...);
  stk::io::set_field_output_type(elem_binding_rates_field, stk::io::FieldOutputType::VECTOR_2D);
  stk::io::set_field_output_type(elem_unbinding_rates_field, stk::io::FieldOutputType::VECTOR_2D);
  stk::io::set_field_output_type(elem_rng_field, scalar_io_type);
  stk::io::set_field_output_type(elem_chain_id_field, scalar_io_type);
  stk::io::set_field_output_type(elem_e_state_field, scalar_io_type);
  stk::io::set_field_output_type(elem_e_state_change_next_time_field, scalar_io_type);
  stk::io::set_field_output_type(elem_e_state_time_field, scalar_io_type);

  // Sew it all together. Start off fields as uninitialized.
  // Give all nodes and elements a random number generator counter.
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_rng_field, meta_data.universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rng_field, meta_data.universal_part(), 1, nullptr);

  // Heterochromatin and euchromatin spheres are used for hydrodynamics. They move and
  // have forces applied to them. If brownian motion is enabled, they will have a
  // stocastic velocity. Heterochromatin spheres are considered for hp1 binding and
  // require an AABB for neighbor detection.
  const stk::mesh::Selector hydro_spheres_part = e_spheres_part | h_spheres_part;
  stk::mesh::put_field_on_mesh(node_velocity_field, hydro_spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, hydro_spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_hydrodynamic_radius_field, hydro_spheres_part, 1, nullptr);

  // Backbone segs apply spring forces and act as spherocylinders for the sake of
  // collision. They apply forces to their nodes and have a collision radius. The
  // difference between ee, eh, and hh segs is that ee segs can exert an active
  // dipole.
  stk::mesh::put_field_on_mesh(node_force_field, backbone_segs_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_collision_radius_field, backbone_segs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_spring_constant_field, backbone_segs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_spring_r0_field, backbone_segs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_youngs_modulus_field, backbone_segs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_poissons_ratio_field, backbone_segs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_aabb_field, backbone_segs_part, 6, nullptr);
  stk::mesh::put_field_on_mesh(elem_aabb_displacement_field, backbone_segs_part, 6, nullptr);

  // HP1 crosslinkers are used for binding/unbinding and apply forces to their nodes.
  stk::mesh::put_field_on_mesh(node_force_field, hp1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_binding_rates_field, hp1_part, 2, nullptr);
  stk::mesh::put_field_on_mesh(elem_unbinding_rates_field, hp1_part, 2, nullptr);
  stk::mesh::put_field_on_mesh(elem_binding_radius_field, hp1_part, 1, nullptr);

  // Bind sites are use for binding/unbinding. They are merely a point in space and have no
  //   inherent field besides node_coords.

  // That's it for the mesh. Commit it's structure and create the bulk data.
  meta_data.commit();
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
  mundy::mesh::BulkData &bulk_data = *bulk_data_ptr;

  // Perform restart (optional)
  bool restart_performed = false;
  if (!restart_performed) {
    /* Declare the chromatin and HP1
    //  E : euchromatin spheres
    //  H : heterochromatin spheres
    //  | : crosslinkers
    // ---: backbone springs/backbone segments
    //
    //  |   |                           |   |
    //  H---H---E---E---E---E---E---E---H---H
    //
    // The actual connectivity looks like this:
    //  n : node, s : segment and or spring, c : crosslinker
    //
    // c1_      c3_       c5_       c7_
    // | /      | /       | /       | /
    // n1       n3        n5        n7
    //  \      /  \      /  \      /
    //   s1   s2   s3   s4   s5   s6
    //    \  /      \  /      \  /
    //     n2        n4        n6
    //     | \       | \       | \
    //     c2⎻       c4⎻       c6⎻
    //
    // If you look at this long enough, the pattern is clear.
    //  - One less segment than nodes.
    //  - Same number of crosslinkers as heterochromatin nodes.
    //  - Segment i connects to nodes i and i+1.
    //  - Crosslinker i connects to nodes i and i.
    //
    // We need to use this information to populate the node and element info vectors.
    // Mundy will handle passing off this information to the bulk data. Just make sure that all
    // MPI ranks contain the same node and element info. This way, we can determine which nodes
    // should become shared.
    //
    // Rules (non-exhaustive):
    //  - Neither nodes nor elements need to have parts or fields.
    //  - The rank and type of the fields must be consistant. You can't pass an element field to a node,
    //    nor can you set the value of a field to a different type or size than it was declared as.
    //  - The owner of a node must be the same as one of the elements that connects to it.
    //  - A node connected to an element not on the same rank as the node will be shared with the owner of the
    element.
    //  - Field/Part names are case-sensitive but don't attempt to declare "field_1" and "Field_1" as if
    //    that will give two different fields since STKIO will not be able to distinguish between them.
    //  - A negative node id in the element connection list can be used to indicate that a node should be left
    unassigned.
    //  - All parts need to be able to contain an element of the given topology.
    */

    // Fill the declare entities helper
    mundy::mesh::DeclareEntitiesHelper dec_helper;
    size_t node_count = 0;
    size_t element_count = 0;

    // Setup the periphery bind sites
    {
      const std::string bind_sites_type = periphery_binding_params.get<std::string>("bind_sites_type");
      if (bind_sites_type == "RANDOM") {
        const size_t num_bind_sites = periphery_binding_params.get<size_t>("num_bind_sites");
        const std::string periphery_shape = periphery_binding_params.get<std::string>("shape");
        openrand::Philox rng(0, 0);
        if (periphery_shape == "SPHERE") {
          const double radius = periphery_binding_params.get<double>("radius");

          for (size_t i = 0; i < num_bind_sites; i++) {
            // Generate a random point on the unit sphere
            const double u1 = rng.rand<double>();
            const double u2 = rng.rand<double>();
            const double theta = 2.0 * M_PI * u1;
            const double phi = std::acos(2.0 * u2 - 1.0);
            double node_coords[3] = {radius * std::sin(phi) * std::cos(theta),  //
                                     radius * std::sin(phi) * std::sin(theta),  //
                                     radius * std::cos(phi)};

            // Declare the node
            dec_helper.create_node()
                .owning_proc(0)                 //
                .id(node_count + 1)             //
                .add_part(&binding_sites_part)  //
                .add_field_data<double>(&node_coords_field, {node_coords[0], node_coords[1], node_coords[2]});
            node_count++;
          }
        } else if (periphery_shape == "ELLIPSOID") {
          const double a = periphery_binding_params.get<double>("axis_radius1");
          const double b = periphery_binding_params.get<double>("axis_radius2");
          const double c = periphery_binding_params.get<double>("axis_radius3");
          const double inv_mu_max = 1.0 / std::max({b * c, a * c, a * b});
          openrand::Philox rng(0, 0);
          auto keep = [&a, &b, &c, &inv_mu_max, &rng](double x, double y, double z) {
            const double mu_xyz =
                std::sqrt((b * c * x) * (b * c * x) + (a * c * y) * (a * c * y) + (a * b * z) * (a * b * z));
            return inv_mu_max * mu_xyz > rng.rand<double>();
          };

          for (size_t i = 0; i < num_bind_sites; i++) {
            // Rejection sampling to place the periphery binding sites
            double node_coords[3];
            while (true) {
              // Generate a random point on the unit sphere
              const double u1 = rng.rand<double>();
              const double u2 = rng.rand<double>();
              const double theta = 2.0 * M_PI * u1;
              const double phi = std::acos(2.0 * u2 - 1.0);
              node_coords[0] = std::sin(phi) * std::cos(theta);
              node_coords[1] = std::sin(phi) * std::sin(theta);
              node_coords[2] = std::cos(phi);

              // Keep this point with probability proportional to the surface area element
              if (keep(node_coords[0], node_coords[1], node_coords[2])) {
                // Pushforward the point to the ellipsoid
                node_coords[0] *= a;
                node_coords[1] *= b;
                node_coords[2] *= c;
                break;
              }
            }

            // Declare the node
            dec_helper.create_node()
                .owning_proc(0)                 //
                .id(node_count + 1)             //
                .add_part(&binding_sites_part)  //
                .add_field_data<double>(&node_coords_field, {node_coords[0], node_coords[1], node_coords[2]});
            node_count++;
          }
        }
      }
    }

    // Setup the chromatin fibers
    {
      const size_t num_chromosomes = sim_params.get<size_t>("num_chromosomes");
      const size_t num_he_repeats = sim_params.get<size_t>("num_hetero_euchromatin_repeats");
      const size_t num_h_per_repeat = sim_params.get<size_t>("num_heterochromatin_per_repeat");
      const size_t num_e_per_repeat = sim_params.get<size_t>("num_euchromatin_per_repeat");
      const size_t num_nodes_per_chromosome = num_he_repeats * (num_h_per_repeat + num_e_per_repeat);
      const double segment_length = sim_params.get<double>("initial_chromosome_separation");

      std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions;
      if (sim_params.get<std::string>("initialization_type") == "GRID") {
        all_chromosome_positions =
            get_chromosome_positions_grid(num_chromosomes, num_nodes_per_chromosome, segment_length);
      } else if (sim_params.get<std::string>("initialization_type") == "RANDOM_UNIT_CELL") {
        auto domain_low = sim_params.get<Teuchos::Array<double>>("domain_low");
        auto domain_high = sim_params.get<Teuchos::Array<double>>("domain_high");
        all_chromosome_positions = get_chromosome_positions_random_unit_cell(
            num_chromosomes, num_nodes_per_chromosome, segment_length, domain_low.getRawPtr(), domain_high.getRawPtr());
      } else if (sim_params.get<std::string>("initialization_type") == "HIILBERT_RANDOM_UNIT_CELL") {
        auto domain_low = sim_params.get<Teuchos::Array<double>>("domain_low");
        auto domain_high = sim_params.get<Teuchos::Array<double>>("domain_high");
        all_chromosome_positions = get_chromosome_positions_hilbert_random_unit_cell(
            num_chromosomes, num_nodes_per_chromosome, segment_length, domain_low.getRawPtr(), domain_high.getRawPtr());
      } else if (sim_params.get<std::string>("initialization_type") == "FROM_DAT") {
        all_chromosome_positions = get_chromosome_positions_from_file(
            sim_params.get<std::string>("initialize_from_dat_filename"), num_chromosomes);
      } else {
        MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Invalid initialization type.");
      }

      for (size_t f = 0; f < num_chromosomes; f++) {
        // Declare the nodes, segments, and heterochromatin/euchromatin
        for (size_t r = 0; r < num_he_repeats; ++r) {
          // Heterochromatin
          for (size_t h = 0; h < num_h_per_repeat; ++h) {
            const size_t node_index = r * (num_h_per_repeat + num_e_per_repeat) + h;

            dec_helper.create_node()
                .owning_proc(0)                                                                            //
                .id(node_count + 1)                                                                        //
                .add_field_data<unsigned>(&node_rng_field, {0})                                            //
                .add_field_data<double>(&node_coords_field, {all_chromosome_positions[f][node_index][0],   //
                                                             all_chromosome_positions[f][node_index][1],   //
                                                             all_chromosome_positions[f][node_index][2]})  //
                .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})                                //
                .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0});
            node_count++;

            // Only create the segment if the node is not the last node in this fiber
            if (node_index < num_nodes_per_chromosome - 1) {
              auto segment = dec_helper.create_element();
              segment
                  .owning_proc(0)                   //
                  .id(element_count + 1)            //
                  .topology(stk::topology::BEAM_2)  //
                  .add_part(&backbone_segs_part)    //
                  .nodes({node_count, node_count + 1})
                  .add_field_data<unsigned>(&elem_rng_field, {0});
              element_count++;

              if (sim_params.get<bool>("enable_backbone_springs")) {
                segment
                    .add_field_data<double>(&elem_spring_constant_field,
                                            {backbone_springs_params.get<double>("spring_constant")})  //
                    .add_field_data<double>(&elem_spring_r0_field, {backbone_springs_params.get<double>("spring_r0")});
              }

              if (sim_params.get<bool>("enable_backbone_collision")) {
                segment
                    .add_field_data<double>(&elem_collision_radius_field,
                                            {backbone_collision_params.get<double>("radius")})
                    .add_field_data<double>(&elem_youngs_modulus_field,
                                            {backbone_collision_params.get<double>("youngs_modulus")})
                    .add_field_data<double>(&elem_poissons_ratio_field,
                                            {backbone_collision_params.get<double>("poissons_ratio")})
                    .add_field_data<double>(&elem_aabb_field, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
                    .add_field_data<double>(&elem_aabb_displacement_field, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
              }

              // Determine if the segment is hh or eh
              const bool left_and_right_node_in_heterochromatin =
                  (h != 0 || r == 0) && (h != num_h_per_repeat - 1 || r == num_he_repeats - 1);
              if (left_and_right_node_in_heterochromatin) {
                segment.add_part(&hh_segs_part);
              } else {
                segment.add_part(&eh_segs_part);
              }

            }

            // Declare the heterochromatin sphere
            dec_helper.create_element()
                .owning_proc(0)                     //
                .id(element_count + 1)              //
                .topology(stk::topology::PARTICLE)  //
                .add_part(&h_spheres_part)          //
                .nodes({node_count})
                .add_field_data<double>(&elem_hydrodynamic_radius_field,
                                        {sim_params.get<double>("backbone_sphere_hydrodynamic_radius")});
            element_count++;
          }

          for (size_t e = 0; e < num_e_per_repeat; ++e) {
            const size_t node_index = r * (num_h_per_repeat + num_e_per_repeat) + num_h_per_repeat + e;
            dec_helper.create_node()
                .owning_proc(0)                                                                            //
                .id(node_count + 1)                                                                        //
                .add_field_data<unsigned>(&node_rng_field, {0})                                            //
                .add_field_data<double>(&node_coords_field, {all_chromosome_positions[f][node_index][0],   //
                                                             all_chromosome_positions[f][node_index][1],   //
                                                             all_chromosome_positions[f][node_index][2]})  //
                .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})                                //
                .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0});
            node_count++;

            // Only create the segment if the node is not the last node in this fiber
            if (node_index < num_nodes_per_chromosome - 1) {
              auto segment = dec_helper.create_element();
              segment
                  .owning_proc(0)                       //
                  .id(element_count + 1)                //
                  .topology(stk::topology::BEAM_2)      //
                  .add_part(&backbone_segs_part)        //
                  .nodes({node_count, node_count + 1})  //
                  .add_field_data<unsigned>(&elem_rng_field, {0});
              element_count++;

              if (sim_params.get<bool>("enable_backbone_springs")) {
                segment
                    .add_field_data<double>(&elem_spring_constant_field,
                                            {backbone_springs_params.get<double>("spring_constant")})  //
                    .add_field_data<double>(&elem_spring_r0_field, {backbone_springs_params.get<double>("spring_r0")});
              }

              if (sim_params.get<bool>("enable_backbone_collision")) {
                segment
                    .add_field_data<double>(&elem_collision_radius_field,
                                            {backbone_collision_params.get<double>("radius")})
                    .add_field_data<double>(&elem_youngs_modulus_field,
                                            {backbone_collision_params.get<double>("youngs_modulus")})
                    .add_field_data<double>(&elem_poissons_ratio_field,
                                            {backbone_collision_params.get<double>("poissons_ratio")})
                    .add_field_data<double>(&elem_aabb_field, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
                    .add_field_data<double>(&elem_aabb_displacement_field, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
              }

              // Determine if the segement is ee or eh
              const bool left_and_right_node_in_euchromatin =
                  (e != 0 || r == 0) && (e != num_e_per_repeat - 1 || r == num_he_repeats - 1);
              if (left_and_right_node_in_euchromatin) {
                segment.add_part(&ee_segs_part);
              } else {
                segment.add_part(&eh_segs_part);
              }
            }

            // Declare the euchromatin sphere
            dec_helper.create_element()
                .owning_proc(0)                     //
                .id(element_count + 1)              //
                .topology(stk::topology::PARTICLE)  //
                .add_part(&e_spheres_part)          //
                .nodes({node_count})                //
                .add_field_data<double>(&elem_hydrodynamic_radius_field,
                                        {sim_params.get<double>("backbone_sphere_hydrodynamic_radius")});
            element_count++;
          }
        }
      }
    }

    dec_helper.check_consistency(bulk_data);

    // Declare the entities
    bulk_data.modification_begin();
    dec_helper.declare_entities(bulk_data);
    bulk_data.modification_end();

    // Write the mesh to file
    size_t step = 1;  // Step = 0 doesn't write out fields...
    stk::io::write_mesh_with_fields("ngp_hp1.exo", bulk_data, step);
  }

  // // Post-setup but pre-run
  // if (sim_params.get<bool>("loadbalance_post_initialization")) {
  //   stk::balance::balanceStkMesh(RcbSettings{}, bulk_data);
  // }

  // // Get the NGP stuff
  // stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  // auto &ngp_node_coords_field = stk::mesh::get_updated_ngp_field<double>(node_coords_field);
  // auto &ngp_node_velocity_field = stk::mesh::get_updated_ngp_field<double>(node_velocity_field);
  // auto &ngp_node_force_field = stk::mesh::get_updated_ngp_field<double>(node_force_field);
  // auto &ngp_node_rng_field = stk::mesh::get_updated_ngp_field<unsigned>(node_rng_field);
  // auto &ngp_elem_hydrodynamic_radius_field =
  // stk::mesh::get_updated_ngp_field<double>(elem_hydrodynamic_radius_field); auto &ngp_elem_binding_radius_field =
  // stk::mesh::get_updated_ngp_field<double>(elem_binding_radius_field); auto &ngp_elem_collision_radius_field =
  // stk::mesh::get_updated_ngp_field<double>(elem_collision_radius_field); auto &ngp_elem_hookean_spring_constant_field
  // =
  //     stk::mesh::get_updated_ngp_field<double>(elem_hookean_spring_constant_field);
  // auto &ngp_elem_hookean_spring_rest_length_field =
  //     stk::mesh::get_updated_ngp_field<double>(elem_hookean_spring_rest_length_field);
  // auto &ngp_elem_youngs_modulus_field = stk::mesh::get_updated_ngp_field<double>(elem_youngs_modulus_field);
  // auto &ngp_elem_poissons_ratio_field = stk::mesh::get_updated_ngp_field<double>(elem_poissons_ratio_field);
  // auto &ngp_elem_aabb_field = stk::mesh::get_updated_ngp_field<double>(elem_aabb_field);
  // auto &ngp_elem_aabb_displacement_field = stk::mesh::get_updated_ngp_field<double>(elem_aabb_displacement_field);
  // auto &ngp_elem_binding_rates_field = stk::mesh::get_updated_ngp_field<double>(elem_binding_rates_field);
  // auto &ngp_elem_unbinding_rates_field = stk::mesh::get_updated_ngp_field<double>(elem_unbinding_rates_field);
  // auto &ngp_elem_perform_state_change_field =
  //     stk::mesh::get_updated_ngp_field<unsigned>(elem_perform_state_change_field);
  // auto &ngp_elem_rng_field = stk::mesh::get_updated_ngp_field<unsigned>(elem_rng_field);

  // // Time loop
  // print_rank0(std::string("Running the simulation for ") + std::to_string(sim_params.get<size_t>("num_time_steps")) +
  //             " timesteps.");

  // unsigned num_surface_nodes = 0;
  // DoubleMatDeviceView inv_self_interaction_matrix(
  //     Kokkos::view_alloc(Kokkos::WithoutInitializing, "inv_self_interaction_matrix"), 0, 0);
  // DoubleVecDeviceView surface_positions(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_positions"), 0);
  // DoubleVecDeviceView surface_normals(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_normals"), 0);
  // DoubleVecDeviceView surface_weights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_weights"), 0);
  // DoubleVecDeviceView surface_radii(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_radii"), 0);
  // DoubleVecDeviceView surface_velocities(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_velocities"), 0);
  // DoubleVecDeviceView surface_forces(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_forces"), 0);
  // DoubleVecDeviceView::HostMirror surface_positions_host = Kokkos::create_mirror_view(surface_positions);
  // DoubleVecDeviceView::HostMirror surface_normals_host = Kokkos::create_mirror_view(surface_normals);
  // DoubleVecDeviceView::HostMirror surface_weights_host = Kokkos::create_mirror_view(surface_weights);
  // DoubleVecDeviceView::HostMirror surface_radii_host = Kokkos::create_mirror_view(surface_radii);
  // DoubleVecDeviceView::HostMirror surface_velocities_host = Kokkos::create_mirror_view(surface_velocities);
  // DoubleVecDeviceView::HostMirror surface_forces_host = Kokkos::create_mirror_view(surface_forces);

  // if (sim_params.get<bool>("enable_periphery_hydro")) {
  //   // Initialize the periphery points, weights, normals, and radii
  //   if ((periphery_hydro_quadrature_ == PERIPHERY_QUADRATURE::GAUSS_LEGENDRE) &&
  //       ((periphery_hydro_shape_ == PERIPHERY_SHAPE::SPHERE) ||
  //        ((periphery_hydro_shape_ == PERIPHERY_SHAPE::ELLIPSOID) &&
  //         (periphery_hydro_axis_radius1_ == periphery_hydro_axis_radius2_) &&
  //         (periphery_hydro_axis_radius2_ == periphery_hydro_axis_radius3_) &&
  //         (periphery_hydro_axis_radius3_ == periphery_hydro_axis_radius1_)))) {
  //     // Generate the quadrature points and weights for the sphere using GL quadrature
  //     std::vector<double> points_vec;
  //     std::vector<double> weights_vec;
  //     std::vector<double> normals_vec;
  //     const bool invert = true;
  //     const bool include_poles = false;
  //     const size_t spectral_order = periphery_hydro_spectral_order_;
  //     const double radius =
  //         (periphery_hydro_shape_ == PERIPHERY_SHAPE::SPHERE) ? periphery_hydro_radius_ :
  //         periphery_hydro_axis_radius1_;
  //     mundy::alens::periphery::gen_sphere_quadrature(spectral_order, radius, &points_vec, &weights_vec, &normals_vec,
  //                                                    include_poles, invert);

  //     // Allocate the views. Note, resizing does not automatically update the mirror views.
  //     num_surface_nodes = weights_vec.size();
  //     Kokkos::resize(surface_positions, 3 * num_surface_nodes);
  //     Kokkos::resize(surface_normals, 3 * num_surface_nodes);
  //     Kokkos::resize(surface_weights, num_surface_nodes);
  //     Kokkos::resize(surface_radii, num_surface_nodes);
  //     Kokkos::resize(surface_velocities, 3 * num_surface_nodes);
  //     Kokkos::resize(surface_forces, 3 * num_surface_nodes);
  //     surface_positions_host = Kokkos::create_mirror_view(surface_positions);
  //     surface_normals_host = Kokkos::create_mirror_view(surface_normals);
  //     surface_weights_host = Kokkos::create_mirror_view(surface_weights);
  //     surface_radii_host = Kokkos::create_mirror_view(surface_radii);
  //     surface_velocities_host = Kokkos::create_mirror_view(surface_velocities);
  //     surface_forces_host = Kokkos::create_mirror_view(surface_forces);

  //     // Copy the raw data into the views
  //     for (unsigned i = 0; i < num_surface_nodes; i++) {
  //       surface_positions_host(3 * i + 0) = points_vec[3 * i + 0];
  //       surface_positions_host(3 * i + 1) = points_vec[3 * i + 1];
  //       surface_positions_host(3 * i + 2) = points_vec[3 * i + 2];
  //       surface_normals_host(3 * i + 0) = normals_vec[3 * i + 0];
  //       surface_normals_host(3 * i + 1) = normals_vec[3 * i + 1];
  //       surface_normals_host(3 * i + 2) = normals_vec[3 * i + 2];
  //       surface_velocities_host(3 * i + 0) = 0.0;
  //       surface_velocities_host(3 * i + 1) = 0.0;
  //       surface_velocities_host(3 * i + 2) = 0.0;
  //       surface_forces_host(3 * i + 0) = 0.0;
  //       surface_forces_host(3 * i + 1) = 0.0;
  //       surface_forces_host(3 * i + 2) = 0.0;
  //       surface_weights_host(i) = weights_vec[i];
  //       surface_radii_host(i) = 0.0;
  //     }

  //     // Copy the views to the device
  //     Kokkos::deep_copy(surface_positions, surface_positions_host);
  //     Kokkos::deep_copy(surface_normals, surface_normals_host);
  //     Kokkos::deep_copy(surface_weights, surface_weights_host);
  //     Kokkos::deep_copy(surface_radii, surface_radii_host);
  //     Kokkos::deep_copy(surface_velocities, surface_velocities_host);
  //     Kokkos::deep_copy(surface_forces, surface_forces_host);
  //   } else if (periphery_hydro_quadrature_ == PERIPHERY_QUADRATURE::FROM_FILE) {
  //     read_vector_from_file(quadrature_weights_filename, num_surface_nodes, surface_weights_host);
  //     read_vector_from_file(quadrature_points_filename, 3 * num_surface_nodes, surface_positions_host);
  //     read_vector_from_file(quadrature_normals_filename, 3 * num_surface_nodes, surface_normals_host);
  //     Kokkos::deep_copy(surface_positions, surface_positions_host);
  //     Kokkos::deep_copy(surface_normals, surface_normals_host);
  //     Kokkos::deep_copy(surface_weights, surface_weights_host);

  //     // Zero out the radii, forces, and velocities (on host and device)
  //     Kokkos::deep_copy(surface_radii_host, 0.0);
  //     Kokkos::deep_copy(surface_velocities_host, 0.0);
  //     Kokkos::deep_copy(surface_forces_host, 0.0);
  //     Kokkos::deep_copy(surface_radii, 0.0);
  //     Kokkos::deep_copy(surface_velocities, 0.0);
  //     Kokkos::deep_copy(surface_forces, 0.0);
  //   } else {
  //     MUNDY_THROW_REQUIRE(false, std::invalid_argument,
  //                         "We currently only support GAUSS_LEGENDRE quadrature for "
  //                         "spheres and ellipsoids with equal radii or direct specification of the quadrature from a "
  //                         "file using FROM_FILE.");
  //   }

  //   // Run the precomputation for the inverse self-interaction matrix
  //   const bool write_to_file = true;
  //   const bool use_values_from_file_if_present = true;
  //   const std::string inverse_self_interaction_matrix_filename = "inverse_self_interaction_matrix.dat";
  //   Kokkos::resize(inv_self_interaction_matrix, 3 * num_surface_nodes, 3 * num_surface_nodes);

  //   bool matrix_read_from_file = false;
  //   if (use_values_from_file_if_present) {
  //     auto does_file_exist = [](const std::string &filename) {
  //       std::ifstream f(inverse_self_interaction_matrix_filename.c_str());
  //       return f.good();
  //     };

  //     if (does_file_exist(inverse_self_interaction_matrix_filename)) {
  //       read_matrix_from_file(inverse_self_interaction_matrix_filename, inv_self_interaction_matrix);
  //       matrix_read_from_file = true;
  //     }
  //   }

  //   if (!matrix_read_from_file) {
  //     DoubleMatDeviceView self_interaction_matrix("self_interaction_matrix", 3 * num_surface_nodes,
  //                                                 3 * num_surface_nodes);
  //     fill_skfie_matrix(stk::ngp::ExecSpace(), viscosity, num_surface_nodes, num_surface_nodes, surface_positions,
  //                       surface_positions, surface_normals, surface_weights, self_interaction_matrix);
  //     invert_matrix(stk::ngp::ExecSpace(), self_interaction_matrix, inv_self_interaction_matrix);

  //     if (write_to_file) {
  //       write_matrix_to_file(inverse_self_interaction_matrix_filename, inv_self_interaction_matrix);
  //     }
  //   }
  // }

  // bool rebuild_neighbors = true;
  // Kokkos::Timer overall_timer;
  // Kokkos::Timer timer;
  // for (size_t timestep_idx = 0; timestep_idx < sim_params.get<size_t>("num_time_steps"); timestep_idx++) {
  //   // Prepare the current configuration.
  //   ngp_node_velocity_field.sync_to_device();
  //   ngp_node_force_field.sync_to_device();
  //   ngp_elem_binding_rates_field.sync_to_device();
  //   ngp_elem_unbinding_rates_field.sync_to_device();
  //   ngp_elem_perform_state_change_field.sync_to_device();
  //   ngp_elem_constraint_perform_state_change_field.sync_to_device();
  //   ngp_elem_constraint_state_change_rate_field.sync_to_device();
  //   ngp_elem_constraint_potential_force_field.sync_to_device();

  //   ngp_node_velocity_field.set_all(ngp_mesh, 0.0);
  //   ngp_node_force_field.set_all(ngp_mesh, 0.0);
  //   ngp_elem_binding_rates_field.set_all(ngp_mesh, 0.0);
  //   ngp_elem_unbinding_rates_field.set_all(ngp_mesh, 0.0);
  //   ngp_elem_perform_state_change_field.set_all(ngp_mesh, 0u);
  //   ngp_elem_constraint_perform_state_change_field.set_all(ngp_mesh, 0u);
  //   ngp_elem_constraint_state_change_rate_field.set_all(ngp_mesh, 0.0);
  //   ngp_elem_constraint_potential_force_field.set_all(ngp_mesh, 0.0);

  //   ngp_node_velocity_field.modify_on_device();
  //   ngp_node_force_field.modify_on_device();
  //   ngp_elem_binding_rates_field.modify_on_device();
  //   ngp_elem_unbinding_rates_field.modify_on_device();
  //   ngp_elem_perform_state_change_field.modify_on_device();
  //   ngp_elem_constraint_perform_state_change_field.modify_on_device();
  //   ngp_elem_constraint_state_change_rate_field.modify_on_device();
  //   ngp_elem_constraint_potential_force_field.modify_on_device();

  //   rotate_field_states();  // TODO(palmerb4): Add "old" fields where necessary

  //   //////////////////////
  //   // Detect neighbors //
  //   //////////////////////
  //   mundy::geom::compute_aabb_spheres(ngp_mesh, neighbor_list_params.get<double>("skin_distance"),
  //                                     ngp_node_coords_field, ngp_elem_radius_field, ngp_elem_aabb_field,
  //                                     backbone_segs_part);
  //   mundy::geom::accumulate_aabb_displacements(ngp_mesh, ngp_old_elem_aabb_field, ngp_elem_aabb_field,
  //                                              ngp_elem_aabb_displacement_field, backbone_segs_part);
  //   double max_aabb_displacement =
  //       stk::mesh::get_field_max(ngp_mesh, ngp_elem_aabb_displacement_field, max_aabb_displacement,
  //       backbone_segs_part);
  //   if (max_aabb_displacement > neighbor_list_params.get<double>("skin_distance")) {
  //     rebuild_neighbors = true;
  //   }

  //   if (rebuild_neighbors) {
  //     print_rank0("Rebuilding neighbors.");
  //     search_aabbs = create_search_aabbs(ngp_mesh, ngp_elem_aabb_field, backbone_segs_part);

  //     stk::search::SearchMethod search_method = stk::search::MORTON_LBVH;

  //     // WARNING: auto_swap_domain_and_range must be true to avoid double counting forces.
  //     const bool results_parallel_symmetry = true;   // create source -> target and target -> source pairs
  //     const bool auto_swap_domain_and_range = true;  // swap source and target if target is owned and source is not
  //     stk::search::coarse_search(search_aabbs, search_aabbs, search_method, bulk_data.parallel(), search_results,
  //                                DeviceExecutionSpace{}, results_parallel_symmetry, auto_swap_domain_and_range);
  //     num_neighbor_pairs = search_results.extent(0);
  //     std::cout << "Search time: " << search_timer.seconds() << " with " << num_neighbor_pairs << " results."
  //               << std::endl;

  //     // Ghost the non-owned spheres
  //     Kokkos::Timer ghost_timer;
  //     ghost_neighbors(bulk_data, search_results);
  //     std::cout << "Ghost time: " << ghost_timer.seconds() << std::endl;

  //     // Create local neighbor indices
  //     Kokkos::Timer local_index_conversion_timer;
  //     local_search_results = get_local_neighbor_indices(bulk_data, stk::topology::ELEMENT_RANK, search_results);
  //     std::cout << "Local index conversion time: " << local_index_conversion_timer.seconds() << std::endl;

  //     // Only resize the collision views if the number of neighbor pairs has changed
  //     // Otherwise we can reuse the previous lagrange multipliers as the initial guess
  //     signed_sep_dist = Kokkos::View<double *, DeviceMemorySpace>("signed_sep_dist", num_neighbor_pairs);
  //     con_normal_ij = Kokkos::View<double **, DeviceMemorySpace>("con_normal_ij", num_neighbor_pairs, 3);
  //     lagrange_multipliers = Kokkos::View<double *, DeviceMemorySpace>("lagrange_multipliers", num_neighbor_pairs);
  //     Kokkos::deep_copy(lagrange_multipliers, 0.0);  // initial guess

  //     // Reset the accumulated displacements and the rebuild flag
  //     ngp_elem_aabb_displacement_field.sync_to_device();
  //     ngp_elem_aabb_displacement_field.set_all(ngp_mesh, 0.0);
  //     ngp_elem_aabb_displacement_field.modify_on_device();
  //     rebuild_neighbors = false;
  //   }

  //   /////////
  //   // KMC //
  //   /////////
  //   if (sim_params.get<bool>("enable_crosslinkers")) {
  //     kmc_compute_state_change_rate_attach_left_bound_attach_to_node(
  //         binding_kt, dynamic_springs, dynamic_springs_to_node_linkers, c_state_change_rate_field);
  //     kmc_decide_state_change_left_bound_attach_to_node(timestep_size, dynamic_springs,
  //     dynamic_springs_to_node_linkers,
  //                                                       c_state_change_rate_field, c_perform_state_change_field);
  //     kmc_decide_state_change_detach_doubly_bound_from_node(timestep_size, dynamic_springs);
  //     kmc_perform_state_change(dynamic_springs, dynamic_springs_to_node_linkers, c_perform_state_change_field);
  //   }

  //   /////////////////////////////
  //   // Evaluate forces f(x(t)) //
  //   /////////////////////////////
  //   if (enable_backbone_collision_) {
  //     Kokkos::Profiling::pushRegion("HP1::compute_hertzian_contact_forces");

  //     // Potential evaluation (Hertzian contact)
  //     auto backbone_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
  //     auto backbone_backbone_neighbor_genx_selector =
  //     stk::mesh::Selector(*backbone_backbone_neighbor_genx_part_ptr_);

  //     compute_ssd_and_cn_ptr_->execute(backbone_backbone_neighbor_genx_selector);
  //     evaluate_linker_potentials_ptr_->execute(backbone_backbone_neighbor_genx_selector);
  //     linker_potential_force_reduction_ptr_->execute(backbone_selector);

  //     Kokkos::Profiling::popRegion();
  //   }
  //   if (enable_backbone_springs_) {
  //     compute_hookean_spring_forces();
  //     compute_fene_spring_forces();
  //   }
  //   if (enable_crosslinkers_) {
  //     compute_hookean_spring_forces();
  //     compute_fene_spring_forces();
  //   }
  //   if (enable_periphery_collision_) {
  //     Kokkos::Profiling::pushRegion("HP1::compute_periphery_collision_forces");
  //     if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
  //       compute_spherical_periphery_collision_forces();
  //     } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
  //       if (periphery_collision_use_fast_approx_) {
  //         compute_ellipsoidal_periphery_collision_forces_fast_approximate();
  //       } else {
  //         compute_ellipsoidal_periphery_collision_forces();
  //       }
  //     } else {
  //       MUNDY_THROW_REQUIRE(false, std::logic_error, "Invalid periphery type.");
  //     }
  //     Kokkos::Profiling::popRegion();
  //   }

  //   ////////////////////////
  //   // Compute velocities //
  //   ////////////////////////
  //   if (enable_chromatin_brownian_motion_) compute_brownian_velocity();
  //   if (enable_backbone_n_body_hydrodynamics_) {
  //     // Before performing the hydro call, check if the spheres are within the periphery (optional)
  //     if (check_maximum_periphery_overlap_) {
  //       check_maximum_overlap_with_hydro_periphery();
  //     }
  //     compute_rpy_hydro();
  //     compute_confined_rpy_hydro();

  //     // Copy the sphere forces and velocities back to STK fields
  //     Kokkos::parallel_for(
  //         stk::ngp::DeviceRangePolicy(0, num_spheres), KOKKOS_LAMBDA(const int &vector_index) {
  //           stk::mesh::Entity sphere = ngp_sphere_entities.device_get(vector_index);
  //           auto sphere_index = ngp_mesh.fast_mesh_index(sphere);

  //           node_force_field(sphere_index, 0) = sphere_forces(vector_index * 3 + 0);
  //           node_force_field(sphere_index, 1) = sphere_forces(vector_index * 3 + 1);
  //           node_force_field(sphere_index, 2) = sphere_forces(vector_index * 3 + 2);

  //           node_velocity_field(sphere_index, 0) = sphere_velocities(vector_index * 3 + 0);
  //           node_velocity_field(sphere_index, 1) = sphere_velocities(vector_index * 3 + 1);
  //           node_velocity_field(sphere_index, 2) = sphere_velocities(vector_index * 3 + 2);
  //         });

  //   } else {
  //     compute_dry_velocity();
  //   }
  // }
}
//@}

}  // namespace chromalens

}  // namespace mundy

///////////////////////////
// Main program          //
///////////////////////////
int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Run the simulation!
  mundy::chromalens::run(argc, argv);

  // Before exiting, sleep for some amount of time to force Kokkos to print better at the end.
  std::this_thread::sleep_for(std::chrono::milliseconds(stk::parallel_machine_rank(MPI_COMM_WORLD)));

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
