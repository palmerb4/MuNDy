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

#ifndef MUNDY_GEOM_AGGREGATES_ENTITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_ENTITYVIEW_HPP_

// C++ core
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh
#include <stk_topology/topology.hpp>      // for stk::topology::topology_t

// Mundy mesh
#include <mundy_geom/primitives/Ellipsoid.hpp>  // for mundy::geom::ValidEllipsoidType
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// \brief A view of an entity with a specific topology
///
/// Use this class access the entity and its connections without repeatedly fetching the data.
/// Think of this class as the first of a long chain of augmented entity views. Rather than
/// having every entity view store the entity and its connections, we can store the entity and
/// its connections in this class and provide an interface to access them.
template <typename OurTopology,
          typename OurRank = std::integral_constant<stk::topology::rank_t,
                                                    stk::topology_detail::topology_data<OurTopology::value>::rank>>
class EntityView {
 public:
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;
  using topology_data_t = stk::topology_detail::topology_data<topology_t>;
  static constexpr stk::topology::rank_t rank = topology_data_t::rank;

  EntityView(const stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity)
    requires(rank == stk::topology::NODE_RANK)
      : bulk_data_(bulk_data), entity_(entity) {
    MUNDY_THROW_ASSERT(bulk_data.is_valid(entity), std::invalid_argument, "The given entity is not valid");
    MUNDY_THROW_ASSERT(bulk_data.entity_rank(entity) == rank, std::invalid_argument,
                       "The entity rank must have the same rank as the entity view");
  }

  EntityView(const stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity)
    requires(rank == stk::topology::EDGE_RANK)
      : bulk_data_(bulk_data), entity_(entity), connected_nodes_(bulk_data_.begin_nodes(entity)) {
  }

  EntityView(const stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity)
    requires(rank == stk::topology::FACE_RANK && topology_t != stk::topology::INVALID_TOPOLOGY &&
             topology_data_t::num_edges == 0)
      : bulk_data_(bulk_data), entity_(entity), connected_nodes_(bulk_data_.begin_nodes(entity)) {
  }

  EntityView(const stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity)
    requires(rank == stk::topology::FACE_RANK && topology_t != stk::topology::INVALID_TOPOLOGY &&
             topology_data_t::num_edges != 0)
      : bulk_data_(bulk_data),
        entity_(entity),
        connected_nodes_(bulk_data_.begin_nodes(entity)),
        connected_edges_(bulk_data_.begin_edges(entity)) {
  }

  EntityView(const stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity)
    requires(rank == stk::topology::ELEM_RANK && topology_t != stk::topology::INVALID_TOPOLOGY &&
             topology_data_t::num_edges == 0 && topology_data_t::num_faces == 0)
      : bulk_data_(bulk_data), entity_(entity), connected_nodes_(bulk_data_.begin_nodes(entity)) {
  }

  EntityView(const stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity)
    requires(rank == stk::topology::ELEM_RANK && topology_t != stk::topology::INVALID_TOPOLOGY &&
             topology_data_t::num_edges != 0 && topology_data_t::num_faces == 0)
      : bulk_data_(bulk_data),
        entity_(entity),
        connected_nodes_(bulk_data_.begin_nodes(entity)),
        connected_edges_(bulk_data_.begin_edges(entity)) {
  }

  EntityView(const stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity)
    requires(rank == stk::topology::ELEM_RANK && topology_t != stk::topology::INVALID_TOPOLOGY &&
             topology_data_t::num_edges != 0 && topology_data_t::num_faces != 0)
      : bulk_data_(bulk_data),
        entity_(entity),
        connected_nodes_(bulk_data_.begin_nodes(entity)),
        connected_edges_(bulk_data_.begin_edges(entity)),
        connected_faces_(bulk_data_.begin_faces(entity)) {
  }

  static constexpr stk::topology::topology_t get_topology() {
    return topology_t;
  }

  static constexpr stk::topology::rank_t get_rank() {
    return rank;
  }

  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }

  stk::mesh::Entity entity() const {
    return entity_;
  }

  //! \name Connected entities
  //@{

  const stk::mesh::Entity* connected_nodes() const {
    return connected_nodes_;
  }

  const stk::mesh::Entity* connected_edges() const
    requires(topology_data_t::num_edges != 0)
  {
    return connected_edges_;
  }

  const stk::mesh::Entity* connected_faces() const
    requires(topology_data_t::num_faces != 0)
  {
    return connected_faces_;
  }
  //@}

  //! \name Connected entity access
  //@{

  stk::mesh::Entity connected_node(int i) const {
    return connected_nodes_[i];
  }

  stk::mesh::Entity connected_edge(int i) const
    requires(topology_data_t::num_edges != 0)
  {
    return connected_edges_[i];
  }

  stk::mesh::Entity connected_face(int i) const
    requires(topology_data_t::num_faces != 0)
  {
    return connected_faces_[i];
  }
  //@}

  /// \brief Chainable function to add augments to this entity view
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto augment_view(Args&&... args) const {
    return NextAugment<EntityView, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  const stk::mesh::BulkData& bulk_data_;
  const stk::mesh::Entity entity_;
  const stk::mesh::Entity* connected_nodes_;
  const stk::mesh::Entity* connected_edges_;
  const stk::mesh::Entity* connected_faces_;
};  // EntityView

/// @brief An ngp-compatible view of an STK entity
template <typename OurTopology,
          typename OurRank = std::integral_constant<stk::topology::rank_t,
                                                    stk::topology_detail::topology_data<OurTopology::value>::rank>>
class NgpEntityView {
 public:
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;
  using topology_data_t = stk::topology_detail::topology_data<topology_t>;
  static constexpr stk::topology::rank_t rank = topology_data_t::rank;

  static constexpr bool can_have_connected_nodes = rank != stk::topology::NODE_RANK;
  static constexpr bool can_have_connected_edges =
      (topology_t == stk::topology::INVALID_TOPOLOGY) || (topology_data_t::num_edges != 0);
  static constexpr bool can_have_connected_faces =
      (topology_t == stk::topology::INVALID_TOPOLOGY) || (topology_data_t::num_faces != 0);

  KOKKOS_INLINE_FUNCTION
  NgpEntityView(const stk::mesh::NgpMesh& ngp_mesh, stk::mesh::FastMeshIndex entity_index)
    requires(!can_have_connected_nodes && !can_have_connected_edges && !can_have_connected_faces)
      : ngp_mesh_(ngp_mesh), entity_index_(entity_index) {
  }

  KOKKOS_INLINE_FUNCTION
  NgpEntityView(const stk::mesh::NgpMesh& ngp_mesh, stk::mesh::FastMeshIndex entity_index)
    requires(can_have_connected_nodes && !can_have_connected_edges && !can_have_connected_faces)
      : ngp_mesh_(ngp_mesh), entity_index_(entity_index), connected_nodes_(ngp_mesh_.get_nodes(rank, entity_index_)) {
  }

  KOKKOS_INLINE_FUNCTION
  NgpEntityView(const stk::mesh::NgpMesh& ngp_mesh, stk::mesh::FastMeshIndex entity_index)
    requires(can_have_connected_nodes && can_have_connected_edges && !can_have_connected_faces)
      : ngp_mesh_(ngp_mesh),
        entity_index_(entity_index),
        connected_nodes_(ngp_mesh_.get_nodes(rank, entity_index_)),
        connected_edges_(ngp_mesh_.get_edges(rank, entity_index_)) {
  }

  KOKKOS_INLINE_FUNCTION
  NgpEntityView(const stk::mesh::NgpMesh& ngp_mesh, stk::mesh::FastMeshIndex entity_index)
    requires(can_have_connected_nodes && can_have_connected_edges && can_have_connected_faces)
      : ngp_mesh_(ngp_mesh),
        entity_index_(entity_index),
        connected_nodes_(ngp_mesh_.get_nodes(rank, entity_index_)),
        connected_edges_(ngp_mesh_.get_edges(rank, entity_index_)),
        connected_faces_(ngp_mesh_.get_faces(rank, entity_index_)) {
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t get_topology() {
    return topology_t;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::rank_t get_rank() {
    return rank;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh& ngp_mesh() const {
    return ngp_mesh_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex entity_index() const {
    return entity_index_;
  }

  //! \name Connected entities
  //@{

  KOKKOS_INLINE_FUNCTION
  stk::mesh::ConnectedNodes connected_nodes() const {
    return connected_nodes_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::ConnectedEntities connected_edges() const
    requires(topology_data_t::num_edges != 0)
  {
    return connected_edges_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::ConnectedEntities connected_faces() const
    requires(topology_data_t::num_faces != 0)
  {
    return connected_faces_;
  }
  //@}

  //! \name Connected entity access
  //@{

  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity connected_node(unsigned i) const {
    return connected_nodes_[i];
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity connected_edge(unsigned i) const
    requires(topology_data_t::num_edges != 0)
  {
    return connected_edges_[i];
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity connected_face(unsigned i) const
    requires(topology_data_t::num_faces != 0)
  {
    return connected_faces_[i];
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex connected_node_index(unsigned i) const {
    return ngp_mesh_.fast_mesh_index(connected_node(i));
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex connected_edge_index(unsigned i) const
    requires(topology_data_t::num_edges != 0)
  {
    return ngp_mesh_.fast_mesh_index(connected_edge(i));
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex connected_face_index(unsigned i) const
    requires(topology_data_t::num_faces != 0)
  {
    return ngp_mesh_.fast_mesh_index(connected_face(i));
  }
  //@}

  /// \brief Chainable function to add augments to this entity view
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION auto augment_view(Args&&... args) const {
    return NextAugment<NgpEntityView, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  const stk::mesh::NgpMesh& ngp_mesh_;
  const stk::mesh::FastMeshIndex entity_index_;
  const stk::mesh::ConnectedNodes connected_nodes_;
  const stk::mesh::ConnectedEntities connected_edges_;
  const stk::mesh::ConnectedEntities connected_faces_;
};  // NgpEntityView

/// \brief A helper function to create a ranked entity view (no topology)
template <stk::topology::rank_t OurRank>
auto create_ranked_entity_view(const stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity) {
  return EntityView<std::integral_constant<stk::topology::topology_t, stk::topology::INVALID_TOPOLOGY>,
                    std::integral_constant<stk::topology::rank_t, OurRank>>(bulk_data, entity);
}

/// \brief A helper function to create an entity view for an object with a specific topology
template <stk::topology::topology_t OurTopology>
auto create_topological_entity_view(const stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity) {
  return EntityView<std::integral_constant<stk::topology::topology_t, OurTopology>>(bulk_data, entity);
}

/// \brief A helper function to create an NGP_compatible ranked entity view (no topology) object
template <stk::topology::rank_t OurRank>
auto create_ngp_ranked_entity_view(const stk::mesh::NgpMesh& ngp_mesh, stk::mesh::FastMeshIndex entity_index) {
  return NgpEntityView<std::integral_constant<stk::topology::topology_t, stk::topology::INVALID_TOPOLOGY>,
                       std::integral_constant<stk::topology::rank_t, OurRank>>(ngp_mesh, entity_index);
}

/// \brief A helper function to create an NGP_compatible entity view for an object with a specific topology
template <stk::topology::topology_t OurTopology>
auto create_ngp_topological_entity_view(const stk::mesh::NgpMesh& ngp_mesh, stk::mesh::FastMeshIndex entity_index) {
  return NgpEntityView<std::integral_constant<stk::topology::topology_t, OurTopology>>(ngp_mesh, entity_index);
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_ENTITYVIEW_HPP_