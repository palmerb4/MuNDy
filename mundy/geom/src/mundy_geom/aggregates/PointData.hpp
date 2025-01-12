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

#ifndef MUNDY_GEOM_AGGREGATES_POINTDATA_HPP_
#define MUNDY_GEOM_AGGREGATES_POINTDATA_HPP_

// C++ core
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/aggregates/EntityView.hpp>  // for mundy::geom::EntityView and mundy::geom::create_topological_entity_view
#include <mundy_geom/aggregates/PointDataConcepts.hpp>  // for mundy::geom::ValidPointDataType
#include <mundy_geom/aggregates/PointEntityView.hpp>    // for mundy::geom::PointEntityView
#include <mundy_mesh/BulkData.hpp>                      // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of points
///
/// The topology of a line segment directly effects the access pattern for the underlying data.
/// Regardless of the topology, the node coordinates are stored on the node of the point.
/// However, how we access this node changes based on the rank of the point. Allowable topologies are:
///   - NODE, PARTICLE
template <typename Scalar,  //
          typename OurTopology>
class PointData {
  static_assert(OurTopology::value == stk::topology::NODE || OurTopology::value == stk::topology::PARTICLE,
                "The topology of a point must be NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = stk::mesh::Field<Scalar>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  PointData(const stk::mesh::BulkData& bulk_data, const node_coords_data_t& node_coords_data)
      : bulk_data_(bulk_data), node_coords_data_(node_coords_data) {
    MUNDY_THROW_ASSERT(node_coords_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center data must be a field of NODE_RANK");
  }

  static constexpr stk::topology::topology_t get_topology() {
    return topology_t;
  }

  static constexpr stk::topology::topology_t get_rank() {
    return stk::topology_detail::topology_data<OurTopology::value>::rank();
  }

  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }

  const node_coords_data_t& node_coords_data() const {
    return node_coords_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<PointData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  auto get_entity_view(stk::mesh::Entity entity) {
    using our_t = PointData<Scalar, OurTopology>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<PointEntityView, our_t>(*this);
  }

  const auto get_entity_view(stk::mesh::Entity entity) const {
    using our_t = PointData<Scalar, OurTopology>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<PointEntityView, our_t>(*this);
  }

  auto get_updated_ngp_data() const {
    return create_ngp_point_data<scalar_t, topology_t>(stk::mesh::get_updated_ngp_mesh(bulk_data_),  //
                                                       stk::mesh::get_updated_ngp_field<scalar_t>(node_coords_data_));
  }

 private:
  const stk::mesh::BulkData& bulk_data_;
  const node_coords_data_t& node_coords_data_;
};  // PointData

/// \brief Aggregate to hold the data for a collection of NGP-compatible points
/// See the discussion for PointData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,  //
          typename OurTopology>
class NgpPointData {
  static_assert(OurTopology::value == stk::topology::NODE || OurTopology::value == stk::topology::PARTICLE,
                "The topology of a point must be NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = stk::mesh::NgpField<Scalar>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  KOKKOS_INLINE_FUNCTION
  NgpPointData(const stk::mesh::NgpMesh& ngp_mesh, const node_coords_data_t& node_coords_data)
      : ngp_mesh_(ngp_mesh), node_coords_data_(node_coords_data) {
    MUNDY_THROW_ASSERT(node_coords_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center data must be a field of NODE_RANK");
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t get_topology() {
    return topology_t;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t get_rank() {
    return stk::topology_detail::topology_data<OurTopology::value>::rank();
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh& ngp_mesh() {
    return ngp_mesh_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh& ngp_mesh() const {
    return ngp_mesh_;
  }

  KOKKOS_INLINE_FUNCTION
  const node_coords_data_t& node_coords_data() const {
    return node_coords_data_;
  }

  KOKKOS_INLINE_FUNCTION
  node_coords_data_t& node_coords_data() {
    return node_coords_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION auto add_augment(Args&&... args) const {
    return NextAugment<NgpPointData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  auto get_entity_view(stk::mesh::FastMeshIndex entity_index) {
    using our_t = NgpPointData<Scalar, OurTopology>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpPointEntityView, our_t>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  const auto get_entity_view(stk::mesh::FastMeshIndex entity_index) const {
    using our_t = NgpPointData<Scalar, OurTopology>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpPointEntityView, our_t>(*this);
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  node_coords_data_t node_coords_data_;
};  // NgpPointData

/// \brief A helper function to create a PointData object with automatic template deduction
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology>  // Must be provided
auto create_point_data(const stk::mesh::BulkData& bulk_data, const stk::mesh::Field<Scalar>& node_coords_data) {
  return PointData<Scalar, stk::topology_detail::topology_data<OurTopology>>{bulk_data, node_coords_data};
}

/// \brief A helper function to create a NgpPointData object
/// See the discussion for create_point_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology>  // Must be provided
auto create_ngp_point_data(const stk::mesh::NgpMesh& ngp_mesh, const stk::mesh::NgpField<Scalar>& node_coords_data) {
  return NgpPointData<Scalar, stk::topology_detail::topology_data<OurTopology>>{ngp_mesh, node_coords_data};
}

/// \brief A helper function to get an updated NgpPointData object from a PointData object
/// \param data The PointData object to convert
template <typename Scalar, typename OurTopology>  // deduced
auto get_updated_ngp_data(const PointData<Scalar, OurTopology>& data) {
  return data.get_updated_ngp_data();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_POINTDATA_HPP_