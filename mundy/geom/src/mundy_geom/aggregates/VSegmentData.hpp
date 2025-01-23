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

#ifndef MUNDY_GEOM_AGGREGATES_VSEGMENT_HPP_
#define MUNDY_GEOM_AGGREGATES_VSEGMENT_HPP_

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
#include <mundy_geom/aggregates/VSegmentDataConcepts.hpp>  // for mundy::geom::ValidVSegmentDataType
#include <mundy_geom/aggregates/VSegmentEntityView.hpp>    // for mundy::geom::VSegmentEntityView
#include <mundy_geom/primitives/VSegment.hpp>              // for mundy::geom::ValidVSegmentType
#include <mundy_mesh/BulkData.hpp>                         // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of v_segments
///
/// The topology of a v segment directly effects the access pattern for the underlying data.
/// Regardless of the topology, the node coordinates are stored on all nodes of the v segment.
/// The difference between the topologies is how we access those nodes.
///
/// Allowable topologies are and their node access patterns are:
///   - SPRING_3: Left, right, middle
///   - TRI_3: Left, middle, right
///   - SHELL_TRI_3: Left, middle, right
template <typename Scalar, typename OurTopology>
class VSegmentData {
  static_assert(OurTopology::value == stk::topology::SPRING_3 || OurTopology::value == stk::topology::TRI_3 ||
                    OurTopology::value == stk::topology::SHELL_TRI_3,
                "The topology of a v segment must be SPRING_3, TRI_3, or SHELL_TRI_3");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = stk::mesh::Field<Scalar>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  VSegmentData(const stk::mesh::BulkData& bulk_data, const node_coords_data_t& node_coords_data)
      : bulk_data_(bulk_data), node_coords_data_(node_coords_data) {
    MUNDY_THROW_ASSERT(node_coords_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The node_coords data must be a field of NODE_RANK");
  }

  static constexpr stk::topology::topology_t get_topology() {
    return topology_t;
  }

  static constexpr stk::topology::rank_t get_rank() {
    return stk::topology_detail::topology_data<OurTopology::value>::rank;
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
    return NextAugment<VSegmentData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  auto get_entity_view(stk::mesh::Entity entity) {
    using our_t = VSegmentData<Scalar, OurTopology>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<VSegmentEntityView, our_t>(*this);
  }

  const auto get_entity_view(stk::mesh::Entity entity) const {
    using our_t = VSegmentData<Scalar, OurTopology>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<VSegmentEntityView, our_t>(*this);
  }

  auto get_updated_ngp_data() const {
    return create_ngp_v_segment_data<scalar_t, topology_t>(  //
        stk::mesh::get_updated_ngp_mesh(bulk_data_),         //
        stk::mesh::get_updated_ngp_field<scalar_t>(node_coords_data_));
  }

 private:
  const stk::mesh::BulkData& bulk_data_;
  const node_coords_data_t& node_coords_data_;
};  // VSegmentData

/// \brief Aggregate to hold the data for a collection of NGP-compatible line segments
/// See the discussion for VSegmentData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename OurTopology>
class NgpVSegmentData {
  static_assert(OurTopology::value == stk::topology::SPRING_3 || OurTopology::value == stk::topology::TRI_3 ||
                    OurTopology::value == stk::topology::SHELL_TRI_3,
                "The topology of a v segment must be SPRING_3, TRI_3, or SHELL_TRI_3");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = stk::mesh::NgpField<Scalar>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  NgpVSegmentData(const stk::mesh::NgpMesh& ngp_mesh, const node_coords_data_t& node_coords_data)
      : ngp_mesh_(ngp_mesh), node_coords_data_(node_coords_data) {
    MUNDY_THROW_ASSERT(node_coords_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The node_coords data must be a field of NODE_RANK");
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t get_topology() {
    return topology_t;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::rank_t get_rank() {
    return stk::topology_detail::topology_data<OurTopology::value>::rank;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh& ngp_mesh() const {
    return ngp_mesh_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh& ngp_mesh() {
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
  KOKKOS_INLINE_FUNCTION
  auto add_augment(Args&&... args) const {
    return NextAugment<NgpVSegmentData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  auto get_entity_view(stk::mesh::FastMeshIndex entity_index) {
    using our_t = NgpVSegmentData<Scalar, OurTopology>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpVSegmentEntityView, our_t>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  const auto get_entity_view(stk::mesh::FastMeshIndex entity_index) const {
    using our_t = NgpVSegmentData<Scalar, OurTopology>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpVSegmentEntityView, our_t>(*this);
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  node_coords_data_t node_coords_data_;
};  // NgpVSegmentData

/// \brief A helper function to create a VSegmentData object
///
/// This function creates a VSegmentData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology>  // Must be provided
auto create_v_segment_data(const stk::mesh::BulkData& bulk_data, const stk::mesh::Field<Scalar>& node_coords_data) {
  return VSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>>{bulk_data, node_coords_data};
}

/// \brief A helper function to create a NgpVSegmentData object
/// See the discussion for create_v_segment_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology>  // Must be provided
auto create_ngp_v_segment_data(const stk::mesh::NgpMesh& ngp_mesh,
                               const stk::mesh::NgpField<Scalar>& node_coords_data) {
  return NgpVSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>>{ngp_mesh, node_coords_data};
}

/// \brief A helper function to get an updated NgpVSegmentData object from a VSegmentData object
/// \param data The VSegmentData object to convert
template <typename Scalar, typename OurTopology>
auto get_updated_ngp_data(const VSegmentData<Scalar, OurTopology>& data) {
  return data.get_updated_ngp_data();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_VSEGMENT_HPP_