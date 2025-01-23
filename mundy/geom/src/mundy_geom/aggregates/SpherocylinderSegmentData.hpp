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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENTDATA_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENTDATA_HPP_

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
#include <mundy_geom/primitives/SpherocylinderSegment.hpp>  // for mundy::geom::ValidSpherocylinderSegmentType
#include <mundy_mesh/BulkData.hpp>                          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_geom/aggregates/SpherocylinderSegmentDataConcepts.hpp>  // for mundy::geom::ValidSpherocylinderSegmentDataType
#include <mundy_geom/aggregates/SpherocylinderSegmentEntityView.hpp>  // for mundy::geom::SpherocylinderSegmentEntityView
#include <mundy_geom/aggregates/EntityView.hpp>  // for mundy::geom::EntityView and mundy::geom::create_topological_entity_view

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of sphero-cylinder segments
///
/// The topology of a line segment directly effects the access pattern for the underlying data.
/// Regardless of the topology, the node coordinates are stored on all nodes of the line segment, whereas
/// the radius is stored on the rank of the line segment. Allowable topologies are:
///   - LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, SPRING_3
template <typename Scalar,       //
          typename OurTopology,  //
          typename HasSharedRadius = std::false_type>
class SpherocylinderSegmentData {
  static_assert(OurTopology::value == stk::topology::LINE_2 || OurTopology::value == stk::topology::LINE_3 ||
                    OurTopology::value == stk::topology::BEAM_2 || OurTopology::value == stk::topology::BEAM_3 ||
                    OurTopology::value == stk::topology::SPRING_2 || OurTopology::value == stk::topology::SPRING_3,
                "The topology of a line segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = stk::mesh::Field<Scalar>;
  using radius_data_t = std::conditional_t<HasSharedRadius::value, Scalar, stk::mesh::Field<Scalar>>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  SpherocylinderSegmentData(const stk::mesh::BulkData& bulk_data, const node_coords_data_t& node_coords_data,
                            const radius_data_t& radius_data)
      : bulk_data_(bulk_data), node_coords_data_(node_coords_data), radius_data_(radius_data) {
    MUNDY_THROW_ASSERT(node_coords_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The node_coords data must be a field of NODE_RANK");
    stk::topology our_topology = topology_t;
    if constexpr (!HasSharedRadius::value) {
      MUNDY_THROW_ASSERT(radius_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                         "The radius data must be a field of the same rank as the segment");
    }
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

  const radius_data_t& radius_data() const {
    return radius_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<SpherocylinderSegmentData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  auto get_entity_view(stk::mesh::Entity entity) {
    using our_t = SpherocylinderSegmentData<Scalar, OurTopology, HasSharedRadius>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<SpherocylinderSegmentEntityView, our_t>(*this);
  }

  const auto get_entity_view(stk::mesh::Entity entity) const {
    using our_t = SpherocylinderSegmentData<Scalar, OurTopology, HasSharedRadius>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<SpherocylinderSegmentEntityView, our_t>(*this);
  }

  auto get_updated_ngp_data() const {
    if constexpr (!HasSharedRadius::value) {
      return create_ngp_sphere_data<scalar_t, topology_t>(                    //
          stk::mesh::get_updated_ngp_mesh(bulk_data_),                  //
          stk::mesh::get_updated_ngp_field<scalar_t>(node_coords_data_),  //
          stk::mesh::get_updated_ngp_field<scalar_t>(radius_data_));
    } else {
      return create_ngp_sphere_data<scalar_t, topology_t>(                    //
          stk::mesh::get_updated_ngp_mesh(bulk_data_),                  //
          stk::mesh::get_updated_ngp_field<scalar_t>(node_coords_data_),  //
          radius_data_);
    }
  }

 private:
  const stk::mesh::BulkData& bulk_data_;
  const node_coords_data_t& node_coords_data_;
  const radius_data_t& radius_data_;
};  // SpherocylinderSegmentData

/// \brief Aggregate to hold the data for a collection of NGP-compatible sphero-cylinder segments
/// See the discussion for SpherocylinderSegmentData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,       //
          typename OurTopology,  //
          typename HasSharedRadius = std::false_type>
class NgpSpherocylinderSegmentData {
  static_assert(OurTopology::value == stk::topology::LINE_2 || OurTopology::value == stk::topology::LINE_3 ||
                    OurTopology::value == stk::topology::BEAM_2 || OurTopology::value == stk::topology::BEAM_3 ||
                    OurTopology::value == stk::topology::SPRING_2 || OurTopology::value == stk::topology::SPRING_3,
                "The topology of a line segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = stk::mesh::NgpField<Scalar>;
  using radius_data_t = std::conditional_t<HasSharedRadius::value, Scalar, stk::mesh::NgpField<Scalar>>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderSegmentData(const stk::mesh::NgpMesh &ngp_mesh, const node_coords_data_t& node_coords_data,
                               const radius_data_t& radius_data)
      : ngp_mesh_(ngp_mesh), node_coords_data_(node_coords_data), radius_data_(radius_data) {
    MUNDY_THROW_ASSERT(node_coords_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The node_coords data must be a field of NODE_RANK");
    stk::topology our_topology = topology_t;
    if constexpr (!HasSharedRadius::value) {
      MUNDY_THROW_ASSERT(radius_data.get_rank() == our_topology.rank(), std::invalid_argument,
                         "The radius data must be a field of the same rank as the segment");
    }
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

  KOKKOS_INLINE_FUNCTION
  const radius_data_t& radius_data() const {
    return radius_data_;
  }

  KOKKOS_INLINE_FUNCTION
  radius_data_t& radius_data() {
    return radius_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION
  auto add_augment(Args&&... args) const {
    return NextAugment<NgpSpherocylinderSegmentData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  auto get_entity_view(stk::mesh::FastMeshIndex entity_index) {
    using our_t = NgpSpherocylinderSegmentData<Scalar, OurTopology, HasSharedRadius>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpSpherocylinderSegmentEntityView, our_t>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  const auto get_entity_view(stk::mesh::FastMeshIndex entity_index) const {
    using our_t = NgpSpherocylinderSegmentData<Scalar, OurTopology, HasSharedRadius>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpSpherocylinderSegmentEntityView, our_t>(*this);
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  node_coords_data_t node_coords_data_;
  radius_data_t radius_data_;
};  // NgpSpherocylinderSegmentData

/// \brief A helper function to create a SpherocylinderSegmentData object
///
/// This function creates a SpherocylinderSegmentData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename RadiusDataType>                // deduced
auto create_spherocylinder_segment_data(const stk::mesh::BulkData& bulk_data, const stk::mesh::Field<Scalar>& node_coords_data,
                                        const RadiusDataType& radius_data) {
  constexpr bool is_radius_shared = std::is_same_v<std::decay_t<RadiusDataType>, Scalar>;
  if constexpr (is_radius_shared) {
    return SpherocylinderSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type>{
        bulk_data, node_coords_data, radius_data};
  } else {
    return SpherocylinderSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type>{
        bulk_data, node_coords_data, radius_data};
  }
}

/// \brief A helper function to create a NgpSpherocylinderSegmentData object
/// See the discussion for create_spherocylinder_segment_data for more information. Only difference is NgpFields over
/// Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename RadiusDataType>                // deduced
auto create_ngp_spherocylinder_segment_data(const stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::NgpField<Scalar>& node_coords_data,
                                            const RadiusDataType& radius_data) {
  constexpr bool is_radius_shared = std::is_same_v<std::decay_t<RadiusDataType>, Scalar>;
  if constexpr (is_radius_shared) {
    return NgpSpherocylinderSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type>{
        ngp_mesh, node_coords_data, radius_data};
  } else {
    return NgpSpherocylinderSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type>{
        ngp_mesh, node_coords_data, radius_data};
  }
}

/// \brief A helper function to get an updated NgpSpherocylinderSegmentData object from a SpherocylinderSegmentData
/// object \param data The SpherocylinderSegmentData object to convert
template <typename Scalar,                     
          typename OurTopology>
auto get_updated_ngp_data(const SpherocylinderSegmentData<Scalar, OurTopology>& data) {
  return data.get_updated_ngp_data();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENTDATA_HPP_