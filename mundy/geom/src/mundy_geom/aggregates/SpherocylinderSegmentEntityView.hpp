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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENTENTITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENTENTITYVIEW_HPP_

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

namespace mundy {

namespace geom {

/// \brief A traits class to provide abstracted access to a spherocylinder_segment's data via an aggregate
template <typename Agg>
struct SpherocylinderSegmentDataTraits {
  static_assert(ValidSpherocylinderSegmentDataType<Agg>,
                "Agg must satisfy the ValidSpherocylinderSegmentDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSpherocylinderSegmentData but is "
                "free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = decltype(std::declval<Agg>().node_coords_data());
  using radius_data_t = decltype(std::declval<Agg>().radius_data());
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  static decltype(auto) node_coords(Agg agg, stk::mesh::Entity spherocylinder_segment_node) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data(), spherocylinder_segment_node);
  }

  static decltype(auto) radius(Agg agg, stk::mesh::Entity spherocylinder_segment) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return stk::mesh::field_data(agg.radius_data(), spherocylinder_segment)[0];
    }
  }
};  // SpherocylinderSegmentDataTraits

/// \brief A traits class to provide abstracted access to a spherocylinder_segment's data via an NGP-compatible
/// aggregate See the discussion for SpherocylinderSegmentDataTraits for more information. Only difference is
/// Ngp-compatible data.
template <typename Agg>
struct NgpSpherocylinderSegmentDataTraits {
  static_assert(ValidNgpSpherocylinderSegmentDataType<Agg>,
                "Agg must satisfy the ValidNgpSpherocylinderSegmentDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSpherocylinderSegmentData but is "
                "free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = decltype(std::declval<Agg>().node_coords_data());
  using radius_data_t = decltype(std::declval<Agg>().radius_data());

  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) node_coords(Agg agg, stk::mesh::FastMeshIndex spherocylinder_segment_index) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data(), spherocylinder_segment_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) radius(Agg agg, stk::mesh::FastMeshIndex spherocylinder_segment_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return agg.radius_data()(spherocylinder_segment_index, 0);
    }
  }
};  // NgpSpherocylinderSegmentDataTraits

/// @brief A view of an STK entity meant to represent a spherocylinder_segment
///
/// We type specialize this class based on the valid set of topologies for a spherocylinder_segment entity.
///
/// Use \ref create_spherocylinder_segment_entity_view to build an SpherocylinderSegmentEntityView object
/// with automatic template deduction.
template <typename Base,  ValidSpherocylinderSegmentDataType SpherocylinderSegmentDataType>
class SpherocylinderSegmentEntityView;

template <typename Base,  ValidSpherocylinderSegmentDataType SpherocylinderSegmentDataType>
  requires(Base::get_topology() == stk::topology::LINE_2 || Base::get_topology() == stk::topology::LINE_3 ||
           Base::get_topology() == stk::topology::BEAM_2 || Base::get_topology() == stk::topology::BEAM_3 ||
           Base::get_topology() == stk::topology::SPRING_2 || Base::get_topology() == stk::topology::SPRING_3)
class SpherocylinderSegmentEntityView<Base, SpherocylinderSegmentDataType> : public Base {
  static_assert(SpherocylinderSegmentDataType::topology_t == Base::get_topology(),
                "The topology of the spherocylinder segment data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = OurTopology;
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;

  SpherocylinderSegmentEntityView(const Base&base, const SpherocylinderSegmentDataType &data)
      : Base(base), data_(data) {
  }

  stk::mesh::Entity& spherocylinder_segment_entity() {
    return spherocylinder_segment_;
  }

  const stk::mesh::Entity& spherocylinder_segment_entity() const {
    return spherocylinder_segment_;
  }

  stk::mesh::Entity& start_node_entity() {
    return start_node_;
  }

  const stk::mesh::Entity& start_node_entity() const {
    return start_node_;
  }

  stk::mesh::Entity& end_node_entity() {
    return end_node_;
  }

  const stk::mesh::Entity& end_node_entity() const {
    return end_node_;
  }

  decltype(auto) start() {
    return data_access_t::node_coords(data(), start_node_entity());
  }

  decltype(auto) start() const {
    return data_access_t::node_coords(data(), start_node_entity());
  }

  decltype(auto) end() {
    return data_access_t::node_coords(data(), end_node_entity());
  }

  decltype(auto) end() const {
    return data_access_t::node_coords(data(), end_node_entity());
  }

  decltype(auto) radius() {
    return data_access_t::radius(data(), spherocylinder_segment_entity());
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data(), spherocylinder_segment_entity());
  }

 private:
  const SpherocylinderSegmentDataType &data_;
};  // SpherocylinderSegmentEntityView

/// @brief An ngp-compatible view of an STK entity meant to represent a spherocylinder_segment
/// See the discussion for SpherocylinderSegmentEntityView for more information. The only difference is ngp-compatible
/// data access.
template <typename Base, ValidNgpSpherocylinderSegmentDataType NgpSpherocylinderSegmentDataType>
class NgpSpherocylinderSegmentEntityView;

template <typename Base, ValidNgpSpherocylinderSegmentDataType NgpSpherocylinderSegmentDataType>
  requires(Base::get_topology() == stk::topology::LINE_2 || Base::get_topology() == stk::topology::LINE_3 ||
           Base::get_topology() == stk::topology::BEAM_2 || Base::get_topology() == stk::topology::BEAM_3 ||
           Base::get_topology() == stk::topology::SPRING_2 || Base::get_topology() == stk::topology::SPRING_3)
class NgpSpherocylinderSegmentEntityView<Base, NgpSpherocylinderSegmentDataType> : public Base {
  static_assert(NgpSpherocylinderSegmentDataType::topology_t == Base::get_topology(),
                "The topology of the spherocylinder segment data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = OurTopology;
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;

  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderSegmentEntityView(const Base&base, const NgpSpherocylinderSegmentDataType &data)
      : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& spherocylinder_segment_index() {
    return spherocylinder_segment_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& spherocylinder_segment_index() const {
    return spherocylinder_segment_index_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& start_node_index() {
    return start_node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& start_node_index() const {
    return start_node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& end_node_index() {
    return end_node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& end_node_index() const {
    return end_node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) start() {
    return data_access_t::node_coords(data(), start_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) start() const {
    return data_access_t::node_coords(data(), start_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() {
    return data_access_t::node_coords(data(), end_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() const {
    return data_access_t::node_coords(data(), end_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data(), spherocylinder_segment_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data(), spherocylinder_segment_index());
  }

 private:
  const NgpSpherocylinderSegmentDataType &data_;
};  // NgpSpherocylinderSegmentEntityView
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENTENTITYVIEW_HPP_