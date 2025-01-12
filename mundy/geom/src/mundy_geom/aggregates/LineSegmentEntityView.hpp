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

#ifndef MUNDY_GEOM_AGGREGATES_LINESEGMENTENTITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_LINESEGMENTENTITYVIEW_HPP_

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
#include <mundy_geom/primitives/LineSegment.hpp>  // for mundy::geom::ValidLineSegmentType
#include <mundy_mesh/BulkData.hpp>                // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// @brief A view of an STK entity meant to represent a line_segment
///
/// We type specialize this class based on the valid set of topologies for an line entity.
///
/// Use \ref create_line_entity_view to build a LineSegmentEntityView object with automatic template deduction.
template <typename Base, ValidLineSegmentDataType LineSegmentDataType>
class LineSegmentEntityView;

/// @brief A view of a STK entity meant to represent a line_segment
template <typename Base, ValidLineSegmentDataType LineSegmentDataType>
  requires(Base::get_topology() == stk::topology::LINE_2 || Base::get_topology() == stk::topology::LINE_3 ||
           Base::get_topology() == stk::topology::BEAM_2 || Base::get_topology() == stk::topology::BEAM_3 ||
           Base::get_topology() == stk::topology::SPRING_2 || Base::get_topology() == stk::topology::SPRING_3)
class LineSegmentEntityView<Base, LineSegmentDataType> : public Base {
  static_assert(LineSegmentDataType::topology_t == Base::get_topology(),
                "The topology of the line segment data must match the view");

 public:
  using scalar_t = typename LineSegmentDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Base::get_topology();
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<topology_t>::rank;

  LineSegmentEntityView(const Base& base, LineSegmentDataType &data)
      : Base(base), 
        data_(data) {
  }

  const stk::mesh::Entity& line_segment_entity() const {
    return Base::entity();
  }

  const stk::mesh::Entity& start_node_entity() const {
    return Base::connected_node(0);
  }

  const stk::mesh::Entity& end_node_entity() const {
    return Base::connected_node(1);
  }

  decltype(auto) start() {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), start_node_entity());
  }

  decltype(auto) start() const {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), start_node_entity());
  }

  decltype(auto) end() {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), end_node_entity());
  }

  decltype(auto) end() const {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), end_node_entity());
  }

 private:
  const LineSegmentDataType &data_;
};  // LineSegmentEntityView

/// @brief An ngp-compatible view of an STK entity meant to represent a line_segment
/// See the discussion for LineSegmentEntityView for more information. The only difference is ngp-compatible data
/// access.
template <typename Base, ValidNgpLineSegmentDataType NgpLineSegmentDataType>
class NgpLineSegmentEntityView;

/// @brief An ngp-compatible view of an STK entity meant to represent a line_segment
template <typename Base, ValidNgpLineSegmentDataType NgpLineSegmentDataType>
  requires(Base::get_topology() == stk::topology::LINE_2 || Base::get_topology() == stk::topology::LINE_3 ||
           Base::get_topology() == stk::topology::BEAM_2 || Base::get_topology() == stk::topology::BEAM_3 ||
           Base::get_topology() == stk::topology::SPRING_2 || Base::get_topology() == stk::topology::SPRING_3)
class NgpLineSegmentEntityView<Base, NgpLineSegmentDataType> : public Base {
  static_assert(NgpLineSegmentDataType::topology_t == Base::get_topology(),
                "The topology of the line segment data must match the view");

 public:
  using scalar_t = typename NgpLineSegmentDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Base::get_topology();
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<topology_t>::rank;

  KOKKOS_INLINE_FUNCTION
  NgpLineSegmentEntityView(const Base&base, const NgpLineSegmentDataType &data)
      : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& line_segment_index() const {
    return Base::entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& start_node_index() const {
    return Base::connected_node_index(0);
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& end_node_index() const {
    return Base::connected_node_index(1);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) start() {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), start_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) start() const {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), start_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), end_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() const {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), end_node_index());
  }

 private:
  const NgpLineSegmentDataType &data_;
};  // NgpLineSegmentEntityView
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_LINESEGMENTENTITYVIEW_HPP_