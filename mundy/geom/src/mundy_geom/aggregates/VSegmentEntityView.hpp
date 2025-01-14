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

#ifndef MUNDY_GEOM_AGGREGATES_VSEGMENTENTITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_VSEGMENTENTITYVIEW_HPP_

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
#include <mundy_geom/primitives/VSegment.hpp>  // for mundy::geom::ValidVSegmentType
#include <mundy_mesh/BulkData.hpp>             // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// @brief A view of a TRI_3, SHELL_TRI_3, or SPRING_3 STK entity meant to represent a v_segment
///
/// Use \ref create_v_segment_entity_view to build an VSegmentEntityView object with automatic template deduction.
template <typename Base, ValidVSegmentDataType VSegmentDataType>
class VSegmentEntityView;

template <typename Base, ValidVSegmentDataType VSegmentDataType>
  requires(Base::get_topology() == stk::topology::TRI_3 || Base::get_topology() == stk::topology::SHELL_TRI_3 ||
           Base::get_topology() == stk::topology::SPRING_3)
class VSegmentEntityView<Base, VSegmentDataType> : public Base {
  static_assert(VSegmentDataType::topology_t == Base::get_topology(),
                "The topology of the v_segment data must match the view");

 public:
  using scalar_t = typename VSegmentDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Base::get_topology();
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<topology_t>::rank;

  VSegmentEntityView(const Base& base, const VSegmentDataType& data) : Base(base), data_(data) {
  }

  stk::mesh::Entity v_segment_entity() const {
    return Base::entity();
  }

  stk::mesh::Entity start_node_entity() const {
    return Base::connected_node(0);
  }

  stk::mesh::Entity middle_node_entity() const {
    if constexpr (topology_t == stk::topology::TRI_3 || topology_t == stk::topology::SHELL_TRI_3) {
      return Base::connected_node(2);
    } else {
      return Base::connected_node(1);
    }
  }

  stk::mesh::Entity end_node_entity() const {
    if constexpr (topology_t == stk::topology::TRI_3 || topology_t == stk::topology::SHELL_TRI_3) {
      return Base::connected_node(1);
    } else {
      return Base::connected_node(2);
    }
  }

  decltype(auto) start() {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), start_node_entity());
  }

  decltype(auto) start() const {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), start_node_entity());
  }

  decltype(auto) middle() {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), middle_node_entity());
  }

  decltype(auto) middle() const {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), middle_node_entity());
  }

  decltype(auto) end() {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), end_node_entity());
  }

  decltype(auto) end() const {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), end_node_entity());
  }

 private:
  const VSegmentDataType& data_;
};  // VSegmentEntityView

/// @brief An ngp-compatible view of an STK entity meant to represent a v_segment
/// See the discussion for VSegmentEntityView for more information. The only difference is ngp-compatible data access.
template <typename Base, ValidNgpVSegmentDataType NgpVSegmentDataType>
class NgpVSegmentEntityView;

template <typename Base, ValidNgpVSegmentDataType NgpVSegmentDataType>
  requires(Base::get_topology() == stk::topology::TRI_3 || Base::get_topology() == stk::topology::SHELL_TRI_3 ||
           Base::get_topology() == stk::topology::SPRING_3)
class NgpVSegmentEntityView<Base, NgpVSegmentDataType> : public Base {
  static_assert(NgpVSegmentDataType::topology_t == Base::get_topology(),
                "The topology of the v_segment data must match the view");

 public:
  using scalar_t = typename NgpVSegmentDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Base::get_topology();
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<topology_t>::rank;

  KOKKOS_INLINE_FUNCTION
  NgpVSegmentEntityView(const Base& base, const NgpVSegmentDataType& data) : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex v_segment_index() const {
    return Base::entity_index();
  }


  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex start_node_index() const {
    return Base::connected_node_index(0);
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex middle_node_index() const {
    if constexpr (topology_t == stk::topology::TRI_3 || topology_t == stk::topology::SHELL_TRI_3) {
      return Base::connected_node_index(2);
    } else {
      return Base::connected_node_index(1);
    }
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex end_node_index() const {
    if constexpr (topology_t == stk::topology::TRI_3 || topology_t == stk::topology::SHELL_TRI_3) {
      return Base::connected_node_index(1);
    } else {
      return Base::connected_node_index(2);
    }
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
  decltype(auto) middle() {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), middle_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) middle() const {
    return mundy::mesh::vector3_field_data(data_.node_coords_data(), middle_node_index());
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
  const NgpVSegmentDataType& data_;
};  // NgpVSegmentEntityView

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_VSEGMENTENTITYVIEW_HPP_