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

#ifndef MUNDY_GEOM_AGGREGATES_LINEENETITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_LINEENETITYVIEW_HPP_

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
#include <mundy_geom/primitives/Line.hpp>  // for mundy::geom::ValidLineType
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// @brief A view of an STK entity meant to represent a line
/// We type specialize this class based on the valid set of topologies for an line entity.
///
/// Use \ref create_line_data to build a LineData object with automatic template deduction.
template <typename Base, ValidLineDataType LineDataType>
class LineEntityView;

/// @brief A view of a NODE STK entity meant to represent a line
template <typename Base,  ValidLineDataType LineDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class LineEntityView<Base, OurTopology, LineDataType> : public Base {
  static_assert(LineDataType::topology_t == stk::topology::NODE, "The topology of the line data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  LineEntityView(const Base& base, const LineDataType &data) : Base(base), data_(data) {
  }

  stk::mesh::Entity& ellipsoid_entity() {
    return entity();
  }

  const stk::mesh::Entity& ellipsoid_entity() const {
    return entity();
  }

  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), entity());
  }

  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), entity());
  }

  decltype(auto) direction() {
    return mundy::mesh::vector3_field_data(data_.direction_data(), entity());
  }

  decltype(auto) direction() const {
    return mundy::mesh::vector3_field_data(data_.direction_data(), entity());
  }

 private:
  const LineDataType &data_;
};  // LineEntityView<stk::topology::NODE, LineDataType>

/// @brief A view of a PARTICLE STK entity meant to represent a line
template <typename Base, ValidLineDataType LineDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class LineEntityView<Base, OurTopology, LineDataType> : public Base {
  static_assert(LineDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the line data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  LineEntityView(const Base&base, const LineDataType &data)
      : Base(base), data_(data) {
  }

  stk::mesh::Entity& line_entity() {
    return entity();
  }

  const stk::mesh::Entity& line_entity() const {
    return entity();
  }

  stk::mesh::Entity& center_node_entity() {
    return connected_node(0);
  }

  const stk::mesh::Entity& center_node_entity() const {
    return connected_node(0);
  }

  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), connected_node(0));
  }

  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), connected_node(0));
  }

  decltype(auto) direction() {
    return mundy::mesh::vector3_field_data(data_.direction_data(), entity());
  }

  decltype(auto) direction() const {
    return mundy::mesh::vector3_field_data(data_.direction_data(), entity());
  }

 private:
  const LineDataType &data_;
};  // LineEntityView<stk::topology::PARTICLE, LineDataType>

/// @brief An ngp-compatible view of a line entity
/// See the discussion for LineEntityView for more information. The only difference is ngp-compatible data access.
template <typename Base, ValidNgpLineDataType NgpLineDataType>
class NgpLineEntityView;

/// @brief An ngp-compatible view of a NODE STK entity meant to represent a line
template <typename Base, ValidNgpLineDataType NgpLineDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class NgpLineEntityView<Base, NgpLineDataType> : public Base {
  static_assert(NgpLineDataType::topology_t == stk::topology::NODE,
                "The topology of the line data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpLineEntityView(const Base&base, const NgpLineDataType &data) : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& line_index() {
    return entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& line_index() const {
    return entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), entity_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), entity_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) direction() {
    return mundy::mesh::vector3_field_data(data_.direction_data(), entity_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) direction() const {
    return mundy::mesh::vector3_field_data(data_.direction_data(), entity_index());
  }

 private:
  const NgpLineDataType &data_;
};  // NgpLineEntityView<stk::topology::NODE, NgpLineDataType>

/// @brief An ngp-compatible view of a PARTICLE STK entity meant to represent a line
template <typename Base, ValidNgpLineDataType NgpLineDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class NgpLineEntityView<Base, OurTopology, NgpLineDataType> : public Base {
  static_assert(NgpLineDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the line data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpLineEntityView(const Base&base, const NgpLineDataType &data) : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& line_index() {
    return entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& line_index() const {
    return entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& center_node_index() {
    return connected_node_index(0);
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& center_node_index() const {
    return connected_node_index(0);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), connected_node_index(0));
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), connected_node_index(0));
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) direction() {
    return mundy::mesh::vector3_field_data(data_.direction_data(), entity_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) direction() const {
    return mundy::mesh::vector3_field_data(data_.direction_data(), entity_index());
  }

 private:
  const NgpLineDataType &data_;
};  // NgpLineEntityView<stk::topology::PARTICLE, NgpLineDataType>
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_LINEENETITYVIEW_HPP_