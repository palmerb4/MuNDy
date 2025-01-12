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

#ifndef MUNDY_GEOM_AGGREGATES_POINTENTITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_POINTENTITYVIEW_HPP_

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
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::ValidPointType
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>        // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// \brief A traits class to provide abstracted access to a point's data via an aggregate
///
/// By default, this class is compatible with PointData or any class the meets the ValidPointDataType concept.
/// Users can specialize this class to support other aggregate types.
template <typename Agg>
struct PointDataTraits {
  static_assert(ValidPointDataType<Agg>,
                "Agg must satisfy the ValidPointDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpPointData but is free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  static decltype(auto) center(Agg agg, stk::mesh::Entity point_node) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data(), point_node);
  }
};  // PointDataTraits

/// \brief A traits class to provide abstracted access to a point's data via an NGP-compatible aggregate
/// See the discussion for PointDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpPointDataTraits {
  static_assert(ValidNgpPointDataType<Agg>,
                "Agg must satisfy the ValidNgpPointDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpPointData but is free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex point_node_index) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data(), point_node_index);
  }
};  // NgpPointDataTraits

/// @brief A view of an STK entity meant to represent a point
///
/// We type specialize this class based on the valid set of topologies for a point entity.
///
/// Use \ref create_point_entity_view to build a PointEntityView object with automatic template deduction.
template <typename Base, typename PointDataType>
class PointEntityView;

/// @brief A view of a NODE STK entity meant to represent a point
template <typename Base, typename PointDataType>
 requires(Base::get_topology() == stk::topology::NODE)
class PointEntityView<Base, PointDataType>  : public Base {
  static_assert(PointDataType::topology_t == stk::topology::NODE, "The topology of the point data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  PointEntityView(const Base&base, const PointDataType &data) : Base(base), data_(data) {
  }

  stk::mesh::Entity& point_entity() {
    return point_;
  }

  const stk::mesh::Entity& point_entity() const {
    return point_;
  }

  decltype(auto) center() {
    return data_access_t::center(data(), point_entity());
  }

  decltype(auto) center() const {
    return data_access_t::center(data(), point_entity());
  }

  decltype(auto) operator[](int i) {
    return center()[i];
  }

  decltype(auto) operator[](int i) const {
    return center()[i];
  }

 private:
  const PointDataType &data_;
};  // PointEntityView for stk::topology::NODE

template <typename Base, typename PointDataType>
  requires(OurTopology::value == stk::topology::PARTICLE)
class PointEntityView<Base, PointDataType>  : public Base {
  static_assert(PointDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the point data must match the view");
              
 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  PointEntityView(const Base&base, const PointDataType &data)
      : Base(base), data_(data) {
  }

  stk::mesh::Entity& point_entity() {
    return point_;
  }

  const stk::mesh::Entity& point_entity() const {
    return point_;
  }

  stk::mesh::Entity& node_entity() {
    return node_;
  }

  const stk::mesh::Entity& node_entity() const {
    return node_;
  }

  decltype(auto) center() {
    return data_access_t::center(data(), node_entity());
  }

  decltype(auto) center() const {
    return data_access_t::center(data(), node_entity());
  }

  decltype(auto) operator[](int i) {
    return center()[i];
  }

  decltype(auto) operator[](int i) const {
    return center()[i];
  }

 private:
  const PointDataType &data_;
};  // PointEntityView<stk::topology::PARTICLE, PointDataType>

/// @brief An ngp-compatible view of a STK entity meant to represent a point
/// See the discussion for PointEntityView for more information. The only difference is ngp-compatible data access.
template <typename Base, typename NgpPointDataType>
class NgpPointEntityView;

/// @brief An ngp-compatible view of a NODE STK entity meant to represent a point
template <typename Base, typename NgpPointDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class NgpPointEntityView<Base, NgpPointDataType> : public Base {
  static_assert(NgpPointDataType::topology_t == stk::topology::NODE,
                "The topology of the point data must match the view");
     

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpPointEntityView(const Base&base, const NgpPointDataType &data)
      : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& point_index() {
    return point_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& point_index() const {
    return point_index_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data(), point_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data(), point_index());
  }

  decltype(auto) operator[](int i) {
    return center()[i];
  }

  decltype(auto) operator[](int i) const {
    return center()[i];
  }

 private:
  const NgpPointDataType &data_;
};  // NgpPointEntityView<stk::topology::NODE, NgpPointDataType>

/// @brief An ngp-compatible view of a PARTICLE STK entity meant to represent a point
template <typename Base, typename NgpPointDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class NgpPointEntityView<Base, NgpPointDataType> : public Base {
  static_assert(NgpPointDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the point data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpPointEntityView(const Base&base, const NgpPointDataType &data)
      : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& point_index() {
    return point_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& point_index() const {
    return point_index_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& node_index() {
    return node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& node_index() const {
    return node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data(), node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data(), node_index());
  }

  decltype(auto) operator[](int i) {
    return center()[i];
  }

  decltype(auto) operator[](int i) const {
    return center()[i];
  }

 private:
  const NgpPointDataType &data_;
};  // NgpElemPointView
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_POINTENTITYVIEW_HPP_