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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERENTITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERENTITYVIEW_HPP_

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
#include <mundy_geom/primitives/Spherocylinder.hpp>  // for mundy::geom::ValidSpherocylinderType
#include <mundy_mesh/BulkData.hpp>                   // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// \brief A traits class to provide abstracted access to a spherocylinder's data via an aggregate
///
/// By default, this class is compatible with SpherocylinderData or any class the meets the
/// ValidSpherocylinderDataType concept. Users can specialize this class to support other aggregate types.
template <typename Agg>
struct SpherocylinderDataTraits {
  static_assert(ValidSpherocylinderDataType<Agg>,
                "Agg must satisfy the ValidSpherocylinderDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSpherocylinderData but is free to "
                "extend it as "
                "needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<decltype(std::declval<Agg>().radius_data())>, scalar_t>;
  }

  static constexpr bool has_shared_length() {
    return std::is_same_v<std::decay_t<decltype(std::declval<Agg>().length_data())>, scalar_t>;
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity spherocylinder_node) {
    return mundy::mesh::vector3_field_data(agg.center_data(), spherocylinder_node);
  }

  static decltype(auto) orientation(Agg agg, stk::mesh::Entity spherocylinder) {
    return mundy::mesh::quaternion_field_data(agg.orientation_data(), spherocylinder);
  }

  static decltype(auto) radius(Agg agg, stk::mesh::Entity spherocylinder) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return stk::mesh::field_data(agg.radius_data(), spherocylinder)[0];
    }
  }

  static decltype(auto) length(Agg agg, stk::mesh::Entity spherocylinder) {
    if constexpr (has_shared_length()) {
      return agg.length_data();
    } else {
      return stk::mesh::field_data(agg.length_data(), spherocylinder)[0];
    }
  }
};  // SpherocylinderDataTraits

/// \brief A traits class to provide abstracted access to a spherocylinder's data via an NGP-compatible aggregate
/// See the discussion for SpherocylinderDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpSpherocylinderDataTraits {
  static_assert(ValidNgpSpherocylinderDataType<Agg>,
                "Agg must satisfy the ValidNgpSpherocylinderDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSpherocylinderData but is free to "
                "extend it as "
                "needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<decltype(std::declval<Agg>().radius_data())>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_length() {
    return std::is_same_v<std::decay_t<decltype(std::declval<Agg>().length_data())>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex spherocylinder_node_index) {
    return mundy::mesh::vector3_field_data(agg.center_data(), spherocylinder_node_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) orientation(Agg agg, stk::mesh::FastMeshIndex spherocylinder_index) {
    return mundy::mesh::quaternion_field_data(agg.orientation_data(), spherocylinder_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) radius(Agg agg, stk::mesh::FastMeshIndex spherocylinder_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return agg.radius_data()(spherocylinder_index, 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) length(Agg agg, stk::mesh::FastMeshIndex spherocylinder_index) {
    if constexpr (has_shared_length()) {
      return agg.length_data();
    } else {
      return agg.length_data()(spherocylinder_index, 0);
    }
  }
};  // NgpSpherocylinderDataTraits

/// @brief A view of an STK entity meant to represent a spherocylinder
///
/// We type specialize this class based on the valid set of topologies for a spherocylinder entity.
///
/// Use \ref create_spherocylinder_entity_view to build an SpherocylinderEntityView object with automatic template
/// deduction.
template <typename Base,  typename SpherocylinderDataType>
class SpherocylinderEntityView;

/// @brief A view of a NODE STK entity meant to represent a spherocylinder
template <typename Base, typename SpherocylinderDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class SpherocylinderEntityView<Base, SpherocylinderDataType> : public Base {
  static_assert(SpherocylinderDataType::topology_t == stk::topology::NODE,
                "The topology of the spherocylinder data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  SpherocylinderEntityView(const Base&base, const SpherocylinderDataType &data)
      : Base(base), data_(data) {
  }

  stk::mesh::Entity& spherocylinder_entity() {
    return spherocylinder_;
  }

  const stk::mesh::Entity& spherocylinder_entity() const {
    return spherocylinder_;
  }

  decltype(auto) center() {
    return data_access_t::center(data(), spherocylinder_entity());
  }

  decltype(auto) center() const {
    return data_access_t::center(data(), spherocylinder_entity());
  }

  decltype(auto) orientation() {
    return data_access_t::orientation(data(), spherocylinder_entity());
  }

  decltype(auto) orientation() const {
    return data_access_t::orientation(data(), spherocylinder_entity());
  }

  decltype(auto) radius() {
    // For those not familiar with decltype(auto), it allows us to return either an auto or an auto&.
    return data_access_t::radius(data(), spherocylinder_entity());
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data(), spherocylinder_entity());
  }

  decltype(auto) length() {
    return data_access_t::length(data(), spherocylinder_entity());
  }

  decltype(auto) length() const {
    return data_access_t::length(data(), spherocylinder_entity());
  }

 private:
  const SpherocylinderDataType &data_;
};  // SpherocylinderEntityView<stk::topology::NODE, SpherocylinderDataType>

/// @brief A view of a PARTICLE STK entity meant to represent a spherocylinder
template <typename Base,typename SpherocylinderDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class SpherocylinderEntityView<Base, SpherocylinderDataType> {
  static_assert(SpherocylinderDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the spherocylinder data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  SpherocylinderEntityView(const Base&base, const SpherocylinderDataType &data)
      : Base(base), data_(data) {
  }

  stk::mesh::Entity& spherocylinder_entity() {
    return spherocylinder_;
  }

  const stk::mesh::Entity& spherocylinder_entity() const {
    return spherocylinder_;
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

  decltype(auto) orientation() {
    return data_access_t::orientation(data(), spherocylinder_entity());
  }

  decltype(auto) orientation() const {
    return data_access_t::orientation(data(), spherocylinder_entity());
  }

  decltype(auto) radius() {
    return data_access_t::radius(data(), spherocylinder_entity());
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data(), spherocylinder_entity());
  }

  decltype(auto) length() {
    return data_access_t::length(data(), spherocylinder_entity());
  }

  decltype(auto) length() const {
    return data_access_t::length(data(), spherocylinder_entity());
  }

 private:
  const SpherocylinderDataType &data_;
};  // SpherocylinderEntityView<stk::topology::PARTICLE, SpherocylinderDataType>

/// @brief An ngp-compatible view of an STK entity meant to represent a spherocylinder
/// See the discussion for SpherocylinderEntityView for more information. The only difference is ngp-compatible data
/// access.
template <typename Base, typename NgpSpherocylinderDataType>
class NgpSpherocylinderEntityView;

/// @brief An ngp-compatible view of a NODE STK entity meant to represent a spherocylinder
template <typename Base, typename NgpSpherocylinderDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class NgpSpherocylinderEntityView<Base, NgpSpherocylinderDataType> : public Base {
  static_assert(NgpSpherocylinderDataType::topology_t == stk::topology::NODE,
                "The topology of the spherocylinder data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderEntityView(const Base&base, const NgpSpherocylinderDataType &data)
      : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& spherocylinder_index() {
    return spherocylinder_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& spherocylinder_index() const {
    return spherocylinder_index_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return data_access_t::orientation(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return data_access_t::orientation(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() {
    return data_access_t::length(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() const {
    return data_access_t::length(data(), spherocylinder_index());
  }

 private:
  const NgpSpherocylinderDataType &data_;
};  // NgpSpherocylinderEntityView<stk::topology::NODE, NgpSpherocylinderDataType>

/// @brief An ngp-compatible view of a PARTICLE STK entity meant to represent a spherocylinder
template <typename Base,typename NgpSpherocylinderDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class NgpSpherocylinderEntityView<Base, NgpSpherocylinderDataType> : public Base {
  static_assert(NgpSpherocylinderDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the spherocylinder data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderEntityView(const Base&base, const NgpSpherocylinderDataType &data)
      : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& spherocylinder_index() {
    return spherocylinder_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& spherocylinder_index() const {
    return spherocylinder_index_;
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

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return data_access_t::orientation(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return data_access_t::orientation(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() {
    return data_access_t::length(data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() const {
    return data_access_t::length(data(), spherocylinder_index());
  }

 private:
  const NgpSpherocylinderDataType &data_;
};  // NgpSpherocylinderEntityView<stk::topology::PARTICLE, NgpSpherocylinderDataType>
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERENTITYVIEW_HPP_