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

/// @brief A view of an STK entity meant to represent a spherocylinder
///
/// We type specialize this class based on the valid set of topologies for a spherocylinder entity.
///
/// Use \ref create_spherocylinder_entity_view to build an SpherocylinderEntityView object with automatic template
/// deduction.
template <typename Base, ValidSpherocylinderDataType SpherocylinderDataType>
class SpherocylinderEntityView;

/// @brief A view of a NODE STK entity meant to represent a spherocylinder
template <typename Base, ValidSpherocylinderDataType SpherocylinderDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class SpherocylinderEntityView<Base, SpherocylinderDataType> : public Base {
  static_assert(SpherocylinderDataType::topology_t == stk::topology::NODE,
                "The topology of the spherocylinder data must match the view");

 public:
  using scalar_t = typename SpherocylinderDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  SpherocylinderEntityView(const Base&base, const SpherocylinderDataType &data)
      : Base(base), data_(data) {
  }

  const stk::mesh::Entity& spherocylinder_entity() const {
    return Base::entity();
  }

  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), spherocylinder_entity());
  }

  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), spherocylinder_entity());
  }

  decltype(auto) orientation() {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), spherocylinder_entity());
  }

  decltype(auto) orientation() const {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), spherocylinder_entity());
  }

  decltype(auto) radius() {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<SpherocylinderDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return stk::mesh::field_data(data_.radius_data(), spherocylinder_entity())[0];
    }
  }

  decltype(auto) radius() const {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<SpherocylinderDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return stk::mesh::field_data(data_.radius_data(), spherocylinder_entity())[0];
    }
  }

  decltype(auto) length() {
    constexpr bool has_shared_length =
        std::is_same_v<std::decay_t<decltype(std::declval<SpherocylinderDataType>().length_data())>, scalar_t>;
    if constexpr (has_shared_length) {
      return data_.length_data();
    } else {
      return stk::mesh::field_data(data_.length_data(), spherocylinder_entity())[0];
    }
  }

  decltype(auto) length() const {
    constexpr bool has_shared_length =
        std::is_same_v<std::decay_t<decltype(std::declval<SpherocylinderDataType>().length_data())>, scalar_t>;
    if constexpr (has_shared_length) {
      return data_.length_data();
    } else {
      return stk::mesh::field_data(data_.length_data(), spherocylinder_entity())[0];
    }
  }

 private:
  const SpherocylinderDataType &data_;
};  // SpherocylinderEntityView<stk::topology::NODE, SpherocylinderDataType>

/// @brief A view of a PARTICLE STK entity meant to represent a spherocylinder
template <typename Base, ValidSpherocylinderDataType SpherocylinderDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class SpherocylinderEntityView<Base, SpherocylinderDataType> : public Base {
  static_assert(SpherocylinderDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the spherocylinder data must match the view");

 public:
  using scalar_t = typename SpherocylinderDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  SpherocylinderEntityView(const Base&base, const SpherocylinderDataType &data)
      : Base(base), data_(data) {
  }

  const stk::mesh::Entity& spherocylinder_entity() const {
    return Base::entity();
  }

  const stk::mesh::Entity& center_node_entity() const {
    return Base::connected_node(0);
  }

  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), spherocylinder_entity());
  }

  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), spherocylinder_entity());
  }

  decltype(auto) orientation() {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), spherocylinder_entity());
  }

  decltype(auto) orientation() const {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), spherocylinder_entity());
  }

  decltype(auto) radius() {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<SpherocylinderDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return stk::mesh::field_data(data_.radius_data(), spherocylinder_entity())[0];
    }
  }

  decltype(auto) radius() const {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<SpherocylinderDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return stk::mesh::field_data(data_.radius_data(), spherocylinder_entity())[0];
    }
  }

  decltype(auto) length() {
    constexpr bool has_shared_length =
        std::is_same_v<std::decay_t<decltype(std::declval<SpherocylinderDataType>().length_data())>, scalar_t>;
    if constexpr (has_shared_length) {
      return data_.length_data();
    } else {
      return stk::mesh::field_data(data_.length_data(), spherocylinder_entity())[0];
    }
  }

  decltype(auto) length() const {
    constexpr bool has_shared_length =
        std::is_same_v<std::decay_t<decltype(std::declval<SpherocylinderDataType>().length_data())>, scalar_t>;
    if constexpr (has_shared_length) {
      return data_.length_data();
    } else {
      return stk::mesh::field_data(data_.length_data(), spherocylinder_entity())[0];
    }
  }

 private:
  const SpherocylinderDataType &data_;
};  // SpherocylinderEntityView<stk::topology::PARTICLE, SpherocylinderDataType>

/// @brief An ngp-compatible view of an STK entity meant to represent a spherocylinder
/// See the discussion for SpherocylinderEntityView for more information. The only difference is ngp-compatible data
/// access.
template <typename Base, ValidNgpSpherocylinderDataType NgpSpherocylinderDataType>
class NgpSpherocylinderEntityView;

/// @brief An ngp-compatible view of a NODE STK entity meant to represent a spherocylinder
template <typename Base, ValidNgpSpherocylinderDataType NgpSpherocylinderDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class NgpSpherocylinderEntityView<Base, NgpSpherocylinderDataType> : public Base {
  static_assert(NgpSpherocylinderDataType::topology_t == stk::topology::NODE,
                "The topology of the spherocylinder data must match the view");

 public:
  using scalar_t = typename NgpSpherocylinderDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderEntityView(const Base&base, const NgpSpherocylinderDataType &data)
      : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& spherocylinder_index() const {
    return Base::entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSpherocylinderDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return data_.radius_data()(spherocylinder_index(), 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSpherocylinderDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return data_.radius_data()(spherocylinder_index(), 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() {
    constexpr bool has_shared_length =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSpherocylinderDataType>().length_data())>, scalar_t>;
    if constexpr (has_shared_length) {
      return data_.length_data();
    } else {
      return data_.length_data()(spherocylinder_index(), 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() const {
    constexpr bool has_shared_length =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSpherocylinderDataType>().length_data())>, scalar_t>;
    if constexpr (has_shared_length) {
      return data_.length_data();
    } else {
      return data_.length_data()(spherocylinder_index(), 0);
    }
  }

 private:
  const NgpSpherocylinderDataType &data_;
};  // NgpSpherocylinderEntityView<stk::topology::NODE, NgpSpherocylinderDataType>

/// @brief An ngp-compatible view of a PARTICLE STK entity meant to represent a spherocylinder
template <typename Base, ValidNgpSpherocylinderDataType NgpSpherocylinderDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class NgpSpherocylinderEntityView<Base, NgpSpherocylinderDataType> : public Base {
  static_assert(NgpSpherocylinderDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the spherocylinder data must match the view");

 public:
  using scalar_t = typename NgpSpherocylinderDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderEntityView(const Base&base, const NgpSpherocylinderDataType &data)
      : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& spherocylinder_index() const {
    return Base::entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& center_node_index() const {
    return Base::connected_node_index(0);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), center_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), center_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), spherocylinder_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSpherocylinderDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return data_.radius_data()(spherocylinder_index(), 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSpherocylinderDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return data_.radius_data()(spherocylinder_index(), 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() {
    constexpr bool has_shared_length =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSpherocylinderDataType>().length_data())>, scalar_t>;
    if constexpr (has_shared_length) {
      return data_.length_data();
    } else {
      return data_.length_data()(spherocylinder_index(), 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) length() const {
    constexpr bool has_shared_length =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSpherocylinderDataType>().length_data())>, scalar_t>;
    if constexpr (has_shared_length) {
      return data_.length_data();
    } else {
      return data_.length_data()(spherocylinder_index(), 0);
    }
  }

 private:
  const NgpSpherocylinderDataType &data_;
};  // NgpSpherocylinderEntityView<stk::topology::PARTICLE, NgpSpherocylinderDataType>
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERENTITYVIEW_HPP_