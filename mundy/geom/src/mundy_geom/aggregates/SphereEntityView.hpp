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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEREENTITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEREENTITYVIEW_HPP_

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
#include <mundy_geom/primitives/Sphere.hpp>  // for mundy::geom::ValidSphereType
#include <mundy_mesh/BulkData.hpp>           // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>         // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// @brief A view of an STK entity meant to represent a sphere
///
/// We type specialize this class based on the valid set of topologies for a sphere entity.
template <typename Base, ValidSphereDataType SphereDataType>
class SphereEntityView;

/// @brief A view of a NODE STK entity meant to represent a sphere
template <typename Base, ValidSphereDataType SphereDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class SphereEntityView<Base, SphereDataType> : public Base {
  static_assert(SphereDataType::topology_t == stk::topology::NODE,
                "The topology of the sphere data must match the view");

 public:
  using scalar_t = typename SphereDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  SphereEntityView(const Base& base, const SphereDataType& data) : Base(base), data_(data) {
  }

  stk::mesh::Entity sphere_entity() const {
    return Base::entity();
  }

  decltype(auto) radius() {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<SphereDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return stk::mesh::field_data(data_.radius_data(), sphere_entity())[0];
    }
  }

  decltype(auto) radius() const {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<SphereDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return stk::mesh::field_data(data_.radius_data(), sphere_entity())[0];
    }
  }

  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), sphere_entity());
  }

  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), sphere_entity());
  }

 private:
  const SphereDataType& data_;
};  // SphereEntityView<stk::topology::NODE, SphereDataType>

/// @brief A view of a PARTICLE STK entity meant to represent a sphere
template <typename Base, ValidSphereDataType SphereDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class SphereEntityView<Base, SphereDataType> : public Base {
  static_assert(SphereDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the sphere data must match the view");

 public:
  using scalar_t = typename SphereDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  SphereEntityView(const Base& base, const SphereDataType& data) : Base(base), data_(data) {
  }

  stk::mesh::Entity sphere_entity() const {
    return Base::entity();
  }

  stk::mesh::Entity center_node_entity() const {
    return Base::connected_node(0);
  }

  decltype(auto) radius() {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<SphereDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return stk::mesh::field_data(data_.radius_data(), sphere_entity())[0];
    }
  }

  decltype(auto) radius() const {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<SphereDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return stk::mesh::field_data(data_.radius_data(), sphere_entity())[0];
    }
  }

  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), center_node_entity());
  }

  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), center_node_entity());
  }

 private:
  const SphereDataType& data_;
};  // SphereEntityView<stk::topology::PARTICLE, SphereDataType>

/// @brief An ngp-compatible view of a STK entity meant to represent a sphere
/// See the discussion for SphereEntityView for more information. The only difference is ngp-compatible data access.
template <typename Base, ValidNgpSphereDataType NgpSphereDataType>
class NgpSphereEntityView;

/// @brief An ngp-compatible view of a NODE STK entity meant to represent a sphere
template <typename Base, ValidNgpSphereDataType NgpSphereDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class NgpSphereEntityView<Base, NgpSphereDataType> : public Base {
  static_assert(NgpSphereDataType::topology_t == stk::topology::NODE,
                "The topology of the sphere data must match the view");

 public:
  using scalar_t = typename NgpSphereDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSphereEntityView(const Base& base, const NgpSphereDataType& data) : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex sphere_index() const {
    return Base::entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSphereDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return data_.radius_data()(sphere_index(), 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSphereDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return data_.radius_data()(sphere_index(), 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), sphere_index());
  }

 private:
  const NgpSphereDataType& data_;
};  // NgpSphereEntityView<stk::topology::NODE, NgpSphereDataType>

/// @brief An ngp-compatible view of a PARTICLE STK entity meant to represent a sphere
template <typename Base, ValidNgpSphereDataType NgpSphereDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class NgpSphereEntityView<Base, NgpSphereDataType> : public Base {
  static_assert(NgpSphereDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the sphere data must match the view");

 public:
  using scalar_t = typename NgpSphereDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSphereEntityView(const Base& base, const NgpSphereDataType& data) : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex sphere_index() const {
    return Base::entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex center_node_index() const {
    return Base::connected_node_index(0);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSphereDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return data_.radius_data()(sphere_index(), 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    constexpr bool has_shared_radius =
        std::is_same_v<std::decay_t<decltype(std::declval<NgpSphereDataType>().radius_data())>, scalar_t>;
    if constexpr (has_shared_radius) {
      return data_.radius_data();
    } else {
      return data_.radius_data()(sphere_index(), 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), center_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), center_node_index());
  }

 private:
  const NgpSphereDataType& data_;
};  // NgpSphereEntityView<stk::topology::PARTICLE, NgpSphereDataType>
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEREENTITYVIEW_HPP_