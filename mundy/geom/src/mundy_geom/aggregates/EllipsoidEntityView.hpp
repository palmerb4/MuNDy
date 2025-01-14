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

#ifndef MUNDY_GEOM_AGGREGATES_ELLIPSOIDENTITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_ELLIPSOIDENTITYVIEW_HPP_

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
#include <mundy_geom/aggregates/EllipsoidDataConcepts.hpp>  // for mundy::geom::ValidEllipsoidDataType
#include <mundy_geom/aggregates/EntityView.hpp>  // for mundy::geom::EntityView and mundy::geom::create_topological_entity_view
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// @brief A view of an STK entity meant to represent a ellipsoid
///
/// We type specialize this class based on the valid set of topologies for an ellipsoid entity.
///
/// Use \ref create_ellipsoid_entity_view to build an EllipsoidEntityView object with automatic template deduction.
template <typename Base, ValidEllipsoidDataType EllipsoidDataType>
class EllipsoidEntityView;

/// @brief A view of an STK entity meant to represent a ellipsoid
template <typename Base, ValidEllipsoidDataType EllipsoidDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class EllipsoidEntityView<Base, EllipsoidDataType> : public Base {
  static_assert(EllipsoidDataType::topology_t == stk::topology::NODE,
                "The topology of the ellipsoid data must match the view");

 public:
  using scalar_t = typename EllipsoidDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  EllipsoidEntityView(const Base& base, const EllipsoidDataType& data) : Base(base), data_(data) {
  }

  stk::mesh::Entity ellipsoid_entity() const {
    return Base::entity();
  }

  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), ellipsoid_entity());
  }

  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), ellipsoid_entity());
  }

  decltype(auto) orientation() {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_entity());
  }

  decltype(auto) orientation() const {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_entity());
  }

  decltype(auto) axis_lengths() {
    constexpr bool has_shared_axis_lengths =
        mundy::math::is_vector3_v<std::decay_t<decltype(std::declval<EllipsoidDataType>().axis_lengths_data())>>;
    if constexpr (has_shared_axis_lengths) {
      return data_.axis_lengths_data();
    } else {
      return mundy::mesh::vector3_field_data(data_.axis_lengths_data(), ellipsoid_entity());
    }
  }

  decltype(auto) axis_lengths() const {
    constexpr bool has_shared_axis_lengths =
        mundy::math::is_vector3_v<std::decay_t<decltype(std::declval<EllipsoidDataType>().axis_lengths_data())>>;
    if constexpr (has_shared_axis_lengths) {
      return data_.axis_lengths_data();
    } else {
      return mundy::mesh::vector3_field_data(data_.axis_lengths_data(), ellipsoid_entity());
    }
  }

  /// \brief Chainable function to add augments to this entity view
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto augment_view(Args&&... args) const {
    return NextAugment<EllipsoidEntityView, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  const EllipsoidDataType& data_;
};  // EllipsoidEntityView for NODEs

template <typename Base, ValidEllipsoidDataType EllipsoidDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class EllipsoidEntityView<Base, EllipsoidDataType> : public Base {
  static_assert(EllipsoidDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the ellipsoid data must match the view");

 public:
  using scalar_t = typename EllipsoidDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;
  static constexpr bool has_shared_axis_lengths =
      mundy::math::is_vector3_v<std::decay_t<decltype(std::declval<EllipsoidDataType>().axis_lengths_data())>>;

  EllipsoidEntityView(const Base& base, const EllipsoidDataType& data) : Base(base), data_(data) {
  }

  stk::mesh::Entity ellipsoid_entity() const {
    return Base::entity();
  }

  stk::mesh::Entity center_node_entity() const {
    return Base::connected_node(0);
  }

  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), center_node_entity());
  }

  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), center_node_entity());
  }

  decltype(auto) orientation() {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_entity());
  }

  decltype(auto) orientation() const {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_entity());
  }

  decltype(auto) axis_lengths() {
    if constexpr (has_shared_axis_lengths) {
      return data_.axis_lengths_data();
    } else {
      return mundy::mesh::vector3_field_data(data_.axis_lengths_data(), ellipsoid_entity());
    }
  }

  decltype(auto) axis_lengths() const {
    if constexpr (has_shared_axis_lengths) {
      return data_.axis_lengths_data();
    } else {
      return mundy::mesh::vector3_field_data(data_.axis_lengths_data(), ellipsoid_entity());
    }
  }

  /// \brief Chainable function to add augments to this entity view
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto augment_view(Args&&... args) const {
    return NextAugment<EllipsoidEntityView, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  const EllipsoidDataType& data_;
};  // EllipsoidEntityView for PARTICLEs

/// @brief An ngp-compatible view of an ellipsoid entity
template <typename Base, ValidNgpEllipsoidDataType NgpEllipsoidDataType>
class NgpEllipsoidEntityView;

template <typename Base, ValidNgpEllipsoidDataType NgpEllipsoidDataType>
  requires(Base::get_topology() == stk::topology::NODE)
class NgpEllipsoidEntityView<Base, NgpEllipsoidDataType> : public Base {
  static_assert(NgpEllipsoidDataType::topology_t == stk::topology::NODE,
                "The topology of the ellipsoid data must match the view");

 public:
  using scalar_t = typename NgpEllipsoidDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpEllipsoidEntityView(const Base& base, const NgpEllipsoidDataType& data) : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex ellipsoid_index() const {
    return Base::entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return mundy::mesh::vector3_field_data(data_.center_data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return mundy::mesh::vector3_field_data(data_.center_data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() {
    constexpr bool has_shared_axis_lengths =
        mundy::math::is_vector3_v<std::decay_t<decltype(std::declval<NgpEllipsoidDataType>().axis_lengths_data())>>;

    if constexpr (has_shared_axis_lengths) {
      return data_.axis_lengths_data();
    } else {
      return mundy::mesh::vector3_field_data(data_.axis_lengths_data(), ellipsoid_index());
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() const {
    constexpr bool has_shared_axis_lengths =
        mundy::math::is_vector3_v<std::decay_t<decltype(std::declval<NgpEllipsoidDataType>().axis_lengths_data())>>;

    if constexpr (has_shared_axis_lengths) {
      return data_.axis_lengths_data();
    } else {
      return mundy::mesh::vector3_field_data(data_.axis_lengths_data(), ellipsoid_index());
    }
  }

  /// \brief Chainable function to add augments to this entity view
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION auto augment_view(Args&&... args) const {
    return NextAugment<NgpEllipsoidEntityView, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  const NgpEllipsoidDataType& data_;
};  // NgpEllipsoidEntityView for NODEs

/// @brief An ngp-compatible view of an STK entity meant to represent a ellipsoid
template <typename Base, ValidNgpEllipsoidDataType NgpEllipsoidDataType>
  requires(Base::get_topology() == stk::topology::PARTICLE)
class NgpEllipsoidEntityView<Base, NgpEllipsoidDataType> : public Base {
  static_assert(NgpEllipsoidDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the ellipsoid data must match the view");

 public:
  using scalar_t = typename NgpEllipsoidDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpEllipsoidEntityView(const Base& base, const NgpEllipsoidDataType& data) : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex ellipsoid_index() const {
    return Base::entity_index();
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex center_node_index() const {
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
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() {
    constexpr bool has_shared_axis_lengths =
        mundy::math::is_vector3_v<std::decay_t<decltype(std::declval<NgpEllipsoidDataType>().axis_lengths_data())>>;

    if constexpr (has_shared_axis_lengths) {
      return data_.axis_lengths_data();
    } else {
      return mundy::mesh::vector3_field_data(data_.axis_lengths_data(), ellipsoid_index());
    }
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() const {
    constexpr bool has_shared_axis_lengths =
        mundy::math::is_vector3_v<std::decay_t<decltype(std::declval<NgpEllipsoidDataType>().axis_lengths_data())>>;

    if constexpr (has_shared_axis_lengths) {
      return data_.axis_lengths_data();
    } else {
      return mundy::mesh::vector3_field_data(data_.axis_lengths_data(), ellipsoid_index());
    }
  }

  /// \brief Chainable function to add augments to this entity view
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION auto augment_view(Args&&... args) const {
    return NextAugment<NgpEllipsoidEntityView, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  const NgpEllipsoidDataType& data_;
};  // NgpEllipsoidEntityView for PARTICLEs

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_ELLIPSOIDENTITYVIEW_HPP_