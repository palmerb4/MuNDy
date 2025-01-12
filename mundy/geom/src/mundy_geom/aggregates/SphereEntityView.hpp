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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEREDATA_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEREDATA_HPP_

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

/// \brief A traits class to provide abstracted access to a sphere's data via an aggregate
///
/// By default, this class is compatible with SphereData or any class the meets the ValidSphereDataType concept.
/// Users can specialize this class to support other aggregate types.
template <typename Agg>
struct SphereDataTraits {
  static_assert(ValidSphereDataType<Agg>,
                "Agg must satisfy the ValidSphereDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSphereData but is free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = decltype(std::declval<Agg>().center_data());
  using radius_data_t = decltype(std::declval<Agg>().radius_data());
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity sphere_node) {
    return mundy::mesh::vector3_field_data(agg.center_data(), sphere_node);
  }

  static decltype(auto) radius(Agg agg, stk::mesh::Entity sphere) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return stk::mesh::field_data(agg.radius_data(), sphere)[0];
    }
  }
};  // SphereDataTraits

/// \brief A traits class to provide abstracted access to a sphere's data via an NGP-compatible aggregate
/// See the discussion for SphereDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpSphereDataTraits {
  static_assert(ValidNgpSphereDataType<Agg>,
                "Agg must satisfy the ValidNgpSphereDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSphereData but is free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<decltype(std::declval<Agg>().radius_data())>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex sphere_node_index) {
    return mundy::mesh::vector3_field_data(agg.center_data(), sphere_node_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) radius(Agg agg, stk::mesh::FastMeshIndex sphere_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return agg.radius_data()(sphere_index, 0);
    }
  }
};  // NgpSphereDataTraits

/// @brief A view of an STK entity meant to represent a sphere
///
/// We type specialize this class based on the valid set of topologies for a sphere entity.
///
/// Use \ref create_sphere_entity_view to build a SphereEntityView object with automatic template deduction.
template <stk::topology::topology_t OurTopology, typename SphereDataType>
class SphereEntityView;

/// @brief A view of a NODE STK entity meant to represent a sphere
template <typename SphereDataType>
class SphereEntityView<stk::topology::NODE, SphereDataType> {
  static_assert(SphereDataType::topology_t == stk::topology::NODE,
                "The topology of the sphere data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  SphereEntityView(const Base&base, const SphereDataType &data) : Base(base), data_(data) {
  }

  stk::mesh::Entity& sphere_entity() {
    return sphere_;
  }

  const stk::mesh::Entity& sphere_entity() const {
    return sphere_;
  }

  decltype(auto) radius() {
    return data_access_t::radius(data(), sphere_entity());
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data(), sphere_entity());
  }

  decltype(auto) center() {
    return data_access_t::center(data(), sphere_entity());
  }

  decltype(auto) center() const {
    return data_access_t::center(data(), sphere_entity());
  }

 private:
  const SphereDataType &data_;
};  // SphereEntityView<stk::topology::NODE, SphereDataType>

/// @brief A view of a PARTICLE STK entity meant to represent a sphere
template <typename SphereDataType>
class SphereEntityView<stk::topology::PARTICLE, SphereDataType> {
  static_assert(SphereDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the sphere data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  SphereEntityView(const Base&base, const SphereDataType &data)
      : Base(base), data_(data){

      }

  stk::mesh::Entity& sphere_entity() {
    return sphere_;
  }

  const stk::mesh::Entity& sphere_entity() const {
    return sphere_;
  }

  stk::mesh::Entity& node_entity() {
    return node_;
  }

  const stk::mesh::Entity& node_entity() const {
    return node_;
  }

  decltype(auto) radius() {
    // For those not familiar with decltype(auto), it allows us to return either an auto or an auto&.
    return data_access_t::radius(data(), sphere_entity());
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data(), sphere_entity());
  }

  decltype(auto) center() {
    return data_access_t::center(data(), node_entity());
  }

  decltype(auto) center() const {
    return data_access_t::center(data(), node_entity());
  }

 private:
  const SphereDataType &data_;
};  // SphereEntityView<stk::topology::PARTICLE, SphereDataType>

/// @brief An ngp-compatible view of a STK entity meant to represent a sphere
/// See the discussion for SphereEntityView for more information. The only difference is ngp-compatible data access.
template <stk::topology::topology_t OurTopology, typename NgpSphereDataType>
class NgpSphereEntityView;

/// @brief An ngp-compatible view of a NODE STK entity meant to represent a sphere
template <typename NgpSphereDataType>
class NgpSphereEntityView<stk::topology::NODE, NgpSphereDataType> {
  static_assert(NgpSphereDataType::topology_t == stk::topology::NODE,
                "The topology of the sphere data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSphereEntityView(const Base&base, const NgpSphereDataType &data)
      : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& sphere_index() {
    return sphere_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& sphere_index() const {
    return sphere_index_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data(), sphere_index());
  }

 private:
  const NgpSphereDataType &data_;
};  // NgpSphereEntityView<stk::topology::NODE, NgpSphereDataType>

/// @brief An ngp-compatible view of a PARTICLE STK entity meant to represent a sphere
template <typename NgpSphereDataType>
class NgpSphereEntityView<stk::topology::PARTICLE, NgpSphereDataType> {
  static_assert(NgpSphereDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the sphere data must match the view");

 public:
  using scalar_t = typename data_access_t::scalar_t;
  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSphereEntityView(const Base&base, const NgpSphereDataType &data)
      : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& sphere_index() {
    return sphere_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& sphere_index() const {
    return sphere_index_;
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
  decltype(auto) radius() {
    return data_access_t::radius(data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data(), node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data(), node_index());
  }

 private:
  const NgpSphereDataType &data_;
};  // NgpSphereEntityView<stk::topology::PARTICLE, NgpSphereDataType>
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEREDATA_HPP_