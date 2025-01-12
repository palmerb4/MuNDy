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

#ifndef MUNDY_GEOM_AUGMENTS_AABBDATA_HPP_
#define MUNDY_GEOM_AUGMENTS_AABBDATA_HPP_

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
#include <mundy_geom/primitives/AABB.hpp>  // for mundy::geom::ValidAABBType
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// @brief A view of an aabb entity. May take any rank.
template <typename Base, ValidAABBDataType AABBDataType>
class AABBEntityView : public Base {
 public:
  using scalar_t = typename AABBDataType::scalar_t;

  AABBEntityView(const Base &base, const AABBDataType &data) : Base(base), data_(data) {
  }

  decltype(auto) min_corner() {
    return mundy::math::get_vector3_view<scalar_t>(stk::mesh::field_data(data_.aabb_data(), Base::entity()));
  }

  decltype(auto) min_corner() const {
    return mundy::math::get_vector3_view<scalar_t>(stk::mesh::field_data(data_.aabb_data(), Base::entity()));
  }

  decltype(auto) max_corner() {
    constexpr size_t shift = 3;
    auto shifted_data_accessor =
        mundy::math::get_shifted_view<scalar_t, shift>(stk::mesh::field_data(data_.aabb_data(), Base::entity()));
    return mundy::math::get_owning_vector3<scalar_t>(std::move(shifted_data_accessor));
  }

  decltype(auto) max_corner() const {
    constexpr size_t shift = 3;
    auto shifted_data_accessor =
        mundy::math::get_shifted_view<scalar_t, shift>(stk::mesh::field_data(data_.aabb_data(), Base::entity()));
    return mundy::math::get_owning_vector3<scalar_t>(std::move(shifted_data_accessor));
  }

  decltype(auto) x_min() {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[0];
  }

  decltype(auto) x_min() const {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[0];
  }

  decltype(auto) y_min() {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[1];
  }

  decltype(auto) y_min() const {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[1];
  }

  decltype(auto) z_min() {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[2];
  }

  decltype(auto) z_min() const {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[2];
  }

  decltype(auto) x_max() {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[3];
  }

  decltype(auto) x_max() const {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[3];
  }

  decltype(auto) y_max() {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[4];
  }

  decltype(auto) y_max() const {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[4];
  }

  decltype(auto) z_max() {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[5];
  }

  decltype(auto) z_max() const {
    return stk::mesh::field_data(data_.aabb_data(), Base::entity())[5];
  }

 private:
  const AABBDataType &data_;
};  // AABBEntityView

/// @brief An ngp-compatible view of an STK entity meant to represent a aabb
/// See the discussion for AABBEntityView for more information. The only difference is ngp-compatible data access.
template <typename Base, ValidNgpAABBDataType NgpAABBDataType>
class NgpAABBEntityView : public Base {
 public:
  using scalar_t = typename AABBDataType::scalar_t;

  KOKKOS_INLINE_FUNCTION
  NgpAABBEntityView(const Base& base, const NgpAABBDataType &data) : Base(base), data_(data) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) min_corner() {
    return mundy::math::get_owning_vector3<scalar_t>(data_.aabb_data()(Base::entity_index()));
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) min_corner() const {
    return mundy::math::get_owning_vector3<scalar_t>(data_.aabb_data()(Base::entity_index()));
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) max_corner() {
    constexpr size_t shift = 3;
    auto shifted_data_accessor = mundy::math::get_owning_shifted_accessor<scalar_t, shift>(data_.aabb_data()(Base::entity_index()));
    return mundy::math::get_owning_vector3<scalar_t>(std::move(shifted_data_accessor));
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) max_corner() const {
    constexpr size_t shift = 3;
    auto shifted_data_accessor = mundy::math::get_owning_shifted_accessor<scalar_t, shift>(data_.aabb_data()(Base::entity_index()));
    return mundy::math::get_owning_vector3<scalar_t>(std::move(shifted_data_accessor));
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_min() {
    return data_.aabb_data()(Base::entity_index(), 0);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_min() const {
    return data_.aabb_data()(Base::entity_index(), 0);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_min() {
    return data_.aabb_data()(Base::entity_index(), 1);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_min() const {
    return data_.aabb_data()(Base::entity_index(), 1);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_min() {
    return data_.aabb_data()(Base::entity_index(), 2);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_min() const {
    return data_.aabb_data()(Base::entity_index(), 2);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_max() {
    return data_.aabb_data()(Base::entity_index(), 3);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_max() const {
    return data_.aabb_data()(Base::entity_index(), 3);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_max() {
    return data_.aabb_data()(Base::entity_index(), 4);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_max() const {
    return data_.aabb_data()(Base::entity_index(), 4);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_max() {
    return data_.aabb_data()(Base::entity_index(), 5);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_max() const {
    return data_.aabb_data()(Base::entity_index(), 5);
  }

 private:
  const NgpAABBDataType &data_;
};  // NgpAABBEntityView

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AUGMENTS_AABBDATA_HPP_