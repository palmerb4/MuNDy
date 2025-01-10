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

#ifndef MUNDY_GEOM_AGGREGATES_AABB_HPP_
#define MUNDY_GEOM_AGGREGATES_AABB_HPP_

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

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of aabbs
///
/// The rank of an AABB does not change the access pattern for the underlying data, so we need not store it
/// as a constexpr. It only enforces a single check that the aabb data is of the same rank as the aabb.
///
/// \note Typically, aabb data will be augmented on top of other data, such as adding aabb data to a collection of
/// spheres. In and of itself, the aabb data isn't very useful.
///
/// \tparam Scalar The scalar type of the aabb's aabb.
template <typename Scalar>
class AABBData {
 public:
  using scalar_t = Scalar;
  using aabb_data_t = stk::mesh::Field<scalar_t>;

  AABBData(stk::mesh::BulkData& bulk_data, aabb_data_t& aabb_data) : bulk_data_(bulk_data), aabb_data_(aabb_data) {
  }

  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }

  stk::mesh::BulkData& bulk_data() {
    return bulk_data_;
  }

  const aabb_data_t& aabb_data() const {
    return aabb_data_;
  }

  aabb_data_t& aabb_data() {
    return aabb_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<AABBData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  stk::mesh::BulkData& bulk_data_;
  aabb_data_t& aabb_data_;
};  // AABBData

/// \brief A struct to hold the data for a collection of NGP-compatible aabbs
/// See the discussion for AABBData for more information. Only difference is NgpFields over Fields.
template <typename Scalar>
class NgpAABBData {
 public:
  using scalar_t = Scalar;
  using aabb_data_t = stk::mesh::NgpField<scalar_t>;

  NgpAABBData(stk::mesh::NgpMesh ngp_mesh, aabb_data_t& aabb_data) : ngp_mesh_(ngp_mesh), aabb_data_(aabb_data) {
  }

  stk::mesh::NgpMesh &ngp_mesh() {
    return ngp_mesh_;
  }

  const stk::mesh::NgpMesh &ngp_mesh() const {
    return ngp_mesh_;
  }


  const aabb_data_t& aabb_data() const {
    return aabb_data_;
  }

  aabb_data_t& aabb_data() {
    return aabb_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<NgpAABBData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  stk::mesh::NgpMesh &ngp_mesh_;
  aabb_data_t& aabb_data_;
};  // NgpAABBData

/// \brief A helper function to create an AABBData object
///
/// Only provided to keep the interface consistent with other aggregate types.
/// The aabb data doesn't need automatic deduction.
template <typename Scalar>  // Must be provided
auto create_aabb_data(stk::mesh::BulkData& bulk_data, stk::mesh::Field<Scalar>& aabb_data) {
  return AABBData<Scalar>{bulk_data, aabb_data};
}

/// \brief A helper function to create a NgpAABBData object
/// See the discussion for create_aabb_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar>  // Must be provided
auto create_ngp_aabb_data(stk::mesh::NgpMesh ngp_mesh, stk::mesh::NgpField<Scalar>& aabb_data) {
  return NgpAABBData<Scalar>{ngp_mesh, aabb_data};
}

/// \brief A concept to check if a type provides the same data as AABBData
template <typename Agg>
concept ValidAABBDataType =
    requires(Agg agg) { typename Agg::scalar_t; } &&
    std::convertible_to<decltype(std::declval<Agg>().bulk_data()), stk::mesh::BulkData&> &&
    std::convertible_to<decltype(std::declval<Agg>().aabb_data()), stk::mesh::Field<typename Agg::scalar_t>&>;

/// \brief A concept to check if a type provides the same data as NgpAABBData
template <typename Agg>
concept ValidNgpAABBDataType =
    requires(Agg agg) { typename Agg::scalar_t; } &&
    std::convertible_to<decltype(std::declval<Agg>().ngp_mesh()), stk::mesh::NgpMesh&> &&
    std::convertible_to<decltype(std::declval<Agg>().aabb_data()), stk::mesh::NgpField<typename Agg::scalar_t>&>;

/// \brief A helper function to get an updated NgpAABBData object from a AABBData object
/// \param data The AABBData object to convert
template <ValidAABBDataType AABBDataType>
auto get_updated_ngp_data(AABBDataType data) {
  using scalar_t = typename AABBDataType::scalar_t;
  return create_ngp_aabb_data(stk::mesh::get_updated_ngp_mesh(data.bulk_data()),  //
                              stk::mesh::get_updated_ngp_field<scalar_t>(data.aabb_data()));
}

/// \brief A traits class to provide abstracted access to a aabb's data via an aggregate
///
/// By default, this class is compatible with AABBData or any class the meets the ValidAABBDataType concept.
/// Users can specialize this class to support other aggregate types.
template <typename Agg>
struct AABBDataTraits {
  static_assert(ValidAABBDataType<Agg>,
                "Agg must satisfy the ValidAABBDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpAABBData but is free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;

  static decltype(auto) min_corner(Agg agg, stk::mesh::Entity aabb_entity) {
    return mundy::math::get_vector3_view<scalar_t>(stk::mesh::field_data(agg.aabb_data(), aabb_entity));
  }

  static decltype(auto) max_corner(Agg agg, stk::mesh::Entity aabb_entity) {
    constexpr size_t shift = 3;
    auto shifted_data_accessor =
        mundy::math::get_shifted_view<scalar_t, shift>(stk::mesh::field_data(agg.aabb_data(), aabb_entity));
    return mundy::math::get_owning_vector3<scalar_t>(std::move(shifted_data_accessor));
  }

  static decltype(auto) x_min(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data(), aabb_entity)[0];
  }

  static decltype(auto) y_min(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data(), aabb_entity)[1];
  }

  static decltype(auto) z_min(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data(), aabb_entity)[2];
  }

  static decltype(auto) x_max(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data(), aabb_entity)[3];
  }

  static decltype(auto) y_max(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data(), aabb_entity)[4];
  }

  static decltype(auto) z_max(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data(), aabb_entity)[5];
  }
};  // AABBDataTraits

/// \brief A traits class to provide abstracted access to a aabb's data via an NGP-compatible aggregate
/// See the discussion for AABBDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpAABBDataTraits {
  static_assert(ValidNgpAABBDataType<Agg>,
                "Agg must satisfy the ValidNgpAABBDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpAABBData but is free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) min_corner(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return mundy::math::get_owning_vector3<scalar_t>(agg.aabb_data()(aabb_index));
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) max_corner(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    constexpr size_t shift = 3;
    auto shifted_data_accessor = mundy::math::get_owning_shifted_accessor<scalar_t, shift>(agg.aabb_data()(aabb_index));
    return mundy::math::get_owning_vector3<scalar_t>(std::move(shifted_data_accessor));
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) x_min(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data()(aabb_index)[0];
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) y_min(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data()(aabb_index)[1];
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) z_min(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data()(aabb_index)[2];
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) x_max(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data()(aabb_index)[3];
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) y_max(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data()(aabb_index)[4];
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) z_max(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data()(aabb_index)[5];
  }
};  // NgpAABBDataTraits

/// @brief A view of an aabb entity. May take any rank.
template <typename AABBDataType>
class AABBEntityView {
 public:
  using data_access_t = AABBDataTraits<AABBDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::min_corner(std::declval<AABBDataType>(), std::declval<stk::mesh::Entity>()));

  AABBEntityView(AABBDataType data, stk::mesh::Entity aabb_entity) : data_(data), aabb_entity_(aabb_entity) {
  }

  decltype(auto) data() {
    return data_;
  }

  decltype(auto) data() const {
    return data_;
  }

  stk::mesh::Entity& aabb_entity() {
    return aabb_entity_;
  }

  const stk::mesh::Entity& aabb_entity() const {
    return aabb_entity_;
  }

  decltype(auto) min_corner() {
    return data_access_t::min_corner(data(), aabb_entity());
  }

  decltype(auto) min_corner() const {
    return data_access_t::min_corner(data(), aabb_entity());
  }

  decltype(auto) max_corner() {
    return data_access_t::max_corner(data(), aabb_entity());
  }

  decltype(auto) max_corner() const {
    return data_access_t::max_corner(data(), aabb_entity());
  }

  decltype(auto) x_min() {
    return data_access_t::x_min(data(), aabb_entity());
  }

  decltype(auto) x_min() const {
    return data_access_t::x_min(data(), aabb_entity());
  }

  decltype(auto) y_min() {
    return data_access_t::y_min(data(), aabb_entity());
  }

  decltype(auto) y_min() const {
    return data_access_t::y_min(data(), aabb_entity());
  }

  decltype(auto) z_min() {
    return data_access_t::z_min(data(), aabb_entity());
  }

  decltype(auto) z_min() const {
    return data_access_t::z_min(data(), aabb_entity());
  }

  decltype(auto) x_max() {
    return data_access_t::x_max(data(), aabb_entity());
  }

  decltype(auto) x_max() const {
    return data_access_t::x_max(data(), aabb_entity());
  }

  decltype(auto) y_max() {
    return data_access_t::y_max(data(), aabb_entity());
  }

  decltype(auto) y_max() const {
    return data_access_t::y_max(data(), aabb_entity());
  }

  decltype(auto) z_max() {
    return data_access_t::z_max(data(), aabb_entity());
  }

  decltype(auto) z_max() const {
    return data_access_t::z_max(data(), aabb_entity());
  }

 private:
  AABBDataType data_;
  stk::mesh::Entity aabb_entity_;
};  // AABBEntityView

/// @brief An ngp-compatible view of an STK entity meant to represent a aabb
/// See the discussion for AABBEntityView for more information. The only difference is ngp-compatible data access.
template <typename NgpAABBDataType>
class NgpAABBEntityView {
 public:
  using data_access_t = NgpAABBDataTraits<NgpAABBDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::min_corner(std::declval<NgpAABBDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpAABBEntityView(NgpAABBDataType data, stk::mesh::FastMeshIndex aabb_index) : data_(data), aabb_index_(aabb_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) data() {
    return data_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) data() const {
    return data_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& aabb_index() {
    return aabb_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& aabb_index() const {
    return aabb_index_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) min_corner() {
    return data_access_t::min_corner(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) min_corner() const {
    return data_access_t::min_corner(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) max_corner() {
    return data_access_t::max_corner(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) max_corner() const {
    return data_access_t::max_corner(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_min() {
    return data_access_t::x_min(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_min() const {
    return data_access_t::x_min(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_min() {
    return data_access_t::y_min(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_min() const {
    return data_access_t::y_min(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_min() {
    return data_access_t::z_min(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_min() const {
    return data_access_t::z_min(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_max() {
    return data_access_t::x_max(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_max() const {
    return data_access_t::x_max(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_max() {
    return data_access_t::y_max(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_max() const {
    return data_access_t::y_max(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_max() {
    return data_access_t::z_max(data(), aabb_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_max() const {
    return data_access_t::z_max(data(), aabb_index());
  }

 private:
  NgpAABBDataType data_;
  stk::mesh::FastMeshIndex aabb_index_;
};  // NgpAABBEntityView

/// \brief A helper function to create a AABBEntityView object with type deduction
template <typename AABBDataType>
auto create_aabb_entity_view(AABBDataType& data, stk::mesh::Entity aabb_entity) {
  return AABBEntityView<AABBDataType>(data, aabb_entity);
}

/// \brief A helper function to create a NgpAABBEntityView object with type deduction
template <typename NgpAABBDataType>
auto create_ngp_aabb_entity_view(NgpAABBDataType data, stk::mesh::FastMeshIndex aabb_index) {
  return NgpAABBEntityView<NgpAABBDataType>(data, aabb_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_AABB_HPP_