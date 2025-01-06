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

/// \brief A struct to hold the data for a collection of aabbs
///
/// The rank of an AABB does not change the access pattern for the underlying data, so we need not store it 
/// as a constexpr. It only enforces a single check that the aabb data is of the same rank as the aabb.
///
/// \note Typically, aabb data will be augmented on top of other data, such as adding aabb data to a collection of
/// spheres. In and of itself, the aabb data isn't very useful.
///
/// \tparam Scalar The scalar type of the aabb's aabb.
/// \tparam AABBDataType The type of the aabb data. Can be a const or non-const stk::mesh::Field of scalars.
template <typename Scalar, typename AABBDataType = stk::mesh::Field<Scalar>>
struct AABBData {
  static_assert(std::is_same_v<std::decay_t<AABBDataType>, stk::mesh::Field<Scalar>>,
                "AABBDataType must be a const or non-const field of scalars");

  using scalar_t = Scalar;
  using aabb_data_t = AABBDataType;

  stk::topology::rank_t aabb_rank;
  aabb_data_t& aabb_data;
};  // AABBData

/// \brief A struct to hold the data for a collection of NGP-compatible aabbs
/// See the discussion for AABBData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename AABBDataType = stk::mesh::NgpField<Scalar>>
struct NgpAABBData {
  static_assert(std::is_same_v<std::decay_t<AABBDataType>, stk::mesh::NgpField<Scalar>>,
                "AABBDataType must be a const or non-const ngp field of scalars");

  using scalar_t = Scalar;
  using aabb_data_t = AABBDataType;

  stk::topology::rank_t aabb_rank;
  aabb_data_t& aabb_data;
};  // NgpAABBData

/// \brief A helper function to create a AABBData object
///
/// This function creates a AABBData object given its rank and data
/// and is used to automatically deduce the template parameters.
template <typename Scalar, typename AABBDataType>
auto create_aabb_data(stk::topology::rank_t aabb_rank, AABBDataType& aabb_data) {
  MUNDY_THROW_ASSERT(aabb_data.entity_rank() == aabb_rank, std::invalid_argument,
                      "The aabb_data must be a field of the same rank as the aabb");
  return AABBData<Scalar, AABBDataType>{aabb_rank, aabb_data};
}

/// \brief A helper function to create a NgpAABBData object
/// See the discussion for create_aabb_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename AABBDataType>
auto create_ngp_aabb_data(stk::topology::rank_t aabb_rank, AABBDataType& aabb_data) {
  MUNDY_THROW_ASSERT(aabb_data.get_rank() == aabb_rank, std::invalid_argument,
                      "The aabb_data must be a field of the same rank as the aabb");
  return NgpAABBData<Scalar, AABBDataType>{aabb_rank, aabb_data};
}

/// \brief A concept to check if a type provides the same data as AABBData
template <typename Agg>
concept ValidDefaultAABBDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::aabb_data_t;
  std::is_same_v<std::decay_t<typename Agg::aabb_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { agg.aabb_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.aabb_data } -> std::convertible_to<typename Agg::aabb_data_t&>;
};  // ValidDefaultAABBDataType

/// \brief A concept to check if a type provides the same data as NgpAABBData
template <typename Agg>
concept ValidDefaultNgpAABBDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::aabb_data_t;
  std::is_same_v<std::decay_t<typename Agg::aabb_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { agg.aabb_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.aabb_data } -> std::convertible_to<typename Agg::aabb_data_t&>;
};  // ValidDefaultNgpAABBDataType

static_assert(ValidDefaultAABBDataType<AABBData<float, stk::mesh::Field<float>>> && ValidDefaultAABBDataType<AABBData<float, const stk::mesh::Field<float>>>,
              "AABBData must satisfy the ValidDefaultAABBDataType concept");

static_assert(
    ValidDefaultNgpAABBDataType<NgpAABBData<float, stk::mesh::NgpField<float>>> &&
        ValidDefaultNgpAABBDataType<NgpAABBData<float, const stk::mesh::NgpField<float>>>,
    "NgpAABBData must satisfy the ValidDefaultNgpAABBDataType concept");

/// \brief A helper function to get an updated NgpAABBData object from a AABBData object
/// \param data The AABBData object to convert
template <ValidDefaultAABBDataType AABBDataType>
auto get_updated_ngp_data(AABBDataType data) {
  using scalar_t = typename AABBDataType::scalar_t;
  using aabb_data_t = typename AABBDataType::aabb_data_t;
  return create_ngp_aabb_data<scalar_t>(data.aabb_rank,  //
                                        stk::mesh::get_updated_ngp_field<scalar_t>(data.aabb_data));
}

/// \brief A traits class to provide abstracted access to a aabb's data via an aggregate
///
/// By default, this class is compatible with AABBData or any class the meets the ValidDefaultAABBDataType concept.
/// Users can specialize this class to support other aggregate types.
template <typename Agg>
struct AABBDataTraits {
  static_assert(
      ValidDefaultAABBDataType<Agg>,
      "Agg must satisfy the ValidDefaultAABBDataType concept.\n"
      "Basically, Agg must have all the same things as NgpAABBData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using aabb_data_t = typename Agg::aabb_data_t;

  static decltype(auto) min_corner(Agg agg, stk::mesh::Entity aabb_entity) {
    return mundy::math::get_vector3_view<scalar_t>(stk::mesh::field_data(agg.aabb_data, aabb_entity));
  }

  static decltype(auto) max_corner(Agg agg, stk::mesh::Entity aabb_entity) {
    constexpr size_t shift = 3;
    auto shifted_data_accessor =
        mundy::math::get_shifted_view<scalar_t, shift>(stk::mesh::field_data(agg.aabb_data, aabb_entity));
    return mundy::math::get_owning_vector3<scalar_t>(std::move(shifted_data_accessor));
  }

  static decltype(auto) x_min(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data, aabb_entity)[0];
  }

  static decltype(auto) y_min(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data, aabb_entity)[1];
  }

  static decltype(auto) z_min(Agg agg, stk::mesh::Entity aabb_entity) {
      return stk::mesh::field_data(agg.aabb_data, aabb_entity)[2];
  }

  static decltype(auto) x_max(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data, aabb_entity)[3];
  }

  static decltype(auto) y_max(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data, aabb_entity)[4];
  }

  static decltype(auto) z_max(Agg agg, stk::mesh::Entity aabb_entity) {
    return stk::mesh::field_data(agg.aabb_data, aabb_entity)[5];
  }
};  // AABBDataTraits

/// \brief A traits class to provide abstracted access to a aabb's data via an NGP-compatible aggregate
/// See the discussion for AABBDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpAABBDataTraits {
  static_assert(
      ValidDefaultNgpAABBDataType<Agg>,
      "Agg must satisfy the ValidDefaultNgpAABBDataType concept.\n"
      "Basically, Agg must have all the same things as NgpAABBData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using aabb_data_t = typename Agg::aabb_data_t;

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) min_corner(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return mundy::math::get_owning_vector3<scalar_t>(agg.aabb_data(aabb_index));
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) max_corner(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    constexpr size_t shift = 3;
    auto shifted_data_accessor = mundy::math::get_owning_shifted_accessor<scalar_t, shift>(agg.aabb_data(aabb_index));
    return mundy::math::get_owning_vector3<scalar_t>(std::move(shifted_data_accessor));
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) x_min(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data(aabb_index)[0];
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) y_min(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data(aabb_index)[1];
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) z_min(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data(aabb_index)[2];
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) x_max(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data(aabb_index)[3];
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) y_max(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data(aabb_index)[4];
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) z_max(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    return agg.aabb_data(aabb_index)[5];
  }
};  // NgpAABBDataTraits

/// @brief A view of an aabb entity. May take any rank.
template <typename AABBDataType>
class AABBEntityView {
 public:
  using data_access_t = AABBDataTraits<AABBDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::min_corner(std::declval<AABBDataType>(), std::declval<stk::mesh::Entity>()));

  AABBEntityView([[maybe_unused]] const stk::mesh::BulkData& bulk_data, AABBDataType data, stk::mesh::Entity aabb_entity)
      : data_(data), aabb_entity_(aabb_entity) {
  }

  decltype(auto) min_corner() {
    return data_access_t::min_corner(data_, aabb_entity_);
  }

  decltype(auto) min_corner() const {
    return data_access_t::min_corner(data_, aabb_entity_);
  }

  decltype(auto) max_corner() {
    return data_access_t::max_corner(data_, aabb_entity_);
  }

  decltype(auto) max_corner() const {
    return data_access_t::max_corner(data_, aabb_entity_);
  }

  decltype(auto) x_min() {
    return data_access_t::x_min(data_, aabb_entity_);
  }

  decltype(auto) x_min() const {
    return data_access_t::x_min(data_, aabb_entity_);
  }

  decltype(auto) y_min() {
    return data_access_t::y_min(data_, aabb_entity_);
  }

  decltype(auto) y_min() const {
    return data_access_t::y_min(data_, aabb_entity_);
  }

  decltype(auto) z_min() {
    return data_access_t::z_min(data_, aabb_entity_);
  }

  decltype(auto) z_min() const {
    return data_access_t::z_min(data_, aabb_entity_);
  }

  decltype(auto) x_max() {
    return data_access_t::x_max(data_, aabb_entity_);
  }

  decltype(auto) x_max() const {
    return data_access_t::x_max(data_, aabb_entity_);
  }

  decltype(auto) y_max() {
    return data_access_t::y_max(data_, aabb_entity_);
  }

  decltype(auto) y_max() const {
    return data_access_t::y_max(data_, aabb_entity_);
  }

  decltype(auto) z_max() {
    return data_access_t::z_max(data_, aabb_entity_);
  }

  decltype(auto) z_max() const {
    return data_access_t::z_max(data_, aabb_entity_);
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
  NgpAABBEntityView([[maybe_unused]] stk::mesh::NgpMesh ngp_mesh, NgpAABBDataType data, stk::mesh::FastMeshIndex aabb_index)
      : data_(data), aabb_index_(aabb_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) min_corner() {
    return data_access_t::min_corner(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) min_corner() const {
    return data_access_t::min_corner(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) max_corner() {
    return data_access_t::max_corner(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) max_corner() const {
    return data_access_t::max_corner(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_min() {
    return data_access_t::x_min(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_min() const {
    return data_access_t::x_min(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_min() {
    return data_access_t::y_min(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_min() const {
    return data_access_t::y_min(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_min() {
    return data_access_t::z_min(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_min() const {
    return data_access_t::z_min(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_max() {
    return data_access_t::x_max(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) x_max() const {
    return data_access_t::x_max(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_max() {
    return data_access_t::y_max(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) y_max() const {
    return data_access_t::y_max(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_max() {
    return data_access_t::z_max(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) z_max() const {
    return data_access_t::z_max(data_, aabb_index_);
  }

 private:
  NgpAABBDataType data_;
  stk::mesh::FastMeshIndex aabb_index_;
};  // NgpAABBEntityView

static_assert(ValidAABBType<AABBEntityView<AABBData<float, stk::mesh::Field<float>>>> &&
                  ValidAABBType<AABBEntityView<AABBData<float, const stk::mesh::Field<float>>>> &&
                  ValidAABBType<NgpAABBEntityView<NgpAABBData<float, stk::mesh::NgpField<float>>>> &&
                  ValidAABBType<NgpAABBEntityView<NgpAABBData<float, const stk::mesh::NgpField<float>>>>,
              "AABBEntityView and NgpAABBEntityView must be valid AABB types");

/// \brief A helper function to create a AABBEntityView object with type deduction
template <typename AABBDataType>
auto create_aabb_entity_view(const stk::mesh::BulkData& bulk_data, AABBDataType& data, stk::mesh::Entity aabb_entity) {
  return AABBEntityView<AABBDataType>(bulk_data, data, aabb_entity);
}

/// \brief A helper function to create a NgpAABBEntityView object with type deduction
template <typename NgpAABBDataType>
auto create_ngp_aabb_entity_view(stk::mesh::NgpMesh ngp_mesh, NgpAABBDataType data,
                                 stk::mesh::FastMeshIndex aabb_index) {
  return NgpAABBEntityView<NgpAABBDataType>(ngp_mesh, data, aabb_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_AABB_HPP_