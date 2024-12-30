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

#ifndef MUNDY_GEOM_AGGREGATES_SPHERE_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHERE_HPP_

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
/// This struct is a data aggregate that holds the aabbs for a collection of aabbs. Typically, the aabb data will be
/// augmented on top of other data, such as adding aabb data to a collection of spheres. In and of itself, the aabb data
/// is not very useful.
///
/// \tparam Scalar The scalar type of the aabb's aabb.
/// \tparam AABBDataType The type of the aabb data. Can either be a scalar or an stk::mesh::Field of scalars.
template <typename Scalar, typename AABBDataType = stk::mesh::Field<Scalar>>
struct AABBData {
  static_assert((is_aabb_v<std::decay_t<AABBDataType>> ||
                 std::is_same_v<std::decay_t<AABBDataType>, stk::mesh::Field<Scalar>>),
                "AABBDataType must be either an AABB object or a field of scalars");

  using scalar_t = Scalar;
  using aabb_t = AABBDataType;

  stk::topology::rank_t aabb_rank;
  aabb_t& aabb;
};  // AABBData

/// \brief A struct to hold the data for a collection of NGP-compatible aabbs
/// See the discussion for AABBData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename AABBDataType = stk::mesh::NgpField<Scalar>>
struct NgpAABBData {
  static_assert((is_aabb_v<std::decay_t<AABBDataType>> ||
                 std::is_same_v<std::decay_t<AABBDataType>, stk::mesh::NgpField<Scalar>>),
                "AABBDataType must be either an AABB object or an ngp field of scalars");

  using scalar_t = Scalar;
  using aabb_t = AABBDataType;

  stk::topology::rank_t aabb_rank;
  aabb_t& aabb;
};  // NgpAABBData

/// \brief A helper function to create a AABBData object
///
/// This function creates a AABBData object with the given aabb and center data
/// and is used to automatically deduce the template parameters.
/// The data may be either a scalar or an stk::mesh::Field of scalars.
template <typename Scalar, typename AABBDataType>
auto create_aabb_data(stk::topology::rank_t aabb_rank, AABBDataType& aabb_data) {
  constexpr bool is_aabb_a_field = std::is_same_v<std::decay_t<AABBDataType>, stk::mesh::Field<Scalar>>;
  if constexpr (is_aabb_a_field) {
    MUNDY_THROW_ASSERT(aabb_data.entity_rank() == aabb_rank, std::invalid_argument,
                       "The aabb data must be a field of the same rank as the aabb");
  }
  return AABBData<Scalar, AABBDataType>{aabb_rank, aabb_data};
}

/// \brief A helper function to create a NgpAABBData object
/// See the discussion for create_aabb_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename AABBDataType>
auto create_ngp_aabb_data(stk::topology::rank_t aabb_rank, AABBDataType& aabb_data) {
  constexpr bool is_aabb_a_field = std::is_same_v<std::decay_t<AABBDataType>, stk::mesh::NgpField<Scalar>>;
  if constexpr (is_aabb_a_field) {
    MUNDY_THROW_ASSERT(aabb_data.get_rank() == aabb_rank, std::invalid_argument,
                       "The aabb data must be a field of the same rank as the aabb");
  }
  return NgpAABBData<Scalar, AABBDataType>{aabb_rank, aabb_data};
}

/// \brief A concept to check if a type provides the same data as AABBData
template <typename Agg>
concept ValidDefaultAABBDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::aabb_t;
  is_aabb_v<std::decay_t<typename Agg::aabb_t>> ||
      std::is_same_v<std::decay_t<typename Agg::aabb_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { agg.aabb_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.aabb } -> std::convertible_to<typename Agg::aabb_t&>;
};  // ValidDefaultAABBDataType

/// \brief A concept to check if a type provides the same data as NgpAABBData
template <typename Agg>
concept ValidDefaultNgpAABBDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::aabb_t;
  is_aabb_v<std::decay_t<typename Agg::center_t>> ||
      std::is_same_v<std::decay_t<typename Agg::aabb_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { agg.aabb_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.aabb } -> std::convertible_to<typename Agg::aabb_t&>;
};  // ValidDefaultNgpAABBDataType

static_assert(ValidDefaultAABBDataType<AABBData<float, float, AABB<float>>> &&
                  ValidDefaultAABBDataType<AABBData<float, stk::mesh::Field<float>, AABB<float>>> &&
                  ValidDefaultAABBDataType<AABBData<float, float, stk::mesh::Field<float>>> &&
                  ValidDefaultAABBDataType<AABBData<float, stk::mesh::Field<float>, stk::mesh::Field<float>>>,
              "AABBData must satisfy the ValidDefaultAABBDataType concept");

static_assert(
    ValidDefaultNgpAABBDataType<NgpAABBData<float, float, AABB<float>>> &&
        ValidDefaultNgpAABBDataType<NgpAABBData<float, stk::mesh::NgpField<float>, AABB<float>>> &&
        ValidDefaultNgpAABBDataType<NgpAABBData<float, float, stk::mesh::NgpField<float>>> &&
        ValidDefaultNgpAABBDataType<NgpAABBData<float, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>,
    "NgpAABBData must satisfy the ValidDefaultNgpAABBDataType concept");

/// \brief A helper function to get an updated NgpAABBData object from a AABBData object
/// \param data The AABBData object to convert
template <ValidDefaultAABBDataType AABBDataType>
auto get_updated_ngp_data(AABBDataType data) {
  using scalar_t = typename AABBDataType::scalar_t;
  using aabb_t = typename AABBDataType::aabb_t;

  constexpr bool is_aabb_a_field = std::is_same_v<std::decay_t<aabb_t>, stk::mesh::Field<scalar_t>>;
  if constexpr (is_aabb_a_field) {
    return create_ngp_aabb_data<scalar_t>(data.aabb_rank,  //
                                          stk::mesh::get_updated_ngp_field<scalar_t>(data.aabb));
  } else {
    return create_ngp_aabb_data<scalar_t>(data.aabb_rank, data.aabb);
  }
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
  using aabb_t = AABB<scalar_t>;

  static constexpr bool has_shared_aabb() {
    return std::is_same_v<std::decay_t<typename Agg::aabb_t>, scalar_t>;
  }

  static decltype(auto) aabb(Agg agg, stk::mesh::Entity aabb_entity) {
    if constexpr (has_shared_aabb()) {
      return agg.aabb;
    } else {
      return mundy::geom::aabb_field_data(agg.aabb, aabb_entity);
    }
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
  using aabb_t = AABB<scalar_t>;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_aabb() {
    return std::is_same_v<std::decay_t<typename Agg::aabb_t>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) aabb(Agg agg, stk::mesh::FastMeshIndex aabb_index) {
    if constexpr (has_shared_aabb()) {
      return agg.aabb;
    } else {
      return mundy::geom::aabb_field_data(agg.aabb, aabb_index);
    }
  }
};  // NgpAABBDataTraits

/// @brief A view of an ELEM_RANK, PARTICLE topology aabb entity
template <typename AABBDataType>
class AABBEntityView {
 public:
  using scalar_t = typename AABBDataType::scalar_t;
  using aabb_t = AABB<scalar_t>;
  using data_access_t = AABBDataTraits<AABBDataType>;

  AABBEntityView(const stk::mesh::BulkData& bulk_data, AABBDataType data, stk::mesh::Entity aabb_entity)
      : data_(data), aabb_entity_(aabb_entity) {
  }

  decltype(auto) aabb() {
    // For those not familiar with decltype(auto), it allows us to return either an auto or an auto&.
    return data_access_t::aabb(data_, aabb_entity_);
  }

  decltype(auto) aabb() const {
    return data_access_t::aabb(data_, aabb_entity_);
  }

 private:
  AABBDataType data_;
  stk::mesh::Entity aabb_entity_;
};  // AABBEntityView

/// @brief An ngp-compatible view of an ELEM_RANK, PARTICLE topology aabb entity
template <typename NgpAABBDataType>
class NgpAABBEntityView {
 public:
  using scalar_t = typename NgpAABBDataType::scalar_t;
  using aabb_t = AABB<scalar_t>;
  using data_access_t = NgpAABBDataTraits<NgpAABBDataType>;

  KOKKOS_INLINE_FUNCTION
  NgpAABBEntityView(stk::mesh::NgpMesh ngp_mesh, NgpAABBDataType data, stk::mesh::FastMeshIndex aabb_index)
      : data_(data), aabb_index_(aabb_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) aabb() {
    return NgpAABBDataTraits<NgpAABBDataType>::aabb(data_, aabb_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) aabb() const {
    return NgpAABBDataTraits<NgpAABBDataType>::aabb(data_, aabb_index_);
  }

 private:
  NgpAABBDataType data_;
  stk::mesh::FastMeshIndex aabb_index_;
};  // NgpAABBEntityView

static_assert(ValidAABBType<AABBEntityView<AABBData<float, float, Point<float>>>>,
              "AABBEntityView must be a valid AABB type");
static_assert(ValidAABBType<AABBEntityView<AABBData<float, stk::mesh::Field<float>, Point<float>>>>,
              "AABBEntityView must be a valid AABB type");
static_assert(ValidAABBType<AABBEntityView<AABBData<float, float, stk::mesh::Field<float>>>>,
              "AABBEntityView must be a valid AABB type");
static_assert(ValidAABBType<AABBEntityView<AABBData<float, stk::mesh::Field<float>, stk::mesh::Field<float>>>>,
              "AABBEntityView must be a valid AABB type");
static_assert(ValidAABBType<NgpAABBEntityView<NgpAABBData<float, float, Point<float>>>>,
              "NgpAABBEntityView must be a valid AABB type");
static_assert(ValidAABBType<NgpAABBEntityView<NgpAABBData<float, stk::mesh::NgpField<float>, Point<float>>>>,
              "NgpAABBEntityView must be a valid AABB type");
static_assert(ValidAABBType<NgpAABBEntityView<NgpAABBData<float, float, stk::mesh::NgpField<float>>>>,
              "NgpAABBEntityView must be a valid AABB type");
static_assert(
    ValidAABBType<NgpAABBEntityView<NgpAABBData<float, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>>,
    "NgpAABBEntityView must be a valid AABB type");

/// \brief A helper function to create a AABBEntityView object with type deduction
template <typename AABBDataType>
auto create_aabb_entity_view(const stk::mesh::BulkData& bulk_data, AABBDataType& data, stk::mesh::Entity aabb) {
  return AABBEntityView<AABBDataType>(bulk_data, data, aabb);
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

#endif  // MUNDY_GEOM_AGGREGATES_SPHERE_HPP_