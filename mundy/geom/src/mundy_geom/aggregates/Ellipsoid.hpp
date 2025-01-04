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

#ifndef MUNDY_GEOM_AGGREGATES_ELLIPSOID_HPP_
#define MUNDY_GEOM_AGGREGATES_ELLIPSOID_HPP_

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/primitives/Ellipsoid.hpp>  // for mundy::geom::ValidEllipsoidType
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief A struct to hold the data for a collection of ellipsoids
///
/// \tparam Scalar The scalar type of the ellipsoid's radius and center.
/// \tparam RadiusDataType The type of the radius data. Can either be a scalar or an stk::mesh::Field of scalars.
/// \tparam CenterDataType The type of the center data. Can either be a Point<Scalar> or an stk::mesh::Field of scalars.
template <typename Scalar, typename CenterDataType = stk::mesh::Field<Scalar>,
          typename OrientationDataType = stk::mesh::Field<Scalar>,
          typename AxisLengthsDataType = stk::mesh::Field<Scalar>>
struct EllipsoidData {
  static_assert((is_point_v<std::decay_t<CenterDataType>> ||
                 std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::Field<Scalar>>) &&
                    (std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::Field<Scalar>> ||
                     std::is_same_v<std::decay_t<OrientationDataType>, mundy::math::Quaternion<Scalar>>) &&
                    (mundy::math::is_vector3_v<std::decay_t<AxisLengthsDataType>> ||
                     std::is_same_v<std::decay_t<AxisLengthsDataType>, stk::mesh::Field<Scalar>>),
                "CenterDataType must be either a Point<Scalar> or an stk::mesh::Field<Scalar>, \n"
                "OrientationDataType must be either a Quaternion<Scalar> or an stk::mesh::Field<Scalar>, \n"
                "AxisLengthsDataType must be either a Vector3<Scalar> or an stk::mesh::Field<Scalar>");

  using scalar_t = Scalar;
  using center_data_t = CenterDataType;
  using orientation_data_t = OrientationDataType;
  using axis_lengths_data_t = AxisLengthsDataType;

  stk::topology::rank_t ellipsoid_rank;
  center_data_t& center_data;
  orientation_data_t& orientation_data;
  axis_lengths_data_t& axis_lengths_data;
};  // EllipsoidData

/// \brief A struct to hold the data for a collection of NGP-compatible ellipsoids
/// See the discussion for EllipsoidData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename CenterDataType = stk::mesh::NgpField<Scalar>,
          typename OrientationDataType = stk::mesh::NgpField<Scalar>,
          typename AxisLengthsDataType = stk::mesh::NgpField<Scalar>>
struct NgpEllipsoidData {
  static_assert((is_point_v<std::decay_t<CenterDataType>> ||
                 std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::NgpField<Scalar>>) &&
                    (std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::NgpField<Scalar>> ||
                     std::is_same_v<std::decay_t<OrientationDataType>, mundy::math::Quaternion<Scalar>>) &&
                    (mundy::math::is_vector3_v<std::decay_t<AxisLengthsDataType>> ||
                     std::is_same_v<std::decay_t<AxisLengthsDataType>, stk::mesh::NgpField<Scalar>>),
                "CenterDataType must be either a Point<Scalar> or an stk::mesh::NgpField<Scalar>, \n"
                "OrientationDataType must be either a Quaternion<Scalar> or an stk::mesh::NgpField<Scalar>, \n"
                "AxisLengthsDataType must be either a Vector3<Scalar> or an stk::mesh::NgpField<Scalar>");

  using scalar_t = Scalar;
  using center_data_t = CenterDataType;
  using orientation_data_t = OrientationDataType;
  using axis_lengths_data_t = AxisLengthsDataType;

  stk::topology::rank_t ellipsoid_rank;
  center_data_t& center_data;
  orientation_data_t& orientation_data;
  axis_lengths_data_t& axis_lengths_data;
};  // NgpEllipsoidData

/// \brief A helper function to create a EllipsoidData object
///
/// This function creates a EllipsoidData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar, typename CenterDataType, typename OrientationDataType, typename AxisLengthsDataType>
auto create_ellipsoid_data(stk::topology::rank_t ellipsoid_rank, CenterDataType& center_data,
                           OrientationDataType& orientation_data, AxisLengthsDataType& axis_lengths_data) {
  constexpr bool is_center_a_field = std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::Field<Scalar>>;
  constexpr bool is_orientation_a_field = std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::Field<Scalar>>;
  constexpr bool is_axis_lengths_a_field = std::is_same_v<std::decay_t<AxisLengthsDataType>, stk::mesh::Field<Scalar>>;

  if constexpr (is_center_a_field) {
    MUNDY_THROW_ASSERT(center_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center_data data must be a field of NODE_RANK");
  }
  if constexpr (is_orientation_a_field) {
    MUNDY_THROW_ASSERT(orientation_data.entity_rank() == ellipsoid_rank, std::invalid_argument,
                       "The orientation data must be a field of the same rank as the ellipsoid");
  }
  if constexpr (is_axis_lengths_a_field) {
    MUNDY_THROW_ASSERT(axis_lengths_data.entity_rank() == ellipsoid_rank, std::invalid_argument,
                       "The axis lengths data must be a field of the same rank as the ellipsoid");
  }
  return EllipsoidData<Scalar, CenterDataType, OrientationDataType, AxisLengthsDataType>{
      ellipsoid_rank, center_data, orientation_data, axis_lengths_data};
}

/// \brief A helper function to create a NgpEllipsoidData object
/// See the discussion for create_ellipsoid_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename CenterDataType, typename OrientationDataType, typename AxisLengthsDataType>
auto create_ngp_ellipsoid_data(stk::topology::rank_t ellipsoid_rank, CenterDataType& center_data,
                               OrientationDataType& orientation_data, AxisLengthsDataType& axis_lengths_data) {
  constexpr bool is_center_a_field = std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::NgpField<Scalar>>;
  constexpr bool is_orientation_a_field =
      std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::NgpField<Scalar>>;
  constexpr bool is_axis_lengths_a_field =
      std::is_same_v<std::decay_t<AxisLengthsDataType>, stk::mesh::NgpField<Scalar>>;

  if constexpr (is_center_a_field) {
    MUNDY_THROW_ASSERT(center_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center_data data must be a field of NODE_RANK");
  }
  if constexpr (is_orientation_a_field) {
    MUNDY_THROW_ASSERT(orientation_data.get_rank() == ellipsoid_rank, std::invalid_argument,
                       "The orientation data must be a field of the same rank as the ellipsoid");
  }
  if constexpr (is_axis_lengths_a_field) {
    MUNDY_THROW_ASSERT(axis_lengths_data.get_rank() == ellipsoid_rank, std::invalid_argument,
                       "The axis lengths data must be a field of the same rank as the ellipsoid");
  }
  return NgpEllipsoidData<Scalar, CenterDataType, OrientationDataType, AxisLengthsDataType>{
      ellipsoid_rank, center_data, orientation_data, axis_lengths_data};
}

/// \brief A concept to check if a type provides the same data as EllipsoidData
template <typename Agg>
concept ValidDefaultEllipsoidDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::center_data_t;
  typename Agg::orientation_data_t;
  typename Agg::axis_lengths_data_t;
  is_point_v<std::decay_t<typename Agg::center_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::center_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  mundy::math::is_quaternion_v<std::decay_t<typename Agg::orientation_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::orientation_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  mundy::math::is_vector3_v<std::decay_t<typename Agg::axis_lengths_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::axis_lengths_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { agg.ellipsoid_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.center_data } -> std::convertible_to<typename Agg::center_data_t&>;
  { agg.orientation_data } -> std::convertible_to<typename Agg::orientation_data_t&>;
  { agg.axis_lengths_data } -> std::convertible_to<typename Agg::axis_lengths_data_t&>;
};  // ValidDefaultEllipsoidDataType

/// \brief A concept to check if a type provides the same data as NgpEllipsoidData
template <typename Agg>
concept ValidDefaultNgpEllipsoidDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::center_data_t;
  typename Agg::orientation_data_t;
  typename Agg::axis_lengths_data_t;
  is_point_v<std::decay_t<typename Agg::center_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::center_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  mundy::math::is_quaternion_v<std::decay_t<typename Agg::orientation_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::orientation_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  mundy::math::is_vector3_v<std::decay_t<typename Agg::axis_lengths_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::axis_lengths_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { agg.ellipsoid_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.center_data } -> std::convertible_to<typename Agg::center_data_t&>;
  { agg.orientation_data } -> std::convertible_to<typename Agg::orientation_data_t&>;
  { agg.axis_lengths_data } -> std::convertible_to<typename Agg::axis_lengths_data_t&>;
};  // ValidDefaultNgpEllipsoidDataType

static_assert(
    ValidDefaultEllipsoidDataType<EllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, Point<float>>> &&
        ValidDefaultEllipsoidDataType<
            EllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, stk::mesh::Field<float>>> &&
        ValidDefaultEllipsoidDataType<
            EllipsoidData<float, Point<float>, stk::mesh::Field<float>, stk::mesh::Field<float>>> &&
        ValidDefaultEllipsoidDataType<
            EllipsoidData<float, stk::mesh::Field<float>, mundy::math::Quaternion<float>, stk::mesh::Field<float>>>,
    "EllipsoidData must satisfy the ValidDefaultEllipsoidDataType concept");

static_assert(
    ValidDefaultNgpEllipsoidDataType<
        NgpEllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, Point<float>>> &&
        ValidDefaultNgpEllipsoidDataType<
            NgpEllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, stk::mesh::NgpField<float>>> &&
        ValidDefaultNgpEllipsoidDataType<
            NgpEllipsoidData<float, Point<float>, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>> &&
        ValidDefaultNgpEllipsoidDataType<NgpEllipsoidData<float, stk::mesh::NgpField<float>,
                                                          mundy::math::Quaternion<float>, stk::mesh::NgpField<float>>>,
    "NgpEllipsoidData must satisfy the ValidDefaultNgpEllipsoidDataType concept");

/// \brief A helper function to get an updated NgpEllipsoidData object from a EllipsoidData object
/// \param data The EllipsoidData object to convert
template <ValidDefaultEllipsoidDataType EllipsoidDataType>
auto get_updated_ngp_data(EllipsoidDataType data) {
  using scalar_t = typename EllipsoidDataType::scalar_t;
  using center_data_t = typename EllipsoidDataType::center_data_t;
  using orientation_data_t = typename EllipsoidDataType::orientation_data_t;
  using axis_lengths_data_t = typename EllipsoidDataType::axis_lengths_data_t;

  constexpr bool is_center_a_field = std::is_same_v<std::decay_t<center_data_t>, stk::mesh::Field<scalar_t>>;
  constexpr bool is_orientation_a_field = std::is_same_v<std::decay_t<orientation_data_t>, stk::mesh::Field<scalar_t>>;
  constexpr bool is_axis_lengths_a_field =
      std::is_same_v<std::decay_t<axis_lengths_data_t>, stk::mesh::Field<scalar_t>>;
  if constexpr (is_center_a_field && is_orientation_a_field && is_axis_lengths_a_field) {
    return create_ngp_ellipsoid_data<scalar_t>(data.ellipsoid_rank,                                           //
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data),  //
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data),
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.axis_lengths_data));
  } else if constexpr (!is_center_a_field && is_orientation_a_field && is_axis_lengths_a_field) {
    return create_ngp_ellipsoid_data<scalar_t>(data.ellipsoid_rank,  //
                                               data.center_data,     //
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data),
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.axis_lengths_data));
  } else if constexpr (is_center_a_field && !is_orientation_a_field && is_axis_lengths_a_field) {
    return create_ngp_ellipsoid_data<scalar_t>(data.ellipsoid_rank,                                           //
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data),  //
                                               data.orientation_data,
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.axis_lengths_data));
  } else if constexpr (is_center_a_field && is_orientation_a_field && !is_axis_lengths_a_field) {
    return create_ngp_ellipsoid_data<scalar_t>(data.ellipsoid_rank,                                           //
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data_data),  //
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data),
                                               data.axis_lengths_data);
  } else if constexpr (!is_center_a_field && !is_orientation_a_field && is_axis_lengths_a_field) {
    return create_ngp_ellipsoid_data<scalar_t>(data.ellipsoid_rank,  //
                                               data.center_data,     //
                                               data.orientation_data,
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.axis_lengths_data));
  } else if constexpr (!is_center_a_field && is_orientation_a_field && !is_axis_lengths_a_field) {
    return create_ngp_ellipsoid_data<scalar_t>(data.ellipsoid_rank,  //
                                               data.center_data,     //
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data),
                                               data.axis_lengths_data);
  } else if constexpr (is_center_a_field && !is_orientation_a_field && !is_axis_lengths_a_field) {
    return create_ngp_ellipsoid_data<scalar_t>(data.ellipsoid_rank,                                           //
                                               stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data),  //
                                               data.orientation_data, data.axis_lengths_data);
  } else {
    return create_ngp_ellipsoid_data<scalar_t>(data.ellipsoid_rank, data.center_data, data.orientation_data,
                                               data.axis_lengths_data);
  }
}

/// \brief A traits class to provide abstracted access to a ellipsoid's data via an aggregate
///
/// By default, this class is compatible with EllipsoidData or any class the meets the ValidDefaultEllipsoidDataType
/// concept. Users can specialize this class to support other aggregate types.
template <typename Agg>
struct EllipsoidDataTraits {
  static_assert(
      ValidDefaultEllipsoidDataType<Agg>,
      "Agg must satisfy the ValidDefaultEllipsoidDataType concept.\n"
      "Basically, Agg must have all the same things as NgpEllipsoidData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;
  using orientation_data_t = typename Agg::orientation_data_t;
  using axis_lengths_data_t = typename Agg::axis_lengths_data_t;

  static constexpr bool has_shared_center() {
    return is_point_v<center_data_t>;
  }

  static constexpr bool has_shared_orientation() {
    return mundy::math::is_quaternion_v<orientation_data_t>;
  }

  static constexpr bool has_shared_axis_lengths() {
    return mundy::math::is_vector3_v<axis_lengths_data_t>;
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity ellipsoid_node) {
    if constexpr (has_shared_center()) {
      return agg.center_data;
    } else {
      // This returns a copy of a view into the raw data of the field.
      return mundy::mesh::vector3_field_data(agg.center_data, ellipsoid_node);
    }
  }

  static decltype(auto) orientation(Agg agg, stk::mesh::Entity ellipsoid) {
    if constexpr (has_shared_orientation()) {
      return agg.orientation_data;
    } else {
      return mundy::mesh::quaternion_field_data(agg.orientation_data, ellipsoid);
    }
  }

  static decltype(auto) axis_lengths(Agg agg, stk::mesh::Entity ellipsoid) {
    if constexpr (has_shared_axis_lengths()) {
      return agg.axis_lengths_data;
    } else {
      return mundy::mesh::vector3_field_data(agg.axis_lengths_data, ellipsoid);
    }
  }
};  // EllipsoidDataTraits

/// \brief A traits class to provide abstracted access to a ellipsoid's data via an NGP-compatible aggregate
/// See the discussion for EllipsoidDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpEllipsoidDataTraits {
  static_assert(
      ValidDefaultNgpEllipsoidDataType<Agg>,
      "Agg must satisfy the ValidDefaultNgpEllipsoidDataType concept.\n"
      "Basically, Agg must have all the same things as NgpEllipsoidData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;
  using orientation_data_t = typename Agg::orientation_data_t;
  using axis_lengths_data_t = typename Agg::axis_lengths_data_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_center() {
    return is_point_v<center_data_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_orientation() {
    return mundy::math::is_quaternion_v<orientation_data_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_axis_lengths() {
    return mundy::math::is_vector3_v<axis_lengths_data_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex ellipsoid_node_index) {
    if constexpr (has_shared_center()) {
      return agg.center_data;
    } else {
      return mundy::mesh::vector3_field_data(agg.center_data, ellipsoid_node_index);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) orientation(Agg agg, stk::mesh::FastMeshIndex ellipsoid_index) {
    if constexpr (has_shared_orientation()) {
      return agg.orientation_data;
    } else {
      return mundy::mesh::quaternion_field_data(agg.orientation_data, ellipsoid_index);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) axis_lengths(Agg agg, stk::mesh::FastMeshIndex ellipsoid_index) {
    if constexpr (has_shared_axis_lengths()) {
      return agg.axis_lengths_data;
    } else {
      return mundy::mesh::vector3_field_data(agg.axis_lengths_data, ellipsoid_index);
    }
  }
};  // NgpEllipsoidDataTraits

/// @brief A view of an STK entity meant to represent a ellipsoid
/// If ellipsoid_rank is NODE_RANK, then the ellipsoid is just a node entity with node center, orientation, and axis
/// lengths.
/// If ellipsoid_rank is ELEM_RANK, then the ellipsoid is a particle entity with node center, elem orientation, and elem
/// axis lengths.
template <typename EllipsoidDataType>
class ElemEllipsoidView {
 public:
  using data_access_t = EllipsoidDataTraits<EllipsoidDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<EllipsoidDataType>(), std::declval<stk::mesh::Entity>()));

  ElemEllipsoidView(const stk::mesh::BulkData& bulk_data, EllipsoidDataType data, stk::mesh::Entity ellipsoid)
      : data_(data), ellipsoid_(ellipsoid), node_(bulk_data.begin_nodes(ellipsoid_)[0]) {
  }

  decltype(auto) center() {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) orientation() {
    return data_access_t::orientation(data_, ellipsoid_);
  }

  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, ellipsoid_);
  }

  decltype(auto) axis_lengths() {
    return data_access_t::axis_lengths(data_, ellipsoid_);
  }

  decltype(auto) axis_lengths() const {
    return data_access_t::axis_lengths(data_, ellipsoid_);
  }

 private:
  EllipsoidDataType data_;
  stk::mesh::Entity ellipsoid_;
  stk::mesh::Entity node_;
};  // ElemEllipsoidView

/// @brief An ngp-compatible view of an STK entity meant to represent a ellipsoid
/// See the discussion for ElemEllipsoidView for more information. The only difference is ngp-compatible data access.
template <typename NgpEllipsoidDataType>
class NgpElemEllipsoidView {
 public:
  using data_access_t = NgpEllipsoidDataTraits<NgpEllipsoidDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<NgpEllipsoidDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpElemEllipsoidView(stk::mesh::NgpMesh ngp_mesh, NgpEllipsoidDataType data, stk::mesh::FastMeshIndex ellipsoid_index)
      : data_(data),
        ellipsoid_index_(ellipsoid_index),
        node_index_(ngp_mesh.fast_mesh_index(ngp_mesh.get_nodes(data_.ellipsoid_rank, ellipsoid_index_)[0])) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data_, node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data_, node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return data_access_t::orientation(data_, ellipsoid_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, ellipsoid_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() {
    return data_access_t::axis_lengths(data_, ellipsoid_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() const {
    return data_access_t::axis_lengths(data_, ellipsoid_index_);
  }

 private:
  NgpEllipsoidDataType data_;
  stk::mesh::FastMeshIndex ellipsoid_index_;
  stk::mesh::FastMeshIndex node_index_;
};  // NgpElemEllipsoidView

/// @brief A view of an STK entity meant to represent a ellipsoid
/// If the ellipsoid_rank is NODE_RANK, then the ellipsoid is just a node entity with node radius and node center.
/// If the ellipsoid_rank is ELEM_RANK, then the ellipsoid is a particle entity with element radius and node center.
template <typename EllipsoidDataType>
class NodeEllipsoidView {
 public:
  using data_access_t = EllipsoidDataTraits<EllipsoidDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<EllipsoidDataType>(), std::declval<stk::mesh::Entity>()));

  NodeEllipsoidView([[maybe_unused]] const stk::mesh::BulkData& bulk_data, EllipsoidDataType data, stk::mesh::Entity ellipsoid)
      : data_(data), ellipsoid_(ellipsoid) {
  }

  decltype(auto) center() {
    return data_access_t::center(data_, ellipsoid_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, ellipsoid_);
  }

  decltype(auto) orientation() {
    return data_access_t::orientation(data_, ellipsoid_);
  }

  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, ellipsoid_);
  }

  decltype(auto) axis_lengths() {
    return data_access_t::axis_lengths(data_, ellipsoid_);
  }

  decltype(auto) axis_lengths() const {
    return data_access_t::axis_lengths(data_, ellipsoid_);
  }


 private:
  EllipsoidDataType data_;
  stk::mesh::Entity ellipsoid_;
};  // NodeEllipsoidView

/// @brief An ngp-compatible view of an ELEM_RANK, PARTICLE topology ellipsoid entity
template <typename NgpEllipsoidDataType>
class NgpNodeEllipsoidView {
 public:
  using data_access_t = NgpEllipsoidDataTraits<NgpEllipsoidDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<NgpEllipsoidDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpNodeEllipsoidView([[maybe_unused]] stk::mesh::NgpMesh ngp_mesh, NgpEllipsoidDataType data, stk::mesh::FastMeshIndex ellipsoid_index)
      : data_(data),
        ellipsoid_index_(ellipsoid_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data_, ellipsoid_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data_, ellipsoid_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return data_access_t::orientation(data_, ellipsoid_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, ellipsoid_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() {
    return data_access_t::axis_lengths(data_, ellipsoid_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() const {
    return data_access_t::axis_lengths(data_, ellipsoid_index_);
  }

 private:
  NgpEllipsoidDataType data_;
  stk::mesh::FastMeshIndex ellipsoid_index_;
};  // NgpNodeEllipsoidView

static_assert(
  ValidEllipsoidType<ElemEllipsoidView<EllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, Point<float>>>>
  && ValidEllipsoidType<ElemEllipsoidView<EllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, stk::mesh::Field<float>>>>
  && ValidEllipsoidType<ElemEllipsoidView<EllipsoidData<float, Point<float>, stk::mesh::Field<float>, stk::mesh::Field<float>>>>
  && ValidEllipsoidType<ElemEllipsoidView<EllipsoidData<float, stk::mesh::Field<float>, mundy::math::Quaternion<float>, stk::mesh::Field<float>>>>
  && ValidEllipsoidType<NgpElemEllipsoidView<NgpEllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, Point<float>>>>
  && ValidEllipsoidType<NgpElemEllipsoidView<NgpEllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, stk::mesh::NgpField<float>>>>
  && ValidEllipsoidType<NgpElemEllipsoidView<NgpEllipsoidData<float, Point<float>, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>>
  && ValidEllipsoidType<NgpElemEllipsoidView<NgpEllipsoidData<float, stk::mesh::NgpField<float>, mundy::math::Quaternion<float>, stk::mesh::NgpField<float>>>>,
  "ElemEllipsoidView and NgpElemEllipsoidView must be valid Ellipsoid types");

static_assert(
  ValidEllipsoidType<NodeEllipsoidView<EllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, Point<float>>>>
  && ValidEllipsoidType<NodeEllipsoidView<EllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, stk::mesh::Field<float>>>>
  && ValidEllipsoidType<NodeEllipsoidView<EllipsoidData<float, Point<float>, stk::mesh::Field<float>, stk::mesh::Field<float>>>>
  && ValidEllipsoidType<NodeEllipsoidView<EllipsoidData<float, stk::mesh::Field<float>, mundy::math::Quaternion<float>, stk::mesh::Field<float>>>>
  && ValidEllipsoidType<NgpNodeEllipsoidView<NgpEllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, Point<float>>>>
  && ValidEllipsoidType<NgpNodeEllipsoidView<NgpEllipsoidData<float, Point<float>, mundy::math::Quaternion<float>, stk::mesh::NgpField<float>>>>
  && ValidEllipsoidType<NgpNodeEllipsoidView<NgpEllipsoidData<float, Point<float>, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>>
  && ValidEllipsoidType<NgpNodeEllipsoidView<NgpEllipsoidData<float, stk::mesh::NgpField<float>, mundy::math::Quaternion<float>, stk::mesh::NgpField<float>>>>,
  "NodeEllipsoidView and NgpNodeEllipsoidView must be valid Ellipsoid types");

/// \brief A helper function to create a ElemEllipsoidView object with type deduction
template <typename EllipsoidDataType>
auto create_elem_ellipsoid_view(const stk::mesh::BulkData& bulk_data, EllipsoidDataType& data,
                                stk::mesh::Entity ellipsoid) {
  return ElemEllipsoidView<EllipsoidDataType>(bulk_data, data, ellipsoid);
}

/// \brief A helper function to create a NgpElemEllipsoidView object with type deduction
template <typename NgpEllipsoidDataType>
auto create_ngp_elem_ellipsoid_view(stk::mesh::NgpMesh ngp_mesh, NgpEllipsoidDataType data,
                                    stk::mesh::FastMeshIndex ellipsoid_index) {
  return NgpElemEllipsoidView<NgpEllipsoidDataType>(ngp_mesh, data, ellipsoid_index);
}

/// \brief A helper function to create a NodeEllipsoidView object with type deduction
template <typename EllipsoidDataType>
auto create_node_ellipsoid_view(const stk::mesh::BulkData& bulk_data, EllipsoidDataType& data,
                                stk::mesh::Entity ellipsoid) {
  return NodeEllipsoidView<EllipsoidDataType>(bulk_data, data, ellipsoid);
}

/// \brief A helper function to create a NgpNodeEllipsoidView object with type deduction
template <typename NgpEllipsoidDataType>
auto create_ngp_node_ellipsoid_view(stk::mesh::NgpMesh ngp_mesh, NgpEllipsoidDataType data,
                                    stk::mesh::FastMeshIndex ellipsoid_index) {
  return NgpNodeEllipsoidView<NgpEllipsoidDataType>(ngp_mesh, data, ellipsoid_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_ELLIPSOID_HPP_