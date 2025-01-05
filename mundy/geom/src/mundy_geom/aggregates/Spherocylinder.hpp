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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEROCYLINDER_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEROCYLINDER_HPP_

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

//! \name Aggregate traits
//@{

/// \brief A struct to hold the data for a collection of spherocylinders
///
/// \tparam Scalar The scalar type of the spherocylinder's radius and center.
/// \tparam CenterDataType The type of the center data. Can either be a const or non-const stk::mesh::Field of scalars.
/// \tparam OrientationDataType The type of the orientation data. Can either be a Quaternion or a const or non-const
/// \tparam RadiusDataType The type of the radius data. Can either be a scalar or an stk::mesh::Field of scalars.
template <typename Scalar, typename CenterDataType = stk::mesh::Field<Scalar>,
          typename OrientationDataType = stk::mesh::Field<Scalar>, typename RadiusDataType = stk::mesh::Field<Scalar>>
struct SpherocylinderData {
  static_assert((std::is_same_v<std::decay_t<RadiusDataType>, Scalar> ||
                 std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::Field<Scalar>>) &&
                    std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::Field<Scalar>> &&
                    (mundy::math::is_quaternion_v<std::decay_t<OrientationDataType>> ||
                     std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::Field<Scalar>>),
                "RadiusDataType must be either a scalar or a field of scalars\n"
                "CenterDataType must be either a const or non-const field of scalars\n"
                "OrientationDataType must be either a quaternion or a field of scalars");

  using scalar_t = Scalar;
  using center_data_t = CenterDataType;
  using orientation_data_t = OrientationDataType;
  using radius_data_t = RadiusDataType;

  stk::topology::rank_t spherocylinder_rank;
  center_data_t& center_data;
  orientation_data_t& orientation_data;
  radius_data_t& radius_data;
};  // SpherocylinderData

/// \brief A struct to hold the data for a collection of NGP-compatible spherocylinders
/// See the discussion for SpherocylinderData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename CenterDataType = stk::mesh::NgpField<Scalar>,
          typename OrientationDataType = stk::mesh::NgpField<Scalar>,
          typename RadiusDataType = stk::mesh::NgpField<Scalar>>
struct NgpSpherocylinderData {
  static_assert((std::is_same_v<std::decay_t<RadiusDataType>, Scalar> ||
                 std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::NgpField<Scalar>>) &&
                    std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::NgpField<Scalar>> &&
                    (mundy::math::is_quaternion_v<std::decay_t<OrientationDataType>> ||
                     std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::NgpField<Scalar>>),
                "RadiusDataType must be either a scalar or a field of scalars\n"
                "CenterDataType must be either a const or non-const field of scalars\n"
                "OrientationDataType must be either a quaternion or a field of scalars");

  using scalar_t = Scalar;
  using center_data_t = CenterDataType;
  using orientation_data_t = OrientationDataType;
  using radius_data_t = RadiusDataType;

  stk::topology::rank_t spherocylinder_rank;
  center_data_t& center_data;
  orientation_data_t& orientation_data;
  radius_data_t& radius_data;
};  // NgpSpherocylinderData

/// \brief A helper function to create a SpherocylinderData object
///
/// This function creates a SpherocylinderData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar, typename CenterDataType, typename OrientationDataType, typename RadiusDataType>
auto create_spherocylinder_data(stk::topology::rank_t spherocylinder_rank, CenterDataType& center_data,
                                OrientationDataType& orientation_data, RadiusDataType& radius_data) {
  MUNDY_THROW_ASSERT(center_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                     "The center data must be a field of NODE_RANK");
  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::Field<Scalar>>;
  constexpr bool is_orientation_a_field = std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::Field<Scalar>>;
  if constexpr (is_radius_a_field) {
    MUNDY_THROW_ASSERT(radius_data.entity_rank() == spherocylinder_rank, std::invalid_argument,
                       "The radius data must be a field of the same rank as the spherocylinder");
  }
  if constexpr (is_orientation_a_field) {
    MUNDY_THROW_ASSERT(orientation_data.entity_rank() == spherocylinder_rank, std::invalid_argument,
                       "The orientation data must be a field of the same rank as the spherocylinder");
  }
  return SpherocylinderData<Scalar, CenterDataType, OrientationDataType, RadiusDataType>{
      spherocylinder_rank, center_data, orientation_data, radius_data};
}

/// \brief A helper function to create a NgpSpherocylinderData object
/// See the discussion for create_spherocylinder_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename CenterDataType, typename OrientationDataType, typename RadiusDataType>
auto create_ngp_spherocylinder_data(stk::topology::rank_t spherocylinder_rank, CenterDataType& center_data,
                                    OrientationDataType& orientation_data, RadiusDataType& radius_data) {
  MUNDY_THROW_ASSERT(center_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                     "The center data must be a field of NODE_RANK");
  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::NgpField<Scalar>>;
  constexpr bool is_orientation_a_field =
      std::is_same_v<std::decay_t<OrientationDataType>, stk::mesh::NgpField<Scalar>>;
  if constexpr (is_radius_a_field) {
    MUNDY_THROW_ASSERT(radius_data.get_rank() == spherocylinder_rank, std::invalid_argument,
                       "The radius data must be a field of the same rank as the spherocylinder");
  }
  if constexpr (is_orientation_a_field) {
    MUNDY_THROW_ASSERT(orientation_data.get_rank() == spherocylinder_rank, std::invalid_argument,
                       "The orientation data must be a field of the same rank as the spherocylinder");
  }
  return NgpSpherocylinderData<Scalar, CenterDataType, OrientationDataType, RadiusDataType>{
      spherocylinder_rank, center_data, orientation_data, radius_data};
}

/// \brief A concept to check if a type provides the same data as SpherocylinderData
template <typename Agg>
concept ValidDefaultSpherocylinderDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::center_data_t;
  typename Agg::orientation_data_t;
  typename Agg::radius_data_t;
  std::is_same_v<std::decay_t<typename Agg::radius_data_t>, typename Agg::scalar_t> ||
      std::is_same_v<std::decay_t<typename Agg::radius_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  mundy::math::is_quaternion_v<std::decay_t<typename Agg::orientation_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::orientation_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  std::is_same_v<std::decay_t<typename Agg::center_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { agg.spherocylinder_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.center_data } -> std::convertible_to<typename Agg::center_data_t&>;
  { agg.orientation_data } -> std::convertible_to<typename Agg::orientation_data_t&>;
  { agg.radius_data } -> std::convertible_to<typename Agg::radius_data_t&>;
};  // ValidDefaultSpherocylinderDataType

/// \brief A concept to check if a type provides the same data as NgpSpherocylinderData
template <typename Agg>
concept ValidDefaultNgpSpherocylinderDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::center_data_t;
  typename Agg::orientation_data_t;
  typename Agg::radius_data_t;
  std::is_same_v<std::decay_t<typename Agg::radius_data_t>, typename Agg::scalar_t> ||
      std::is_same_v<std::decay_t<typename Agg::radius_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  mundy::math::is_quaternion_v<std::decay_t<typename Agg::orientation_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::orientation_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  std::is_same_v<std::decay_t<typename Agg::center_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { agg.spherocylinder_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.center_data } -> std::convertible_to<typename Agg::center_data_t&>;
  { agg.orientation_data } -> std::convertible_to<typename Agg::orientation_data_t&>;
  { agg.radius_data } -> std::convertible_to<typename Agg::radius_data_t&>;
};  // ValidDefaultNgpSpherocylinderDataType

static_assert(
    ValidDefaultSpherocylinderDataType<
        SpherocylinderData<float, stk::mesh::Field<float>, mundy::math::Quaternion<float>, float>> &&
        ValidDefaultSpherocylinderDataType<
            SpherocylinderData<float, stk::mesh::Field<float>, stk::mesh::Field<float>, stk::mesh::Field<float>>> &&
        ValidDefaultSpherocylinderDataType<SpherocylinderData<float, const stk::mesh::Field<float>,
                                                              const mundy::math::Quaternion<float>, const float>> &&
        ValidDefaultSpherocylinderDataType<SpherocylinderData<
            float, const stk::mesh::Field<float>, const stk::mesh::Field<float>, const stk::mesh::Field<float>>>,
    "SpherocylinderData must satisfy the ValidDefaultSpherocylinderDataType concept");

static_assert(ValidDefaultNgpSpherocylinderDataType<
                  NgpSpherocylinderData<float, stk::mesh::NgpField<float>, mundy::math::Quaternion<float>, float>> &&
                  ValidDefaultNgpSpherocylinderDataType<NgpSpherocylinderData<
                      float, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>> &&
                  ValidDefaultNgpSpherocylinderDataType<NgpSpherocylinderData<
                      float, const stk::mesh::NgpField<float>, const mundy::math::Quaternion<float>, const float>> &&
                  ValidDefaultNgpSpherocylinderDataType<
                      NgpSpherocylinderData<float, const stk::mesh::NgpField<float>, const stk::mesh::NgpField<float>,
                                            const stk::mesh::NgpField<float>>>,
              "NgpSpherocylinderData must satisfy the ValidDefaultNgpSpherocylinderDataType concept");

/// \brief A helper function to get an updated NgpSpherocylinderData object from a SpherocylinderData object
/// \param data The SpherocylinderData object to convert
template <ValidDefaultSpherocylinderDataType SpherocylinderDataType>
auto get_updated_ngp_data(SpherocylinderDataType data) {
  using scalar_t = typename SpherocylinderDataType::scalar_t;
  using center_data_t = typename SpherocylinderDataType::center_data_t;
  using orientation_data_t = typename SpherocylinderDataType::orientation_data_t;
  using radius_data_t = typename SpherocylinderDataType::radius_data_t;

  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<radius_data_t>, stk::mesh::Field<scalar_t>>;
  constexpr bool is_orientation_a_field = std::is_same_v<std::decay_t<orientation_data_t>, stk::mesh::Field<scalar_t>>;
  if constexpr (is_radius_a_field && is_orientation_a_field) {
    return create_ngp_spherocylinder_data<scalar_t>(data.spherocylinder_rank,                                      //
                                                    stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data),  //
                                                    stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data),
                                                    stk::mesh::get_updated_ngp_field<scalar_t>(data.radius_data));
  } else if constexpr (is_radius_a_field && !is_orientation_a_field) {
    return create_ngp_spherocylinder_data<scalar_t>(data.spherocylinder_rank,                                      //
                                                    stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data),  //
                                                    data.orientation_data,
                                                    stk::mesh::get_updated_ngp_field<scalar_t>(data.radius_data));
  } else if constexpr (!is_radius_a_field && is_orientation_a_field) {
    return create_ngp_spherocylinder_data<scalar_t>(data.spherocylinder_rank,                                      //
                                                    stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data),  //
                                                    stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data),
                                                    data.radius_data);
  } else {
    return create_ngp_spherocylinder_data<scalar_t>(data.spherocylinder_rank,                                      //
                                                    stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data),  //
                                                    data.orientation_data, data.radius_data);
  }
}

/// \brief A traits class to provide abstracted access to a spherocylinder's data via an aggregate
///
/// By default, this class is compatible with SpherocylinderData or any class the meets the
/// ValidDefaultSpherocylinderDataType concept. Users can specialize this class to support other aggregate types.
template <typename Agg>
struct SpherocylinderDataTraits {
  static_assert(ValidDefaultSpherocylinderDataType<Agg>,
                "Agg must satisfy the ValidDefaultSpherocylinderDataType concept.\n"
                "Basically, Agg must have all the same things as NgpSpherocylinderData but is free to extend it as "
                "needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;
  using orientation_data_t = typename Agg::orientation_data_t;
  using radius_data_t = typename Agg::radius_data_t;

  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  static constexpr decltype(auto) has_shared_orientation() {
    return mundy::math::is_quaternion_v<std::decay_t<orientation_data_t>>;
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity spherocylinder_node) {
    return mundy::mesh::vector3_field_data(agg.center_data, spherocylinder_node);
  }

  static decltype(auto) orientation(Agg agg, stk::mesh::Entity spherocylinder) {
    if constexpr (has_shared_orientation()) {
      return agg.orientation_data;
    } else {
      return mundy::mesh::quaternion_field_data(agg.orientation_data, spherocylinder);
    }
  }

  static decltype(auto) radius(Agg agg, stk::mesh::Entity spherocylinder) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data;
    } else {
      return stk::mesh::field_data(agg.radius_data, spherocylinder)[0];
    }
  }
};  // SpherocylinderDataTraits

/// \brief A traits class to provide abstracted access to a spherocylinder's data via an NGP-compatible aggregate
/// See the discussion for SpherocylinderDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpSpherocylinderDataTraits {
  static_assert(ValidDefaultNgpSpherocylinderDataType<Agg>,
                "Agg must satisfy the ValidDefaultNgpSpherocylinderDataType concept.\n"
                "Basically, Agg must have all the same things as NgpSpherocylinderData but is free to extend it as "
                "needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;
  using orientation_data_t = typename Agg::orientation_data_t;
  using radius_data_t = typename Agg::radius_data_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_orientation() {
    return mundy::math::is_quaternion_v<std::decay_t<orientation_data_t>>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex spherocylinder_node_index) {
    return mundy::mesh::vector3_field_data(agg.center_data, spherocylinder_node_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) orientation(Agg agg, stk::mesh::FastMeshIndex spherocylinder_index) {
    if constexpr (has_shared_orientation()) {
      return agg.orientation_data;
    } else {
      return mundy::mesh::quaternion_field_data(agg.orientation_data, spherocylinder_index);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) radius(Agg agg, stk::mesh::FastMeshIndex spherocylinder_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data;
    } else {
      return agg.radius_data(spherocylinder_index, 0);
    }
  }
};  // NgpSpherocylinderDataTraits

/// @brief A view of an STK entity meant to represent a spherocylinder
/// If the spherocylinder_rank is NODE_RANK, then the spherocylinder is just a node entity with node radius and node
/// center. If the spherocylinder_rank is ELEM_RANK, then the spherocylinder is a particle entity with element radius
/// and node center.
template <typename SpherocylinderDataType>
class ElemSpherocylinderView {
 public:
  using data_access_t = SpherocylinderDataTraits<SpherocylinderDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<SpherocylinderDataType>(), std::declval<stk::mesh::Entity>()));
  using orientation_t =
      decltype(data_access_t::orientation(std::declval<SpherocylinderDataType>(), std::declval<stk::mesh::Entity>()));

  ElemSpherocylinderView(const stk::mesh::BulkData& bulk_data, SpherocylinderDataType data,
                         stk::mesh::Entity spherocylinder)
      : data_(data), spherocylinder_(spherocylinder), node_(bulk_data.begin_nodes(spherocylinder_)[0]) {
    MUNDY_THROW_ASSERT(bulk_data.entity_rank(spherocylinder_) == stk::topology::ELEM_RANK &&
                           data_.spherocylinder_rank == stk::topology::ELEM_RANK,
                       std::invalid_argument,
                       "Both the spherocylinder entity rank and the spherocylinder data rank must be ELEM_RANK");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(spherocylinder_), std::invalid_argument,
                       "The given spherocylinder entity is not valid");
    MUNDY_THROW_ASSERT(bulk_data.num_nodes(spherocylinder_) == 1, std::invalid_argument,
                       "The given spherocylinder entity must have exactly one node");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(node_), std::invalid_argument,
                       "The node entity associated with the spherocylinder is not valid");
  }

  decltype(auto) center() {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) orientation() {
    return data_access_t::orientation(data_, spherocylinder_);
  }

  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, spherocylinder_);
  }

  decltype(auto) radius() {
    return data_access_t::radius(data_, spherocylinder_);
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data_, spherocylinder_);
  }

 private:
  SpherocylinderDataType data_;
  stk::mesh::Entity spherocylinder_;
  stk::mesh::Entity node_;
};  // ElemSpherocylinderView

/// @brief An ngp-compatible view of an ELEM_RANK STK entity meant to represent a spherocylinder
/// See the discussion for ElemSpherocylinderView for more information. The only difference is ngp-compatible data
/// access.
template <typename NgpSpherocylinderDataType>
class NgpElemSpherocylinderView {
 public:
  using data_access_t = NgpSpherocylinderDataTraits<NgpSpherocylinderDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<NgpSpherocylinderDataType>(),
                                                 std::declval<stk::mesh::FastMeshIndex>()));
  using orientation_t = decltype(data_access_t::orientation(std::declval<NgpSpherocylinderDataType>(),
                                                            std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpElemSpherocylinderView(stk::mesh::NgpMesh ngp_mesh, NgpSpherocylinderDataType data,
                            stk::mesh::FastMeshIndex spherocylinder_index)
      : data_(data),
        spherocylinder_index_(spherocylinder_index),
        node_index_(ngp_mesh.fast_mesh_index(ngp_mesh.get_nodes(data_.spherocylinder_rank, spherocylinder_index_)[0])) {
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
    return data_access_t::orientation(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data_, spherocylinder_index_);
  }

 private:
  NgpSpherocylinderDataType data_;
  stk::mesh::FastMeshIndex spherocylinder_index_;
  stk::mesh::FastMeshIndex node_index_;
};  // NgpElemSpherocylinderView

/// @brief A view of a NODE_RANK STK entity meant to represent a spherocylinder
template <typename SpherocylinderDataType>
class NodeSpherocylinderView {
 public:
  using data_access_t = SpherocylinderDataTraits<SpherocylinderDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<SpherocylinderDataType>(), std::declval<stk::mesh::Entity>()));
  using orientation_t =
      decltype(data_access_t::orientation(std::declval<SpherocylinderDataType>(), std::declval<stk::mesh::Entity>()));

  NodeSpherocylinderView([[maybe_unused]] const stk::mesh::BulkData& bulk_data, SpherocylinderDataType data,
                         stk::mesh::Entity spherocylinder)
      : data_(data), spherocylinder_(spherocylinder) {
    MUNDY_THROW_ASSERT(bulk_data.entity_rank(spherocylinder_) == stk::topology::NODE_RANK &&
                           data_.spherocylinder_rank == stk::topology::NODE_RANK,
                       std::invalid_argument,
                       "Both the spherocylinder entity rank and the spherocylinder data rank must be NODE_RANK");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(spherocylinder_), std::invalid_argument,
                       "The given spherocylinder entity is not valid");
  }

  decltype(auto) center() {
    return data_access_t::center(data_, spherocylinder_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, spherocylinder_);
  }

  decltype(auto) orientation() {
    return data_access_t::orientation(data_, spherocylinder_);
  }

  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, spherocylinder_);
  }

  decltype(auto) radius() {
    // For those not familiar with decltype(auto), it allows us to return either an auto or an auto&.
    return data_access_t::radius(data_, spherocylinder_);
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data_, spherocylinder_);
  }

 private:
  SpherocylinderDataType data_;
  stk::mesh::Entity spherocylinder_;
};  // NodeSpherocylinderView

/// @brief An ngp-compatible view of a NODE_RANK STK entity meant to represent a spherocylinder
/// See the discussion for NodeSpherocylinderView for more information. The only difference is ngp-compatible data
/// access.
template <typename NgpSpherocylinderDataType>
class NgpNodeSpherocylinderView {
 public:
  using data_access_t = NgpSpherocylinderDataTraits<NgpSpherocylinderDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<NgpSpherocylinderDataType>(),
                                                 std::declval<stk::mesh::FastMeshIndex>()));
  using orientation_t = decltype(data_access_t::orientation(std::declval<NgpSpherocylinderDataType>(),
                                                            std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpNodeSpherocylinderView([[maybe_unused]] stk::mesh::NgpMesh ngp_mesh, NgpSpherocylinderDataType data,
                            stk::mesh::FastMeshIndex spherocylinder_index)
      : data_(data), spherocylinder_index_(spherocylinder_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return data_access_t::orientation(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return data_access_t::orientation(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data_, spherocylinder_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data_, spherocylinder_index_);
  }

 private:
  NgpSpherocylinderDataType data_;
  stk::mesh::FastMeshIndex spherocylinder_index_;
};  // NgpNodeSpherocylinderView

static_assert(
    ValidSpherocylinderType<ElemSpherocylinderView<
            SpherocylinderData<float, stk::mesh::Field<float>, mundy::math::Quaternion<float>, float>>> &&
        ValidSpherocylinderType<ElemSpherocylinderView<
            SpherocylinderData<float, stk::mesh::Field<float>, stk::mesh::Field<float>, stk::mesh::Field<float>>>> &&
        ValidSpherocylinderType<NgpElemSpherocylinderView<
            NgpSpherocylinderData<float, stk::mesh::NgpField<float>, mundy::math::Quaternion<float>, float>>> &&
        ValidSpherocylinderType<NgpElemSpherocylinderView<NgpSpherocylinderData<
            float, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>>,
    "ElemSpherocylinderView and NgpElemSpherocylinderView must be valid Spherocylinder types");

static_assert(
    ValidSpherocylinderType<NodeSpherocylinderView<
            SpherocylinderData<float, stk::mesh::Field<float>, mundy::math::Quaternion<float>, float>>> &&
        ValidSpherocylinderType<NodeSpherocylinderView<
            SpherocylinderData<float, stk::mesh::Field<float>, stk::mesh::Field<float>, stk::mesh::Field<float>>>> &&
        ValidSpherocylinderType<NgpNodeSpherocylinderView<
            NgpSpherocylinderData<float, stk::mesh::NgpField<float>, mundy::math::Quaternion<float>, float>>> &&
        ValidSpherocylinderType<NgpNodeSpherocylinderView<NgpSpherocylinderData<
            float, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>>,
    "NodeSpherocylinderView and NgpNodeSpherocylinderView must be valid Spherocylinder types");

/// \brief A helper function to create a ElemSpherocylinderView object with type deduction
template <typename SpherocylinderDataType>
auto create_elem_spherocylinder_view(const stk::mesh::BulkData& bulk_data, SpherocylinderDataType& data,
                                     stk::mesh::Entity spherocylinder) {
  return ElemSpherocylinderView<SpherocylinderDataType>(bulk_data, data, spherocylinder);
}

/// \brief A helper function to create a NgpElemSpherocylinderView object with type deduction
template <typename NgpSpherocylinderDataType>
auto create_ngp_elem_spherocylinder_view(stk::mesh::NgpMesh ngp_mesh, NgpSpherocylinderDataType data,
                                         stk::mesh::FastMeshIndex spherocylinder_index) {
  return NgpElemSpherocylinderView<NgpSpherocylinderDataType>(ngp_mesh, data, spherocylinder_index);
}

/// \brief A helper function to create a NodeSpherocylinderView object with type deduction
template <typename SpherocylinderDataType>
auto create_node_spherocylinder_view(const stk::mesh::BulkData& bulk_data, SpherocylinderDataType& data,
                                     stk::mesh::Entity spherocylinder) {
  MUNDY_THROW_ASSERT(bulk_data.entity_rank(spherocylinder) == stk::topology::NODE_RANK &&
                         data.spherocylinder_rank == stk::topology::NODE_RANK,
                     std::invalid_argument,
                     "Both the spherocylinder entity rank and the spherocylinder data rank must be NODE_RANK");
  return NodeSpherocylinderView<SpherocylinderDataType>(bulk_data, data, spherocylinder);
}

/// \brief A helper function to create a NgpNodeSpherocylinderView object with type deduction
template <typename NgpSpherocylinderDataType>
auto create_ngp_node_spherocylinder_view(stk::mesh::NgpMesh ngp_mesh, NgpSpherocylinderDataType data,
                                         stk::mesh::FastMeshIndex spherocylinder_index) {
  return NgpNodeSpherocylinderView<NgpSpherocylinderDataType>(ngp_mesh, data, spherocylinder_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDER_HPP_