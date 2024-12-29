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
#include <mundy_geom/primitives/Sphere.hpp>  // for mundy::geom::ValidSphereType
#include <mundy_mesh/BulkData.hpp>           // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>         // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief A struct to hold the data for a collection of spheres
///
/// This struct is a data aggregate that holds the data for a collection of spheres. The radius and center of each
/// sphere can be stored as either a reference to scalar or to an stk::mesh::Field of scalars, either of which may be
/// const or non-const.
///
/// \tparam Scalar The scalar type of the sphere's radius and center.
/// \tparam RadiusDataType The type of the radius data. Can either be a scalar or an stk::mesh::Field of scalars.
/// \tparam CenterDataType The type of the center data. Can either be a Point<Scalar> or an stk::mesh::Field of scalars.
template <typename Scalar, typename RadiusDataType = stk::mesh::Field<Scalar>,
          typename CenterDataType = stk::mesh::Field<Scalar>>
struct SphereData {
  static_assert((std::is_same_v<std::decay_t<RadiusDataType>, Scalar> ||
                 std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::Field<Scalar>>) &&
                    (std::is_same_v<std::decay_t<CenterDataType>, Point<Scalar>> ||
                     std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::Field<Scalar>>),
                "RadiusDataType and CenterDataType must be either a scalar or a field of scalars");

  using scalar_t = Scalar;
  using radius_t = RadiusDataType;
  using center_t = CenterDataType;

  radius_t& radius;
  center_t& center;
};  // SphereData

/// \brief A struct to hold the data for a collection of NGP-compatible spheres
/// See the discussion for SphereData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename RadiusDataType = stk::mesh::NgpField<Scalar>,
          typename CenterDataType = stk::mesh::NgpField<Scalar>>
struct NgpSphereData {
  static_assert((std::is_same_v<std::decay_t<RadiusDataType>, Scalar> ||
                 std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::NgpField<Scalar>>) &&
                    (std::is_same_v<std::decay_t<CenterDataType>, Point<Scalar>> ||
                     std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::NgpField<Scalar>>),
                "RadiusDataType and CenterDataType must be either a scalar or a field of scalars");

  using scalar_t = Scalar;
  using radius_t = RadiusDataType;
  using center_t = CenterDataType;

  radius_t& radius;
  center_t& center;
};  // NgpSphereData

/// \brief A helper function to create a SphereData object
///
/// This function creates a SphereData object with the given radius and center data
/// and is used to automatically deduce the template parameters.
/// The data may be either a scalar or an stk::mesh::Field of scalars.
template <typename Scalar, typename RadiusDataType, typename CenterDataType>
auto create_sphere_data(RadiusDataType& radius_data, CenterDataType& center_data) {
  return SphereData<Scalar, RadiusDataType, CenterDataType>{radius_data, center_data};
}

/// \brief A helper function to create a NgpSphereData object
/// See the discussion for create_sphere_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename RadiusDataType, typename CenterDataType>
auto create_ngp_sphere_data(RadiusDataType& radius_data, CenterDataType& center_data) {
  return NgpSphereData<Scalar, RadiusDataType, CenterDataType>{radius_data, center_data};
}

/// \brief A concept to check if a type provides the same data as SphereData
template <typename Agg>
concept ValidDefaultSphereDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::radius_t;
  typename Agg::center_t;
  std::is_same_v<std::decay_t<typename Agg::radius_t>, typename Agg::scalar_t> ||
      std::is_same_v<std::decay_t<typename Agg::radius_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  is_point_v<std::decay_t<typename Agg::center_t>> ||
      std::is_same_v<std::decay_t<typename Agg::center_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { agg.radius } -> std::convertible_to<typename Agg::radius_t&>;
  { agg.center } -> std::convertible_to<typename Agg::center_t&>;
};  // ValidDefaultSphereDataType

/// \brief A concept to check if a type provides the same data as NgpSphereData
template <typename Agg>
concept ValidDefaultNgpSphereDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::radius_t;
  typename Agg::center_t;
  std::is_same_v<std::decay_t<typename Agg::radius_t>, typename Agg::scalar_t> ||
      std::is_same_v<std::decay_t<typename Agg::radius_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  is_point_v<std::decay_t<typename Agg::center_t>> ||
      std::is_same_v<std::decay_t<typename Agg::center_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { agg.radius } -> std::convertible_to<typename Agg::radius_t&>;
  { agg.center } -> std::convertible_to<typename Agg::center_t&>;
};  // ValidDefaultNgpSphereDataType

static_assert(ValidDefaultSphereDataType<SphereData<float, float, Point<float>>> &&
                  ValidDefaultSphereDataType<SphereData<float, stk::mesh::Field<float>, Point<float>>> &&
                  ValidDefaultSphereDataType<SphereData<float, float, stk::mesh::Field<float>>> &&
                  ValidDefaultSphereDataType<SphereData<float, stk::mesh::Field<float>, stk::mesh::Field<float>>>,
              "SphereData must satisfy the ValidDefaultSphereDataType concept");

static_assert(
    ValidDefaultNgpSphereDataType<NgpSphereData<float, float, Point<float>>> &&
        ValidDefaultNgpSphereDataType<NgpSphereData<float, stk::mesh::NgpField<float>, Point<float>>> &&
        ValidDefaultNgpSphereDataType<NgpSphereData<float, float, stk::mesh::NgpField<float>>> &&
        ValidDefaultNgpSphereDataType<NgpSphereData<float, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>,
    "NgpSphereData must satisfy the ValidDefaultNgpSphereDataType concept");

/// \brief A helper function to get an updated NgpSphereData object from a SphereData object
/// \param data The SphereData object to convert
template <ValidDefaultSphereDataType SphereDataType>
auto get_updated_ngp_data(SphereDataType data) {
  using scalar_t = typename SphereDataType::scalar_t;
  using radius_t = typename SphereDataType::radius_t;
  using center_t = typename SphereDataType::center_t;

  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<radius_t>, stk::mesh::Field<scalar_t>>;
  constexpr bool is_center_a_field = std::is_same_v<std::decay_t<center_t>, stk::mesh::Field<scalar_t>>;
  if constexpr (is_radius_a_field && is_center_a_field) {
    return create_ngp_sphere_data<scalar_t>(stk::mesh::get_updated_ngp_field<scalar_t>(data.radius),
                                            stk::mesh::get_updated_ngp_field<scalar_t>(data.center));
  } else if constexpr (is_radius_a_field && !is_center_a_field) {
    return create_ngp_sphere_data<scalar_t>(stk::mesh::get_updated_ngp_field<scalar_t>(data.radius), data.center);
  } else if constexpr (!is_radius_a_field && is_center_a_field) {
    return create_ngp_sphere_data<scalar_t>(data.radius, stk::mesh::get_updated_ngp_field<scalar_t>(data.center));
  } else {
    return create_ngp_sphere_data<scalar_t>(data.radius, data.center);
  }
}

/// \brief A traits class to provide abstracted access to a sphere's data via an aggregate
///
/// By default, this class is compatible with SphereData or any class the meets the ValidDefaultSphereDataType concept.
/// Users can specialize this class to support other aggregate types.
template <typename Agg>
struct SphereDataTraits {
  static_assert(
      ValidDefaultSphereDataType<Agg>,
      "Agg must satisfy the ValidDefaultSphereDataType concept.\n"
      "Basically, Agg must have all the same things as NgpSphereData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using point_t = Point<scalar_t>;

  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<typename Agg::radius_t>, scalar_t>;
  }

  static constexpr bool has_shared_center() {
    return std::is_same_v<std::decay_t<typename Agg::center_t>, point_t>;
  }

  static decltype(auto) radius(Agg agg, stk::mesh::Entity sphere_elem) {
    if constexpr (has_shared_radius()) {
      return agg.radius;
    } else {
      return stk::mesh::field_data(agg.radius, sphere_elem)[0];
    }
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity sphere_node) {
    if constexpr (has_shared_center()) {
      return agg.center;
    } else {
      // This returns a copy of a view into the raw data of the field.
      return mundy::mesh::vector3_field_data(agg.center, sphere_node);
    }
  }
};  // SphereDataTraits

/// \brief A traits class to provide abstracted access to a sphere's data via an NGP-compatible aggregate
/// See the discussion for SphereDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpSphereDataTraits {
  static_assert(
      ValidDefaultNgpSphereDataType<Agg>,
      "Agg must satisfy the ValidDefaultNgpSphereDataType concept.\n"
      "Basically, Agg must have all the same things as NgpSphereData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using point_t = Point<scalar_t>;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<typename Agg::radius_t>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_center() {
    return std::is_same_v<std::decay_t<typename Agg::center_t>, point_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) radius(Agg agg, stk::mesh::FastMeshIndex sphere_elem_index) {
    if constexpr (has_shared_radius()) {
      return (agg.radius);
    } else {
      return agg.radius(sphere_elem_index, 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex sphere_node_index) {
    if constexpr (has_shared_center()) {
      return (agg.center);
    } else {
      return mundy::mesh::vector3_field_data(agg.center, sphere_node_index);
    }
  }
};  // NgpSphereDataTraits

/// @brief A view of an ELEM_RANK, PARTICLE topology sphere entity
template <typename SphereDataType>
class SphereEntityView {
 public:
  using scalar_t = typename SphereDataType::scalar_t;
  using point_t = Point<scalar_t>;
  using data_access_t = SphereDataTraits<SphereDataType>;

  SphereEntityView(const stk::mesh::BulkData& bulk_data, SphereDataType data, stk::mesh::Entity sphere)
      : data_(data), sphere_(sphere), node_(bulk_data.begin_nodes(sphere_)[0]) {
  }

  decltype(auto) radius() {
    // For those not familiar with decltype(auto), it allows us to return either an auto or an auto&.
    return data_access_t::radius(data_, sphere_);
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data_, sphere_);
  }

  decltype(auto) center() {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, node_);
  }

 private:
  SphereDataType data_;
  stk::mesh::Entity sphere_;
  stk::mesh::Entity node_;
};  // SphereEntityView

/// @brief An ngp-compatible view of an ELEM_RANK, PARTICLE topology sphere entity
template <typename NgpSphereDataType>
class NgpSphereEntityView {
 public:
  using scalar_t = typename NgpSphereDataType::scalar_t;
  using point_t = Point<scalar_t>;
  using data_access_t = NgpSphereDataTraits<NgpSphereDataType>;

  KOKKOS_INLINE_FUNCTION
  NgpSphereEntityView(stk::mesh::NgpMesh ngp_mesh, NgpSphereDataType data, stk::mesh::FastMeshIndex sphere_index)
      : data_(data),
        sphere_index_(sphere_index),
        node_index_(ngp_mesh.fast_mesh_index(ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index_)[0])) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return NgpSphereDataTraits<NgpSphereDataType>::radius(data_, sphere_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return NgpSphereDataTraits<NgpSphereDataType>::radius(data_, sphere_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return NgpSphereDataTraits<NgpSphereDataType>::center(data_, node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return NgpSphereDataTraits<NgpSphereDataType>::center(data_, node_index_);
  }

 private:
  NgpSphereDataType data_;
  stk::mesh::FastMeshIndex sphere_index_;
  stk::mesh::FastMeshIndex node_index_;
};  // NgpSphereEntityView

static_assert(ValidSphereType<SphereEntityView<SphereData<float, float, Point<float>>>>,
              "SphereEntityView must be a valid Sphere type");
static_assert(ValidSphereType<SphereEntityView<SphereData<float, stk::mesh::Field<float>, Point<float>>>>,
              "SphereEntityView must be a valid Sphere type");
static_assert(ValidSphereType<SphereEntityView<SphereData<float, float, stk::mesh::Field<float>>>>,
              "SphereEntityView must be a valid Sphere type");
static_assert(ValidSphereType<SphereEntityView<SphereData<float, stk::mesh::Field<float>, stk::mesh::Field<float>>>>,
              "SphereEntityView must be a valid Sphere type");
static_assert(ValidSphereType<NgpSphereEntityView<NgpSphereData<float, float, Point<float>>>>,
              "NgpSphereEntityView must be a valid Sphere type");
static_assert(ValidSphereType<NgpSphereEntityView<NgpSphereData<float, stk::mesh::NgpField<float>, Point<float>>>>,
              "NgpSphereEntityView must be a valid Sphere type");
static_assert(ValidSphereType<NgpSphereEntityView<NgpSphereData<float, float, stk::mesh::NgpField<float>>>>,
              "NgpSphereEntityView must be a valid Sphere type");
static_assert(
    ValidSphereType<NgpSphereEntityView<NgpSphereData<float, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>>,
    "NgpSphereEntityView must be a valid Sphere type");

/// \brief A helper function to create a SphereEntityView object with type deduction
template <typename SphereDataType>
auto create_sphere_entity_view(const stk::mesh::BulkData& bulk_data, SphereDataType& data, stk::mesh::Entity sphere) {
  return SphereEntityView<SphereDataType>(bulk_data, data, sphere);
}

/// \brief A helper function to create a NgpSphereEntityView object with type deduction
template <typename NgpSphereDataType>
auto create_ngp_sphere_entity_view(stk::mesh::NgpMesh ngp_mesh, NgpSphereDataType data,
                                   stk::mesh::FastMeshIndex sphere_index) {
  return NgpSphereEntityView<NgpSphereDataType>(ngp_mesh, data, sphere_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHERE_HPP_