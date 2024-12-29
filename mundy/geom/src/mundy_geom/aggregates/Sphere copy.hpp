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

template <typename Scalar, bool has_shared_radius_ = false, bool has_shared_center_ = false>
struct SphereData {
  static constexpr bool has_shared_radius = has_shared_radius_;
  static constexpr bool has_shared_center = has_shared_center_;
  using scalar_t = Scalar;
  using radius_t = std::conditional_t<has_shared_radius, Scalar&, stk::mesh::Field<Scalar>&>;
  using center_t = std::conditional_t<has_shared_center, Point<Scalar>&, stk::mesh::Field<Scalar>&>;

  radius_t radius;
  center_t center;
};  // SphereData

template <typename Scalar, bool has_shared_radius_ = false, bool has_shared_center_ = false>
struct NgpSphereData {
  static constexpr bool has_shared_radius = has_shared_radius_;
  static constexpr bool has_shared_center = has_shared_center_;
  using scalar_t = Scalar;
  using radius_t = std::conditional_t<has_shared_radius, Scalar&, stk::mesh::NgpField<Scalar>&>;
  using center_t = std::conditional_t<has_shared_center, Point<Scalar>&, stk::mesh::NgpField<Scalar>&>;

  radius_t radius;
  center_t center;
};  // NgpSphereData

template <typename Scalar>
auto create_sphere_data(auto& radius_or_radius_field, auto& center_or_center_field) {
  using radius_t = std::remove_cv_t<std::remove_reference_t<decltype(radius_or_radius_field)>>;
  using center_t = std::remove_cv_t<std::remove_reference_t<decltype(center_or_center_field)>>;

  constexpr bool has_shared_radius = std::is_same_v<radius_t, Scalar>;
  constexpr bool has_shared_center = std::is_same_v<center_t, Point<Scalar>>;

  static_assert(std::is_same_v<radius_t, Scalar> || std::is_same_v<radius_t, stk::mesh::Field<Scalar>>,
                "radius_or_radius_field must be either a scalar or a field of scalars");
  static_assert(std::is_same_v<center_t, Point<Scalar>> || std::is_same_v<center_t, stk::mesh::Field<Scalar>>,
                "center_or_center_field must be either a Point of scalar value type or a field of scalars");

  return SphereData<Scalar, has_shared_radius, has_shared_center>{radius_or_radius_field, center_or_center_field};
}

template <typename Scalar>
auto create_ngp_sphere_data(auto& radius_or_radius_field, auto& center_or_center_field) {
  using radius_t = std::remove_cv_t<std::remove_reference_t<decltype(radius_or_radius_field)>>;
  using center_t = std::remove_cv_t<std::remove_reference_t<decltype(center_or_center_field)>>;

  constexpr bool has_shared_radius = std::is_same_v<radius_t, Scalar>;
  constexpr bool has_shared_center = std::is_same_v<center_t, Point<Scalar>>;

  static_assert(std::is_same_v<radius_t, Scalar> || std::is_same_v<radius_t, stk::mesh::NgpField<Scalar>>,
                "radius_or_radius_field must be either a scalar or an ngp field of scalars");
  static_assert(std::is_same_v<center_t, Point<Scalar>> || std::is_same_v<center_t, stk::mesh::NgpField<Scalar>>,
                "center_or_center_field must be either a Point of scalar value type or an ngp field of scalars");

  return NgpSphereData<Scalar, has_shared_radius, has_shared_center>{radius_or_radius_field, center_or_center_field};
}

template <typename Agg>
concept ValidDefaultSphereDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::radius_t;
  typename Agg::center_t;
  { Agg::has_shared_radius } -> std::convertible_to<bool>;
  { Agg::has_shared_center } -> std::convertible_to<bool>;
  {
    agg.radius
  } -> std::convertible_to<std::conditional_t<Agg::has_shared_radius,   //
                                              typename Agg::scalar_t&,  //
                                              stk::mesh::Field<typename Agg::scalar_t>&>>;
  {
    agg.center
  } -> std::convertible_to<std::conditional_t<Agg::has_shared_center,          //
                                              Point<typename Agg::scalar_t>&,  //
                                              stk::mesh::Field<typename Agg::scalar_t>&>>;
};  // ValidDefaultSphereDataType

template <typename Agg>
concept ValidDefaultNgpSphereDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::radius_t;
  typename Agg::center_t;
  { Agg::has_shared_radius } -> std::convertible_to<bool>;
  { Agg::has_shared_center } -> std::convertible_to<bool>;
  {
    agg.radius
  } -> std::convertible_to<std::conditional_t<Agg::has_shared_radius,   //
                                              typename Agg::scalar_t&,  //
                                              stk::mesh::NgpField<typename Agg::scalar_t>&>>;
  {
    agg.center
  } -> std::convertible_to<std::conditional_t<Agg::has_shared_center,          //
                                              Point<typename Agg::scalar_t>&,  //
                                              stk::mesh::NgpField<typename Agg::scalar_t>&>>;
};  // ValidDefaultNgpSphereDataType

static_assert(ValidDefaultSphereDataType<SphereData<float, true, true>> &&
                  ValidDefaultSphereDataType<SphereData<float, true, false>> &&
                  ValidDefaultSphereDataType<SphereData<float, false, true>> &&
                  ValidDefaultSphereDataType<SphereData<float, false, false>>,
              "SphereData must satisfy the ValidDefaultSphereDataType concept");

static_assert(ValidDefaultNgpSphereDataType<NgpSphereData<float, true, true>> &&
                  ValidDefaultNgpSphereDataType<NgpSphereData<float, true, false>> &&
                  ValidDefaultNgpSphereDataType<NgpSphereData<float, false, true>> &&
                  ValidDefaultNgpSphereDataType<NgpSphereData<float, false, false>>,
              "NgpSphereData must satisfy the ValidDefaultNgpSphereDataType concept");

template <ValidDefaultSphereDataType SphereDataType>
auto get_updated_ngp_data(SphereDataType& data) {
  using scalar_t = typename SphereDataType::scalar_t;
  if constexpr (SphereDataType::has_shared_radius && SphereDataType::has_shared_center) {
    return create_ngp_sphere_data<scalar_t>(data.radius, data.center);
  } else if constexpr (SphereDataType::has_shared_radius && !SphereDataType::has_shared_center) {
    return create_ngp_sphere_data<scalar_t>(data.radius, stk::mesh::get_updated_ngp_field<scalar_t>(data.center));
  } else if constexpr (!SphereDataType::has_shared_radius && SphereDataType::has_shared_center) {
    return create_ngp_sphere_data<scalar_t>(stk::mesh::get_updated_ngp_field<scalar_t>(data.radius), data.center);
  } else {
    return create_ngp_sphere_data<scalar_t>(stk::mesh::get_updated_ngp_field<scalar_t>(data.radius),
                                            stk::mesh::get_updated_ngp_field<scalar_t>(data.center));
  }
}

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
    return Agg::has_shared_radius;
  }

  static constexpr bool has_shared_center() {
    return Agg::has_shared_center;
  }

  static const scalar_t& radius(const Agg& agg, stk::mesh::Entity sphere_elem) {
    if constexpr (has_shared_radius()) {
      return agg.radius;
    } else {
      return stk::mesh::field_data(agg.radius, sphere_elem)[0];
    }
  }

  static scalar_t& radius(Agg& agg, stk::mesh::Entity sphere_elem) {
    if constexpr (has_shared_radius()) {
      return agg.radius;
    } else {
      return stk::mesh::field_data(agg.radius, sphere_elem)[0];
    }
  }

  static const point_t& center(const Agg& agg, stk::mesh::Entity sphere_node) {
    if constexpr (has_shared_center()) {
      return agg.center;
    } else {
      return mundy::mesh::vector3_field_data(agg.center, sphere_node);
    }
  }

  static point_t& center(Agg& agg, stk::mesh::Entity sphere_node) {
    if constexpr (has_shared_center()) {
      return agg.center;
    } else {
      return mundy::mesh::vector3_field_data(agg.center, sphere_node);
    }
  }
};  // SphereDataTraits

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
    return Agg::has_shared_radius;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_center() {
    return Agg::has_shared_center;
  }

  KOKKOS_INLINE_FUNCTION
  static const scalar_t& radius(const Agg& agg, stk::mesh::FastMeshIndex sphere_elem_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius;
    } else {
      return agg.radius(sphere_elem_index, 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static scalar_t& radius(Agg& agg, stk::mesh::FastMeshIndex sphere_elem_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius;
    } else {
      return agg.radius(sphere_elem_index, 0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static const point_t& center(const Agg& agg, stk::mesh::FastMeshIndex sphere_node_index) {
    if constexpr (has_shared_center()) {
      return agg.center;
    } else {
      return mundy::mesh::vector3_field_data(agg.center, sphere_node_index);
    }
  }

  KOKKOS_INLINE_FUNCTION
  static point_t& center(Agg& agg, stk::mesh::FastMeshIndex sphere_node_index) {
    if constexpr (has_shared_center()) {
      return agg.center;
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

  SphereEntityView(const stk::mesh::BulkData& bulk_data, const SphereDataType& data, stk::mesh::Entity sphere)
      : data_(data), sphere_(sphere), node_(bulk_data.begin_nodes(sphere_)[0]) {
  }

  scalar_t& radius() {
    return data_access_t::radius(data_, sphere_);
  }

  const scalar_t& radius() const {
    return data_access_t::radius(data_, sphere_);
  }

  point_t& center() {
    return data_access_t::center(data_, node_);
  }

  const point_t& center() const {
    return data_access_t::center(data_, node_);
  }

 private:
  const SphereDataType& data_;
  stk::mesh::Entity sphere_;
  stk::mesh::Entity node_;
};  // SphereEntityView

/// @brief A view of an ELEM_RANK, PARTICLE topology sphere entity
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
  scalar_t& radius() {
    return SphereDataTraits<NgpSphereDataType>::radius(data_, sphere_index_);
  }

  KOKKOS_INLINE_FUNCTION
  const scalar_t& radius() const {
    return SphereDataTraits<NgpSphereDataType>::radius(data_, sphere_index_);
  }

  KOKKOS_INLINE_FUNCTION
  point_t& center() {
    return SphereDataTraits<NgpSphereDataType>::center(data_, node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  const point_t& center() const {
    return SphereDataTraits<NgpSphereDataType>::center(data_, node_index_);
  }

 private:
  NgpSphereDataType data_;
  stk::mesh::FastMeshIndex sphere_index_;
  stk::mesh::FastMeshIndex node_index_;
};  // NgpSphereEntityView

static_assert(ValidSphereType<SphereEntityView<SphereData<float, true, true>>> &&
                  ValidSphereType<SphereEntityView<SphereData<float, true, false>>> &&
                  ValidSphereType<SphereEntityView<SphereData<float, false, true>>> &&
                  ValidSphereType<SphereEntityView<SphereData<float, false, false>>>,
              "SphereEntityView must be a valid Sphere type");

static_assert(ValidSphereType<NgpSphereEntityView<NgpSphereData<float, true, true>>> &&
                  ValidSphereType<NgpSphereEntityView<NgpSphereData<float, true, false>>> &&
                  ValidSphereType<NgpSphereEntityView<NgpSphereData<float, false, true>>> &&
                  ValidSphereType<NgpSphereEntityView<NgpSphereData<float, false, false>>>,
              "NgpSphereEntityView must be a valid Sphere type");
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHERE_HPP_