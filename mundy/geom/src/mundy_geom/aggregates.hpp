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

#ifndef MUNDY_GEOM_AGGREGATES_HPP_
#define MUNDY_GEOM_AGGREGATES_HPP_

// C++ core

// Kokkos and Kokkos-Kernels
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK Mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/GetNgpMesh.hpp>   // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh
#include <stk_mesh/base/Part.hpp>         // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>     // for stk::mesh::Selector

// Mundy mesh
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MeshBuilder.hpp>      // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>     // for mundy::mesh::field_fill, mundy::mesh::field_copy, etc
#include <mundy_mesh/fmt_stk_types.hpp>    // adds fmt::format for stk types
#include <mundy_mesh/utils/FillFieldWithValue.hpp>  // for mundy::mesh::utils::fill_field_with_value

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/*
Notes, I would like to be able to construct a view of an ELEM_RANK PARTICLE topology entity meant to represent a sphere.
The view is constructed from the entity itself and the mesh. During construction, the nodes of the entity are fetched
and stored in the view. This way, we aren't repeatedly fetching the nodes of the entity every time we want to access
their data.

This view is a meant to satisfy the SphereTraits concept and provide a way to access the sphere's center and radius
agnostic of how we choose to store the sphere's data. This way, we can easily switch between different storage schemes
without changing the code that uses the sphere.


We will use the NGP mesh ONLY.
*/

template <typename Scalar, bool has_shared_radius_, bool has_shared_center_>
struct SphereData {
  static constexpr bool has_shared_radius = has_shared_radius_;
  static constexpr bool has_shared_center = has_shared_center_;
  using scalar_t = Scalar;
  using radius_t = std::conditional_t<has_shared_radius, Scalar, stk::mesh::NgpField<Scalar>>;
  using center_t = std::conditional_t<has_shared_center, mundy::geom::Point<Scalar>, stk::mesh::NgpField<Scalar>>;

  radius_t radius;
  center_t center;
};

template <typename Scalar>
SphereData create_sphere_data(auto& radius_or_radius_field, auto& center_or_center_field) {
  constexpr bool has_shared_radius = std::is_same_v<decltype(radius_or_radius_field), Scalar>;
  constexpr bool has_shared_center = std::is_same_v<decltype(center_or_center_field), mundy::geom::Point<Scalar>>;

  static_assert(std::is_same_v<decltype(radius_or_radius_field), Scalar> ||
                    std::is_same_v<decltype(radius_or_radius_field), stk::mesh::NgpField<Scalar>>,
                "radius_or_radius_field must be either a scalar or an ngp field of scalars");
  static_assert(std::is_same_v<decltype(center_or_center_field), mundy::geom::Point<Scalar>> ||
                    std::is_same_v<decltype(center_or_center_field), stk::mesh::NgpField<Scalar>>,
                "center_or_center_field must be either a Point of scalar value type or an ngp field of scalars");

  return SphereData<Scalar, has_shared_radius, has_shared_center>{radius_or_radius_field, center_or_center_field};
}

template <typename Agg>
concept ValidDefaultSphereDataType = requires(Agg agg) {
  { Agg::scalar_t } -> std::convertible_to<float>;
  { Agg::has_shared_radius } -> std::convertible_to<bool>;
  { Agg::has_shared_center } -> std::convertible_to<bool>;
  {
    Agg::radius_t
  } -> std::convertible_to<std::conditional_t<Agg::has_shared_radius,  //
                                              float,                   //
                                              stk::mesh::NgpField<float>>>;
  {
    Agg::center_t
  } -> std::convertible_to < std::conditional_t<Agg::has_shared_center,     //
                                                mundy::geom::Point<float>,  //
                                                stk::mesh::NgpField<float>>;
  {
    agg.radius
  } -> std::convertible_to<std::conditional_t<Agg::has_shared_radius,  //
                                              float,                   //
                                              stk::mesh::NgpField<float>>>;
  {
    agg.center
  } -> std::convertible_to < std::conditional_t<Agg::has_shared_center,     //
                                                mundy::geom::Point<float>,  //
                                                stk::mesh::NgpField<float>>;
};  // ValidDefaultSphereDataType

template <typename Agg>
struct SphereDataTraits {
  static_assert(ValidDefaultSphereDataType<Agg>,
                "Agg must satisfy the ValidDefaultSphereDataType concept.\n"
                "Basically, Agg must have all the same things as SphereData but is free to extend it as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  static constexpr bool has_shared_radius() {
    return typename Agg::has_shared_radius;
  }

  static constexpr bool has_shared_center() {
    return typename Agg::has_shared_center;
  }

  static const scalar_t& radius(const Agg& agg, stk::mesh::FastMeshIndex sphere_elem_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius;
    } else {
      return agg.radius(sphere_elem_index, 0);
    }
  }

  static scalar_t& radius(Agg& agg, stk::mesh::FastMeshIndex sphere_elem_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius;
    } else {
      return agg.radius(sphere_elem_index, 0);
    }
  }

  static const mundy::mesh::Vector3<scalar_t>& center(const Agg& agg, stk::mesh::FastMeshIndex sphere_node_index) {
    if constexpr (has_shared_center()) {
      return agg.center;
    } else {
      return mundy::mesh::vector3_field_data(agg.center, sphere_node_index);
    }
  }

  static mundy::mesh::Vector3<scalar_t>& center(Agg& agg, stk::mesh::FastMeshIndex sphere_node_index) {
    if constexpr (has_shared_center()) {
      return agg.center;
    } else {
      return mundy::mesh::vector3_field_data(agg.center, sphere_node_index);
    }
  }
};

/// @brief A view of an ELEM_RANK, PARTICLE topology sphere entity
template <typename SphereDataType>
struct SphereEntityView {
  stk::mesh::NgpMesh ngp_mesh;
  SphereDataType data;
  stk::mesh::FastMeshIndex sphere_index;
  stk::mesh::NgpMesh::ConnectedNodes nodes;

  using scalar_t = typename FieldsType::scalar_t;

  SphereEntityView(stk::mesh::NgpMesh ngp_mesh_, SphereDataType data_, stk::mesh::FastMeshIndex sphere_index_)
      : ngp_mesh(ngp_mesh_), data(data_), sphere_index(sphere_index_), nodes(ngp_mesh.get_nodes(sphere_index)) {
  }

  scalar_t& radius() {
    return SphereDataTraits<SphereDataType>::radius(data, sphere_index);
  }

  const scalar_t& radius() const {
    return SphereDataTraits<SphereDataType>::radius(data, sphere_index);
  }

  mundy::geom::Point<scalar_t>& center() {
    return SphereDataTraits<SphereDataType>::center(data, ngp_mesh.fast_mesh_index(nodes[0]));
  }

  const mundy::geom::Point<scalar_t>& center() const {
    return SphereDataTraits<SphereDataType>::center(data, ngp_mesh.fast_mesh_index(nodes[0]));
  }
};

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_HPP_