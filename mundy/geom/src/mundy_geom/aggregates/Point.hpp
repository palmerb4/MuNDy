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

#ifndef MUNDY_GEOM_AGGREGATES_POINT_HPP_
#define MUNDY_GEOM_AGGREGATES_POINT_HPP_

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::ValidPointType
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>        // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief A struct to hold the data for a collection of points
///
/// While the center of a point is just the point's position, this naming convention allows any PARTICLE or NODE
/// topology aggregate (such as Spheres or Ellipsoids) to be treated as a collection of points.
template <typename Scalar, typename CenterDataType = stk::mesh::Field<Scalar>>
struct PointData {
  static_assert(is_point_v<std::decay_t<CenterDataType>> ||
                    std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::Field<Scalar>>,
                "CenterDataType must be either a scalar or a field of scalars");

  using scalar_t = Scalar;
  using center_data_t = CenterDataType;

  stk::topology::rank_t point_rank;
  center_data_t& center_data;
};  // PointData

/// \brief A struct to hold the data for a collection of NGP-compatible points
/// See the discussion for PointData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename CenterDataType = stk::mesh::NgpField<Scalar>>
struct NgpPointData {
  static_assert(is_point_v<std::decay_t<CenterDataType>> ||
                    std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::NgpField<Scalar>>,
                "CenterDataType must be either a scalar or a field of scalars");

  using scalar_t = Scalar;
  using center_data_t = CenterDataType;

  stk::topology::rank_t point_rank;
  center_data_t& center_data;
};  // NgpPointData

/// \brief A helper function to create a PointData object
///
/// This function creates a PointData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar, typename CenterDataType>
auto create_point_data(stk::topology::rank_t point_rank, CenterDataType& center_data) {
  constexpr bool is_center_a_field = std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::Field<Scalar>>;
  if constexpr (is_center_a_field) {
    MUNDY_THROW_ASSERT(center_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center data must be a field of NODE_RANK");
  }
  return PointData<Scalar, CenterDataType>{point_rank, center_data};
}

/// \brief A helper function to create a NgpPointData object
/// See the discussion for create_point_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename CenterDataType>
auto create_ngp_point_data(stk::topology::rank_t point_rank, CenterDataType& center_data) {
  constexpr bool is_center_a_field = std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::NgpField<Scalar>>;
  if constexpr (is_center_a_field) {
    MUNDY_THROW_ASSERT(center_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center data must be a field of NODE_RANK");
  }
  return NgpPointData<Scalar, CenterDataType>{point_rank, center_data};
}

/// \brief A concept to check if a type provides the same data as PointData
template <typename Agg>
concept ValidDefaultPointDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::center_data_t;
  is_point_v<std::decay_t<typename Agg::center_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::center_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { agg.point_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.center_data } -> std::convertible_to<typename Agg::center_data_t&>;
};  // ValidDefaultPointDataType

/// \brief A concept to check if a type provides the same data as NgpPointData
template <typename Agg>
concept ValidDefaultNgpPointDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::center_data_t;
  is_point_v<std::decay_t<typename Agg::center_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::center_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { agg.point_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.center_data } -> std::convertible_to<typename Agg::center_data_t&>;
};  // ValidDefaultNgpPointDataType

static_assert(ValidDefaultPointDataType<PointData<float, Point<float>>> &&
                  ValidDefaultPointDataType<PointData<float, stk::mesh::Field<float>>>,
              "PointData must satisfy the ValidDefaultPointDataType concept");

static_assert(ValidDefaultNgpPointDataType<NgpPointData<float, Point<float>>> &&
                  ValidDefaultNgpPointDataType<NgpPointData<float, stk::mesh::NgpField<float>>>,
              "NgpPointData must satisfy the ValidDefaultNgpPointDataType concept");

/// \brief A helper function to get an updated NgpPointData object from a PointData object
/// \param data The PointData object to convert
template <ValidDefaultPointDataType PointDataType>
auto get_updated_ngp_data(PointDataType data) {
  using scalar_t = typename PointDataType::scalar_t;
  using center_data_t = typename PointDataType::center_data_t;

  constexpr bool is_center_a_field = std::is_same_v<std::decay_t<center_data_t>, stk::mesh::Field<scalar_t>>;
  if constexpr (is_center_a_field) {
    return create_ngp_point_data<scalar_t>(data.point_rank,  //
                                           stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data));
  } else {
    return create_ngp_point_data<scalar_t>(data.point_rank, data.center_data);
  }
}

/// \brief A traits class to provide abstracted access to a point's data via an aggregate
///
/// By default, this class is compatible with PointData or any class the meets the ValidDefaultPointDataType concept.
/// Users can specialize this class to support other aggregate types.
template <typename Agg>
struct PointDataTraits {
  static_assert(
      ValidDefaultPointDataType<Agg>,
      "Agg must satisfy the ValidDefaultPointDataType concept.\n"
      "Basically, Agg must have all the same things as NgpPointData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;

  static constexpr bool has_shared_center() {
    return is_point_v<center_data_t>;
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity point_node) {
    if constexpr (has_shared_center()) {
      return agg.center_data;
    } else {
      return mundy::mesh::vector3_field_data(agg.center_data, point_node);
    }
  }
};  // PointDataTraits

/// \brief A traits class to provide abstracted access to a point's data via an NGP-compatible aggregate
/// See the discussion for PointDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpPointDataTraits {
  static_assert(
      ValidDefaultNgpPointDataType<Agg>,
      "Agg must satisfy the ValidDefaultNgpPointDataType concept.\n"
      "Basically, Agg must have all the same things as NgpPointData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_center() {
    return is_point_v<center_data_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex point_node_index) {
    if constexpr (has_shared_center()) {
      return agg.center_data;
    } else {
      return mundy::mesh::vector3_field_data(agg.center_data, point_node_index);
    }
  }
};  // NgpPointDataTraits

/// @brief A view of an STK entity meant to represent a point
/// If the point_rank is NODE_RANK, then the point is just a node entity with node center.
/// If the point_rank is ELEM_RANK, then the point is a particle entity with node center.
template <typename PointDataType>
class ElemPointView {
 public:
  using data_access_t = PointDataTraits<PointDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<PointDataType>(), std::declval<stk::mesh::Entity>()));

  ElemPointView(const stk::mesh::BulkData& bulk_data, PointDataType data, stk::mesh::Entity point)
      : data_(data), point_(point), node_(bulk_data.begin_nodes(point_)[0]) {
    MUNDY_THROW_ASSERT(
        bulk_data.entity_rank(point_) == stk::topology::ELEM_RANK && data_.point_rank == stk::topology::ELEM_RANK,
        std::invalid_argument, "Both the point entity rank and the point data rank must be ELEM_RANK");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(point_), std::invalid_argument, "The given point entity is not valid");
    MUNDY_THROW_ASSERT(bulk_data.num_nodes(point_) == 1, std::invalid_argument,
                       "The given point entity must have exactly one node");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(node_), std::invalid_argument,
                       "The node entity associated with the point is not valid");
  }

  decltype(auto) center() {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) operator[](int i) {
    return center()[i];
  }

  decltype(auto) operator[](int i) const {
    return center()[i];
  }

 private:
  PointDataType data_;
  stk::mesh::Entity point_;
  stk::mesh::Entity node_;
};  // ElemPointView

/// @brief An ngp-compatible view of an ELEM_RANK STK entity meant to represent a point
/// See the discussion for ElemPointView for more information. The only difference is ngp-compatible data access.
template <typename NgpPointDataType>
class NgpElemPointView {
 public:
  using data_access_t = NgpPointDataTraits<NgpPointDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<NgpPointDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpElemPointView(stk::mesh::NgpMesh ngp_mesh, NgpPointDataType data, stk::mesh::FastMeshIndex point_index)
      : data_(data),
        point_index_(point_index),
        node_index_(ngp_mesh.fast_mesh_index(ngp_mesh.get_nodes(data_.point_rank, point_index_)[0])) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data_, node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data_, node_index_);
  }

  decltype(auto) operator[](int i) {
    return center()[i];
  }

  decltype(auto) operator[](int i) const {
    return center()[i];
  }

 private:
  NgpPointDataType data_;
  stk::mesh::FastMeshIndex point_index_;
  stk::mesh::FastMeshIndex node_index_;
};  // NgpElemPointView

/// @brief A view of a NODE_RANK STK entity meant to represent a point
template <typename PointDataType>
class NodePointView {
 public:
  using data_access_t = PointDataTraits<PointDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<PointDataType>(), std::declval<stk::mesh::Entity>()));

  NodePointView([[maybe_unused]] const stk::mesh::BulkData& bulk_data, PointDataType data, stk::mesh::Entity point)
      : data_(data), point_(point) {
    MUNDY_THROW_ASSERT(
        bulk_data.entity_rank(point_) == stk::topology::NODE_RANK && data_.point_rank == stk::topology::NODE_RANK,
        std::invalid_argument, "Both the point entity rank and the point data rank must be NODE_RANK");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(point_), std::invalid_argument, "The given point entity is not valid");
  }

  decltype(auto) center() {
    return data_access_t::center(data_, point_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, point_);
  }

  decltype(auto) operator[](int i) {
    return center()[i];
  }

  decltype(auto) operator[](int i) const {
    return center()[i];
  }

 private:
  PointDataType data_;
  stk::mesh::Entity point_;
};  // NodePointView

/// @brief An ngp-compatible view of a NODE_RANK STK entity meant to represent a point
/// See the discussion for NodePointView for more information. The only difference is ngp-compatible data access.
template <typename NgpPointDataType>
class NgpNodePointView {
 public:
  using data_access_t = NgpPointDataTraits<NgpPointDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<NgpPointDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpNodePointView([[maybe_unused]] stk::mesh::NgpMesh ngp_mesh, NgpPointDataType data,
                   stk::mesh::FastMeshIndex point_index)
      : data_(data), point_index_(point_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data_, point_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data_, point_index_);
  }

  decltype(auto) operator[](int i) {
    return center()[i];
  }

  decltype(auto) operator[](int i) const {
    return center()[i];
  }

 private:
  NgpPointDataType data_;
  stk::mesh::FastMeshIndex point_index_;
};  // NgpNodePointView

static_assert(ValidPointType<ElemPointView<PointData<float, Point<float>>>> &&
                  ValidPointType<ElemPointView<PointData<float, stk::mesh::Field<float>>>> &&
                  ValidPointType<NgpElemPointView<NgpPointData<float, Point<float>>>> &&
                  ValidPointType<NgpElemPointView<NgpPointData<float, stk::mesh::NgpField<float>>>>,
              "ElemPointView and NgpElemPointView must be valid Point types");

static_assert(ValidPointType<NodePointView<PointData<float, Point<float>>>> &&
                  ValidPointType<NodePointView<PointData<float, stk::mesh::Field<float>>>> &&
                  ValidPointType<NgpNodePointView<NgpPointData<float, Point<float>>>> &&
                  ValidPointType<NgpNodePointView<NgpPointData<float, stk::mesh::NgpField<float>>>>,
              "NodePointView and NgpNodePointView must be valid Point types");

/// \brief A helper function to create a ElemPointView object with type deduction
template <typename PointDataType>
auto create_elem_point_view(const stk::mesh::BulkData& bulk_data, PointDataType& data, stk::mesh::Entity point) {
  return ElemPointView<PointDataType>(bulk_data, data, point);
}

/// \brief A helper function to create a NgpElemPointView object with type deduction
template <typename NgpPointDataType>
auto create_ngp_elem_point_view(stk::mesh::NgpMesh ngp_mesh, NgpPointDataType data,
                                stk::mesh::FastMeshIndex point_index) {
  return NgpElemPointView<NgpPointDataType>(ngp_mesh, data, point_index);
}

/// \brief A helper function to create a NodePointView object with type deduction
template <typename PointDataType>
auto create_node_point_view(const stk::mesh::BulkData& bulk_data, PointDataType& data, stk::mesh::Entity point) {
  MUNDY_THROW_ASSERT(
      bulk_data.entity_rank(point) == stk::topology::NODE_RANK && data.point_rank == stk::topology::NODE_RANK,
      std::invalid_argument, "Both the point entity rank and the point data rank must be NODE_RANK");
  return NodePointView<PointDataType>(bulk_data, data, point);
}

/// \brief A helper function to create a NgpNodePointView object with type deduction
template <typename NgpPointDataType>
auto create_ngp_node_point_view(stk::mesh::NgpMesh ngp_mesh, NgpPointDataType data,
                                stk::mesh::FastMeshIndex point_index) {
  return NgpNodePointView<NgpPointDataType>(ngp_mesh, data, point_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_POINT_HPP_