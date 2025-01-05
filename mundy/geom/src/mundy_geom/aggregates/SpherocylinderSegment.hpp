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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENT_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENT_HPP_

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/primitives/SpherocylinderSegment.hpp>  // for mundy::geom::ValidSpherocylinderSegmentType
#include <mundy_mesh/BulkData.hpp>                          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief A struct to hold the data for a collection of sphero-cylinder segments
template <typename Scalar, typename NodeCoordsDataType = stk::mesh::Field<Scalar>,
          typename RadiusDataType = stk::mesh::Field<Scalar>>
struct SpherocylinderSegmentData {
  static_assert(std::is_same_v<std::decay_t<NodeCoordsDataType>, stk::mesh::Field<Scalar>> &&
                    (std::is_same_v<std::decay_t<RadiusDataType>, Scalar> ||
                     std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::Field<Scalar>>),
                "NodeCoordsDataType must be either a const or non-const field of scalars\n"
                "RadiusDataType must be either a scalar or a field of scalars");

  using scalar_t = Scalar;
  using node_coords_data_t = NodeCoordsDataType;
  using radius_data_t = RadiusDataType;

  stk::topology::rank_t spherocylinder_segment_rank;
  node_coords_data_t& node_coords_data;
  radius_data_t& radius_data;
};  // SpherocylinderSegmentData

/// \brief A struct to hold the data for a collection of NGP-compatible sphero-cylinder segments
/// See the discussion for SpherocylinderSegmentData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename NodeCoordsDataType = stk::mesh::NgpField<Scalar>,
          typename RadiusDataType = stk::mesh::Field<Scalar>>
struct NgpSpherocylinderSegmentData {
  static_assert(std::is_same_v<std::decay_t<NodeCoordsDataType>, stk::mesh::NgpField<Scalar>> &&
                    (std::is_same_v<std::decay_t<RadiusDataType>, Scalar> ||
                     std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::NgpField<Scalar>>),
                "NodeCoordsDataType must be either a const or non-const field of scalars\n"
                "RadiusDataType must be either a scalar or a field of scalars");

  using scalar_t = Scalar;
  using node_coords_data_t = NodeCoordsDataType;
  using radius_data_t = RadiusDataType;

  stk::topology::rank_t spherocylinder_segment_rank;
  node_coords_data_t& node_coords_data;
  radius_data_t& radius_data;
};  // NgpSpherocylinderSegmentData

/// \brief A helper function to create a SpherocylinderSegmentData object
///
/// This function creates a SpherocylinderSegmentData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar, typename NodeCoordsDataType, typename RadiusDataType>
auto create_spherocylinder_segment_data(stk::topology::rank_t spherocylinder_segment_rank,
                                        NodeCoordsDataType& node_coords_data, RadiusDataType& radius_data) {
  MUNDY_THROW_ASSERT(node_coords_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                     "The node_coords data must be a field of NODE_RANK");
  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::Field<Scalar>>;
  if constexpr (is_radius_a_field) {
    MUNDY_THROW_ASSERT(radius_data.entity_rank() == spherocylinder_segment_rank, std::invalid_argument,
                       "The radius data must be a field of the same rank as the segment");
  }
  return SpherocylinderSegmentData<Scalar, NodeCoordsDataType, RadiusDataType>{spherocylinder_segment_rank,
                                                                               node_coords_data, radius_data};
}

/// \brief A helper function to create a NgpSpherocylinderSegmentData object
/// See the discussion for create_spherocylinder_segment_data for more information. Only difference is NgpFields over
/// Fields.
template <typename Scalar, typename NodeCoordsDataType, typename RadiusDataType>
auto create_ngp_spherocylinder_segment_data(stk::topology::rank_t spherocylinder_segment_rank,
                                            NodeCoordsDataType& node_coords_data, RadiusDataType& radius_data) {
  MUNDY_THROW_ASSERT(node_coords_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                     "The node_coords data must be a field of NODE_RANK");
  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::NgpField<Scalar>>;
  if constexpr (is_radius_a_field) {
    MUNDY_THROW_ASSERT(radius_data.get_rank() == spherocylinder_segment_rank, std::invalid_argument,
                       "The radius data must be a field of the same rank as the segment");
  }
  return NgpSpherocylinderSegmentData<Scalar, NodeCoordsDataType, RadiusDataType>{spherocylinder_segment_rank,
                                                                                  node_coords_data, radius_data};
}

/// \brief A concept to check if a type provides the same data as SpherocylinderSegmentData
template <typename Agg>
concept ValidDefaultSpherocylinderSegmentDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::node_coords_data_t;
  typename Agg::radius_data_t;
  std::is_same_v<std::decay_t<typename Agg::node_coords_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  std::is_same_v<std::decay_t<typename Agg::radius_data_t>, typename Agg::scalar_t> ||
      std::is_same_v<std::decay_t<typename Agg::radius_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { agg.spherocylinder_segment_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.node_coords_data } -> std::convertible_to<typename Agg::node_coords_data_t&>;
  { agg.radius_data } -> std::convertible_to<typename Agg::radius_data_t&>;
};  // ValidDefaultSpherocylinderSegmentDataType

/// \brief A concept to check if a type provides the same data as NgpSpherocylinderSegmentData
template <typename Agg>
concept ValidDefaultNgpSpherocylinderSegmentDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::node_coords_data_t;
  std::is_same_v<std::decay_t<typename Agg::node_coords_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  std::is_same_v<std::decay_t<typename Agg::radius_data_t>, typename Agg::scalar_t> ||
      std::is_same_v<std::decay_t<typename Agg::radius_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { agg.spherocylinder_segment_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.node_coords_data } -> std::convertible_to<typename Agg::node_coords_data_t&>;
  { agg.radius_data } -> std::convertible_to<typename Agg::radius_data_t&>;
};  // ValidDefaultNgpSpherocylinderSegmentDataType

static_assert(
    ValidDefaultSpherocylinderSegmentDataType<SpherocylinderSegmentData<float, stk::mesh::Field<float>, float>> &&
        ValidDefaultSpherocylinderSegmentDataType<
            SpherocylinderSegmentData<float, const stk::mesh::Field<float>, float>> &&
        ValidDefaultSpherocylinderSegmentDataType<
            SpherocylinderSegmentData<float, stk::mesh::Field<float>, stk::mesh::Field<float>>> &&
        ValidDefaultSpherocylinderSegmentDataType<
            SpherocylinderSegmentData<float, const stk::mesh::Field<float>, stk::mesh::Field<float>>>,
    "SpherocylinderSegmentData must satisfy the ValidDefaultSpherocylinderSegmentDataType concept");

static_assert(
    ValidDefaultNgpSpherocylinderSegmentDataType<
        NgpSpherocylinderSegmentData<float, stk::mesh::NgpField<float>, float>> &&
        ValidDefaultNgpSpherocylinderSegmentDataType<
            NgpSpherocylinderSegmentData<float, const stk::mesh::NgpField<float>, float>> &&
        ValidDefaultNgpSpherocylinderSegmentDataType<
            NgpSpherocylinderSegmentData<float, stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>> &&
        ValidDefaultNgpSpherocylinderSegmentDataType<
            NgpSpherocylinderSegmentData<float, const stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>,
    "NgpSpherocylinderSegmentData must satisfy the ValidDefaultNgpSpherocylinderSegmentDataType concept");

/// \brief A helper function to get an updated NgpSpherocylinderSegmentData object from a SpherocylinderSegmentData
/// object \param data The SpherocylinderSegmentData object to convert
template <ValidDefaultSpherocylinderSegmentDataType SpherocylinderSegmentDataType>
auto get_updated_ngp_data(SpherocylinderSegmentDataType data) {
  using scalar_t = typename SpherocylinderSegmentDataType::scalar_t;
  using radius_data_t = typename SpherocylinderSegmentDataType::radius_data_t;

  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<radius_data_t>, stk::mesh::Field<scalar_t>>;
  if constexpr (is_radius_a_field) {
    return create_ngp_sphere_data<scalar_t>(data.spherocylinder_segment_rank,                                   //
                                            stk::mesh::get_updated_ngp_field<scalar_t>(data.node_coords_data),  //
                                            stk::mesh::get_updated_ngp_field<scalar_t>(data.radius_data));
  } else {
    return create_ngp_sphere_data<scalar_t>(data.spherocylinder_segment_rank,                                   //
                                            stk::mesh::get_updated_ngp_field<scalar_t>(data.node_coords_data),  //
                                            data.radius_data);
  }
}

/// \brief A traits class to provide abstracted access to a spherocylinder_segment's data via an aggregate
template <typename Agg>
struct SpherocylinderSegmentDataTraits {
  static_assert(ValidDefaultSpherocylinderSegmentDataType<Agg>,
                "Agg must satisfy the ValidDefaultSpherocylinderSegmentDataType concept.\n"
                "Basically, Agg must have all the same things as NgpSpherocylinderSegmentData but is free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = typename Agg::node_coords_data_t;
  using radius_data_t = typename Agg::radius_data_t;

  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  static decltype(auto) node_coords(Agg agg, stk::mesh::Entity spherocylinder_segment_node) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data, spherocylinder_segment_node);
  }

  static decltype(auto) radius(Agg agg, stk::mesh::Entity spherocylinder_segment) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data;
    } else {
      return stk::mesh::field_data(agg.radius_data, spherocylinder_segment)[0];
    }
  }
};  // SpherocylinderSegmentDataTraits

/// \brief A traits class to provide abstracted access to a spherocylinder_segment's data via an NGP-compatible
/// aggregate See the discussion for SpherocylinderSegmentDataTraits for more information. Only difference is
/// Ngp-compatible data.
template <typename Agg>
struct NgpSpherocylinderSegmentDataTraits {
  static_assert(ValidDefaultNgpSpherocylinderSegmentDataType<Agg>,
                "Agg must satisfy the ValidDefaultNgpSpherocylinderSegmentDataType concept.\n"
                "Basically, Agg must have all the same things as NgpSpherocylinderSegmentData but is free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = typename Agg::node_coords_data_t;
  using radius_data_t = typename Agg::radius_data_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) node_coords(Agg agg, stk::mesh::FastMeshIndex spherocylinder_segment_index) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data, spherocylinder_segment_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) radius(Agg agg, stk::mesh::FastMeshIndex spherocylinder_segment_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data;
    } else {
      return agg.radius_data(spherocylinder_segment_index, 0);
    }
  }
};  // NgpSpherocylinderSegmentDataTraits

/// @brief A view of an STK entity meant to represent a spherocylinder_segment
/// If the spherocylinder_segment_rank is NODE_RANK, then the spherocylinder_segment is just a node entity with node
/// node_coords and direction. If the spherocylinder_segment_rank is ELEM_RANK, then the spherocylinder_segment is a
/// particle entity with node node_coords and element direction.
template <typename SpherocylinderSegmentDataType>
class ElemSpherocylinderSegmentView {
 public:
  using data_access_t = SpherocylinderSegmentDataTraits<SpherocylinderSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<SpherocylinderSegmentDataType>(),
                                                      std::declval<stk::mesh::Entity>()));

  ElemSpherocylinderSegmentView(const stk::mesh::BulkData& bulk_data, SpherocylinderSegmentDataType data,
                                stk::mesh::Entity spherocylinder_segment)
      : data_(data),
        spherocylinder_segment_(spherocylinder_segment),
        start_node_(bulk_data.begin_nodes(spherocylinder_segment_)[0]),
        end_node_(bulk_data.begin_nodes(spherocylinder_segment_)[1]) {
    MUNDY_THROW_ASSERT(
        bulk_data.entity_rank(spherocylinder_segment_) == stk::topology::ELEM_RANK &&
            data_.spherocylinder_segment_rank == stk::topology::ELEM_RANK,
        std::invalid_argument,
        "Both the spherocylinder_segment entity rank and the spherocylinder_segment data rank must be ELEM_RANK");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(spherocylinder_segment_), std::invalid_argument,
                       "The given spherocylinder_segment entity is not valid");
    MUNDY_THROW_ASSERT(bulk_data.num_nodes(spherocylinder_segment_) == 2, std::invalid_argument,
                       "The given spherocylinder_segment entity must have exactly one node");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(start_node_), std::invalid_argument,
                       "The start node entity associated with the spherocylinder_segment is not valid");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(end_node_), std::invalid_argument,
                       "The end node entity associated with the spherocylinder_segment is not valid");
  }

  decltype(auto) start() {
    return data_access_t::node_coords(data_, start_node_);
  }

  decltype(auto) start() const {
    return data_access_t::node_coords(data_, start_node_);
  }

  decltype(auto) end() {
    return data_access_t::node_coords(data_, end_node_);
  }

  decltype(auto) end() const {
    return data_access_t::node_coords(data_, end_node_);
  }

  decltype(auto) radius() {
    return data_access_t::radius(data_, spherocylinder_segment_);
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data_, spherocylinder_segment_);
  }

 private:
  SpherocylinderSegmentDataType data_;
  stk::mesh::Entity spherocylinder_segment_;
  stk::mesh::Entity start_node_;
  stk::mesh::Entity end_node_;
};  // ElemSpherocylinderSegmentView

/// @brief An ngp-compatible view of an ELEM_RANK STK entity meant to represent a spherocylinder_segment
/// See the discussion for ElemSpherocylinderSegmentView for more information. The only difference is ngp-compatible
/// data access.
template <typename NgpSpherocylinderSegmentDataType>
class NgpElemSpherocylinderSegmentView {
 public:
  using data_access_t = NgpSpherocylinderSegmentDataTraits<NgpSpherocylinderSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<NgpSpherocylinderSegmentDataType>(),
                                                      std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpElemSpherocylinderSegmentView(stk::mesh::NgpMesh ngp_mesh, NgpSpherocylinderSegmentDataType data,
                                   stk::mesh::FastMeshIndex spherocylinder_segment_index)
      : data_(data),
        spherocylinder_segment_index_(spherocylinder_segment_index),
        start_node_index_(ngp_mesh.fast_mesh_index(
            ngp_mesh.get_nodes(data_.spherocylinder_segment_rank, spherocylinder_segment_index_)[0])),
        end_node_index_(ngp_mesh.fast_mesh_index(
            ngp_mesh.get_nodes(data_.spherocylinder_segment_rank, spherocylinder_segment_index_)[1])) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) start() {
    return data_access_t::node_coords(data_, start_node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) start() const {
    return data_access_t::node_coords(data_, start_node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() {
    return data_access_t::node_coords(data_, end_node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() const {
    return data_access_t::node_coords(data_, end_node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data_, spherocylinder_segment_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data_, spherocylinder_segment_index_);
  }

 private:
  NgpSpherocylinderSegmentDataType data_;
  stk::mesh::FastMeshIndex spherocylinder_segment_index_;
  stk::mesh::FastMeshIndex start_node_index_;
  stk::mesh::FastMeshIndex end_node_index_;
};  // NgpElemSpherocylinderSegmentView

static_assert(
    ValidSpherocylinderSegmentType<
        ElemSpherocylinderSegmentView<SpherocylinderSegmentData<float, stk::mesh::Field<float>, float>>> &&
        ValidSpherocylinderSegmentType<ElemSpherocylinderSegmentView<
            SpherocylinderSegmentData<float, const stk::mesh::Field<float>, stk::mesh::Field<float>>>> &&
        ValidSpherocylinderSegmentType<
            NgpElemSpherocylinderSegmentView<NgpSpherocylinderSegmentData<float, stk::mesh::NgpField<float>, float>>> &&
        ValidSpherocylinderSegmentType<NgpElemSpherocylinderSegmentView<
            NgpSpherocylinderSegmentData<float, const stk::mesh::NgpField<float>, stk::mesh::NgpField<float>>>>,
    "ElemSpherocylinderSegmentView and NgpElemSpherocylinderSegmentView must be valid SpherocylinderSegment types");

/// \brief A helper function to create a ElemSpherocylinderSegmentView object with type deduction
template <typename SpherocylinderSegmentDataType>
auto create_elem_spherocylinder_segment_view(const stk::mesh::BulkData& bulk_data, SpherocylinderSegmentDataType& data,
                                             stk::mesh::Entity spherocylinder_segment) {
  return ElemSpherocylinderSegmentView<SpherocylinderSegmentDataType>(bulk_data, data, spherocylinder_segment);
}

/// \brief A helper function to create a NgpElemSpherocylinderSegmentView object with type deduction
template <typename NgpSpherocylinderSegmentDataType>
auto create_ngp_elem_spherocylinder_segment_view(stk::mesh::NgpMesh ngp_mesh, NgpSpherocylinderSegmentDataType data,
                                                 stk::mesh::FastMeshIndex spherocylinder_segment_index) {
  return NgpElemSpherocylinderSegmentView<NgpSpherocylinderSegmentDataType>(ngp_mesh, data,
                                                                            spherocylinder_segment_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENT_HPP_