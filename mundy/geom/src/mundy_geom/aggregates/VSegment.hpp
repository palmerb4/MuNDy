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

#ifndef MUNDY_GEOM_AGGREGATES_VSEGMENT_HPP_
#define MUNDY_GEOM_AGGREGATES_VSEGMENT_HPP_

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/primitives/VSegment.hpp>  // for mundy::geom::ValidVSegmentType
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>        // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief A struct to hold the data for a collection of line segments
template <typename Scalar, typename NodeCoordsDataType = stk::mesh::Field<Scalar>>
struct VSegmentData {
  static_assert(std::is_same_v<std::decay_t<NodeCoordsDataType>, stk::mesh::Field<Scalar>>,
                "NodeCoordsDataType must be either a const or non-const field of scalars");

  using scalar_t = Scalar;
  using node_coords_data_t = NodeCoordsDataType;

  stk::topology::rank_t v_segment_rank;
  node_coords_data_t& node_coords_data;
};  // VSegmentData

/// \brief A struct to hold the data for a collection of NGP-compatible line segments
/// See the discussion for VSegmentData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename NodeCoordsDataType = stk::mesh::NgpField<Scalar>>
struct NgpVSegmentData {
  static_assert(std::is_same_v<std::decay_t<NodeCoordsDataType>, stk::mesh::NgpField<Scalar>>,
                "NodeCoordsDataType must be either a const or non-const field of scalars");

  using scalar_t = Scalar;
  using node_coords_data_t = NodeCoordsDataType;

  stk::topology::rank_t v_segment_rank;
  node_coords_data_t& node_coords_data;
};  // NgpVSegmentData

/// \brief A helper function to create a VSegmentData object
///
/// This function creates a VSegmentData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar, typename NodeCoordsDataType>
auto create_v_segment_data(stk::topology::rank_t v_segment_rank, NodeCoordsDataType& node_coords_data) {
  MUNDY_THROW_ASSERT(node_coords_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                      "The node_coords data must be a field of NODE_RANK");
  return VSegmentData<Scalar, NodeCoordsDataType>{v_segment_rank, node_coords_data};
}

/// \brief A helper function to create a NgpVSegmentData object
/// See the discussion for create_v_segment_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename NodeCoordsDataType>
auto create_ngp_v_segment_data(stk::topology::rank_t v_segment_rank, NodeCoordsDataType& node_coords_data) {
  MUNDY_THROW_ASSERT(node_coords_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                      "The node_coords data must be a field of NODE_RANK");
  return NgpVSegmentData<Scalar, NodeCoordsDataType>{v_segment_rank, node_coords_data};
}

/// \brief A concept to check if a type provides the same data as VSegmentData
template <typename Agg>
concept ValidDefaultVSegmentDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::node_coords_data_t;
  std::is_same_v<std::decay_t<typename Agg::node_coords_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { agg.v_segment_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.node_coords_data } -> std::convertible_to<typename Agg::node_coords_data_t&>;
};  // ValidDefaultVSegmentDataType

/// \brief A concept to check if a type provides the same data as NgpVSegmentData
template <typename Agg>
concept ValidDefaultNgpVSegmentDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::node_coords_data_t;
  std::is_same_v<std::decay_t<typename Agg::node_coords_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { agg.v_segment_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.node_coords_data } -> std::convertible_to<typename Agg::node_coords_data_t&>;
};  // ValidDefaultNgpVSegmentDataType

static_assert(ValidDefaultVSegmentDataType<VSegmentData<float, stk::mesh::Field<float>>> &&
                  ValidDefaultVSegmentDataType<VSegmentData<float, const stk::mesh::Field<float>>>,
              "VSegmentData must satisfy the ValidDefaultVSegmentDataType concept");

static_assert(ValidDefaultNgpVSegmentDataType<NgpVSegmentData<float, stk::mesh::NgpField<float>>> &&
                  ValidDefaultNgpVSegmentDataType<NgpVSegmentData<float, const stk::mesh::NgpField<float>>>,
              "NgpVSegmentData must satisfy the ValidDefaultNgpVSegmentDataType concept");

/// \brief A helper function to get an updated NgpVSegmentData object from a VSegmentData object
/// \param data The VSegmentData object to convert
template <ValidDefaultVSegmentDataType VSegmentDataType>
auto get_updated_ngp_data(VSegmentDataType data) {
  using scalar_t = typename VSegmentDataType::scalar_t;
  return create_ngp_v_segment_data<scalar_t>(data.v_segment_rank,  //
                                        stk::mesh::get_updated_ngp_field<scalar_t>(data.node_coords_data));
}

/// \brief A traits class to provide abstracted access to a v_segment's data via an aggregate
template <typename Agg>
struct VSegmentDataTraits {
  static_assert(
      ValidDefaultVSegmentDataType<Agg>,
      "Agg must satisfy the ValidDefaultVSegmentDataType concept.\n"
      "Basically, Agg must have all the same things as NgpVSegmentData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = typename Agg::node_coords_data_t;

  static decltype(auto) node_coords(Agg agg, stk::mesh::Entity v_segment_node) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data, v_segment_node);
  }
};  // VSegmentDataTraits

/// \brief A traits class to provide abstracted access to a v_segment's data via an NGP-compatible aggregate
/// See the discussion for VSegmentDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpVSegmentDataTraits {
  static_assert(
      ValidDefaultNgpVSegmentDataType<Agg>,
      "Agg must satisfy the ValidDefaultNgpVSegmentDataType concept.\n"
      "Basically, Agg must have all the same things as NgpVSegmentData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = typename Agg::node_coords_data_t;

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) node_coords(Agg agg, stk::mesh::FastMeshIndex v_segment_index) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data, v_segment_index);
  }
};  // NgpVSegmentDataTraits

/// @brief A view of an STK entity meant to represent a v_segment
/// If the v_segment_rank is ELEM_RANK, then the v_segment is a particle entity with node node_coords.
template <typename VSegmentDataType>
class ElemVSegmentView {
 public:
  using data_access_t = VSegmentDataTraits<VSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<VSegmentDataType>(), std::declval<stk::mesh::Entity>()));

  ElemVSegmentView(const stk::mesh::BulkData& bulk_data, VSegmentDataType data, stk::mesh::Entity v_segment)
      : data_(data), v_segment_(v_segment), 
      start_node_(bulk_data.begin_nodes(v_segment_)[0]), 
      middle_node_(bulk_data.begin_nodes(v_segment_)[1]),      // WARNING for when we switch to topologies: A TRI_3 has the middle as 1 and a SPRING_3 has the middle as 2.
      end_node_(bulk_data.begin_nodes(v_segment_)[2]) {
    MUNDY_THROW_ASSERT(
        bulk_data.entity_rank(v_segment_) == stk::topology::ELEM_RANK && data_.v_segment_rank == stk::topology::ELEM_RANK,
        std::invalid_argument, "Both the v_segment entity rank and the v_segment data rank must be ELEM_RANK");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(v_segment_), std::invalid_argument, "The given v_segment entity is not valid");
    MUNDY_THROW_ASSERT(bulk_data.num_nodes(v_segment_) == 3, std::invalid_argument,
                       "The given v_segment entity must have exactly one node");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(start_node_), std::invalid_argument,
                       "The start node entity associated with the v_segment is not valid");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(middle_node_), std::invalid_argument,
                        "The middle node entity associated with the v_segment is not valid");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(end_node_), std::invalid_argument,
                        "The end node entity associated with the v_segment is not valid");
  }

  decltype(auto) start() {
    return data_access_t::node_coords(data_, start_node_);
  }

  decltype(auto) start() const {
    return data_access_t::node_coords(data_, start_node_);
  }

  decltype(auto) middle() {
    return data_access_t::node_coords(data_, middle_node_);
  }

  decltype(auto) middle() const {
    return data_access_t::node_coords(data_, middle_node_);
  }

  decltype(auto) end() {
    return data_access_t::node_coords(data_, end_node_);
  }

  decltype(auto) end() const {
    return data_access_t::node_coords(data_, end_node_);
  }

 private:
  VSegmentDataType data_;
  stk::mesh::Entity v_segment_;
  stk::mesh::Entity start_node_;
  stk::mesh::Entity middle_node_;
  stk::mesh::Entity end_node_;
};  // ElemVSegmentView

/// @brief An ngp-compatible view of an ELEM_RANK STK entity meant to represent a v_segment
/// See the discussion for ElemVSegmentView for more information. The only difference is ngp-compatible data access.
template <typename NgpVSegmentDataType>
class NgpElemVSegmentView {
 public:
  using data_access_t = NgpVSegmentDataTraits<NgpVSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<NgpVSegmentDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpElemVSegmentView(stk::mesh::NgpMesh ngp_mesh, NgpVSegmentDataType data, stk::mesh::FastMeshIndex v_segment_index)
      : data_(data),
        v_segment_index_(v_segment_index),
        start_node_index_(ngp_mesh.fast_mesh_index(ngp_mesh.get_nodes(data_.v_segment_rank, v_segment_index_)[0])),
        middle_node_index_(ngp_mesh.fast_mesh_index(ngp_mesh.get_nodes(data_.v_segment_rank, v_segment_index_)[1])),
        end_node_index_(ngp_mesh.fast_mesh_index(ngp_mesh.get_nodes(data_.v_segment_rank, v_segment_index_)[2])) {
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
  decltype(auto) middle() {
    return data_access_t::node_coords(data_, middle_node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) middle() const {
    return data_access_t::node_coords(data_, middle_node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() {
    return data_access_t::node_coords(data_, end_node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() const {
    return data_access_t::node_coords(data_, end_node_index_);
  }

 private:
  NgpVSegmentDataType data_;
  stk::mesh::FastMeshIndex v_segment_index_;
  stk::mesh::FastMeshIndex start_node_index_;
  stk::mesh::FastMeshIndex middle_node_index_;
  stk::mesh::FastMeshIndex end_node_index_;
};  // NgpElemVSegmentView

static_assert(ValidVSegmentType<ElemVSegmentView<VSegmentData<float, stk::mesh::Field<float>>>> &&
              ValidVSegmentType<ElemVSegmentView<VSegmentData<float, const stk::mesh::Field<float>>>> &&
              ValidVSegmentType<NgpElemVSegmentView<NgpVSegmentData<float, stk::mesh::NgpField<float>>>> &&
              ValidVSegmentType<NgpElemVSegmentView<NgpVSegmentData<float, const stk::mesh::NgpField<float>>>>,
              "ElemVSegmentView and NgpElemVSegmentView must be valid VSegment types");

/// \brief A helper function to create a ElemVSegmentView object with type deduction
template <typename VSegmentDataType>
auto create_elem_v_segment_view(const stk::mesh::BulkData& bulk_data, VSegmentDataType& data, stk::mesh::Entity v_segment) {
  return ElemVSegmentView<VSegmentDataType>(bulk_data, data, v_segment);
}

/// \brief A helper function to create a NgpElemVSegmentView object with type deduction
template <typename NgpVSegmentDataType>
auto create_ngp_elem_v_segment_view(stk::mesh::NgpMesh ngp_mesh, NgpVSegmentDataType data,
                                stk::mesh::FastMeshIndex v_segment_index) {
  return NgpElemVSegmentView<NgpVSegmentDataType>(ngp_mesh, data, v_segment_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_VSEGMENT_HPP_