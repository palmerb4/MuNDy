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
#include <mundy_mesh/BulkData.hpp>             // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of v_segments
///
/// The topology of a v segment directly effects the access pattern for the underlying data.
/// Regardless of the topology, the node coordinates are stored on all nodes of the v segment.
/// The difference between the topologies is how we access those nodes.
///
/// Allowable topologies are and their node access patterns are:
///   - SPRING_3: Left, right, middle
///   - TRI_3: Left, middle, right
///   - SHELL_TRI_3: Left, middle, right
template <typename Scalar,                        //
          stk::topology::topology_t OurTopology,  //
          typename NodeCoordsDataType = stk::mesh::Field<Scalar>>
class VSegmentData {
  static_assert(OurTopology == stk::topology::SPRING_3 || OurTopology == stk::topology::TRI_3 ||
                    OurTopology == stk::topology::SHELL_TRI_3,
                "The topology of a v segment must be SPRING_3, TRI_3, or SHELL_TRI_3");
  static_assert(std::is_same_v<std::decay_t<NodeCoordsDataType>, stk::mesh::Field<Scalar>>,
                "NodeCoordsDataType must be either a const or non-const field of scalars");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = NodeCoordsDataType;
  static constexpr stk::topology::topology_t topology_t = OurTopology;

  /// \brief Constructor
  VSegmentData(stk::mesh::BulkData& bulk_data, node_coords_data_t& node_coords_data)
      : bulk_data_(bulk_data), node_coords_data_(node_coords_data) {
    MUNDY_THROW_ASSERT(node_coords_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The node_coords data must be a field of NODE_RANK");
  }

  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }

  stk::mesh::BulkData& bulk_data() {
    return bulk_data_;
  }

  const node_coords_data_t& node_coords_data() const {
    return node_coords_data_;
  }

  node_coords_data_t& node_coords_data() {
    return node_coords_data_;
  }

 private:
  stk::mesh::BulkData& bulk_data_;
  node_coords_data_t& node_coords_data_;
};  // VSegmentData

/// \brief Aggregate to hold the data for a collection of NGP-compatible line segments
/// See the discussion for VSegmentData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        //
          stk::topology::topology_t OurTopology,  //
          typename NodeCoordsDataType = stk::mesh::NgpField<Scalar>>
class NgpVSegmentData {
  static_assert(OurTopology == stk::topology::SPRING_3 || OurTopology == stk::topology::TRI_3 ||
                    OurTopology == stk::topology::SHELL_TRI_3,
                "The topology of a v segment must be SPRING_3, TRI_3, or SHELL_TRI_3");
  static_assert(std::is_same_v<std::decay_t<NodeCoordsDataType>, stk::mesh::NgpField<Scalar>>,
                "NodeCoordsDataType must be either a const or non-const field of scalars");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = NodeCoordsDataType;
  static constexpr stk::topology::topology_t topology_t = OurTopology;

  /// \brief Constructor
  NgpVSegmentData(stk::mesh::NgpMesh ngp_mesh, node_coords_data_t& node_coords_data)
      : ngp_mesh_(ngp_mesh), node_coords_data_(node_coords_data) {
    MUNDY_THROW_ASSERT(node_coords_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The node_coords data must be a field of NODE_RANK");
  }

  const stk::mesh::NgpMesh& ngp_mesh() const {
    return ngp_mesh_;
  }

  stk::mesh::NgpMesh& ngp_mesh() {
    return ngp_mesh_;
  }

  const node_coords_data_t& node_coords_data() const {
    return node_coords_data_;
  }

  node_coords_data_t& node_coords_data() {
    return node_coords_data_;
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  node_coords_data_t& node_coords_data_;
};  // NgpVSegmentData

/// \brief A helper function to create a VSegmentData object
///
/// This function creates a VSegmentData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename NodeCoordsDataType>            // deduced
auto create_v_segment_data(stk::mesh::BulkData& bulk_data, NodeCoordsDataType& node_coords_data) {
  return VSegmentData<Scalar, OurTopology, NodeCoordsDataType>{bulk_data, node_coords_data};
}

/// \brief A helper function to create a NgpVSegmentData object
/// See the discussion for create_v_segment_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename NodeCoordsDataType>            // deduced
auto create_ngp_v_segment_data(stk::mesh::NgpMesh ngp_mesh, NodeCoordsDataType& node_coords_data) {
  return NgpVSegmentData<Scalar, OurTopology, NodeCoordsDataType>{ngp_mesh, node_coords_data};
}

/// \brief A concept to check if a type provides the same data as VSegmentData
template <typename Agg>
concept ValidVSegmentDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::node_coords_data_t;
  std::is_same_v<std::decay_t<typename Agg::node_coords_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
  { agg.bulk_data() } -> std::convertible_to<stk::mesh::BulkData&>;
  { agg.node_coords_data() } -> std::convertible_to<typename Agg::node_coords_data_t&>;
};  // ValidVSegmentDataType

/// \brief A concept to check if a type provides the same data as NgpVSegmentData
template <typename Agg>
concept ValidNgpVSegmentDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::node_coords_data_t;
  std::is_same_v<std::decay_t<typename Agg::node_coords_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
  { agg.ngp_mesh() } -> std::convertible_to<stk::mesh::NgpMesh>;
  { agg.node_coords_data() } -> std::convertible_to<typename Agg::node_coords_data_t&>;
};  // ValidNgpVSegmentDataType

static_assert(ValidVSegmentDataType<VSegmentData<float,                    //
                                                 stk::topology::SPRING_3,  //
                                                 stk::mesh::Field<float>>> &&
                  ValidVSegmentDataType<VSegmentData<float,                 //
                                                     stk::topology::TRI_3,  //
                                                     stk::mesh::Field<float>>>,
              "VSegmentData must satisfy the ValidVSegmentDataType concept");

static_assert(ValidNgpVSegmentDataType<NgpVSegmentData<float,                    //
                                                       stk::topology::SPRING_3,  //
                                                       stk::mesh::NgpField<float>>> &&
                  ValidNgpVSegmentDataType<NgpVSegmentData<float,                 //
                                                           stk::topology::TRI_3,  //
                                                           stk::mesh::NgpField<float>>>,
              "NgpVSegmentData must satisfy the ValidNgpVSegmentDataType concept");

/// \brief A helper function to get an updated NgpVSegmentData object from a VSegmentData object
/// \param data The VSegmentData object to convert
template <ValidVSegmentDataType VSegmentDataType>
auto get_updated_ngp_data(VSegmentDataType data) {
  using scalar_t = typename VSegmentDataType::scalar_t;
  constexpr stk::topology::topology_t topology_t = VSegmentDataType::topology_t;

  return create_ngp_v_segment_data<scalar_t, topology_t>(  //
      stk::mesh::get_updated_ngp_mesh(data.bulk_data()),   //
      stk::mesh::get_updated_ngp_field<scalar_t>(data.node_coords_data));
}

/// \brief A traits class to provide abstracted access to a v_segment's data via an aggregate
template <typename Agg>
struct VSegmentDataTraits {
  static_assert(ValidVSegmentDataType<Agg>,
                "Agg must satisfy the ValidVSegmentDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpVSegmentData but is free to extend "
                "it as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = typename Agg::node_coords_data_t;

  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  static decltype(auto) node_coords(Agg agg, stk::mesh::Entity v_segment_node) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data(), v_segment_node);
  }
};  // VSegmentDataTraits

/// \brief A traits class to provide abstracted access to a v_segment's data via an NGP-compatible aggregate
/// See the discussion for VSegmentDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpVSegmentDataTraits {
  static_assert(ValidNgpVSegmentDataType<Agg>,
                "Agg must satisfy the ValidNgpVSegmentDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpVSegmentData but is free to extend "
                "it as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = typename Agg::node_coords_data_t;

  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) node_coords(Agg agg, stk::mesh::FastMeshIndex v_segment_index) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data(), v_segment_index);
  }
};  // NgpVSegmentDataTraits

/// @brief A view of a TRI_3, SHELL_TRI_3, or SPRING_3 STK entity meant to represent a v_segment
///
/// Use \ref create_v_segment_entity_view to build an VSegmentEntityView object with automatic template deduction.
template <stk::topology::topology_t OurTopology, typename VSegmentDataType>
class VSegmentEntityView {
  static_assert(OurTopology == stk::topology::TRI_3 || OurTopology == stk::topology::SHELL_TRI_3 ||
                    OurTopology == stk::topology::SPRING_3,
                "The topology of the v_segment entity must be TRI_3, SHELL_TRI_3, or SPRING_3");
  static_assert(VSegmentDataType::topology_t == OurTopology, "The topology of the v_segment data must match the view");

 public:
  using data_access_t = VSegmentDataTraits<VSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::node_coords(std::declval<VSegmentDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology_t = OurTopology;
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;

  VSegmentEntityView(VSegmentDataType data, stk::mesh::Entity v_segment)
    requires(OurTopology == stk::topology::TRI_3 || OurTopology == stk::topology::SHELL_TRI_3)
      : data_(data),
        v_segment_(v_segment),
        start_node_(data_.bulk_data().begin_nodes(v_segment_)[0]),
        middle_node_(data_.bulk_data().begin_nodes(v_segment_)[1]),
        end_node_(data_.bulk_data().begin_nodes(v_segment_)[2]) {
    MUNDY_THROW_ASSERT(data_.bulk_data().entity_rank(v_segment_) == stk::topology::ELEM_RANK, std::invalid_argument,
                       "Both the v_segment entity rank and the v_segment data rank must be ELEM_RANK");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(v_segment_), std::invalid_argument,
                       "The given v_segment entity is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().num_nodes(v_segment_) >= 3, std::invalid_argument,
                       "The given v_segment entity must have at least three nodes.");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(start_node_), std::invalid_argument,
                       "The start node entity associated with the v_segment is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(middle_node_), std::invalid_argument,
                       "The middle node entity associated with the v_segment is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(end_node_), std::invalid_argument,
                       "The end node entity associated with the v_segment is not valid");
  }

  VSegmentEntityView(VSegmentDataType data, stk::mesh::Entity v_segment)
    requires(OurTopology == stk::topology::SPRING_3)
      : data_(data),
        v_segment_(v_segment),
        start_node_(data_.bulk_data().begin_nodes(v_segment_)[0]),
        middle_node_(data_.bulk_data().begin_nodes(v_segment_)[2]),
        end_node_(data_.bulk_data().begin_nodes(v_segment_)[1]) {
    MUNDY_THROW_ASSERT(data_.bulk_data().entity_rank(v_segment_) == rank, std::invalid_argument,
                       "Both the v_segment entity rank and the v_segment data rank must match the view's topology");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(v_segment_), std::invalid_argument,
                       "The given v_segment entity is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().num_nodes(v_segment_) >= 3, std::invalid_argument,
                       "The given v_segment entity must have at least three nodes.");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(start_node_), std::invalid_argument,
                       "The start node entity associated with the v_segment is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(middle_node_), std::invalid_argument,
                       "The middle node entity associated with the v_segment is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(end_node_), std::invalid_argument,
                       "The end node entity associated with the v_segment is not valid");
  }

  decltype(auto) data() {
    return data_;
  }

  decltype(auto) data() const {
    return data_;
  }

  stk::mesh::Entity& v_segment_entity() {
    return v_segment_;
  }

  const stk::mesh::Entity& v_segment_entity() const {
    return v_segment_;
  }

  stk::mesh::Entity& start_node_entity() {
    return start_node_;
  }

  const stk::mesh::Entity& start_node_entity() const {
    return start_node_;
  }

  stk::mesh::Entity& middle_node_entity() {
    return middle_node_;
  }

  const stk::mesh::Entity& middle_node_entity() const {
    return middle_node_;
  }

  stk::mesh::Entity& end_node_entity() {
    return end_node_;
  }

  const stk::mesh::Entity& end_node_entity() const {
    return end_node_;
  }

  decltype(auto) start() {
    return data_access_t::node_coords(data(), start_node_entity());
  }

  decltype(auto) start() const {
    return data_access_t::node_coords(data(), start_node_entity());
  }

  decltype(auto) middle() {
    return data_access_t::node_coords(data(), middle_node_entity());
  }

  decltype(auto) middle() const {
    return data_access_t::node_coords(data(), middle_node_entity());
  }

  decltype(auto) end() {
    return data_access_t::node_coords(data(), end_node_entity());
  }

  decltype(auto) end() const {
    return data_access_t::node_coords(data(), end_node_entity());
  }

 private:
  VSegmentDataType data_;
  stk::mesh::Entity v_segment_;
  stk::mesh::Entity start_node_;
  stk::mesh::Entity middle_node_;
  stk::mesh::Entity end_node_;
};  // VSegmentEntityView<OurTopology, VSegmentDataType>

/// @brief An ngp-compatible view of an STK entity meant to represent a v_segment
/// See the discussion for VSegmentEntityView for more information. The only difference is ngp-compatible data access.
template <stk::topology::topology_t OurTopology, typename NgpVSegmentDataType>
class NgpVSegmentEntityView {
  static_assert(OurTopology == stk::topology::TRI_3 || OurTopology == stk::topology::SHELL_TRI_3 ||
                    OurTopology == stk::topology::SPRING_3,
                "The topology of the v_segment entity must be TRI_3, SHELL_TRI_3, or SPRING_3");
  static_assert(NgpVSegmentDataType::topology_t == OurTopology,
                "The topology of the v_segment data must match the view");

 public:
  using data_access_t = NgpVSegmentDataTraits<NgpVSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<NgpVSegmentDataType>(),
                                                      std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology_t = OurTopology;
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;

  KOKKOS_INLINE_FUNCTION
  NgpVSegmentEntityView(NgpVSegmentDataType data, stk::mesh::FastMeshIndex v_segment_index)
    requires(OurTopology == stk::topology::TRI_3 || OurTopology == stk::topology::SHELL_TRI_3)
      : data_(data),
        v_segment_index_(v_segment_index),
        start_node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, v_segment_index_)[0])),
        middle_node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, v_segment_index_)[1])),
        end_node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, v_segment_index_)[2])) {
  }

  KOKKOS_INLINE_FUNCTION
  NgpVSegmentEntityView(NgpVSegmentDataType data, stk::mesh::FastMeshIndex v_segment_index)
    requires(OurTopology == stk::topology::SPRING_3)
      : data_(data),
        v_segment_index_(v_segment_index),
        start_node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, v_segment_index_)[0])),
        middle_node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, v_segment_index_)[2])),
        end_node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, v_segment_index_)[1])) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) data() {
    return data_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) data() const {
    return data_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& v_segment_index() {
    return v_segment_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& v_segment_index() const {
    return v_segment_index_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& start_node_index() {
    return start_node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& start_node_index() const {
    return start_node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& middle_node_index() {
    return middle_node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& middle_node_index() const {
    return middle_node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& end_node_index() {
    return end_node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& end_node_index() const {
    return end_node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) start() {
    return data_access_t::node_coords(data(), start_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) start() const {
    return data_access_t::node_coords(data(), start_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) middle() {
    return data_access_t::node_coords(data(), middle_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) middle() const {
    return data_access_t::node_coords(data(), middle_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() {
    return data_access_t::node_coords(data(), end_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() const {
    return data_access_t::node_coords(data(), end_node_index());
  }

 private:
  NgpVSegmentDataType data_;
  stk::mesh::FastMeshIndex v_segment_index_;
  stk::mesh::FastMeshIndex start_node_index_;
  stk::mesh::FastMeshIndex middle_node_index_;
  stk::mesh::FastMeshIndex end_node_index_;
};  // NgpVSegmentEntityView<OurTopology, NgpVSegmentDataType>

static_assert(ValidVSegmentType<VSegmentEntityView<stk::topology::SPRING_3,
                                                   VSegmentData<float,                    //
                                                                stk::topology::SPRING_3,  //
                                                                stk::mesh::Field<float>>>> &&
                  ValidVSegmentType<VSegmentEntityView<stk::topology::TRI_3,
                                                       VSegmentData<float,                 //
                                                                    stk::topology::TRI_3,  //
                                                                    stk::mesh::Field<float>>>> &&
                  ValidVSegmentType<NgpVSegmentEntityView<stk::topology::SPRING_3,
                                                          NgpVSegmentData<float,                    //
                                                                          stk::topology::SPRING_3,  //
                                                                          stk::mesh::NgpField<float>>>> &&
                  ValidVSegmentType<NgpVSegmentEntityView<stk::topology::TRI_3,
                                                          NgpVSegmentData<float,                 //
                                                                          stk::topology::TRI_3,  //
                                                                          stk::mesh::NgpField<float>>>>,
              "VSegmentEntityView and NgpVSegmentEntityView must be valid VSegment types");

/// \brief A helper function to create a VSegmentEntityView object with type deduction
template <typename VSegmentDataType>  // deduced
auto create_v_segment_entity_view(VSegmentDataType& data, stk::mesh::Entity v_segment) {
  return VSegmentEntityView<VSegmentDataType::topology_t, VSegmentDataType>(data, v_segment);
}

/// \brief A helper function to create a NgpVSegmentEntityView object with type deduction
template <typename NgpVSegmentDataType>  // deduced
auto create_ngp_v_segment_entity_view(NgpVSegmentDataType data, stk::mesh::FastMeshIndex v_segment_index) {
  return NgpVSegmentEntityView<NgpVSegmentDataType::topology_t, NgpVSegmentDataType>(data, v_segment_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_VSEGMENT_HPP_