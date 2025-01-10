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

#ifndef MUNDY_GEOM_AGGREGATES_LINESEGMENT_HPP_
#define MUNDY_GEOM_AGGREGATES_LINESEGMENT_HPP_

// C++ core
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/primitives/LineSegment.hpp>  // for mundy::geom::ValidLineSegmentType
#include <mundy_mesh/BulkData.hpp>                // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of line segments
///
/// The topology of a line segment directly effects the access pattern for the underlying data.
/// Regardless of the topology, the node coordinates are stored on all nodes of the line segment.
/// However, how we access those nodes changes based on the rank of the line segment. Allowable topologies are:
///   - LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, SPRING_3
template <typename Scalar,                        //
          typename OurTopology>
class LineSegmentData {
  static_assert(OurTopology::value == stk::topology::LINE_2 || OurTopology::value == stk::topology::LINE_3 ||
                    OurTopology::value == stk::topology::BEAM_2 || OurTopology::value == stk::topology::BEAM_3 ||
                    OurTopology::value == stk::topology::SPRING_2 || OurTopology::value == stk::topology::SPRING_3,
                "The topology of a line segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = stk::mesh::Field<Scalar>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  LineSegmentData(stk::mesh::BulkData& bulk_data, node_coords_data_t& node_coords_data)
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

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<LineSegmentData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  stk::mesh::BulkData& bulk_data_;
  node_coords_data_t& node_coords_data_;
};  // LineSegmentData

/// \brief Aggregate to hold the data for a collection of NGP-compatible line segments
/// See the discussion for LineSegmentData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        //
          typename OurTopology>
class NgpLineSegmentData {
  static_assert(OurTopology::value == stk::topology::LINE_2 || OurTopology::value == stk::topology::LINE_3 ||
                    OurTopology::value == stk::topology::BEAM_2 || OurTopology::value == stk::topology::BEAM_3 ||
                    OurTopology::value == stk::topology::SPRING_2 || OurTopology::value == stk::topology::SPRING_3,
                "The topology of a line segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = stk::mesh::NgpField<Scalar>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  NgpLineSegmentData(stk::mesh::NgpMesh ngp_mesh, node_coords_data_t& node_coords_data)
      : ngp_mesh_(ngp_mesh), node_coords_data_(node_coords_data) {
    MUNDY_THROW_ASSERT(node_coords_data_.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The node_coords data must be a field of NODE_RANK");
  }

  stk::mesh::NgpMesh &ngp_mesh() {
    return ngp_mesh_;
  }

  const stk::mesh::NgpMesh &ngp_mesh() const {
    return ngp_mesh_;
  }

  const node_coords_data_t& node_coords_data() const {
    return node_coords_data_;
  }

  node_coords_data_t& node_coords_data() {
    return node_coords_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<NgpLineSegmentData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  node_coords_data_t& node_coords_data_;
};  // NgpLineSegmentData

/// \brief A helper function to create a LineSegmentData object
///
/// This function creates a LineSegmentData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology>  // Must be provided
auto create_line_segment_data(stk::mesh::BulkData& bulk_data, stk::mesh::Field<Scalar>& node_coords_data) {
  return LineSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>>{bulk_data, node_coords_data};
}

/// \brief A helper function to create a NgpLineSegmentData object
/// See the discussion for create_line_segment_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology>  // Must be provided
auto create_ngp_line_segment_data(stk::mesh::NgpMesh ngp_mesh, stk::mesh::NgpField<Scalar>& node_coords_data) {
  return NgpLineSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>>{ngp_mesh, node_coords_data};
}

/// \brief Check if the type provides the same data as LineSegmentData
template <typename Agg>
concept ValidLineSegmentDataType = requires(Agg agg) { 
      typename Agg::scalar_t; 
    { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
    }
  && std::convertible_to<decltype(std::declval<Agg>().bulk_data()), stk::mesh::BulkData&>
  && std::convertible_to<decltype(std::declval<Agg>().node_coords_data()), stk::mesh::Field<typename Agg::scalar_t>&>;

/// \brief Check if the type provides the same data as NgpLineSegmentData
template <typename Agg>
concept ValidNgpLineSegmentDataType = requires(Agg agg) { 
      typename Agg::scalar_t; 
    { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
    }
  && std::convertible_to<decltype(std::declval<Agg>().ngp_mesh()), stk::mesh::NgpMesh&>
  && std::convertible_to<decltype(std::declval<Agg>().node_coords_data()), stk::mesh::NgpField<typename Agg::scalar_t>&>;

/// \brief A helper function to get an updated NgpLineSegmentData object from a LineSegmentData object
/// \param data The LineSegmentData object to convert
template <ValidLineSegmentDataType LineSegmentDataType>
auto get_updated_ngp_data(LineSegmentDataType data) {
  using scalar_t = typename LineSegmentDataType::scalar_t;
  constexpr stk::topology::topology_t topology_t = LineSegmentDataType::topology_t;
  return create_ngp_line_segment_data<scalar_t, topology_t>(
      stk::mesh::get_updated_ngp_mesh(data.bulk_data()),  //
      stk::mesh::get_updated_ngp_field<scalar_t>(data.node_coords_data()));
}

/// \brief A traits class to provide abstracted access to a line_segment's data via an aggregate
template <typename Agg>
struct LineSegmentDataTraits {
  static_assert(
      ValidLineSegmentDataType<Agg>,
      "Agg must satisfy the ValidLineSegmentDataType concept.\n"
      "Basically, Agg must have all the same getters and types aliases as NgpLineSegmentData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  static decltype(auto) node_coords(Agg agg, stk::mesh::Entity line_segment_node) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data(), line_segment_node);
  }
};  // LineSegmentDataTraits

/// \brief A traits class to provide abstracted access to a line_segment's data via an NGP-compatible aggregate
/// See the discussion for LineSegmentDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpLineSegmentDataTraits {
  static_assert(
      ValidNgpLineSegmentDataType<Agg>,
      "Agg must satisfy the ValidNgpLineSegmentDataType concept.\n"
      "Basically, Agg must have the same getters and types aliases as NgpLineSegmentData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) node_coords(Agg agg, stk::mesh::FastMeshIndex line_segment_index) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data(), line_segment_index);
  }
};  // NgpLineSegmentDataTraits

/// @brief A view of an STK entity meant to represent a line_segment
///
/// We type specialize this class based on the valid set of topologies for an line entity.
///
/// Use \ref create_line_entity_view to build a LineSegmentEntityView object with automatic template deduction.
template <stk::topology::topology_t OurTopology, typename LineSegmentDataType>
class LineSegmentEntityView;

/// @brief A view of a STK entity meant to represent a line_segment
template <stk::topology::topology_t OurTopology, typename LineSegmentDataType>
class LineSegmentEntityView {
  static_assert(OurTopology == stk::topology::LINE_2 || OurTopology == stk::topology::LINE_3 ||
                    OurTopology == stk::topology::BEAM_2 || OurTopology == stk::topology::BEAM_3 ||
                    OurTopology == stk::topology::SPRING_2 || OurTopology == stk::topology::SPRING_3,
                "The topology of a line segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");
  static_assert(LineSegmentDataType::topology_t == OurTopology,
                "The topology of the line segment data must match the view");

 public:
  using data_access_t = LineSegmentDataTraits<LineSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::node_coords(std::declval<LineSegmentDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology_t = OurTopology;
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;

  LineSegmentEntityView(LineSegmentDataType data, stk::mesh::Entity line_segment)
      : data_(data),
        line_segment_(line_segment),
        start_node_(data_.bulk_data().begin_nodes(line_segment_)[0]),
        end_node_(data_.bulk_data().begin_nodes(line_segment_)[1]) {
    MUNDY_THROW_ASSERT(data_.bulk_data().entity_rank(line_segment_) == rank, std::invalid_argument,
                       "The line_segment entity rank must match the given topology");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(line_segment_), std::invalid_argument,
                       "The given line_segment entity is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().num_nodes(line_segment_) >= 2, std::invalid_argument,
                       "The given line_segment entity must have at least two nodes");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(start_node_), std::invalid_argument,
                       "The start node entity associated with the line_segment is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(end_node_), std::invalid_argument,
                       "The end node entity associated with the line_segment is not valid");
  }

  decltype(auto) data() {
    return data_;
  }

  decltype(auto) data() const {
    return data_;
  }

  stk::mesh::Entity& line_segment_entity() {
    return line_segment_;
  }

  const stk::mesh::Entity& line_segment_entity() const {
    return line_segment_;
  }

  stk::mesh::Entity& start_node_entity() {
    return start_node_;
  }

  const stk::mesh::Entity& start_node_entity() const {
    return start_node_;
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

  decltype(auto) end() {
    return data_access_t::node_coords(data(), end_node_entity());
  }

  decltype(auto) end() const {
    return data_access_t::node_coords(data(), end_node_entity());
  }

 private:
  LineSegmentDataType data_;
  stk::mesh::Entity line_segment_;
  stk::mesh::Entity start_node_;
  stk::mesh::Entity end_node_;
};  // LineSegmentEntityView<OurTopology, LineSegmentDataType>

/// @brief An ngp-compatible view of an STK entity meant to represent a line_segment
/// See the discussion for LineSegmentEntityView for more information. The only difference is ngp-compatible data
/// access.
template <stk::topology::topology_t OurTopology, typename NgpLineSegmentDataType>
class NgpLineSegmentEntityView;

/// @brief An ngp-compatible view of an STK entity meant to represent a line_segment
template <stk::topology::topology_t OurTopology, typename NgpLineSegmentDataType>
class NgpLineSegmentEntityView {
  static_assert(OurTopology == stk::topology::LINE_2 || OurTopology == stk::topology::LINE_3 ||
                    OurTopology == stk::topology::BEAM_2 || OurTopology == stk::topology::BEAM_3 ||
                    OurTopology == stk::topology::SPRING_2 || OurTopology == stk::topology::SPRING_3,
                "The topology of a line segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");
  static_assert(NgpLineSegmentDataType::topology_t == OurTopology,
                "The topology of the line segment data must match the view");

 public:
  using data_access_t = NgpLineSegmentDataTraits<NgpLineSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<NgpLineSegmentDataType>(),
                                                      std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology_t = OurTopology;
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;

  KOKKOS_INLINE_FUNCTION
  NgpLineSegmentEntityView(NgpLineSegmentDataType data, stk::mesh::FastMeshIndex line_segment_index)
      : data_(data),
        line_segment_index_(line_segment_index),
        start_node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, line_segment_index_)[0])),
        end_node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, line_segment_index_)[1])) {
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
  stk::mesh::FastMeshIndex& line_segment_index() {
    return line_segment_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& line_segment_index() const {
    return line_segment_index_;
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
  decltype(auto) end() {
    return data_access_t::node_coords(data(), end_node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) end() const {
    return data_access_t::node_coords(data(), end_node_index());
  }

 private:
  NgpLineSegmentDataType data_;
  stk::mesh::FastMeshIndex line_segment_index_;
  stk::mesh::FastMeshIndex start_node_index_;
  stk::mesh::FastMeshIndex end_node_index_;
};  // NgpLineSegmentEntityView<OurTopology, NgpLineSegmentDataType>

/// \brief A helper function to create a LineSegmentEntityView object with type deduction
template <typename LineSegmentDataType>  // deduced
auto create_line_segment_entity_view(LineSegmentDataType& data, stk::mesh::Entity line_segment) {
  return LineSegmentEntityView<LineSegmentDataType::topology_t, LineSegmentDataType>(data, line_segment);
}

/// \brief A helper function to create a NgpLineSegmentEntityView object with type deduction
template <typename NgpLineSegmentDataType>  // deduced
auto create_ngp_line_segment_entity_view(NgpLineSegmentDataType data, stk::mesh::FastMeshIndex line_segment_index) {
  return NgpLineSegmentEntityView<NgpLineSegmentDataType::topology_t, NgpLineSegmentDataType>(data, line_segment_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_LINESEGMENT_HPP_