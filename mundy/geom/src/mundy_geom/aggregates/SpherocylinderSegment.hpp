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
#include <mundy_geom/primitives/SpherocylinderSegment.hpp>  // for mundy::geom::ValidSpherocylinderSegmentType
#include <mundy_mesh/BulkData.hpp>                          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of sphero-cylinder segments
///
/// The topology of a line segment directly effects the access pattern for the underlying data.
/// Regardless of the topology, the node coordinates are stored on all nodes of the line segment, whereas
/// the radius is stored on the rank of the line segment. Allowable topologies are:
///   - LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, SPRING_3
template <typename Scalar,       //
          typename OurTopology,  //
          typename HasSharedRadius = std::false_type>
class SpherocylinderSegmentData {
  static_assert(OurTopology::value == stk::topology::LINE_2 || OurTopology::value == stk::topology::LINE_3 ||
                    OurTopology::value == stk::topology::BEAM_2 || OurTopology::value == stk::topology::BEAM_3 ||
                    OurTopology::value == stk::topology::SPRING_2 || OurTopology::value == stk::topology::SPRING_3,
                "The topology of a line segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = stk::mesh::Field<Scalar>;
  using radius_data_t = std::conditional_t<HasSharedRadius::value, Scalar, stk::mesh::Field<Scalar>>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  SpherocylinderSegmentData(stk::mesh::BulkData& bulk_data, node_coords_data_t& node_coords_data,
                            radius_data_t& radius_data)
      : bulk_data_(bulk_data), node_coords_data_(node_coords_data), radius_data_(radius_data) {
    MUNDY_THROW_ASSERT(node_coords_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The node_coords data must be a field of NODE_RANK");
    stk::topology our_topology = topology_t;
    if constexpr (!HasSharedRadius::value) {
      MUNDY_THROW_ASSERT(radius_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                         "The radius data must be a field of the same rank as the segment");
    }
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

  const radius_data_t& radius_data() const {
    return radius_data_;
  }

  radius_data_t& radius_data() {
    return radius_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<SpherocylinderSegmentData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  stk::mesh::BulkData& bulk_data_;
  node_coords_data_t& node_coords_data_;
  radius_data_t& radius_data_;
};  // SpherocylinderSegmentData

/// \brief Aggregate to hold the data for a collection of NGP-compatible sphero-cylinder segments
/// See the discussion for SpherocylinderSegmentData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,       //
          typename OurTopology,  //
          typename HasSharedRadius = std::false_type>
class NgpSpherocylinderSegmentData {
  static_assert(OurTopology::value == stk::topology::LINE_2 || OurTopology::value == stk::topology::LINE_3 ||
                    OurTopology::value == stk::topology::BEAM_2 || OurTopology::value == stk::topology::BEAM_3 ||
                    OurTopology::value == stk::topology::SPRING_2 || OurTopology::value == stk::topology::SPRING_3,
                "The topology of a line segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");

 public:
  using scalar_t = Scalar;
  using node_coords_data_t = stk::mesh::NgpField<Scalar>;
  using radius_data_t = std::conditional_t<HasSharedRadius::value, Scalar, stk::mesh::NgpField<Scalar>>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  NgpSpherocylinderSegmentData(stk::mesh::NgpMesh ngp_mesh, node_coords_data_t& node_coords_data,
                               radius_data_t& radius_data)
      : ngp_mesh_(ngp_mesh), node_coords_data_(node_coords_data), radius_data_(radius_data) {
    MUNDY_THROW_ASSERT(node_coords_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The node_coords data must be a field of NODE_RANK");
    stk::topology our_topology = topology_t;
    if constexpr (!HasSharedRadius::value) {
      MUNDY_THROW_ASSERT(radius_data.get_rank() == our_topology.rank(), std::invalid_argument,
                         "The radius data must be a field of the same rank as the segment");
    }
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

  const radius_data_t& radius_data() const {
    return radius_data_;
  }

  radius_data_t& radius_data() {
    return radius_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<NgpSpherocylinderSegmentData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  node_coords_data_t& node_coords_data_;
  radius_data_t& radius_data_;
};  // NgpSpherocylinderSegmentData

/// \brief A helper function to create a SpherocylinderSegmentData object
///
/// This function creates a SpherocylinderSegmentData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename RadiusDataType>                // deduced
auto create_spherocylinder_segment_data(stk::mesh::BulkData& bulk_data, stk::mesh::Field<Scalar>& node_coords_data,
                                        RadiusDataType& radius_data) {
  constexpr bool is_radius_shared = std::is_same_v<std::decay_t<RadiusDataType>, Scalar>;
  if constexpr (is_radius_shared) {
    return SpherocylinderSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type>{
        bulk_data, node_coords_data, radius_data};
  } else {
    return SpherocylinderSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type>{
        bulk_data, node_coords_data, radius_data};
  }
}

/// \brief A helper function to create a NgpSpherocylinderSegmentData object
/// See the discussion for create_spherocylinder_segment_data for more information. Only difference is NgpFields over
/// Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename RadiusDataType>                // deduced
auto create_ngp_spherocylinder_segment_data(stk::mesh::NgpMesh ngp_mesh, stk::mesh::NgpField<Scalar>& node_coords_data,
                                            RadiusDataType& radius_data) {
  constexpr bool is_radius_shared = std::is_same_v<std::decay_t<RadiusDataType>, Scalar>;
  if constexpr (is_radius_shared) {
    return NgpSpherocylinderSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type>{
        ngp_mesh, node_coords_data, radius_data};
  } else {
    return NgpSpherocylinderSegmentData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type>{
        ngp_mesh, node_coords_data, radius_data};
  }
}

/// \brief Check if the type provides the same data as SpherocylinderSegmentData
template <typename Agg>
concept ValidSpherocylinderSegmentDataType = requires(Agg agg) { 
      typename Agg::scalar_t; 
    { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
    }
  && std::convertible_to<decltype(std::declval<Agg>().bulk_data()), stk::mesh::BulkData&>
  && std::convertible_to<decltype(std::declval<Agg>().node_coords_data()), stk::mesh::Field<typename Agg::scalar_t>&>
  && (std::convertible_to<decltype(std::declval<Agg>().radius_data()), stk::mesh::Field<typename Agg::scalar_t>&> ||
      std::convertible_to<decltype(std::declval<Agg>().radius_data()), typename Agg::scalar_t&>);

/// \brief Check if the type provides the same data as NgpSpherocylinderSegmentData
template <typename Agg>
concept ValidNgpSpherocylinderSegmentDataType = requires(Agg agg) { 
      typename Agg::scalar_t; 
    { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
    }
  && std::convertible_to<decltype(std::declval<Agg>().ngp_mesh()), stk::mesh::NgpMesh&>
  && std::convertible_to<decltype(std::declval<Agg>().node_coords_data()), stk::mesh::NgpField<typename Agg::scalar_t>&>
  && (std::convertible_to<decltype(std::declval<Agg>().radius_data()), stk::mesh::NgpField<typename Agg::scalar_t>&> ||
      std::convertible_to<decltype(std::declval<Agg>().radius_data()), typename Agg::scalar_t&>);

/// \brief A helper function to get an updated NgpSpherocylinderSegmentData object from a SpherocylinderSegmentData
/// object \param data The SpherocylinderSegmentData object to convert
template <ValidSpherocylinderSegmentDataType SpherocylinderSegmentDataType>
auto get_updated_ngp_data(SpherocylinderSegmentDataType data) {
  using scalar_t = typename SpherocylinderSegmentDataType::scalar_t;
  using radius_data_t = decltype(data.radius_data());

  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<radius_data_t>, stk::mesh::Field<scalar_t>>;
  constexpr stk::topology::topology_t topology_t = SpherocylinderSegmentDataType::topology_t;
  if constexpr (is_radius_a_field) {
    return create_ngp_sphere_data<scalar_t, topology_t>(                    //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),                  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.node_coords_data),  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.radius_data()));
  } else {
    return create_ngp_sphere_data<scalar_t, topology_t>(                    //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),                  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.node_coords_data),  //
        data.radius_data());
  }
}

/// \brief A traits class to provide abstracted access to a spherocylinder_segment's data via an aggregate
template <typename Agg>
struct SpherocylinderSegmentDataTraits {
  static_assert(ValidSpherocylinderSegmentDataType<Agg>,
                "Agg must satisfy the ValidSpherocylinderSegmentDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSpherocylinderSegmentData but is "
                "free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = decltype(std::declval<Agg>().node_coords_data());
  using radius_data_t = decltype(std::declval<Agg>().radius_data());
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  static decltype(auto) node_coords(Agg agg, stk::mesh::Entity spherocylinder_segment_node) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data(), spherocylinder_segment_node);
  }

  static decltype(auto) radius(Agg agg, stk::mesh::Entity spherocylinder_segment) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return stk::mesh::field_data(agg.radius_data(), spherocylinder_segment)[0];
    }
  }
};  // SpherocylinderSegmentDataTraits

/// \brief A traits class to provide abstracted access to a spherocylinder_segment's data via an NGP-compatible
/// aggregate See the discussion for SpherocylinderSegmentDataTraits for more information. Only difference is
/// Ngp-compatible data.
template <typename Agg>
struct NgpSpherocylinderSegmentDataTraits {
  static_assert(ValidNgpSpherocylinderSegmentDataType<Agg>,
                "Agg must satisfy the ValidNgpSpherocylinderSegmentDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSpherocylinderSegmentData but is "
                "free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = decltype(std::declval<Agg>().node_coords_data());
  using radius_data_t = decltype(std::declval<Agg>().radius_data());

  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) node_coords(Agg agg, stk::mesh::FastMeshIndex spherocylinder_segment_index) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data(), spherocylinder_segment_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) radius(Agg agg, stk::mesh::FastMeshIndex spherocylinder_segment_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return agg.radius_data()(spherocylinder_segment_index, 0);
    }
  }
};  // NgpSpherocylinderSegmentDataTraits

/// @brief A view of an STK entity meant to represent a spherocylinder_segment
///
/// We type specialize this class based on the valid set of topologies for a spherocylinder_segment entity.
///
/// Use \ref create_spherocylinder_segment_entity_view to build an SpherocylinderSegmentEntityView object
/// with automatic template deduction.
template <stk::topology::topology_t OurTopology, typename SpherocylinderSegmentDataType>
class SpherocylinderSegmentEntityView {
  static_assert(
      OurTopology == stk::topology::LINE_2 || OurTopology == stk::topology::LINE_3 ||
          OurTopology == stk::topology::BEAM_2 || OurTopology == stk::topology::BEAM_3 ||
          OurTopology == stk::topology::SPRING_2 || OurTopology == stk::topology::SPRING_3,
      "The topology of a spherocylinder segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");
  static_assert(SpherocylinderSegmentDataType::topology_t == OurTopology,
                "The topology of the spherocylinder segment data must match the view");

 public:
  using data_access_t = SpherocylinderSegmentDataTraits<SpherocylinderSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<SpherocylinderSegmentDataType>(),
                                                      std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology_t = OurTopology;
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;

  SpherocylinderSegmentEntityView(SpherocylinderSegmentDataType data, stk::mesh::Entity spherocylinder_segment)
      : data_(data),
        spherocylinder_segment_(spherocylinder_segment),
        start_node_(data_.bulk_data().begin_nodes(spherocylinder_segment_)[0]),
        end_node_(data_.bulk_data().begin_nodes(spherocylinder_segment_)[1]) {
    MUNDY_THROW_ASSERT(data_.bulk_data().entity_rank(spherocylinder_segment_) == rank, std::invalid_argument,
                       "The spherocylinder_segment's entity rank must match that of the view's topology");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(spherocylinder_segment_), std::invalid_argument,
                       "The given spherocylinder_segment entity is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().num_nodes(spherocylinder_segment_) >= 2, std::invalid_argument,
                       "The given spherocylinder_segment entity must have at least two nodes.");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(start_node_), std::invalid_argument,
                       "The start node entity associated with the spherocylinder_segment is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(end_node_), std::invalid_argument,
                       "The end node entity associated with the spherocylinder_segment is not valid");
  }

  decltype(auto) data() {
    return data_;
  }

  decltype(auto) data() const {
    return data_;
  }

  stk::mesh::Entity& spherocylinder_segment_entity() {
    return spherocylinder_segment_;
  }

  const stk::mesh::Entity& spherocylinder_segment_entity() const {
    return spherocylinder_segment_;
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

  decltype(auto) radius() {
    return data_access_t::radius(data(), spherocylinder_segment_entity());
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data(), spherocylinder_segment_entity());
  }

 private:
  SpherocylinderSegmentDataType data_;
  stk::mesh::Entity spherocylinder_segment_;
  stk::mesh::Entity start_node_;
  stk::mesh::Entity end_node_;
};  // SpherocylinderSegmentEntityView

/// @brief An ngp-compatible view of an STK entity meant to represent a spherocylinder_segment
/// See the discussion for SpherocylinderSegmentEntityView for more information. The only difference is ngp-compatible
/// data access.
template <stk::topology::topology_t OurTopology, typename NgpSpherocylinderSegmentDataType>
class NgpSpherocylinderSegmentEntityView {
  static_assert(
      OurTopology == stk::topology::LINE_2 || OurTopology == stk::topology::LINE_3 ||
          OurTopology == stk::topology::BEAM_2 || OurTopology == stk::topology::BEAM_3 ||
          OurTopology == stk::topology::SPRING_2 || OurTopology == stk::topology::SPRING_3,
      "The topology of a spherocylinder segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");
  static_assert(NgpSpherocylinderSegmentDataType::topology_t == OurTopology,
                "The topology of the spherocylinder segment data must match the view");

 public:
  using data_access_t = NgpSpherocylinderSegmentDataTraits<NgpSpherocylinderSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<NgpSpherocylinderSegmentDataType>(),
                                                      std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology_t = OurTopology;
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;

  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderSegmentEntityView(NgpSpherocylinderSegmentDataType data,
                                     stk::mesh::FastMeshIndex spherocylinder_segment_index)
      : data_(data),
        spherocylinder_segment_index_(spherocylinder_segment_index),
        start_node_index_(
            data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, spherocylinder_segment_index_)[0])),
        end_node_index_(
            data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, spherocylinder_segment_index_)[1])) {
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
  stk::mesh::FastMeshIndex& spherocylinder_segment_index() {
    return spherocylinder_segment_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& spherocylinder_segment_index() const {
    return spherocylinder_segment_index_;
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

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data(), spherocylinder_segment_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data(), spherocylinder_segment_index());
  }

 private:
  NgpSpherocylinderSegmentDataType data_;
  stk::mesh::FastMeshIndex spherocylinder_segment_index_;
  stk::mesh::FastMeshIndex start_node_index_;
  stk::mesh::FastMeshIndex end_node_index_;
};  // NgpSpherocylinderSegmentEntityView

/// \brief A helper function to create a SpherocylinderSegmentEntityView object with type deduction
template <typename SpherocylinderSegmentDataType>  // deduced
auto create_spherocylinder_segment_entity_view(SpherocylinderSegmentDataType& data,
                                               stk::mesh::Entity spherocylinder_segment) {
  return SpherocylinderSegmentEntityView<SpherocylinderSegmentDataType::topology_t, SpherocylinderSegmentDataType>(
      data, spherocylinder_segment);
}

/// \brief A helper function to create a NgpSpherocylinderSegmentEntityView object with type deduction
template <typename NgpSpherocylinderSegmentDataType>  // deduced
auto create_ngp_spherocylinder_segment_entity_view(NgpSpherocylinderSegmentDataType data,
                                                   stk::mesh::FastMeshIndex spherocylinder_segment_index) {
  return NgpSpherocylinderSegmentEntityView<NgpSpherocylinderSegmentDataType::topology_t,
                                            NgpSpherocylinderSegmentDataType>(data, spherocylinder_segment_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENT_HPP_