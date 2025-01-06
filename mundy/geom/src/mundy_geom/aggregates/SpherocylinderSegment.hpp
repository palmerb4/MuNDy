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
///
/// The topology of a line segment directly effects the access pattern for the underlying data.
/// Regardless of the topology, the node coordinates are stored on all nodes of the line segment, whereas
/// the radius is stored on the rank of the line segment. Allowable topologies are:
///   - LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, SPRING_3
template <typename Scalar,                                         //
          stk::topology::topology_t OurTopology,                   //
          typename NodeCoordsDataType = stk::mesh::Field<Scalar>,  //
          typename RadiusDataType = stk::mesh::Field<Scalar>>
struct SpherocylinderSegmentData {
  static_assert(OurTopology == stk::topology::LINE_2 || OurTopology == stk::topology::LINE_3 ||
                    OurTopology == stk::topology::BEAM_2 || OurTopology == stk::topology::BEAM_3 ||
                    OurTopology == stk::topology::SPRING_2 || OurTopology == stk::topology::SPRING_3,
                "The topology of a line segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");
  static_assert(std::is_same_v<std::decay_t<NodeCoordsDataType>, stk::mesh::Field<Scalar>> &&
                    (std::is_same_v<std::decay_t<RadiusDataType>, Scalar> ||
                     std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::Field<Scalar>>),
                "NodeCoordsDataType must be either a const or non-const field of scalars\n"
                "RadiusDataType must be either a scalar or a field of scalars");

  using scalar_t = Scalar;
  using node_coords_data_t = NodeCoordsDataType;
  using radius_data_t = RadiusDataType;

  static constexpr stk::topology::topology_t topology = OurTopology;
  node_coords_data_t& node_coords_data;
  radius_data_t& radius_data;
};  // SpherocylinderSegmentData

/// \brief A struct to hold the data for a collection of NGP-compatible sphero-cylinder segments
/// See the discussion for SpherocylinderSegmentData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                                            //
          stk::topology::topology_t OurTopology,                      //
          typename NodeCoordsDataType = stk::mesh::NgpField<Scalar>,  //
          typename RadiusDataType = stk::mesh::NgpField<Scalar>>
struct NgpSpherocylinderSegmentData {
  static_assert(OurTopology == stk::topology::LINE_2 || OurTopology == stk::topology::LINE_3 ||
                    OurTopology == stk::topology::BEAM_2 || OurTopology == stk::topology::BEAM_3 ||
                    OurTopology == stk::topology::SPRING_2 || OurTopology == stk::topology::SPRING_3,
                "The topology of a line segment must be LINE_2, LINE_3, BEAM_2, BEAM_3, SPRING_2, or SPRING_3");
  static_assert(std::is_same_v<std::decay_t<NodeCoordsDataType>, stk::mesh::NgpField<Scalar>> &&
                    (std::is_same_v<std::decay_t<RadiusDataType>, Scalar> ||
                     std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::NgpField<Scalar>>),
                "NodeCoordsDataType must be either a const or non-const field of scalars\n"
                "RadiusDataType must be either a scalar or a field of scalars");

  using scalar_t = Scalar;
  using node_coords_data_t = NodeCoordsDataType;
  using radius_data_t = RadiusDataType;

  static constexpr stk::topology::topology_t topology = OurTopology;
  node_coords_data_t& node_coords_data;
  radius_data_t& radius_data;
};  // NgpSpherocylinderSegmentData

/// \brief A helper function to create a SpherocylinderSegmentData object
///
/// This function creates a SpherocylinderSegmentData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename NodeCoordsDataType,            // deduced
          typename RadiusDataType>                // deduced
auto create_spherocylinder_segment_data(NodeCoordsDataType& node_coords_data, RadiusDataType& radius_data) {
  MUNDY_THROW_ASSERT(node_coords_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                     "The node_coords data must be a field of NODE_RANK");
  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::Field<Scalar>>;
  stk::topology our_topology = OurTopology;
  if constexpr (is_radius_a_field) {
    MUNDY_THROW_ASSERT(radius_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                       "The radius data must be a field of the same rank as the segment");
  }
  return SpherocylinderSegmentData<Scalar, OurTopology, NodeCoordsDataType, RadiusDataType>{node_coords_data,
                                                                                            radius_data};
}

/// \brief A helper function to create a NgpSpherocylinderSegmentData object
/// See the discussion for create_spherocylinder_segment_data for more information. Only difference is NgpFields over
/// Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename NodeCoordsDataType,            // deduced
          typename RadiusDataType>                // deduced
auto create_ngp_spherocylinder_segment_data(NodeCoordsDataType& node_coords_data, RadiusDataType& radius_data) {
  MUNDY_THROW_ASSERT(node_coords_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                     "The node_coords data must be a field of NODE_RANK");
  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<RadiusDataType>, stk::mesh::NgpField<Scalar>>;
  stk::topology our_topology = OurTopology;
  if constexpr (is_radius_a_field) {
    MUNDY_THROW_ASSERT(radius_data.get_rank() == our_topology.rank(), std::invalid_argument,
                       "The radius data must be a field of the same rank as the segment");
  }
  return NgpSpherocylinderSegmentData<Scalar, OurTopology, NodeCoordsDataType, RadiusDataType>{node_coords_data,
                                                                                               radius_data};
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
  { Agg::topology } -> std::convertible_to<stk::topology::topology_t>;
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
  { Agg::topology } -> std::convertible_to<stk::topology::topology_t>;
  { agg.node_coords_data } -> std::convertible_to<typename Agg::node_coords_data_t&>;
  { agg.radius_data } -> std::convertible_to<typename Agg::radius_data_t&>;
};  // ValidDefaultNgpSpherocylinderSegmentDataType

static_assert(ValidDefaultSpherocylinderSegmentDataType<SpherocylinderSegmentData<float,                    //
                                                                                  stk::topology::LINE_2,    //
                                                                                  stk::mesh::Field<float>,  //
                                                                                  float>> &&
                  ValidDefaultSpherocylinderSegmentDataType<SpherocylinderSegmentData<float,                    //
                                                                                      stk::topology::BEAM_2,    //
                                                                                      stk::mesh::Field<float>,  //
                                                                                      stk::mesh::Field<float>>>,
              "SpherocylinderSegmentData must satisfy the ValidDefaultSpherocylinderSegmentDataType concept");

static_assert(
    ValidDefaultNgpSpherocylinderSegmentDataType<NgpSpherocylinderSegmentData<float,                       //
                                                                              stk::topology::LINE_2,       //
                                                                              stk::mesh::NgpField<float>,  //
                                                                              float>> &&
        ValidDefaultNgpSpherocylinderSegmentDataType<NgpSpherocylinderSegmentData<float,                       //
                                                                                  stk::topology::BEAM_2,       //
                                                                                  stk::mesh::NgpField<float>,  //
                                                                                  stk::mesh::NgpField<float>>>,
    "NgpSpherocylinderSegmentData must satisfy the ValidDefaultNgpSpherocylinderSegmentDataType concept");

/// \brief A helper function to get an updated NgpSpherocylinderSegmentData object from a SpherocylinderSegmentData
/// object \param data The SpherocylinderSegmentData object to convert
template <ValidDefaultSpherocylinderSegmentDataType SpherocylinderSegmentDataType>
auto get_updated_ngp_data(SpherocylinderSegmentDataType data) {
  using scalar_t = typename SpherocylinderSegmentDataType::scalar_t;
  using radius_data_t = typename SpherocylinderSegmentDataType::radius_data_t;

  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<radius_data_t>, stk::mesh::Field<scalar_t>>;
  constexpr stk::topology::topology_t our_topology = SpherocylinderSegmentDataType::topology;
  if constexpr (is_radius_a_field) {
    return create_ngp_sphere_data<scalar_t, our_topology>(                  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.node_coords_data),  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.radius_data));
  } else {
    return create_ngp_sphere_data<scalar_t, our_topology>(                  //
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

  static constexpr stk::topology::topology_t topology = Agg::topology;

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

  static constexpr stk::topology::topology_t topology = Agg::topology;

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
  static_assert(SpherocylinderSegmentDataType::topology == OurTopology,
                "The topology of the spherocylinder segment data must match the view");

 public:
  using data_access_t = SpherocylinderSegmentDataTraits<SpherocylinderSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<SpherocylinderSegmentDataType>(),
                                                      std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology = OurTopology;
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;

  SpherocylinderSegmentEntityView(const stk::mesh::BulkData& bulk_data, SpherocylinderSegmentDataType data,
                                  stk::mesh::Entity spherocylinder_segment)
      : data_(data),
        spherocylinder_segment_(spherocylinder_segment),
        start_node_(bulk_data.begin_nodes(spherocylinder_segment_)[0]),
        end_node_(bulk_data.begin_nodes(spherocylinder_segment_)[1]) {
    MUNDY_THROW_ASSERT(bulk_data.entity_rank(spherocylinder_segment_) == rank, std::invalid_argument,
                       "The spherocylinder_segment's entity rank must match that of the view's topology");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(spherocylinder_segment_), std::invalid_argument,
                       "The given spherocylinder_segment entity is not valid");
    MUNDY_THROW_ASSERT(bulk_data.num_nodes(spherocylinder_segment_) >= 2, std::invalid_argument,
                       "The given spherocylinder_segment entity must have at least two nodes.");
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
  static_assert(NgpSpherocylinderSegmentDataType::topology == OurTopology,
                "The topology of the spherocylinder segment data must match the view");

 public:
  using data_access_t = NgpSpherocylinderSegmentDataTraits<NgpSpherocylinderSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<NgpSpherocylinderSegmentDataType>(),
                                                      std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology = OurTopology;
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;

  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderSegmentEntityView(stk::mesh::NgpMesh ngp_mesh, NgpSpherocylinderSegmentDataType data,
                                     stk::mesh::FastMeshIndex spherocylinder_segment_index)
      : data_(data),
        spherocylinder_segment_index_(spherocylinder_segment_index),
        start_node_index_(ngp_mesh.fast_mesh_index(ngp_mesh.get_nodes(rank, spherocylinder_segment_index_)[0])),
        end_node_index_(ngp_mesh.fast_mesh_index(ngp_mesh.get_nodes(rank, spherocylinder_segment_index_)[1])) {
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
};  // NgpSpherocylinderSegmentEntityView

static_assert(
    ValidSpherocylinderSegmentType<
        SpherocylinderSegmentEntityView<stk::topology::LINE_2,
                                        SpherocylinderSegmentData<float,                    //
                                                                  stk::topology::LINE_2,    //
                                                                  stk::mesh::Field<float>,  //
                                                                  float>>> &&
        ValidSpherocylinderSegmentType<
            SpherocylinderSegmentEntityView<stk::topology::BEAM_2,
                                            SpherocylinderSegmentData<float,                    //
                                                                      stk::topology::BEAM_2,    //
                                                                      stk::mesh::Field<float>,  //
                                                                      stk::mesh::Field<float>>>> &&
        ValidSpherocylinderSegmentType<
            NgpSpherocylinderSegmentEntityView<stk::topology::LINE_2,
                                               NgpSpherocylinderSegmentData<float,                       //
                                                                            stk::topology::LINE_2,       //
                                                                            stk::mesh::NgpField<float>,  //
                                                                            float>>> &&
        ValidSpherocylinderSegmentType<
            NgpSpherocylinderSegmentEntityView<stk::topology::BEAM_2,
                                               NgpSpherocylinderSegmentData<float,                       //
                                                                            stk::topology::BEAM_2,       //
                                                                            stk::mesh::NgpField<float>,  //
                                                                            stk::mesh::NgpField<float>>>>,
    "SpherocylinderSegmentEntityView and NgpSpherocylinderSegmentEntityView must be valid SpherocylinderSegment types");

/// \brief A helper function to create a SpherocylinderSegmentEntityView object with type deduction
template <stk::topology::topology_t OurTopology,   // Must be provided
          typename SpherocylinderSegmentDataType>  // deduced
auto create_spherocylinder_segment_entity_view(const stk::mesh::BulkData& bulk_data,
                                               SpherocylinderSegmentDataType& data,
                                               stk::mesh::Entity spherocylinder_segment) {
  return SpherocylinderSegmentEntityView<OurTopology, SpherocylinderSegmentDataType>(bulk_data, data,
                                                                                     spherocylinder_segment);
}

/// \brief A helper function to create a NgpSpherocylinderSegmentEntityView object with type deduction
template <stk::topology::topology_t OurTopology,      // Must be provided
          typename NgpSpherocylinderSegmentDataType>  // deduced
auto create_ngp_spherocylinder_segment_entity_view(stk::mesh::NgpMesh ngp_mesh, NgpSpherocylinderSegmentDataType data,
                                                   stk::mesh::FastMeshIndex spherocylinder_segment_index) {
  return NgpSpherocylinderSegmentEntityView<OurTopology, NgpSpherocylinderSegmentDataType>(
      ngp_mesh, data, spherocylinder_segment_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERSEGMENT_HPP_