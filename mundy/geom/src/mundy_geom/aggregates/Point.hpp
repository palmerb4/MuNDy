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
/// The topology of a line segment directly effects the access pattern for the underlying data.
/// Regardless of the topology, the node coordinates are stored on the node of the point.
/// However, how we access this node changes based on the rank of the point. Allowable topologies are:
///   - NODE, PARTICLE
template <typename Scalar,                        //
          stk::topology::topology_t OurTopology,  //
          typename NodeCoordsType = stk::mesh::Field<Scalar>>
struct PointData {
  static_assert(OurTopology == stk::topology::NODE || OurTopology == stk::topology::PARTICLE,
                "The topology of a point must be NODE or PARTICLE");
  static_assert(std::is_same_v<std::decay_t<NodeCoordsType>, stk::mesh::Field<Scalar>>,
                "NodeCoordsType must be either a const or non-const field of scalars");

  using scalar_t = Scalar;
  using node_coords_data_t = NodeCoordsType;

  static constexpr stk::topology::topology_t topology = OurTopology;

  stk::mesh::BulkData& bulk_data;
  node_coords_data_t& node_coords_data;
};  // PointData

/// \brief A struct to hold the data for a collection of NGP-compatible points
/// See the discussion for PointData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        //
          stk::topology::topology_t OurTopology,  //
          typename NodeCoordsType = stk::mesh::NgpField<Scalar>>
struct NgpPointData {
  static_assert(OurTopology == stk::topology::NODE || OurTopology == stk::topology::PARTICLE,
                "The topology of a point must be NODE or PARTICLE");
  static_assert(is_point_v<std::decay_t<NodeCoordsType>> ||
                    std::is_same_v<std::decay_t<NodeCoordsType>, stk::mesh::NgpField<Scalar>>,
                "NodeCoordsType must be either a const or non-const field of scalars");

  using scalar_t = Scalar;
  using node_coords_data_t = NodeCoordsType;

  static constexpr stk::topology::topology_t topology = OurTopology;

  stk::mesh::NgpMesh ngp_mesh;
  node_coords_data_t& node_coords_data;
};  // NgpPointData

/// \brief A helper function to create a PointData object with automatic template deduction
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename NodeCoordsType>                // deduced
auto create_point_data(stk::mesh::BulkData& bulk_data, NodeCoordsType& node_coords_data) {
  MUNDY_THROW_ASSERT(node_coords_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                     "The center data must be a field of NODE_RANK");
  return PointData<Scalar, OurTopology, NodeCoordsType>{bulk_data, node_coords_data};
}

/// \brief A helper function to create a NgpPointData object
/// See the discussion for create_point_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename NodeCoordsType>                // deduced
auto create_ngp_point_data(stk::mesh::NgpMesh ngp_mesh, NodeCoordsType& node_coords_data) {
  MUNDY_THROW_ASSERT(node_coords_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                     "The center data must be a field of NODE_RANK");
  return NgpPointData<Scalar, OurTopology, NodeCoordsType>{ngp_mesh, node_coords_data};
}

/// \brief A concept to check if a type provides the same data as PointData
template <typename Agg>
concept ValidDefaultPointDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::node_coords_data_t;
  std::is_same_v<std::decay_t<typename Agg::node_coords_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { Agg::topology } -> std::convertible_to<stk::topology::topology_t>;
  { agg.bulk_data } -> std::convertible_to<stk::mesh::BulkData&>;
  { agg.node_coords_data } -> std::convertible_to<typename Agg::node_coords_data_t&>;
};  // ValidDefaultPointDataType

/// \brief A concept to check if a type provides the same data as NgpPointData
template <typename Agg>
concept ValidDefaultNgpPointDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::node_coords_data_t;
  std::is_same_v<std::decay_t<typename Agg::node_coords_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { Agg::topology } -> std::convertible_to<stk::topology::topology_t>;
  { agg.ngp_mesh } -> std::convertible_to<stk::mesh::NgpMesh>;
  { agg.node_coords_data } -> std::convertible_to<typename Agg::node_coords_data_t&>;
};  // ValidDefaultNgpPointDataType

static_assert(ValidDefaultPointDataType<PointData<float,                //
                                                  stk::topology::NODE,  //
                                                  stk::mesh::Field<float>>> &&
                  ValidDefaultPointDataType<PointData<float,                    //
                                                      stk::topology::PARTICLE,  //
                                                      stk::mesh::Field<float>>>,
              "PointData must satisfy the ValidDefaultPointDataType concept");

static_assert(ValidDefaultNgpPointDataType<NgpPointData<float,                //
                                                        stk::topology::NODE,  //
                                                        stk::mesh::NgpField<float>>> &&
                  ValidDefaultNgpPointDataType<NgpPointData<float,                    //
                                                            stk::topology::PARTICLE,  //
                                                            stk::mesh::NgpField<float>>>,
              "NgpPointData must satisfy the ValidDefaultNgpPointDataType concept");

/// \brief A helper function to get an updated NgpPointData object from a PointData object
/// \param data The PointData object to convert
template <ValidDefaultPointDataType PointDataType>
auto get_updated_ngp_data(PointDataType data) {
  using scalar_t = typename PointDataType::scalar_t;
  using node_coords_data_t = typename PointDataType::node_coords_data_t;
  constexpr stk::topology::topology_t our_topology = PointDataType::topology;

  return create_ngp_point_data<scalar_t, our_topology>(
      stk::mesh::get_updated_ngp_field<scalar_t>(data.node_coords_data));
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
  using node_coords_data_t = typename Agg::node_coords_data_t;
  static constexpr stk::topology::topology_t topology = Agg::topology;

  static decltype(auto) center(Agg agg, stk::mesh::Entity point_node) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data, point_node);
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
  using node_coords_data_t = typename Agg::node_coords_data_t;
  static constexpr stk::topology::topology_t topology = Agg::topology;

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex point_node_index) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data, point_node_index);
  }
};  // NgpPointDataTraits

/// @brief A view of an STK entity meant to represent a point
///
/// We type specialize this class based on the valid set of topologies for a point entity.
///
/// Use \ref create_point_entity_view to build a PointEntityView object with automatic template deduction.
template <stk::topology::topology_t OurTopology, typename PointDataType>
class PointEntityView;

/// @brief A view of a NODE STK entity meant to represent a point
template <typename PointDataType>
class PointEntityView<stk::topology::NODE, PointDataType> {
  static_assert(PointDataType::topology == stk::topology::NODE, "The topology of the point data must match the view");

 public:
  using data_access_t = PointDataTraits<PointDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<PointDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  PointEntityView(PointDataType data, stk::mesh::Entity point) : data_(data), point_(point) {
    MUNDY_THROW_ASSERT(data_.bulk_data.entity_rank(point_) == stk::topology::NODE_RANK, std::invalid_argument,
                       "The point entity rank must be NODE_RANK");
    MUNDY_THROW_ASSERT(data_.bulk_data.is_valid(point_), std::invalid_argument, "The given point entity is not valid");
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
};  // PointEntityView<stk::topology::NODE, PointDataType>

template <typename PointDataType>
class PointEntityView<stk::topology::PARTICLE, PointDataType> {
  static_assert(PointDataType::topology == stk::topology::PARTICLE,
                "The topology of the point data must match the view");

 public:
  using data_access_t = PointDataTraits<PointDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<PointDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  PointEntityView(PointDataType data, stk::mesh::Entity point)
      : data_(data), point_(point), node_(data_.bulk_data.begin_nodes(point_)[0]) {
    MUNDY_THROW_ASSERT(data_.bulk_data.entity_rank(point_) == stk::topology::ELEM_RANK, std::invalid_argument,
                       "The point entity rank must be ELEM_RANK");
    MUNDY_THROW_ASSERT(data_.bulk_data.is_valid(point_), std::invalid_argument, "The given point entity is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data.num_nodes(point_) >= 1, std::invalid_argument,
                       "The given point entity must have at least one node");
    MUNDY_THROW_ASSERT(data_.bulk_data.is_valid(node_), std::invalid_argument,
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
};  // PointEntityView<stk::topology::PARTICLE, PointDataType>

/// @brief An ngp-compatible view of a STK entity meant to represent a point
/// See the discussion for PointEntityView for more information. The only difference is ngp-compatible data access.
template <stk::topology::topology_t OurTopology, typename NgpPointDataType>
class NgpPointEntityView;

/// @brief An ngp-compatible view of a NODE STK entity meant to represent a point
template <typename NgpPointDataType>
class NgpPointEntityView<stk::topology::NODE, NgpPointDataType> {
  static_assert(NgpPointDataType::topology == stk::topology::NODE,
                "The topology of the point data must match the view");

 public:
  using data_access_t = NgpPointDataTraits<NgpPointDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<NgpPointDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpPointEntityView(NgpPointDataType data,
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
};  // NgpPointEntityView<stk::topology::NODE, NgpPointDataType>

/// @brief An ngp-compatible view of a PARTICLE STK entity meant to represent a point
template <typename NgpPointDataType>
class NgpPointEntityView<stk::topology::PARTICLE, NgpPointDataType> {
  static_assert(NgpPointDataType::topology == stk::topology::PARTICLE,
                "The topology of the point data must match the view");

 public:
  using data_access_t = NgpPointDataTraits<NgpPointDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<NgpPointDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpPointEntityView(NgpPointDataType data, stk::mesh::FastMeshIndex point_index)
      : data_(data),
        point_index_(point_index),
        node_index_(data_.ngp_mesh.fast_mesh_index(data_.ngp_mesh.get_nodes(rank, point_index_)[0])) {
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

static_assert(ValidPointType<PointEntityView<stk::topology::NODE,
                                             PointData<float,                //
                                                       stk::topology::NODE,  //
                                                       stk::mesh::Field<float>>>> &&
                  ValidPointType<PointEntityView<stk::topology::PARTICLE,
                                                 PointData<float,                    //
                                                           stk::topology::PARTICLE,  //
                                                           stk::mesh::Field<float>>>> &&
                  ValidPointType<NgpPointEntityView<stk::topology::NODE,
                                                    NgpPointData<float,                //
                                                                 stk::topology::NODE,  //
                                                                 stk::mesh::NgpField<float>>>> &&
                  ValidPointType<NgpPointEntityView<stk::topology::PARTICLE,
                                                    NgpPointData<float,                    //
                                                                 stk::topology::PARTICLE,  //
                                                                 stk::mesh::NgpField<float>>>>,
              "PointEntityView and NgpPointEntityView must be valid Point types");

/// \brief A helper function to create a PointEntityView object with type deduction
template <typename PointDataType>                 // deduced
auto create_point_entity_view(PointDataType& data, stk::mesh::Entity point) {
  return PointEntityView<PointDataType::topology, PointDataType>(data, point);
}

/// \brief A helper function to create a NgpPointEntityView object with type deduction
template <typename NgpPointDataType>              // deduced
auto create_ngp_point_entity_view(NgpPointDataType data,
                                  stk::mesh::FastMeshIndex point_index) {
  return NgpPointEntityView<NgpPointDataType::topology, NgpPointDataType>(data, point_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_POINT_HPP_