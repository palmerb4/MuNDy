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

#ifndef MUNDY_GEOM_AGGREGATES_LINE_HPP_
#define MUNDY_GEOM_AGGREGATES_LINE_HPP_

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/primitives/Line.hpp>  // for mundy::geom::ValidLineType
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief A struct to hold the data for a collection of infinite lines
///
/// The topology of a line directly effects the access pattern for the underlying data:
///   - NODE: All data is stored on a single node
///   - PARTICLE: The center is stored on a node, whereas the direction is stored on the element-rank particle
///
/// Use \ref create_line_data to build a LineData object with automatic template deduction.
template <typename Scalar,                                     //
          stk::topology::topology_t OurTopology,               //
          typename CenterDataType = stk::mesh::Field<Scalar>,  //
          typename DirectionDataType = stk::mesh::Field<Scalar>>
struct LineData {
  static_assert(OurTopology == stk::topology::NODE || OurTopology == stk::topology::PARTICLE,
                "The topology of a line must be either NODE or PARTICLE");
  static_assert(std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::Field<Scalar>>,
                "CenterDataType must be either a const or non-const field of scalars");
  static_assert(mundy::math::is_vector3_v<std::decay_t<DirectionDataType>> ||
                    std::is_same_v<std::decay_t<DirectionDataType>, stk::mesh::Field<Scalar>>,
                "DirectionDataType must be either a vector3 or a field of scalars");

  using scalar_t = Scalar;
  using center_data_t = CenterDataType;
  using direction_data_t = DirectionDataType;

  static constexpr stk::topology::topology_t topology = OurTopology;

  stk::mesh::BulkData& bulk_data;
  center_data_t& center_data;
  direction_data_t& direction_data;
};  // LineData

/// \brief A struct to hold the data for a collection of NGP-compatible lines
/// See the discussion for LineData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                                        //
          stk::topology::topology_t OurTopology,                  //
          typename CenterDataType = stk::mesh::NgpField<Scalar>,  //
          typename DirectionDataType = stk::mesh::NgpField<Scalar>>
struct NgpLineData {
  static_assert(OurTopology == stk::topology::NODE || OurTopology == stk::topology::PARTICLE,
                "The topology of a line must be either NODE or PARTICLE");
  static_assert(std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::NgpField<Scalar>>,
                "CenterDataType must be either a const or non-const field of scalars");
  static_assert(mundy::math::is_vector3_v<std::decay_t<DirectionDataType>> ||
                    std::is_same_v<std::decay_t<DirectionDataType>, stk::mesh::NgpField<Scalar>>,
                "DirectionDataType must be either a vector3 or a field of scalars");

  using scalar_t = Scalar;
  using center_data_t = CenterDataType;
  using direction_data_t = DirectionDataType;

  static constexpr stk::topology::topology_t topology = OurTopology;

  stk::mesh::NgpMesh ngp_mesh;
  center_data_t& center_data;
  direction_data_t& direction_data;
};  // NgpLineData

/// \brief A helper function to create a LineData object
///
/// This function creates a LineData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename CenterDataType,                // deduced
          typename DirectionDataType>             // deduced
auto create_line_data(stk::mesh::BulkData& bulk_data, CenterDataType& center_data, DirectionDataType& direction_data) {
  constexpr bool is_direction_a_field = std::is_same_v<std::decay_t<DirectionDataType>, stk::mesh::Field<Scalar>>;
  stk::topology our_topology = OurTopology;
  MUNDY_THROW_ASSERT(center_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                     "The center data must be a field of NODE_RANK");
  if constexpr (is_direction_a_field) {
    MUNDY_THROW_ASSERT(direction_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                       "The direction data must have the same rank as the line rank.");
  }
  return LineData<Scalar, OurTopology, CenterDataType, DirectionDataType>{bulk_data, center_data, direction_data};
}

/// \brief A helper function to create a NgpLineData object
/// See the discussion for create_line_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename CenterDataType,                // deduced
          typename DirectionDataType>             // deduced
auto create_ngp_line_data(stk::mesh::NgpMesh ngp_mesh, CenterDataType& center_data, DirectionDataType& direction_data) {
  constexpr bool is_direction_a_field = std::is_same_v<std::decay_t<DirectionDataType>, stk::mesh::NgpField<Scalar>>;
  stk::topology our_topology = OurTopology;
  MUNDY_THROW_ASSERT(center_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                     "The center data must be a field of NODE_RANK");
  if constexpr (is_direction_a_field) {
    MUNDY_THROW_ASSERT(direction_data.get_rank() == our_topology.rank(), std::invalid_argument,
                       "The direction data must have the same rank as the line rank.");
  }
  return NgpLineData<Scalar, OurTopology, CenterDataType, DirectionDataType>{ngp_mesh, center_data, direction_data};
}

/// \brief A concept to check if a type provides the same data as LineData
template <typename Agg>
concept ValidDefaultLineDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::center_data_t;
  typename Agg::direction_data_t;
  std::is_same_v<std::decay_t<typename Agg::center_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  mundy::math::is_vector3_v<std::decay_t<typename Agg::direction_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::direction_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { Agg::topology } -> std::convertible_to<stk::topology::topology_t>;
  { agg.bulk_data } -> std::convertible_to<stk::mesh::BulkData&>;
  { agg.center_data } -> std::convertible_to<typename Agg::center_data_t&>;
  { agg.direction_data } -> std::convertible_to<typename Agg::direction_data_t&>;
};  // ValidDefaultLineDataType

/// \brief A concept to check if a type provides the same data as NgpLineData
template <typename Agg>
concept ValidDefaultNgpLineDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::center_data_t;
  typename Agg::direction_data_t;
  std::is_same_v<std::decay_t<typename Agg::center_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  mundy::math::is_vector3_v<std::decay_t<typename Agg::direction_data_t>> ||
      std::is_same_v<std::decay_t<typename Agg::direction_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { Agg::topology } -> std::convertible_to<stk::topology::topology_t>;
  { agg.ngp_mesh } -> std::convertible_to<stk::mesh::NgpMesh>;
  { agg.center_data } -> std::convertible_to<typename Agg::center_data_t&>;
  { agg.direction_data } -> std::convertible_to<typename Agg::direction_data_t&>;
};  // ValidDefaultNgpLineDataType

static_assert(ValidDefaultLineDataType<LineData<float,                    //
                                                stk::topology::NODE,      //
                                                stk::mesh::Field<float>,  //
                                                Point<float>>> &&
                  ValidDefaultLineDataType<LineData<float,                    //
                                                    stk::topology::PARTICLE,  //
                                                    stk::mesh::Field<float>,  //
                                                    stk::mesh::Field<float>>>,
              "LineData must satisfy the ValidDefaultLineDataType concept");

static_assert(ValidDefaultNgpLineDataType<NgpLineData<float,                       //
                                                      stk::topology::NODE,         //
                                                      stk::mesh::NgpField<float>,  //
                                                      Point<float>>> &&
                  ValidDefaultNgpLineDataType<NgpLineData<float,                       //
                                                          stk::topology::PARTICLE,     //
                                                          stk::mesh::NgpField<float>,  //
                                                          stk::mesh::NgpField<float>>>,
              "NgpLineData must satisfy the ValidDefaultNgpLineDataType concept");

/// \brief A helper function to get an updated NgpLineData object from a LineData object
/// \param data The LineData object to convert
template <ValidDefaultLineDataType LineDataType>
auto get_updated_ngp_data(LineDataType data) {
  using scalar_t = typename LineDataType::scalar_t;
  using center_data_t = typename LineDataType::center_data_t;
  using direction_data_t = typename LineDataType::direction_data_t;

  constexpr bool is_direction_a_field = std::is_same_v<std::decay_t<direction_data_t>, stk::mesh::Field<scalar_t>>;
  constexpr stk::topology::topology_t our_topology = LineDataType::topology;
  if constexpr (is_direction_a_field) {
    return create_ngp_line_data<scalar_t, our_topology>(               //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data),               //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data),  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.direction_data));
  } else {
    return create_ngp_line_data<scalar_t, our_topology>(               //
        stk::mesh::get_updated_ngp_mesh(data.bulk_data),               //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data),  //
        data.direction_data);
  }
}

/// \brief A traits class to provide abstracted access to a NODE_RANK line's data via an aggregate
///
/// By default, this class is compatible with LineData or any class the meets the ValidDefaultLineDataType concept.
/// Users can specialize this class to support other aggregate types.
template <typename Agg>
struct NodeLineDataTraits {
  static_assert(
      ValidDefaultLineDataType<Agg>,
      "Agg must satisfy the ValidDefaultLineDataType concept.\n"
      "Basically, Agg must have all the same things as NgpLineData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;
  using direction_data_t = typename Agg::direction_data_t;

  static constexpr bool has_shared_direction() {
    return mundy::math::is_vector3_v<direction_data_t>;
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity line_node) {
    return mundy::mesh::vector3_field_data(agg.center_data, line_node);
  }

  static decltype(auto) direction(Agg agg, stk::mesh::Entity line_node) {
    if constexpr (has_shared_direction()) {
      return agg.direction_data;
    } else {
      return mundy::mesh::vector3_field_data(agg.direction_data, line_node);
    }
  }
};  // NodeLineDataTraits

/// \brief A traits class to provide abstracted access to a line's data via an aggregate
template <typename Agg>
struct LineDataTraits {
  static_assert(
      ValidDefaultLineDataType<Agg>,
      "Agg must satisfy the ValidDefaultLineDataType concept.\n"
      "Basically, Agg must have all the same things as NgpLineData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;
  using direction_data_t = typename Agg::direction_data_t;
  static constexpr stk::topology::topology_t topology = Agg::topology;

  static constexpr bool has_shared_direction() {
    return mundy::math::is_vector3_v<direction_data_t>;
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity line_node) {
    return mundy::mesh::vector3_field_data(agg.center_data, line_node);
  }

  static decltype(auto) direction(Agg agg, stk::mesh::Entity line_entity) {
    if constexpr (has_shared_direction()) {
      return agg.direction_data;
    } else {
      return mundy::mesh::vector3_field_data(agg.direction_data, line_entity);
    }
  }
};  // LineDataTraits

/// \brief A traits class to provide abstracted access to a line's data via an NGP-compatible aggregate
/// See the discussion for LineDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpLineDataTraits {
  static_assert(
      ValidDefaultNgpLineDataType<Agg>,
      "Agg must satisfy the ValidDefaultNgpLineDataType concept.\n"
      "Basically, Agg must have all the same things as NgpLineData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = typename Agg::center_data_t;
  using direction_data_t = typename Agg::direction_data_t;
  static constexpr stk::topology::topology_t topology = Agg::topology;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_direction() {
    return mundy::math::is_vector3_v<direction_data_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex line_index) {
    return mundy::mesh::vector3_field_data(agg.center_data, line_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) direction(Agg agg, stk::mesh::FastMeshIndex line_index) {
    if constexpr (has_shared_direction()) {
      return agg.direction_data;
    } else {
      return mundy::mesh::vector3_field_data(agg.direction_data, line_index);
    }
  }
};  // NgpLineDataTraits

/// @brief A view of an STK entity meant to represent a line
/// We type specialize this class based on the valid set of topologies for an line entity.
///
/// Use \ref create_line_data to build a LineData object with automatic template deduction.
template <stk::topology::topology_t OurTopology, typename LineDataType>
class LineEntityView;

/// @brief A view of a NODE STK entity meant to represent a line
template <typename LineDataType>
class LineEntityView<stk::topology::NODE, LineDataType> {
  static_assert(LineDataType::topology == stk::topology::NODE, "The topology of the line data must match the view");

 public:
  using data_access_t = LineDataTraits<LineDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<LineDataType>(), std::declval<stk::mesh::Entity>()));
  using direction_t =
      decltype(data_access_t::direction(std::declval<LineDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  LineEntityView(LineDataType data, stk::mesh::Entity line) : data_(data), line_(line) {
    MUNDY_THROW_ASSERT(data_.bulk_data.entity_rank(line_) == stk::topology::NODE_RANK, std::invalid_argument,
                       "The line entity rank must be NODE_RANK");
    MUNDY_THROW_ASSERT(data_.bulk_data.is_valid(line_), std::invalid_argument, "The given line entity is not valid");
  }

  decltype(auto) center() {
    return data_access_t::center(data_, line_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, line_);
  }

  decltype(auto) direction() {
    return data_access_t::direction(data_, line_);
  }

  decltype(auto) direction() const {
    return data_access_t::direction(data_, line_);
  }

 private:
  LineDataType data_;
  stk::mesh::Entity line_;
};  // LineEntityView<stk::topology::NODE, LineDataType>

/// @brief A view of a PARTICLE STK entity meant to represent a line
template <typename LineDataType>
class LineEntityView<stk::topology::PARTICLE, LineDataType> {
  static_assert(LineDataType::topology == stk::topology::PARTICLE, "The topology of the line data must match the view");

 public:
  using data_access_t = LineDataTraits<LineDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<LineDataType>(), std::declval<stk::mesh::Entity>()));
  using direction_t =
      decltype(data_access_t::direction(std::declval<LineDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  LineEntityView(LineDataType data, stk::mesh::Entity line)
      : data_(data), line_(line), node_(data_.bulk_data.begin_nodes(line_)[0]) {
    MUNDY_THROW_ASSERT(data_.bulk_data.entity_rank(line_) == stk::topology::ELEM_RANK, std::invalid_argument,
                       "The line entity rank must be ELEM_RANK");
    MUNDY_THROW_ASSERT(data_.bulk_data.is_valid(line_), std::invalid_argument, "The given line entity is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data.num_nodes(line_) >= 1, std::invalid_argument,
                       "The given line entity must have at least one node");
    MUNDY_THROW_ASSERT(data_.bulk_data.is_valid(node_), std::invalid_argument,
                       "The node entity associated with the line is not valid");
  }

  decltype(auto) center() {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) center() const {
    return data_access_t::center(data_, node_);
  }

  decltype(auto) direction() {
    return data_access_t::direction(data_, line_);
  }

  decltype(auto) direction() const {
    return data_access_t::direction(data_, line_);
  }

 private:
  LineDataType data_;
  stk::mesh::Entity line_;
  stk::mesh::Entity node_;
};  // LineEntityView<stk::topology::PARTICLE, LineDataType>

/// @brief An ngp-compatible view of a line entity
/// See the discussion for LineEntityView for more information. The only difference is ngp-compatible data access.
template <stk::topology::topology_t OurTopology, typename NgpLineDataType>
class NgpLineEntityView;

/// @brief An ngp-compatible view of a NODE STK entity meant to represent a line
template <typename NgpLineDataType>
class NgpLineEntityView<stk::topology::NODE, NgpLineDataType> {
  static_assert(NgpLineDataType::topology == stk::topology::NODE, "The topology of the line data must match the view");

 public:
  using data_access_t = NgpLineDataTraits<NgpLineDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<NgpLineDataType>(), std::declval<stk::mesh::FastMeshIndex>()));
  using direction_t =
      decltype(data_access_t::direction(std::declval<NgpLineDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpLineEntityView(NgpLineDataType data, stk::mesh::FastMeshIndex line_index) : data_(data), line_index_(line_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data_, line_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data_, line_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) direction() {
    return data_access_t::direction(data_, line_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) direction() const {
    return data_access_t::direction(data_, line_index_);
  }

 private:
  NgpLineDataType data_;
  stk::mesh::FastMeshIndex line_index_;
};  // NgpLineEntityView<stk::topology::NODE, NgpLineDataType>

/// @brief An ngp-compatible view of a PARTICLE STK entity meant to represent a line
template <typename NgpLineDataType>
class NgpLineEntityView<stk::topology::PARTICLE, NgpLineDataType> {
  static_assert(NgpLineDataType::topology == stk::topology::PARTICLE,
                "The topology of the line data must match the view");

 public:
  using data_access_t = NgpLineDataTraits<NgpLineDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<NgpLineDataType>(), std::declval<stk::mesh::FastMeshIndex>()));
  using direction_t =
      decltype(data_access_t::direction(std::declval<NgpLineDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpLineEntityView(NgpLineDataType data, stk::mesh::FastMeshIndex line_index)
      : data_(data),
        line_index_(line_index),
        node_index_(data_.ngp_mesh.fast_mesh_index(data_.ngp_mesh.get_nodes(rank, line_index_)[0])) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data_, node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data_, node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) direction() {
    return data_access_t::direction(data_, line_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) direction() const {
    return data_access_t::direction(data_, line_index_);
  }

 private:
  NgpLineDataType data_;
  stk::mesh::FastMeshIndex line_index_;
  stk::mesh::FastMeshIndex node_index_;
};  // NgpLineEntityView<stk::topology::PARTICLE, NgpLineDataType>

static_assert(ValidLineType<LineEntityView<stk::topology::NODE,
                                           LineData<float,                    //
                                                    stk::topology::NODE,      //
                                                    stk::mesh::Field<float>,  //
                                                    mundy::math::Vector3<float>>>> &&
                  ValidLineType<LineEntityView<stk::topology::PARTICLE,
                                               LineData<float,                    //
                                                        stk::topology::PARTICLE,  //
                                                        stk::mesh::Field<float>,  //
                                                        stk::mesh::Field<float>>>> &&
                  ValidLineType<NgpLineEntityView<stk::topology::NODE,
                                                  NgpLineData<float,                       //
                                                              stk::topology::NODE,         //
                                                              stk::mesh::NgpField<float>,  //
                                                              mundy::math::Vector3<float>>>> &&
                  ValidLineType<NgpLineEntityView<stk::topology::PARTICLE,
                                                  NgpLineData<float,                       //
                                                              stk::topology::PARTICLE,     //
                                                              stk::mesh::NgpField<float>,  //
                                                              stk::mesh::NgpField<float>>>>,
              "LineEntityView and NgpLineEntityView must be valid Line types");

/// \brief A helper function to create a LineEntityView object with type deduction
template <typename LineDataType>                  // deduced
auto create_line_entity_view(LineDataType& data, stk::mesh::Entity line) {
  return LineEntityView<LineDataType::topology, LineDataType>(data, line);
}

/// \brief A helper function to create a NgpLineEntityView object with type deduction
template <typename NgpLineDataType>               // deduced
auto create_ngp_line_entity_view(NgpLineDataType data, stk::mesh::FastMeshIndex line_index) {
  return NgpLineEntityView<NgpLineDataType::topology, NgpLineDataType>(data, line_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_LINE_HPP_