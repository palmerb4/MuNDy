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
#include <mundy_geom/primitives/LineSegment.hpp>  // for mundy::geom::ValidLineSegmentType
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>        // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief A struct to hold the data for a collection of line segments
template <typename Scalar, typename CenterDataType = stk::mesh::Field<Scalar>>
struct LineSegmentData {
  static_assert(std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::Field<Scalar>>,
                "CenterDataType must be either a const or non-const field of scalars");

  using scalar_t = Scalar;
  using node_coords_data_t = CenterDataType;

  stk::topology::rank_t line_rank;
  node_coords_data_t& node_coords_data;
};  // LineSegmentData

/// \brief A struct to hold the data for a collection of NGP-compatible line segments
/// See the discussion for LineSegmentData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename CenterDataType = stk::mesh::NgpField<Scalar>>
struct NgpLineSegmentData {
  static_assert(std::is_same_v<std::decay_t<CenterDataType>, stk::mesh::NgpField<Scalar>>,
                "CenterDataType must be either a const or non-const field of scalars");

  using scalar_t = Scalar;
  using node_coords_data_t = CenterDataType;

  stk::topology::rank_t line_rank;
  node_coords_data_t& node_coords_data;
};  // NgpLineSegmentData

/// \brief A helper function to create a LineSegmentData object
///
/// This function creates a LineSegmentData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar, typename CenterDataType>
auto create_line_data(stk::topology::rank_t line_rank, CenterDataType& node_coords_data) {
  MUNDY_THROW_ASSERT(node_coords_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                      "The node_coords data must be a field of NODE_RANK");
  return LineSegmentData<Scalar, CenterDataType>{line_rank, node_coords_data};
}

/// \brief A helper function to create a NgpLineSegmentData object
/// See the discussion for create_line_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename CenterDataType>
auto create_ngp_line_data(stk::topology::rank_t line_rank, CenterDataType& node_coords_data) {
  MUNDY_THROW_ASSERT(node_coords_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                      "The node_coords data must be a field of NODE_RANK");
  return NgpLineSegmentData<Scalar, CenterDataType>{line_rank, node_coords_data};
}

/// \brief A concept to check if a type provides the same data as LineSegmentData
template <typename Agg>
concept ValidDefaultLineSegmentDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::node_coords_data_t;
  std::is_same_v<std::decay_t<typename Agg::node_coords_data_t>, stk::mesh::Field<typename Agg::scalar_t>>;
  { agg.line_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.node_coords_data } -> std::convertible_to<typename Agg::node_coords_data_t&>;
};  // ValidDefaultLineSegmentDataType

/// \brief A concept to check if a type provides the same data as NgpLineSegmentData
template <typename Agg>
concept ValidDefaultNgpLineSegmentDataType = requires(Agg agg) {
  typename Agg::scalar_t;
  typename Agg::node_coords_data_t;
  std::is_same_v<std::decay_t<typename Agg::node_coords_data_t>, stk::mesh::NgpField<typename Agg::scalar_t>>;
  { agg.line_rank } -> std::convertible_to<stk::topology::rank_t>;
  { agg.node_coords_data } -> std::convertible_to<typename Agg::node_coords_data_t&>;
};  // ValidDefaultNgpLineSegmentDataType

static_assert(ValidDefaultLineSegmentDataType<LineSegmentData<float, stk::mesh::Field<float>>> &&
                  ValidDefaultLineSegmentDataType<LineSegmentData<float, const stk::mesh::Field<float>>>,
              "LineSegmentData must satisfy the ValidDefaultLineSegmentDataType concept");

static_assert(ValidDefaultNgpLineSegmentDataType<NgpLineSegmentData<float, stk::mesh::NgpField<float>>> &&
                  ValidDefaultNgpLineSegmentDataType<NgpLineSegmentData<float, const stk::mesh::NgpField<float>>>,
              "NgpLineSegmentData must satisfy the ValidDefaultNgpLineSegmentDataType concept");

/// \brief A helper function to get an updated NgpLineSegmentData object from a LineSegmentData object
/// \param data The LineSegmentData object to convert
template <ValidDefaultLineSegmentDataType LineSegmentDataType>
auto get_updated_ngp_data(LineSegmentDataType data) {
  using scalar_t = typename LineSegmentDataType::scalar_t;
  using node_coords_data_t = typename LineSegmentDataType::node_coords_data_t;
  return create_ngp_line_data<scalar_t>(data.line_rank,  //
                                        stk::mesh::get_updated_ngp_field<scalar_t>(data.node_coords_data));
}

/// \brief A traits class to provide abstracted access to a NODE_RANK line's data via an aggregate
///
/// By default, this class is compatible with LineSegmentData or any class the meets the ValidDefaultLineSegmentDataType concept.
/// Users can specialize this class to support other aggregate types.
template <typename Agg>
struct NodeLineSegmentDataTraits {
  static_assert(
      ValidDefaultLineSegmentDataType<Agg>,
      "Agg must satisfy the ValidDefaultLineSegmentDataType concept.\n"
      "Basically, Agg must have all the same things as NgpLineSegmentData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = typename Agg::node_coords_data_t;

  static decltype(auto) node_coords(Agg agg, stk::mesh::Entity line_node) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data, line_node);
  }
};  // NodeLineSegmentDataTraits

/// \brief A traits class to provide abstracted access to a line's data via an aggregate
template <typename Agg>
struct LineSegmentDataTraits {
  static_assert(
      ValidDefaultLineSegmentDataType<Agg>,
      "Agg must satisfy the ValidDefaultLineSegmentDataType concept.\n"
      "Basically, Agg must have all the same things as NgpLineSegmentData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = typename Agg::node_coords_data_t;

  static decltype(auto) node_coords(Agg agg, stk::mesh::Entity line_node) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data, line_node);
  }
};  // LineSegmentDataTraits

/// \brief A traits class to provide abstracted access to a line's data via an NGP-compatible aggregate
/// See the discussion for LineSegmentDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpLineSegmentDataTraits {
  static_assert(
      ValidDefaultNgpLineSegmentDataType<Agg>,
      "Agg must satisfy the ValidDefaultNgpLineSegmentDataType concept.\n"
      "Basically, Agg must have all the same things as NgpLineSegmentData but is free to extend it as needed without "
      "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using node_coords_data_t = typename Agg::node_coords_data_t;

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) node_coords(Agg agg, stk::mesh::FastMeshIndex line_index) {
    return mundy::mesh::vector3_field_data(agg.node_coords_data, line_index);
  }
};  // NgpLineSegmentDataTraits

/// @brief A view of an STK entity meant to represent a line
/// If the line_rank is NODE_RANK, then the line is just a node entity with node node_coords and direction.
/// If the line_rank is ELEM_RANK, then the line is a particle entity with node node_coords and element direction.
template <typename LineSegmentDataType>
class ElemLineSegmentView {
 public:
  using data_access_t = LineSegmentDataTraits<LineSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<LineSegmentDataType>(), std::declval<stk::mesh::Entity>()));

  ElemLineSegmentView(const stk::mesh::BulkData& bulk_data, LineSegmentDataType data, stk::mesh::Entity line)
      : data_(data), line_(line), node_(bulk_data.begin_nodes(line_)[0]) {
    MUNDY_THROW_ASSERT(
        bulk_data.entity_rank(line_) == stk::topology::ELEM_RANK && data_.line_rank == stk::topology::ELEM_RANK,
        std::invalid_argument, "Both the line entity rank and the line data rank must be ELEM_RANK");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(line_), std::invalid_argument, "The given line entity is not valid");
    MUNDY_THROW_ASSERT(bulk_data.num_nodes(line_) == 1, std::invalid_argument,
                       "The given line entity must have exactly one node");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(node_), std::invalid_argument,
                       "The node entity associated with the line is not valid");
  }

  decltype(auto) node_coords() {
    return data_access_t::node_coords(data_, node_);
  }

  decltype(auto) node_coords() const {
    return data_access_t::node_coords(data_, node_);
  }

 private:
  LineSegmentDataType data_;
  stk::mesh::Entity line_;
  stk::mesh::Entity node_;
};  // ElemLineSegmentView

/// @brief An ngp-compatible view of an ELEM_RANK STK entity meant to represent a line
/// See the discussion for ElemLineSegmentView for more information. The only difference is ngp-compatible data access.
template <typename NgpLineSegmentDataType>
class NgpElemLineSegmentView {
 public:
  using data_access_t = NgpLineSegmentDataTraits<NgpLineSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<NgpLineSegmentDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpElemLineSegmentView(stk::mesh::NgpMesh ngp_mesh, NgpLineSegmentDataType data, stk::mesh::FastMeshIndex line_index)
      : data_(data),
        line_index_(line_index),
        node_index_(ngp_mesh.fast_mesh_index(ngp_mesh.get_nodes(data_.line_rank, line_index_)[0])) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) node_coords() {
    return data_access_t::node_coords(data_, node_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) node_coords() const {
    return data_access_t::node_coords(data_, node_index_);
  }

 private:
  NgpLineSegmentDataType data_;
  stk::mesh::FastMeshIndex line_index_;
  stk::mesh::FastMeshIndex node_index_;
};  // NgpElemLineSegmentView

/// @brief A view of a NODE_RANK STK entity meant to represent a line
template <typename LineSegmentDataType>
class NodeLineSegmentView {
 public:
  using data_access_t = LineSegmentDataTraits<LineSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<LineSegmentDataType>(), std::declval<stk::mesh::Entity>()));

  NodeLineSegmentView([[maybe_unused]] const stk::mesh::BulkData& bulk_data, LineSegmentDataType data, stk::mesh::Entity line)
      : data_(data), line_(line) {
    MUNDY_THROW_ASSERT(
        bulk_data.entity_rank(line_) == stk::topology::NODE_RANK && data_.line_rank == stk::topology::NODE_RANK,
        std::invalid_argument, "Both the line entity rank and the line data rank must be NODE_RANK");
    MUNDY_THROW_ASSERT(bulk_data.is_valid(line_), std::invalid_argument, "The given line entity is not valid");
  }

  decltype(auto) node_coords() {
    return data_access_t::node_coords(data_, line_);
  }

  decltype(auto) node_coords() const {
    return data_access_t::node_coords(data_, line_);
  }

 private:
  LineSegmentDataType data_;
  stk::mesh::Entity line_;
};  // NodeLineSegmentView

/// @brief An ngp-compatible view of a NODE_RANK STK entity meant to represent a line
/// See the discussion for NodeLineSegmentView for more information. The only difference is ngp-compatible data access.
template <typename NgpLineSegmentDataType>
class NgpNodeLineSegmentView {
 public:
  using data_access_t = NgpLineSegmentDataTraits<NgpLineSegmentDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::node_coords(std::declval<NgpLineSegmentDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  KOKKOS_INLINE_FUNCTION
  NgpNodeLineSegmentView([[maybe_unused]] stk::mesh::NgpMesh ngp_mesh, NgpLineSegmentDataType data,
                   stk::mesh::FastMeshIndex line_index)
      : data_(data), line_index_(line_index) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) node_coords() {
    return data_access_t::node_coords(data_, line_index_);
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) node_coords() const {
    return data_access_t::node_coords(data_, line_index_);
  }

 private:
  NgpLineSegmentDataType data_;
  stk::mesh::FastMeshIndex line_index_;
};  // NgpNodeLineSegmentView

static_assert(ValidLineSegmentType<ElemLineSegmentView<LineSegmentData<float, stk::mesh::Field<float>>>> &&
              ValidLineSegmentType<ElemLineSegmentView<LineSegmentData<float, const stk::mesh::Field<float>>>> &&
              ValidLineSegmentType<NgpElemLineSegmentView<NgpLineSegmentData<float, stk::mesh::NgpField<float>>>> &&
              ValidLineSegmentType<NgpElemLineSegmentView<NgpLineSegmentData<float, const stk::mesh::NgpField<float>>>>,
              "ElemLineSegmentView and NgpElemLineSegmentView must be valid LineSegment types");

static_assert(ValidLineSegmentType<NodeLineSegmentView<LineSegmentData<float, stk::mesh::Field<float>>>> &&
              ValidLineSegmentType<NodeLineSegmentView<LineSegmentData<float, const stk::mesh::Field<float>>>>&&
              ValidLineSegmentType<NgpNodeLineSegmentView<NgpLineSegmentData<float, stk::mesh::NgpField<float>>>> &&
              ValidLineSegmentType<NgpNodeLineSegmentView<NgpLineSegmentData<float, const stk::mesh::NgpField<float>>>>,
              "NodeLineSegmentView and NgpNodeLineSegmentView must be valid LineSegment types");

/// \brief A helper function to create a ElemLineSegmentView object with type deduction
template <typename LineSegmentDataType>
auto create_elem_line_view(const stk::mesh::BulkData& bulk_data, LineSegmentDataType& data, stk::mesh::Entity line) {
  return ElemLineSegmentView<LineSegmentDataType>(bulk_data, data, line);
}

/// \brief A helper function to create a NgpElemLineSegmentView object with type deduction
template <typename NgpLineSegmentDataType>
auto create_ngp_elem_line_view(stk::mesh::NgpMesh ngp_mesh, NgpLineSegmentDataType data,
                                stk::mesh::FastMeshIndex line_index) {
  return NgpElemLineSegmentView<NgpLineSegmentDataType>(ngp_mesh, data, line_index);
}

/// \brief A helper function to create a NodeLineSegmentView object with type deduction
template <typename LineSegmentDataType>
auto create_node_line_view(const stk::mesh::BulkData& bulk_data, LineSegmentDataType& data, stk::mesh::Entity line) {
  return NodeLineSegmentView<LineSegmentDataType>(bulk_data, data, line);
}

/// \brief A helper function to create a NgpNodeLineSegmentView object with type deduction
template <typename NgpLineSegmentDataType>
auto create_ngp_node_line_view(stk::mesh::NgpMesh ngp_mesh, NgpLineSegmentDataType data,
                                stk::mesh::FastMeshIndex line_index) {
  return NgpNodeLineSegmentView<NgpLineSegmentDataType>(ngp_mesh, data, line_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_LINE_HPP_