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

#ifndef MUNDY_GEOM_AGGREGATES_LINEDATA_HPP_
#define MUNDY_GEOM_AGGREGATES_LINEDATA_HPP_

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
#include <mundy_geom/primitives/Line.hpp>  // for mundy::geom::ValidLineType
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_geom/aggregates/LineDataConcepts.hpp>  // for mundy::geom::ValidLineDataType
#include <mundy_geom/aggregates/LineEntityView.hpp>  // for mundy::geom::LineEntityView
#include <mundy_geom/aggregates/EntityView.hpp>  // for mundy::geom::EntityView and mundy::geom::create_topological_entity_view

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of infinite lines
///
/// The topology of a line directly effects the access pattern for the underlying data:
///   - NODE: All data is stored on a single node
///   - PARTICLE: The center is stored on a node, whereas the direction is stored on the element-rank particle
///
/// Use \ref create_line_data to build a LineData object with automatic template deduction.
template <typename Scalar, typename OurTopology>
class LineData {
  static_assert(OurTopology::value == stk::topology::NODE || OurTopology::value == stk::topology::PARTICLE,
                "The topology of a line must be either NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using center_data_t = stk::mesh::Field<Scalar>;
  using direction_data_t = stk::mesh::Field<Scalar>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  LineData(const stk::mesh::BulkData& bulk_data, const center_data_t& center_data, const direction_data_t& direction_data)
      : bulk_data_(bulk_data), center_data_(center_data), direction_data_(direction_data) {
    stk::topology our_topology = topology_t;
    MUNDY_THROW_ASSERT(center_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center data must be a field of NODE_RANK");
    MUNDY_THROW_ASSERT(direction_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                       "The direction data must have the same rank as the line rank.");
  }

  static constexpr stk::topology::topology_t get_topology() {
    return topology_t;
  }

  static constexpr stk::topology::rank_t get_rank() {
    return stk::topology_detail::topology_data<OurTopology::value>::rank;
  }

  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }


  const center_data_t& center_data() const {
    return center_data_;
  }

  const direction_data_t& direction_data() const {
    return direction_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<LineData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  auto get_entity_view(stk::mesh::Entity entity) {
    using our_t = LineData<Scalar, OurTopology>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<LineEntityView, our_t>(*this);
  }

  const auto get_entity_view(stk::mesh::Entity entity) const {
    using our_t = LineData<Scalar, OurTopology>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<LineEntityView, our_t>(*this);
  }

  auto get_updated_ngp_data() const {
    return create_ngp_line_data<scalar_t, topology_t>(                   //
      stk::mesh::get_updated_ngp_mesh(bulk_data_),               //
        stk::mesh::get_updated_ngp_field<scalar_t>(center_data_),  //
        stk::mesh::get_updated_ngp_field<scalar_t>(direction_data_));
  } 

 private:
  const stk::mesh::BulkData& bulk_data_;
  const center_data_t& center_data_;
  const direction_data_t& direction_data_;
};  // LineData

/// \brief Aggregate to hold the data for a collection of NGP-compatible lines
/// See the discussion for LineData for more information. Only difference is NgpFields over Fields.
template <typename Scalar, typename OurTopology>
class NgpLineData {
  static_assert(OurTopology::value == stk::topology::NODE || OurTopology::value == stk::topology::PARTICLE,
                "The topology of a line must be either NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using center_data_t = stk::mesh::NgpField<Scalar>;
  using direction_data_t = stk::mesh::NgpField<Scalar>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  NgpLineData(const stk::mesh::NgpMesh &ngp_mesh, const center_data_t& center_data, const direction_data_t& direction_data)
      : ngp_mesh_(ngp_mesh), center_data_(center_data), direction_data_(direction_data) {
    stk::topology our_topology = topology_t;
    MUNDY_THROW_ASSERT(center_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center data must be a field of NODE_RANK");
    MUNDY_THROW_ASSERT(direction_data.get_rank() == our_topology.rank(), std::invalid_argument,
                       "The direction data must have the same rank as the line rank.");
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t get_topology() {
    return topology_t;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::rank_t get_rank() {
    return stk::topology_detail::topology_data<OurTopology::value>::rank;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh& ngp_mesh() {
    return ngp_mesh_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh& ngp_mesh() const {
    return ngp_mesh_;
  }

  KOKKOS_INLINE_FUNCTION
  const center_data_t& center_data() const {
    return center_data_;
  }

  KOKKOS_INLINE_FUNCTION
  center_data_t& center_data() {
    return center_data_;
  }

  KOKKOS_INLINE_FUNCTION
  const direction_data_t& direction_data() const {
    return direction_data_;
  }

  KOKKOS_INLINE_FUNCTION
  direction_data_t& direction_data() {
    return direction_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION
  auto add_augment(Args&&... args) const {
    return NextAugment<NgpLineData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  auto get_entity_view(stk::mesh::FastMeshIndex entity_index) {
    using our_t = NgpLineData<Scalar, OurTopology>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpLineEntityView, our_t>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  const auto get_entity_view(stk::mesh::FastMeshIndex entity_index) const {
    using our_t = NgpLineData<Scalar, OurTopology>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpLineEntityView, our_t>(*this);
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  center_data_t center_data_;
  direction_data_t direction_data_;
};  // NgpLineData

/// \brief A helper function to create a LineData object
///
/// This function creates a LineData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology>  // Must be provided
auto create_line_data(const stk::mesh::BulkData& bulk_data, const stk::mesh::Field<Scalar>& center_data,
                      const stk::mesh::Field<Scalar>& direction_data) {
  return LineData<Scalar, stk::topology_detail::topology_data<OurTopology>>{bulk_data, center_data, direction_data};
}

/// \brief A helper function to create a NgpLineData object
/// See the discussion for create_line_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology>  // Must be provided
auto create_ngp_line_data(const stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::NgpField<Scalar>& center_data,
                          const stk::mesh::NgpField<Scalar>& direction_data) {
  return NgpLineData<Scalar, stk::topology_detail::topology_data<OurTopology>>{ngp_mesh, center_data, direction_data};
}

/// \brief A helper function to get an updated NgpLineData object from a LineData object
/// \param data The LineData object to convert
template <typename Scalar, typename OurTopology>
auto get_updated_ngp_data(const LineData<Scalar, OurTopology>& data) {
  return data.get_updated_ngp_data();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_LINEDATA_HPP_