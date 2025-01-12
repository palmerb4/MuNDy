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

#ifndef MUNDY_GEOM_AGGREGATES_ELLIPSOIDDATA_HPP_
#define MUNDY_GEOM_AGGREGATES_ELLIPSOIDDATA_HPP_

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
#include <mundy_geom/aggregates/EllipsoidDataConcepts.hpp>  // for mundy::geom::ValidEllipsoidDataType
#include <mundy_geom/aggregates/EllipsoidEntityView.hpp>    // for mundy::geom::EllipsoidEntityView
#include <mundy_geom/aggregates/EntityView.hpp>  // for mundy::geom::EntityView and mundy::geom::create_topological_entity_view
#include <mundy_geom/primitives/Ellipsoid.hpp>  // for mundy::geom::ValidEllipsoidType
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of ellipsoids
///
/// The topology of an ellipsoid directly effects the access pattern for the underlying data:
///   - NODE: All data is stored on a single node
///   - PARTICLE: The center is stored on a node, whereas the orientation and axis lengths are stored on the
///   element-rank particle
///
/// Shared data is stored as a const ref to the original data and is therefore unmodifiable. This is to prevent
/// accidental non-thread-safe modifications to the shared data. If this is every limiting, let us know and we can
/// consider adding a Kokkos::View to the shared data.
///
/// Use \ref create_ellipsoid_data to build an EllipsoidData object with automatic template deduction.
template <typename Scalar,       //
          typename OurTopology,  //
          typename HasSharedAxisLengths = std::false_type>
class EllipsoidData {
  static_assert(OurTopology::value == stk::topology::NODE || OurTopology::value == stk::topology::PARTICLE,
                "The topology of an ellipsoid must be either NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using center_data_t = stk::mesh::Field<scalar_t>;
  using orientation_data_t = stk::mesh::Field<Scalar>;
  using axis_lengths_data_t = std::conditional_t<HasSharedAxisLengths::value,  //
                                                 mundy::math::Vector3<scalar_t>, stk::mesh::Field<scalar_t>>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  EllipsoidData(const stk::mesh::BulkData& bulk_data, const center_data_t& center_data,
                const orientation_data_t& orientation_data, const axis_lengths_data_t& axis_lengths_data)
      : bulk_data_(bulk_data),
        center_data_(center_data),
        orientation_data_(orientation_data),
        axis_lengths_data_(axis_lengths_data) {
    stk::topology our_topology = topology_t;
    MUNDY_THROW_ASSERT(center_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center_data data must be a field of NODE_RANK");
    MUNDY_THROW_ASSERT(orientation_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                       "The orientation data must be a field of the same rank as the ellipsoid");
    if constexpr (!HasSharedAxisLengths::value) {
      MUNDY_THROW_ASSERT(axis_lengths_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                         "The axis lengths data must be a field of the same rank as the ellipsoid");
    }
  }

  static constexpr stk::topology::topology_t get_topology() {
    return topology_t;
  }

  static constexpr stk::topology::topology_t get_rank() {
    return stk::topology_detail::topology_data<OurTopology::value>::rank();
  }

  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }

  const center_data_t& center_data() const {
    return center_data_;
  }

  const orientation_data_t& orientation_data() const {
    return orientation_data_;
  }

  const axis_lengths_data_t& axis_lengths_data() const {
    return axis_lengths_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto augment_data(Args&&... args) const {
    return NextAugment<EllipsoidData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  auto get_entity_view(stk::mesh::Entity entity) {
    // We ~are~ the top of the chain, so we need to create the topological entity view that starts it all
    //
    // Recursively calls get_entity_view on the next aggregate up in the chain, traversing to the very top of the chain
    // before then adding each augment in the chain to the entity view from the top down.
    using our_t = EllipsoidData<Scalar, OurTopology, HasSharedAxisLengths>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<EllipsoidEntityView, our_t>(*this);
  }

  const auto get_entity_view(stk::mesh::Entity entity) const {
    using our_t = EllipsoidData<Scalar, OurTopology, HasSharedAxisLengths>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<EllipsoidEntityView, our_t>(*this);
  }

  auto get_updated_ngp_data() const {
    if constexpr (!HasSharedAxisLengths::value) {
      return create_ngp_ellipsoid_data<scalar_t, topology_t>(
          stk::mesh::get_updated_ngp_mesh(bulk_data_),                            //
          stk::mesh::get_updated_ngp_field<scalar_t>(center_data_),       //
          stk::mesh::get_updated_ngp_field<scalar_t>(orientation_data_),  //
          stk::mesh::get_updated_ngp_field<scalar_t>(axis_lengths_data_));
    } else {
      return create_ngp_ellipsoid_data<scalar_t, topology_t>(
          stk::mesh::get_updated_ngp_mesh(bulk_data_),                            //
          stk::mesh::get_updated_ngp_field<scalar_t>(center_data_),       //
          stk::mesh::get_updated_ngp_field<scalar_t>(orientation_data_),  //
          axis_lengths_data_);
    }
  }

 private:
  const stk::mesh::BulkData& bulk_data_;
  const center_data_t& center_data_;
  const orientation_data_t& orientation_data_;
  const axis_lengths_data_t& axis_lengths_data_;
};  // EllipsoidData

/// \brief Aggregate to hold the data for a collection of NGP-compatible ellipsoids
/// See the discussion for EllipsoidData for more information. Only difference is NgpFields over Fields.
/// 
/// One additional difference is that, we cannot store a reference to host memory, so we store a const copy of any shared data.
template <typename Scalar,       //
          typename OurTopology,  //
          typename HasSharedAxisLengths = std::false_type>
class NgpEllipsoidData {
  static_assert(OurTopology::value == stk::topology::NODE || OurTopology::value == stk::topology::PARTICLE,
                "The topology of an ellipsoid must be either NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using center_data_t = stk::mesh::NgpField<Scalar>;
  using orientation_data_t = stk::mesh::NgpField<Scalar>;
  using axis_lengths_data_t =
      std::conditional_t<HasSharedAxisLengths::value, mundy::math::Vector3<Scalar>, stk::mesh::NgpField<Scalar>>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  NgpEllipsoidData(const stk::mesh::NgpMesh &ngp_mesh, const center_data_t& center_data,
                   const orientation_data_t& orientation_data, const axis_lengths_data_t& axis_lengths_data)
      : ngp_mesh_(ngp_mesh),
        center_data_(center_data),
        orientation_data_(orientation_data),
        axis_lengths_data_(axis_lengths_data) {
    stk::topology our_topology = topology_t;
    MUNDY_THROW_ASSERT(center_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center_data data must be a field of NODE_RANK");
    MUNDY_THROW_ASSERT(orientation_data.get_rank() == our_topology.rank(), std::invalid_argument,
                       "The orientation data must be a field of the same rank as the ellipsoid");
    if constexpr (!HasSharedAxisLengths::value) {
      MUNDY_THROW_ASSERT(axis_lengths_data.get_rank() == our_topology.rank(), std::invalid_argument,
                         "The axis lengths data must be a field of the same rank as the ellipsoid");
    }
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t get_topology() {
    return topology_t;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t get_rank() {
    return stk::topology_detail::topology_data<OurTopology::value>::rank();
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
  const orientation_data_t& orientation_data() const {
    return orientation_data_;
  }

  KOKKOS_INLINE_FUNCTION
  orientation_data_t& orientation_data() {
    return orientation_data_;
  }

  KOKKOS_INLINE_FUNCTION
  const axis_lengths_data_t& axis_lengths_data() const {
    return axis_lengths_data_;
  }

  KOKKOS_INLINE_FUNCTION
  axis_lengths_data_t& axis_lengths_data() {
    return axis_lengths_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION auto augment_data(Args&&... args) const {
    return NextAugment<NgpEllipsoidData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  auto get_entity_view(stk::mesh::FastMeshIndex entity_index) {
    using our_t = NgpEllipsoidData<Scalar, OurTopology, HasSharedAxisLengths>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpEllipsoidEntityView, our_t>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  const auto get_entity_view(stk::mesh::FastMeshIndex entity_index) const {
    using our_t = NgpEllipsoidData<Scalar, OurTopology, HasSharedAxisLengths>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpEllipsoidEntityView, our_t>(*this);
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  center_data_t center_data_;
  orientation_data_t orientation_data_;
  axis_lengths_data_t axis_lengths_data_;
};  // NgpEllipsoidData

/// \brief A helper function to create an EllipsoidData object
///
/// This function creates a EllipsoidData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename AxisLengthsDataType>           // deduced
auto create_ellipsoid_data(const stk::mesh::BulkData& bulk_data, const stk::mesh::Field<Scalar>& center_data,
                           const stk::mesh::Field<Scalar>& orientation_data, const AxisLengthsDataType& axis_lengths_data) {
  constexpr bool is_axis_lengths_shared = mundy::math::is_vector3_v<AxisLengthsDataType>;
  if constexpr (is_axis_lengths_shared) {
    return EllipsoidData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type>{
        bulk_data, center_data, orientation_data, axis_lengths_data};
  } else {
    return EllipsoidData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type>{
        bulk_data, center_data, orientation_data, axis_lengths_data};
  }
}

/// \brief A helper function to create an NgpEllipsoidData object
/// See the discussion for create_ellipsoid_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename AxisLengthsDataType>           // deduced
auto create_ngp_ellipsoid_data(const stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::NgpField<Scalar>& center_data,
                               const stk::mesh::NgpField<Scalar>& orientation_data, const AxisLengthsDataType& axis_lengths_data) {
  constexpr bool is_axis_lengths_shared = mundy::math::is_vector3_v<AxisLengthsDataType>;
  if constexpr (is_axis_lengths_shared) {
    return NgpEllipsoidData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type>{
        ngp_mesh, center_data, orientation_data, axis_lengths_data};
  } else {
    return NgpEllipsoidData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type>{
        ngp_mesh, center_data, orientation_data, axis_lengths_data};
  }
}

/// \brief A helper function to get an updated NgpEllipsoidData object from a EllipsoidData object
/// \param data The EllipsoidData object to convert
template <typename Scalar, typename OurTopology, typename HasSharedAxisLengths>
auto get_updated_ngp_data(const EllipsoidData<Scalar, OurTopology, HasSharedAxisLengths>& data) {
  return data.get_updated_ngp_data();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_ELLIPSOIDDATA_HPP_