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

#ifndef MUNDY_GEOM_AGGREGATES_ELLIPSOID_HPP_
#define MUNDY_GEOM_AGGREGATES_ELLIPSOID_HPP_

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
  EllipsoidData(stk::mesh::BulkData& bulk_data, center_data_t& center_data, orientation_data_t& orientation_data,
                axis_lengths_data_t& axis_lengths_data)
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

  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }

  stk::mesh::BulkData& bulk_data() {
    return bulk_data_;
  }

  const center_data_t& center_data() const {
    return center_data_;
  }

  center_data_t& center_data() {
    return center_data_;
  }

  const orientation_data_t& orientation_data() const {
    return orientation_data_;
  }

  orientation_data_t& orientation_data() {
    return orientation_data_;
  }

  const axis_lengths_data_t& axis_lengths_data() const {
    return axis_lengths_data_;
  }

  axis_lengths_data_t& axis_lengths_data() {
    return axis_lengths_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<EllipsoidData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  stk::mesh::BulkData& bulk_data_;
  center_data_t& center_data_;
  orientation_data_t& orientation_data_;
  axis_lengths_data_t& axis_lengths_data_;
};  // EllipsoidData

/// \brief Aggregate to hold the data for a collection of NGP-compatible ellipsoids
/// See the discussion for EllipsoidData for more information. Only difference is NgpFields over Fields.
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
  NgpEllipsoidData(stk::mesh::NgpMesh ngp_mesh, center_data_t& center_data, orientation_data_t& orientation_data,
                   axis_lengths_data_t& axis_lengths_data)
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

  stk::mesh::NgpMesh &ngp_mesh() {
    return ngp_mesh_;
  }

  const stk::mesh::NgpMesh &ngp_mesh() const {
    return ngp_mesh_;
  }

  const center_data_t& center_data() const {
    return center_data_;
  }

  center_data_t& center_data() {
    return center_data_;
  }

  const orientation_data_t& orientation_data() const {
    return orientation_data_;
  }

  orientation_data_t& orientation_data() {
    return orientation_data_;
  }

  const axis_lengths_data_t& axis_lengths_data() const {
    return axis_lengths_data_;
  }

  axis_lengths_data_t& axis_lengths_data() {
    return axis_lengths_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<NgpEllipsoidData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  center_data_t& center_data_;
  orientation_data_t& orientation_data_;
  axis_lengths_data_t& axis_lengths_data_;
};  // NgpEllipsoidData

/// \brief A helper function to create an EllipsoidData object
///
/// This function creates a EllipsoidData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename AxisLengthsDataType>           // deduced
auto create_ellipsoid_data(stk::mesh::BulkData& bulk_data, stk::mesh::Field<Scalar>& center_data,
                           stk::mesh::Field<Scalar>& orientation_data, AxisLengthsDataType& axis_lengths_data) {
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
auto create_ngp_ellipsoid_data(stk::mesh::NgpMesh ngp_mesh, stk::mesh::NgpField<Scalar>& center_data,
                               stk::mesh::NgpField<Scalar>& orientation_data, AxisLengthsDataType& axis_lengths_data) {
  constexpr bool is_axis_lengths_shared = mundy::math::is_vector3_v<AxisLengthsDataType>;
  if constexpr (is_axis_lengths_shared) {
    return NgpEllipsoidData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type>{
        ngp_mesh, center_data, orientation_data, axis_lengths_data};
  } else {
    return NgpEllipsoidData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type>{
        ngp_mesh, center_data, orientation_data, axis_lengths_data};
  }
}

/// \brief Check if the type provides the same data as EllipsoidData
template <typename Agg>
concept ValidEllipsoidDataType =
    requires(Agg agg) { 
      typename Agg::scalar_t; 
    { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
    } &&
    std::convertible_to<decltype(std::declval<Agg>().bulk_data()), stk::mesh::BulkData&> &&
    std::convertible_to<decltype(std::declval<Agg>().center_data()), stk::mesh::Field<typename Agg::scalar_t>&> &&
    std::convertible_to<decltype(std::declval<Agg>().orientation_data()), stk::mesh::Field<typename Agg::scalar_t>&> &&
    (std::convertible_to<decltype(std::declval<Agg>().axis_lengths_data()),
                         stk::mesh::Field<typename Agg::scalar_t>&> ||
     std::convertible_to<decltype(std::declval<Agg>().axis_lengths_data()),
                         mundy::math::Vector3<typename Agg::scalar_t>&>);

/// \brief Check if the type provides the same data as NgpEllipsoidData
template <typename Agg>
concept ValidNgpEllipsoidDataType =
    requires(Agg agg) { 
      typename Agg::scalar_t; 
    { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
    } &&
    std::convertible_to<decltype(std::declval<Agg>().ngp_mesh()), stk::mesh::NgpMesh&> &&
    std::convertible_to<decltype(std::declval<Agg>().center_data()), stk::mesh::NgpField<typename Agg::scalar_t>&> &&
    std::convertible_to<decltype(std::declval<Agg>().orientation_data()),
                        stk::mesh::NgpField<typename Agg::scalar_t>&> &&
    (std::convertible_to<decltype(std::declval<Agg>().axis_lengths_data()),
                         stk::mesh::NgpField<typename Agg::scalar_t>&> ||
     std::convertible_to<decltype(std::declval<Agg>().axis_lengths_data()),
                         mundy::math::Vector3<typename Agg::scalar_t>&>);

/// \brief A helper function to get an updated NgpEllipsoidData object from a EllipsoidData object
/// \param data The EllipsoidData object to convert
template <ValidEllipsoidDataType EllipsoidDataType>
auto get_updated_ngp_data(EllipsoidDataType data) {
  using scalar_t = typename EllipsoidDataType::scalar_t;
  constexpr stk::topology::topology_t topology_t = EllipsoidDataType::topology_t;
  using axis_lengths_data_t = decltype(data.axis_lengths_data());

  constexpr bool is_axis_lengths_a_field = std::is_same_v<std::decay_t<axis_lengths_data_t>, stk::mesh::Field<scalar_t>>;
  if constexpr (is_axis_lengths_a_field) {
    return create_ngp_ellipsoid_data<scalar_t, topology_t>(
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),                  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),       //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data()),  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.axis_lengths_data()));
  } else {
    return create_ngp_ellipsoid_data<scalar_t, topology_t>(
        stk::mesh::get_updated_ngp_mesh(data.bulk_data()),                  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),       //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.orientation_data()),  //
        data.axis_lengths_data());
  }
}

/// \brief A traits class to provide abstracted access to a ellipsoid's data via an aggregate
///
/// By default, this class is compatible with EllipsoidData or any class the meets the ValidEllipsoidDataType
/// concept. Users can specialize this class to support other aggregate types.
template <typename Agg>
struct EllipsoidDataTraits {
  static_assert(ValidEllipsoidDataType<Agg>,
                "Agg must satisfy the ValidEllipsoidDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpEllipsoidData but is free to extend "
                "it as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;

  static constexpr bool has_shared_axis_lengths() {
    return mundy::math::is_vector3_v<std::decay_t<decltype(std::declval<Agg>().axis_lengths_data())>>;
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity ellipsoid_node) {
    return mundy::mesh::vector3_field_data(agg.center_data(), ellipsoid_node);
  }

  static decltype(auto) orientation(Agg agg, stk::mesh::Entity ellipsoid) {
    return mundy::mesh::quaternion_field_data(agg.orientation_data(), ellipsoid);
  }

  static decltype(auto) axis_lengths(Agg agg, stk::mesh::Entity ellipsoid) {
    if constexpr (has_shared_axis_lengths()) {
      return agg.axis_lengths_data();
    } else {
      return mundy::mesh::vector3_field_data(agg.axis_lengths_data(), ellipsoid);
    }
  }
};  // EllipsoidDataTraits

/// \brief A traits class to provide abstracted access to a ellipsoid's data via an NGP-compatible aggregate
/// See the discussion for EllipsoidDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpEllipsoidDataTraits {
  static_assert(ValidNgpEllipsoidDataType<Agg>,
                "Agg must satisfy the ValidNgpEllipsoidDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpEllipsoidData but is free to extend "
                "it as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_axis_lengths() {
    return mundy::math::is_vector3_v<std::decay_t<decltype(std::declval<Agg>().axis_lengths_data())>>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex ellipsoid_node_index) {
    return mundy::mesh::vector3_field_data(agg.center_data(), ellipsoid_node_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) orientation(Agg agg, stk::mesh::FastMeshIndex ellipsoid_index) {
    return mundy::mesh::quaternion_field_data(agg.orientation_data(), ellipsoid_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) axis_lengths(Agg agg, stk::mesh::FastMeshIndex ellipsoid_index) {
    if constexpr (has_shared_axis_lengths()) {
      return agg.axis_lengths_data();
    } else {
      return mundy::mesh::vector3_field_data(agg.axis_lengths_data(), ellipsoid_index);
    }
  }
};  // NgpEllipsoidDataTraits

/// @brief A view of an STK entity meant to represent a ellipsoid
///
/// We type specialize this class based on the valid set of topologies for an ellipsoid entity.
///
/// Use \ref create_ellipsoid_entity_view to build an EllipsoidEntityView object with automatic template deduction.
template <stk::topology::topology_t OurTopology, typename EllipsoidDataType>
class EllipsoidEntityView;

/// @brief A view of an STK entity meant to represent a ellipsoid
template <typename EllipsoidDataType>
class EllipsoidEntityView<stk::topology::NODE, EllipsoidDataType> {
  static_assert(EllipsoidDataType::topology_t == stk::topology::NODE,
                "The topology of the ellipsoid data must match the view");

 public:
  using data_access_t = EllipsoidDataTraits<EllipsoidDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<EllipsoidDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  EllipsoidEntityView(EllipsoidDataType data, stk::mesh::Entity ellipsoid) : data_(data), ellipsoid_(ellipsoid) {
  }

  decltype(auto) data() {
    return data_;
  }

  decltype(auto) data() const {
    return data_;
  }

  stk::mesh::Entity& ellipsoid_entity() {
    return ellipsoid_;
  }

  const stk::mesh::Entity& ellipsoid_entity() const {
    return ellipsoid_;
  }

  decltype(auto) center() {
    return data_access_t::center(data(), ellipsoid_entity());
  }

  decltype(auto) center() const {
    return data_access_t::center(data(), ellipsoid_entity());
  }

  decltype(auto) orientation() {
    return data_access_t::orientation(data(), ellipsoid_entity());
  }

  decltype(auto) orientation() const {
    return data_access_t::orientation(data(), ellipsoid_entity());
  }

  decltype(auto) axis_lengths() {
    return data_access_t::axis_lengths(data(), ellipsoid_entity());
  }

  decltype(auto) axis_lengths() const {
    return data_access_t::axis_lengths(data(), ellipsoid_entity());
  }

 private:
  EllipsoidDataType data_;
  stk::mesh::Entity ellipsoid_;
};  // EllipsoidEntityView<NODE, EllipsoidDataType>

template <typename EllipsoidDataType>
class EllipsoidEntityView<stk::topology::PARTICLE, EllipsoidDataType> {
  static_assert(EllipsoidDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the ellipsoid data must match the view");

 public:
  using data_access_t = EllipsoidDataTraits<EllipsoidDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<EllipsoidDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  EllipsoidEntityView(EllipsoidDataType data, stk::mesh::Entity ellipsoid)
      : data_(data), ellipsoid_(ellipsoid), node_(data_.bulk_data().begin_nodes(ellipsoid_)[0]) {
  }

  decltype(auto) data() {
    return data_;
  }

  decltype(auto) data() const {
    return data_;
  }

  stk::mesh::Entity& ellipsoid_entity() {
    return ellipsoid_;
  }

  const stk::mesh::Entity& ellipsoid_entity() const {
    return ellipsoid_;
  }

  stk::mesh::Entity& node_entity() {
    return node_;
  }

  const stk::mesh::Entity& node_entity() const {
    return node_;
  }

  decltype(auto) center() {
    return data_access_t::center(data(), node_entity());
  }

  decltype(auto) center() const {
    return data_access_t::center(data(), node_entity());
  }

  decltype(auto) orientation() {
    return data_access_t::orientation(data(), ellipsoid_entity());
  }

  decltype(auto) orientation() const {
    return data_access_t::orientation(data(), ellipsoid_entity());
  }

  decltype(auto) axis_lengths() {
    return data_access_t::axis_lengths(data(), ellipsoid_entity());
  }

  decltype(auto) axis_lengths() const {
    return data_access_t::axis_lengths(data(), ellipsoid_entity());
  }

 private:
  EllipsoidDataType data_;
  stk::mesh::Entity ellipsoid_;
  stk::mesh::Entity node_;
};  // EllipsoidEntityView<PARTICLE, EllipsoidDataType>

/// @brief An ngp-compatible view of an ellipsoid entity
template <stk::topology::topology_t OurTopology, typename NgpEllipsoidDataType>
class NgpEllipsoidEntityView;

template <typename NgpEllipsoidDataType>
class NgpEllipsoidEntityView<stk::topology::NODE, NgpEllipsoidDataType> {
  static_assert(NgpEllipsoidDataType::topology_t == stk::topology::NODE,
                "The topology of the ellipsoid data must match the view");

 public:
  using data_access_t = NgpEllipsoidDataTraits<NgpEllipsoidDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<NgpEllipsoidDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpEllipsoidEntityView(NgpEllipsoidDataType data, stk::mesh::FastMeshIndex ellipsoid_index)
      : data_(data), ellipsoid_index_(ellipsoid_index) {
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
  stk::mesh::FastMeshIndex& ellipsoid_index() {
    return ellipsoid_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& ellipsoid_index() const {
    return ellipsoid_index_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return data_access_t::orientation(data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return data_access_t::orientation(data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() {
    return data_access_t::axis_lengths(data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() const {
    return data_access_t::axis_lengths(data(), ellipsoid_index());
  }

 private:
  NgpEllipsoidDataType data_;
  stk::mesh::FastMeshIndex ellipsoid_index_;
};  // NgpEllipsoidEntityView<NODE, NgpEllipsoidDataType>

/// @brief An ngp-compatible view of an STK entity meant to represent a ellipsoid
template <typename NgpEllipsoidDataType>
class NgpEllipsoidEntityView<stk::topology::PARTICLE, NgpEllipsoidDataType> {
  static_assert(NgpEllipsoidDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the ellipsoid data must match the view");

 public:
  using data_access_t = NgpEllipsoidDataTraits<NgpEllipsoidDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<NgpEllipsoidDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpEllipsoidEntityView(NgpEllipsoidDataType data, stk::mesh::FastMeshIndex ellipsoid_index)
      : data_(data),
        ellipsoid_index_(ellipsoid_index),
        node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, ellipsoid_index_)[0])) {
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
  stk::mesh::FastMeshIndex& ellipsoid_index() {
    return ellipsoid_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& ellipsoid_index() const {
    return ellipsoid_index_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex& node_index() {
    return node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& node_index() const {
    return node_index_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data(), node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data(), node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() {
    return data_access_t::orientation(data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) orientation() const {
    return data_access_t::orientation(data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() {
    return data_access_t::axis_lengths(data(), ellipsoid_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) axis_lengths() const {
    return data_access_t::axis_lengths(data(), ellipsoid_index());
  }

 private:
  NgpEllipsoidDataType data_;
  stk::mesh::FastMeshIndex ellipsoid_index_;
  stk::mesh::FastMeshIndex node_index_;
};  // NgpEllipsoidEntityView<PARTICLE, NgpEllipsoidDataType>

/// \brief A helper function to create a EllipsoidEntityView object with type deduction
template <typename EllipsoidDataType>  // deduced
auto create_ellipsoid_entity_view(EllipsoidDataType& data, stk::mesh::Entity ellipsoid) {
  return EllipsoidEntityView<EllipsoidDataType::topology_t, EllipsoidDataType>(data, ellipsoid);
}

/// \brief A helper function to create a NgpEllipsoidEntityView object with type deduction
template <typename NgpEllipsoidDataType>  // deduced
auto create_ngp_ellipsoid_entity_view(NgpEllipsoidDataType data, stk::mesh::FastMeshIndex ellipsoid_index) {
  return NgpEllipsoidEntityView<NgpEllipsoidDataType::topology_t, NgpEllipsoidDataType>(data, ellipsoid_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_ELLIPSOID_HPP_