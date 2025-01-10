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

#ifndef MUNDY_GEOM_AGGREGATES_SPHERE_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHERE_HPP_

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
#include <mundy_geom/primitives/Sphere.hpp>  // for mundy::geom::ValidSphereType
#include <mundy_mesh/BulkData.hpp>           // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>         // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of spheres
/// The topology of an sphere directly effects the access pattern for the underlying data:
///   - NODE: All data is stored on a single node
///   - PARTICLE: The center is stored on a node, whereas the radius is stored on the element-rank particle
///
/// Use \ref create_sphere_data to build an SphereData object with automatic template deduction.
template <typename Scalar,       //
          typename OurTopology,  //
          typename HasSharedRadius = std::false_type>
class SphereData {
  static_assert(OurTopology::value == stk::topology::NODE || OurTopology::value == stk::topology::PARTICLE,
                "The topology of a sphere must be either NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using center_data_t = stk::mesh::Field<Scalar>;
  using radius_data_t = std::conditional_t<HasSharedRadius::value, Scalar, stk::mesh::Field<Scalar>>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  SphereData(stk::mesh::BulkData& bulk_data, center_data_t& center_data, radius_data_t& radius_data)
      : bulk_data_(bulk_data), center_data_(center_data), radius_data_(radius_data) {
    stk::topology our_topology = topology_t;
    MUNDY_THROW_ASSERT(center_data.entity_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center data must be a field of NODE_RANK");
    if constexpr (!HasSharedRadius::value) {
      MUNDY_THROW_ASSERT(radius_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                         "The radius data must be a field of the same rank as the sphere");
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
    return NextAugment<SphereData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  stk::mesh::BulkData& bulk_data_;
  center_data_t& center_data_;
  radius_data_t& radius_data_;
};  // SphereData

/// \brief Aggregate to hold the data for a collection of NGP-compatible spheres
/// See the discussion for SphereData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,       //
          typename OurTopology,  //
          typename HasSharedRadius = std::false_type>
class NgpSphereData {
  static_assert(OurTopology::value == stk::topology::NODE || OurTopology::value == stk::topology::PARTICLE,
                "The topology of a sphere must be either NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using center_data_t = stk::mesh::NgpField<Scalar>;
  using radius_data_t = std::conditional_t<HasSharedRadius::value, Scalar, stk::mesh::NgpField<Scalar>>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  NgpSphereData(stk::mesh::NgpMesh ngp_mesh, center_data_t& center_data, radius_data_t& radius_data)
      : ngp_mesh_(ngp_mesh), center_data_(center_data), radius_data_(radius_data) {
    MUNDY_THROW_ASSERT(center_data.get_rank() == stk::topology::NODE_RANK, std::invalid_argument,
                       "The center data must be a field of NODE_RANK");
    stk::topology our_topology = topology_t;
    if constexpr (!HasSharedRadius::value) {
      MUNDY_THROW_ASSERT(radius_data.get_rank() == our_topology.rank(), std::invalid_argument,
                         "The radius data must be a field of the same rank as the sphere");
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
    return NextAugment<NgpSphereData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  center_data_t& center_data_;
  radius_data_t& radius_data_;
};  // NgpSphereData

/// \brief A helper function to create a SphereData object
///
/// This function creates a SphereData object given its data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename RadiusDataType>                // deduced
auto create_sphere_data(stk::mesh::BulkData& bulk_data, stk::mesh::Field<Scalar>& center_data,
                        RadiusDataType& radius_data) {
  constexpr bool is_radius_shared = std::is_same_v<std::decay_t<RadiusDataType>, Scalar>;
  if constexpr (is_radius_shared) {
    return SphereData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type>{bulk_data, center_data,
                                                                                                radius_data};
  } else {
    return SphereData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type>{bulk_data, center_data,
                                                                                                 radius_data};
  }
}

/// \brief A helper function to create a NgpSphereData object
/// See the discussion for create_sphere_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename RadiusDataType>                // deduced
auto create_ngp_sphere_data(stk::mesh::NgpMesh ngp_mesh, stk::mesh::NgpField<Scalar>& center_data,
                            RadiusDataType& radius_data) {
  constexpr bool is_radius_shared = std::is_same_v<std::decay_t<RadiusDataType>, Scalar>;
  if constexpr (is_radius_shared) {
    return NgpSphereData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type>{
        ngp_mesh, center_data, radius_data};
  } else {
    return NgpSphereData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type>{
        ngp_mesh, center_data, radius_data};
  }
}

/// \brief Check if the type provides the same data as SphereData
template <typename Agg>
concept ValidSphereDataType =
    requires(Agg agg) { 
      typename Agg::scalar_t; 
    { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
    } &&
    std::convertible_to<decltype(std::declval<Agg>().bulk_data()), stk::mesh::BulkData&> &&
    std::convertible_to<decltype(std::declval<Agg>().center_data()), stk::mesh::Field<typename Agg::scalar_t>&> &&
    (std::convertible_to<decltype(std::declval<Agg>().radius_data()), stk::mesh::Field<typename Agg::scalar_t>&> ||
     std::convertible_to<decltype(std::declval<Agg>().radius_data()), typename Agg::scalar_t&>);

/// \brief Check if the type provides the same data as NgpSphereData
template <typename Agg>
concept ValidNgpSphereDataType =
    requires(Agg agg) { 
      typename Agg::scalar_t; 
    { Agg::topology_t } -> std::convertible_to<stk::topology::topology_t>;
    } &&
    std::convertible_to<decltype(std::declval<Agg>().ngp_mesh()), stk::mesh::NgpMesh&> &&
    std::convertible_to<decltype(std::declval<Agg>().center_data()), stk::mesh::NgpField<typename Agg::scalar_t>&> &&
    (std::convertible_to<decltype(std::declval<Agg>().radius_data()), stk::mesh::NgpField<typename Agg::scalar_t>&> ||
     std::convertible_to<decltype(std::declval<Agg>().radius_data()), typename Agg::scalar_t&>);

/// \brief A helper function to get an updated NgpSphereData object from a SphereData object
/// \param data The SphereData object to convert
template <ValidSphereDataType SphereDataType>
auto get_updated_ngp_data(SphereDataType data) {
  using scalar_t = typename SphereDataType::scalar_t;
  using radius_data_t = decltype(data.radius_data());

  constexpr bool is_radius_a_field = std::is_same_v<std::decay_t<radius_data_t>, stk::mesh::Field<scalar_t>>;
  constexpr stk::topology::topology_t topology_t = SphereDataType::topology_t;
  if constexpr (is_radius_a_field) {
    return create_ngp_sphere_data<scalar_t, topology_t>(                 //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),  //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.radius_data()));
  } else {
    return create_ngp_sphere_data<scalar_t, topology_t>(                 //
        stk::mesh::get_updated_ngp_field<scalar_t>(data.center_data()),  //
        data.radius_data());
  }
}

/// \brief A traits class to provide abstracted access to a sphere's data via an aggregate
///
/// By default, this class is compatible with SphereData or any class the meets the ValidSphereDataType concept.
/// Users can specialize this class to support other aggregate types.
template <typename Agg>
struct SphereDataTraits {
  static_assert(ValidSphereDataType<Agg>,
                "Agg must satisfy the ValidSphereDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSphereData but is free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  using center_data_t = decltype(std::declval<Agg>().center_data());
  using radius_data_t = decltype(std::declval<Agg>().radius_data());
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<radius_data_t>, scalar_t>;
  }

  static decltype(auto) center(Agg agg, stk::mesh::Entity sphere_node) {
    return mundy::mesh::vector3_field_data(agg.center_data(), sphere_node);
  }

  static decltype(auto) radius(Agg agg, stk::mesh::Entity sphere) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return stk::mesh::field_data(agg.radius_data(), sphere)[0];
    }
  }
};  // SphereDataTraits

/// \brief A traits class to provide abstracted access to a sphere's data via an NGP-compatible aggregate
/// See the discussion for SphereDataTraits for more information. Only difference is Ngp-compatible data.
template <typename Agg>
struct NgpSphereDataTraits {
  static_assert(ValidNgpSphereDataType<Agg>,
                "Agg must satisfy the ValidNgpSphereDataType concept.\n"
                "Basically, Agg must have the same getters and types aliases as NgpSphereData but is free to extend it "
                "as needed without "
                "having to rely on inheritance.");

  using scalar_t = typename Agg::scalar_t;
  static constexpr stk::topology::topology_t topology_t = Agg::topology_t;

  KOKKOS_INLINE_FUNCTION
  static constexpr bool has_shared_radius() {
    return std::is_same_v<std::decay_t<decltype(std::declval<Agg>().radius_data())>, scalar_t>;
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) center(Agg agg, stk::mesh::FastMeshIndex sphere_node_index) {
    return mundy::mesh::vector3_field_data(agg.center_data(), sphere_node_index);
  }

  KOKKOS_INLINE_FUNCTION
  static decltype(auto) radius(Agg agg, stk::mesh::FastMeshIndex sphere_index) {
    if constexpr (has_shared_radius()) {
      return agg.radius_data();
    } else {
      return agg.radius_data()(sphere_index, 0);
    }
  }
};  // NgpSphereDataTraits

/// @brief A view of an STK entity meant to represent a sphere
///
/// We type specialize this class based on the valid set of topologies for a sphere entity.
///
/// Use \ref create_sphere_entity_view to build a SphereEntityView object with automatic template deduction.
template <stk::topology::topology_t OurTopology, typename SphereDataType>
class SphereEntityView;

/// @brief A view of a NODE STK entity meant to represent a sphere
template <typename SphereDataType>
class SphereEntityView<stk::topology::NODE, SphereDataType> {
  static_assert(SphereDataType::topology_t == stk::topology::NODE,
                "The topology of the sphere data must match the view");

 public:
  using data_access_t = SphereDataTraits<SphereDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<SphereDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  SphereEntityView(SphereDataType data, stk::mesh::Entity sphere) : data_(data), sphere_(sphere) {
    MUNDY_THROW_ASSERT(data_.bulk_data().entity_rank(sphere_) == stk::topology::NODE_RANK, std::invalid_argument,
                       "The sphere entity rank must be NODE_RANK");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(sphere_), std::invalid_argument,
                       "The given sphere entity is not valid");
  }

  decltype(auto) data() {
    return data_;
  }

  decltype(auto) data() const {
    return data_;
  }

  stk::mesh::Entity& sphere_entity() {
    return sphere_;
  }

  const stk::mesh::Entity& sphere_entity() const {
    return sphere_;
  }

  decltype(auto) radius() {
    // For those not familiar with decltype(auto), it allows us to return either an auto or an auto&.
    return data_access_t::radius(data(), sphere_entity());
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data(), sphere_entity());
  }

  decltype(auto) center() {
    return data_access_t::center(data(), sphere_entity());
  }

  decltype(auto) center() const {
    return data_access_t::center(data(), sphere_entity());
  }

 private:
  SphereDataType data_;
  stk::mesh::Entity sphere_;
};  // SphereEntityView<stk::topology::NODE, SphereDataType>

/// @brief A view of a PARTICLE STK entity meant to represent a sphere
template <typename SphereDataType>
class SphereEntityView<stk::topology::PARTICLE, SphereDataType> {
  static_assert(SphereDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the sphere data must match the view");

 public:
  using data_access_t = SphereDataTraits<SphereDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t = decltype(data_access_t::center(std::declval<SphereDataType>(), std::declval<stk::mesh::Entity>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  SphereEntityView(SphereDataType data, stk::mesh::Entity sphere)
      : data_(data), sphere_(sphere), node_(data_.bulk_data().begin_nodes(sphere_)[0]) {
    MUNDY_THROW_ASSERT(data_.bulk_data().entity_rank(sphere_) == stk::topology::ELEM_RANK, std::invalid_argument,
                       "The sphere entity rank must be ELEM_RANK");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(sphere_), std::invalid_argument,
                       "The given sphere entity is not valid");
    MUNDY_THROW_ASSERT(data_.bulk_data().num_nodes(sphere_) >= 1, std::invalid_argument,
                       "The given sphere entity must have at least one node");
    MUNDY_THROW_ASSERT(data_.bulk_data().is_valid(node_), std::invalid_argument,
                       "The node entity associated with the sphere is not valid");
  }

  decltype(auto) data() {
    return data_;
  }

  decltype(auto) data() const {
    return data_;
  }

  stk::mesh::Entity& sphere_entity() {
    return sphere_;
  }

  const stk::mesh::Entity& sphere_entity() const {
    return sphere_;
  }

  stk::mesh::Entity& node_entity() {
    return node_;
  }

  const stk::mesh::Entity& node_entity() const {
    return node_;
  }

  decltype(auto) radius() {
    // For those not familiar with decltype(auto), it allows us to return either an auto or an auto&.
    return data_access_t::radius(data(), sphere_entity());
  }

  decltype(auto) radius() const {
    return data_access_t::radius(data(), sphere_entity());
  }

  decltype(auto) center() {
    return data_access_t::center(data(), node_entity());
  }

  decltype(auto) center() const {
    return data_access_t::center(data(), node_entity());
  }

 private:
  SphereDataType data_;
  stk::mesh::Entity sphere_;
  stk::mesh::Entity node_;
};  // SphereEntityView<stk::topology::PARTICLE, SphereDataType>

/// @brief An ngp-compatible view of a STK entity meant to represent a sphere
/// See the discussion for SphereEntityView for more information. The only difference is ngp-compatible data access.
template <stk::topology::topology_t OurTopology, typename NgpSphereDataType>
class NgpSphereEntityView;

/// @brief An ngp-compatible view of a NODE STK entity meant to represent a sphere
template <typename NgpSphereDataType>
class NgpSphereEntityView<stk::topology::NODE, NgpSphereDataType> {
  static_assert(NgpSphereDataType::topology_t == stk::topology::NODE,
                "The topology of the sphere data must match the view");

 public:
  using data_access_t = NgpSphereDataTraits<NgpSphereDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<NgpSphereDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::NODE;
  static constexpr stk::topology::rank_t rank = stk::topology::NODE_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSphereEntityView(NgpSphereDataType data, stk::mesh::FastMeshIndex sphere_index)
      : data_(data), sphere_index_(sphere_index) {
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
  stk::mesh::FastMeshIndex& sphere_index() {
    return sphere_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& sphere_index() const {
    return sphere_index_;
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() {
    return data_access_t::radius(data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data(), sphere_index());
  }

 private:
  NgpSphereDataType data_;
  stk::mesh::FastMeshIndex sphere_index_;
};  // NgpSphereEntityView<stk::topology::NODE, NgpSphereDataType>

/// @brief An ngp-compatible view of a PARTICLE STK entity meant to represent a sphere
template <typename NgpSphereDataType>
class NgpSphereEntityView<stk::topology::PARTICLE, NgpSphereDataType> {
  static_assert(NgpSphereDataType::topology_t == stk::topology::PARTICLE,
                "The topology of the sphere data must match the view");

 public:
  using data_access_t = NgpSphereDataTraits<NgpSphereDataType>;
  using scalar_t = typename data_access_t::scalar_t;
  using point_t =
      decltype(data_access_t::center(std::declval<NgpSphereDataType>(), std::declval<stk::mesh::FastMeshIndex>()));

  static constexpr stk::topology::topology_t topology_t = stk::topology::PARTICLE;
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;

  KOKKOS_INLINE_FUNCTION
  NgpSphereEntityView(NgpSphereDataType data, stk::mesh::FastMeshIndex sphere_index)
      : data_(data),
        sphere_index_(sphere_index),
        node_index_(data_.ngp_mesh().fast_mesh_index(data_.ngp_mesh().get_nodes(rank, sphere_index_)[0])) {
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
  stk::mesh::FastMeshIndex& sphere_index() {
    return sphere_index_;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::FastMeshIndex& sphere_index() const {
    return sphere_index_;
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
  decltype(auto) radius() {
    return data_access_t::radius(data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) radius() const {
    return data_access_t::radius(data(), sphere_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() {
    return data_access_t::center(data(), node_index());
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) center() const {
    return data_access_t::center(data(), node_index());
  }

 private:
  NgpSphereDataType data_;
  stk::mesh::FastMeshIndex sphere_index_;
  stk::mesh::FastMeshIndex node_index_;
};  // NgpSphereEntityView<stk::topology::PARTICLE, NgpSphereDataType>

/// \brief A helper function to create a SphereEntityView object with type deduction
template <typename SphereDataType>  // deduced
auto create_sphere_entity_view(SphereDataType& data, stk::mesh::Entity sphere) {
  return SphereEntityView<SphereDataType::topology_t, SphereDataType>(data, sphere);
}

/// \brief A helper function to create a NgpSphereEntityView object with type deduction
template <typename NgpSphereDataType>  // deduced
auto create_ngp_sphere_entity_view(NgpSphereDataType data, stk::mesh::FastMeshIndex sphere_index) {
  return NgpSphereEntityView<NgpSphereDataType::topology_t, NgpSphereDataType>(data, sphere_index);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHERE_HPP_