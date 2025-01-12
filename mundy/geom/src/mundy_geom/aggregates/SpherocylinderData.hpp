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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERDATA_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERDATA_HPP_

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
#include <mundy_geom/aggregates/EntityView.hpp>  // for mundy::geom::EntityView and mundy::geom::create_topological_entity_view
#include <mundy_geom/aggregates/SpherocylinderDataConcepts.hpp>  // for mundy::geom::ValidSpherocylinderDataType
#include <mundy_geom/aggregates/SpherocylinderEntityView.hpp>    // for mundy::geom::SpherocylinderEntityView
#include <mundy_geom/primitives/Spherocylinder.hpp>              // for mundy::geom::ValidSpherocylinderType
#include <mundy_mesh/BulkData.hpp>                               // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Aggregate to hold the data for a collection of spherocylinders
///
/// The topology of a spherocylinder directly effects the access pattern for the underlying data:
///   - NODE: All data is stored on a single node
///   - PARTICLE: The center is stored on a node, whereas the orientation and radius are stored on the element-rank
///   particle
///
/// Use \ref create_spherocylinder_data to build a SpherocylinderData object with automatic template deduction.
template <typename Scalar,                             //
          typename OurTopology,                        //
          typename HasSharedRadius = std::false_type,  //
          typename HasSharedLength = std::false_type>
class SpherocylinderData {
  static_assert(OurTopology::value == stk::topology::NODE || OurTopology::value == stk::topology::PARTICLE,
                "The topology of a spherocylinder must be either NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using center_data_t = stk::mesh::Field<Scalar>;
  using orientation_data_t = stk::mesh::Field<Scalar>;
  using radius_data_t = std::conditional_t<HasSharedRadius::value, Scalar, stk::mesh::Field<Scalar>>;
  using length_data_t = std::conditional_t<HasSharedLength::value, Scalar, stk::mesh::Field<Scalar>>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  SpherocylinderData(const stk::mesh::BulkData& bulk_data, const center_data_t& center_data, const orientation_data_t& orientation_data,
                     const radius_data_t& radius_data, const length_data_t& length_data)
      : bulk_data_(bulk_data),
        center_data_(center_data),
        orientation_data_(orientation_data),
        radius_data_(radius_data),
        length_data_(length_data) {
    stk::topology our_topology = topology_t;
    MUNDY_THROW_ASSERT(orientation_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                       "The orientation data must be a field of the same rank as the spherocylinder");
    if constexpr (!HasSharedRadius::value) {
      MUNDY_THROW_ASSERT(radius_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                         "The radius data must be a field of the same rank as the spherocylinder");
    }
    if constexpr (!HasSharedLength::value) {
      MUNDY_THROW_ASSERT(length_data.entity_rank() == our_topology.rank(), std::invalid_argument,
                         "The length data must be a field of the same rank as the spherocylinder");
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

  const radius_data_t& radius_data() const {
    return radius_data_;
  }

  const length_data_t& length_data() const {
    return length_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto add_augment(Args&&... args) const {
    return NextAugment<SpherocylinderData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  auto get_entity_view(stk::mesh::Entity entity) {
    using our_t = SpherocylinderData<Scalar, OurTopology, HasSharedRadius, HasSharedLength>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<SpherocylinderEntityView, our_t>(*this);
  }

  const auto get_entity_view(stk::mesh::Entity entity) const {
    using our_t = SpherocylinderData<Scalar, OurTopology, HasSharedRadius, HasSharedLength>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(bulk_data(), entity)
        .template augment_view<SpherocylinderEntityView, our_t>(*this);
  }

  auto get_updated_ngp_data() const {
    if constexpr (!HasSharedRadius::value && !HasSharedLength::value) {
      return create_ngp_spherocylinder_data<scalar_t, topology_t>(        //
          stk::mesh::get_updated_ngp_mesh(bulk_data_),                    //
          stk::mesh::get_updated_ngp_field<scalar_t>(center_data_),       //
          stk::mesh::get_updated_ngp_field<scalar_t>(orientation_data_),  //
          stk::mesh::get_updated_ngp_field<scalar_t>(radius_data_),       //
          stk::mesh::get_updated_ngp_field<scalar_t>(length_data_));
    } else if constexpr (!HasSharedRadius::value && HasSharedLength::value) {
      return create_ngp_spherocylinder_data<scalar_t, topology_t>(        //
          stk::mesh::get_updated_ngp_mesh(bulk_data_),                    //
          stk::mesh::get_updated_ngp_field<scalar_t>(center_data_),       //
          stk::mesh::get_updated_ngp_field<scalar_t>(orientation_data_),  //
          stk::mesh::get_updated_ngp_field<scalar_t>(radius_data_),       //
          length_data_);
    } else if constexpr (HasSharedRadius::value && !HasSharedLength::value) {
      return create_ngp_spherocylinder_data<scalar_t, topology_t>(        //
          stk::mesh::get_updated_ngp_mesh(bulk_data_),                    //
          stk::mesh::get_updated_ngp_field<scalar_t>(center_data_),       //
          stk::mesh::get_updated_ngp_field<scalar_t>(orientation_data_),  //
          radius_data_,                                                   //
          stk::mesh::get_updated_ngp_field<scalar_t>(length_data_));
    } else {
      return create_ngp_spherocylinder_data<scalar_t, topology_t>(        //
          stk::mesh::get_updated_ngp_mesh(bulk_data_),                    //
          stk::mesh::get_updated_ngp_field<scalar_t>(center_data_),       //
          stk::mesh::get_updated_ngp_field<scalar_t>(orientation_data_),  //
          radius_data_,                                                   //
          length_data_);
    }
  }

 private:
  const stk::mesh::BulkData& bulk_data_;
  const center_data_t& center_data_;
  const orientation_data_t& orientation_data_;
  const radius_data_t& radius_data_;
  const length_data_t& length_data_;
};  // SpherocylinderData

/// \brief Aggregate to hold the data for a collection of NGP-compatible spherocylinders
/// See the discussion for SpherocylinderData for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                             //
          typename OurTopology,                        //
          typename HasSharedRadius = std::false_type,  //
          typename HasSharedLength = std::false_type>
class NgpSpherocylinderData {
  static_assert(OurTopology::value == stk::topology::NODE || OurTopology::value == stk::topology::PARTICLE,
                "The topology of a spherocylinder must be either NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using center_data_t = stk::mesh::NgpField<Scalar>;
  using orientation_data_t = stk::mesh::NgpField<Scalar>;
  using radius_data_t = std::conditional_t<HasSharedRadius::value, Scalar, stk::mesh::NgpField<Scalar>>;
  using length_data_t = std::conditional_t<HasSharedLength::value, Scalar, stk::mesh::NgpField<Scalar>>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;

  /// \brief Constructor
  KOKKOS_INLINE_FUNCTION
  NgpSpherocylinderData(const stk::mesh::NgpMesh &ngp_mesh, const center_data_t& center_data, const orientation_data_t& orientation_data,
                        const radius_data_t& radius_data, const length_data_t& length_data)
      : ngp_mesh_(ngp_mesh),
        center_data_(center_data),
        orientation_data_(orientation_data),
        radius_data_(radius_data),
        length_data_(length_data) {
    stk::topology our_topology = topology_t;
    MUNDY_THROW_ASSERT(orientation_data.get_rank() == our_topology.rank(), std::invalid_argument,
                       "The orientation data must be a field of the same rank as the spherocylinder");
    if constexpr (!HasSharedRadius::value) {
      MUNDY_THROW_ASSERT(radius_data.get_rank() == our_topology.rank(), std::invalid_argument,
                         "The radius data must be a field of the same rank as the spherocylinder");
    }
    if constexpr (!HasSharedLength::value) {
      MUNDY_THROW_ASSERT(length_data.get_rank() == our_topology.rank(), std::invalid_argument,
                         "The length data must be a field of the same rank as the spherocylinder");
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
  const radius_data_t& radius_data() const {
    return radius_data_;
  }

  KOKKOS_INLINE_FUNCTION
  radius_data_t& radius_data() {
    return radius_data_;
  }

  KOKKOS_INLINE_FUNCTION
  const length_data_t& length_data() const {
    return length_data_;
  }

  KOKKOS_INLINE_FUNCTION
  length_data_t& length_data() {
    return length_data_;
  }

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION
  auto add_augment(Args&&... args) const {
    return NextAugment<NgpSpherocylinderData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  auto get_entity_view(stk::mesh::FastMeshIndex entity_index) {
    using our_t = NgpSpherocylinderData<Scalar, OurTopology, HasSharedRadius, HasSharedLength>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpSpherocylinderEntityView, our_t>(*this);
  }

  KOKKOS_INLINE_FUNCTION
  const auto get_entity_view(stk::mesh::FastMeshIndex entity_index) const {
    using our_t = NgpSpherocylinderData<Scalar, OurTopology, HasSharedRadius, HasSharedLength>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(ngp_mesh(), entity_index)
        .template augment_view<NgpSpherocylinderEntityView, our_t>(*this);
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
  center_data_t center_data_;
  orientation_data_t orientation_data_;
  radius_data_t radius_data_;
  length_data_t length_data_;
};  // NgpSpherocylinderData

/// \brief A helper function to create a SpherocylinderData object
///
/// This function creates a SpherocylinderData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename RadiusDataType,                // deduced
          typename LengthDataType>                // deduced
auto create_spherocylinder_data(const stk::mesh::BulkData& bulk_data, const stk::mesh::Field<Scalar>& center_data,
                                const stk::mesh::Field<Scalar>& orientation_data, const RadiusDataType& radius_data,
                                const LengthDataType& length_data) {
  constexpr bool is_radius_shared = std::is_same_v<std::decay_t<RadiusDataType>, Scalar>;
  constexpr bool is_length_shared = std::is_same_v<std::decay_t<LengthDataType>, Scalar>;
  if constexpr (is_radius_shared && is_length_shared) {
    return SpherocylinderData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type, std::true_type>{
        bulk_data, center_data, orientation_data, radius_data, length_data};
  } else if constexpr (is_radius_shared && !is_length_shared) {
    return SpherocylinderData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type,
                              std::false_type>{bulk_data, center_data, orientation_data, radius_data, length_data};
  } else if constexpr (!is_radius_shared && is_length_shared) {
    return SpherocylinderData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type,
                              std::true_type>{bulk_data, center_data, orientation_data, radius_data, length_data};
  } else {
    return SpherocylinderData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type,
                              std::false_type>{bulk_data, center_data, orientation_data, radius_data, length_data};
  }
}

/// \brief A helper function to create a NgpSpherocylinderData object
/// See the discussion for create_spherocylinder_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                        // Must be provided
          stk::topology::topology_t OurTopology,  // Must be provided
          typename RadiusDataType,                // deduced
          typename LengthDataType>                // deduced
auto create_ngp_spherocylinder_data(const stk::mesh::NgpMesh& ngp_mesh, const stk::mesh::NgpField<Scalar>& center_data,
                                    const stk::mesh::NgpField<Scalar>& orientation_data,
                                    const RadiusDataType& radius_data, const LengthDataType& length_data) {
  constexpr bool is_radius_shared = std::is_same_v<std::decay_t<RadiusDataType>, Scalar>;
  constexpr bool is_length_shared = std::is_same_v<std::decay_t<LengthDataType>, Scalar>;
  if constexpr (is_radius_shared && is_length_shared) {
    return NgpSpherocylinderData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type,
                                 std::true_type>{ngp_mesh, center_data, orientation_data, radius_data, length_data};
  } else if constexpr (is_radius_shared && !is_length_shared) {
    return NgpSpherocylinderData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::true_type,
                                 std::false_type>{ngp_mesh, center_data, orientation_data, radius_data, length_data};
  } else if constexpr (!is_radius_shared && is_length_shared) {
    return NgpSpherocylinderData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type,
                                 std::true_type>{ngp_mesh, center_data, orientation_data, radius_data, length_data};
  } else {
    return NgpSpherocylinderData<Scalar, stk::topology_detail::topology_data<OurTopology>, std::false_type,
                                 std::false_type>{ngp_mesh, center_data, orientation_data, radius_data, length_data};
  }
}

/// \brief A helper function to get an updated NgpSpherocylinderData object from a SpherocylinderData object
/// \param data The SpherocylinderData object to convert
template <typename Scalar, typename OurTopology, typename HasSharedRadius, typename HasSharedLength>
auto get_updated_ngp_data(const SpherocylinderData<Scalar, OurTopology, HasSharedRadius, HasSharedLength>& data) {
  return data.get_updated_ngp_data();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEROCYLINDERDATA_HPP_