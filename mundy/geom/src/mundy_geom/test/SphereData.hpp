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

#ifndef MUNDY_GEOM_AGGREGATES_SPHEREDATA_HPP_
#define MUNDY_GEOM_AGGREGATES_SPHEREDATA_HPP_

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
#include <mundy_geom/aggregates/SphereDataConcepts.hpp>  // for mundy::geom::ValidSphereDataType
#include <mundy_geom/aggregates/SphereEntityView.hpp>    // for mundy::geom::SphereEntityView
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Base class for all SphereData objects
///
/// This base class simply hold runtime information about the SphereData object.
class SphereDataBase {
 public:
  /// \brief Constructor
  ///
  /// \param name The string name identifying an instance of the SphereData object
  /// \param bulk_data The bulk data object
  /// \param topology The topology of entities within this aggregate
  SphereDataBase(const std::string& name, const stk::mesh::BulkData& bulk_data, stk::topology topology,
                      const stk::mesh::PartVector& parts)
      : name_(name), bulk_data_(bulk_data), topology_(topology), rank_(topology_.rank()), parts_(parts) {
  }

  /// \brief Get the name of the SphereData object
  std::string name() const {
    return name_;
  }

  /// \brief Get the bulk data object
  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }

  /// \brief Get the topology of entities within this aggregate
  stk::topology topology() const {
    return topology_;
  }

  /// \brief Get the rank of entities within this aggregate
  stk::topology::rank_t rank() const {
    return rank_;
  }

  /// \brief Get the parts of entities within this aggregate
  const stk::mesh::PartVector& parts() const {
    return parts_;
  }

 private:
  const std::string name_;                ///< The name of the SphereData object
  const stk::mesh::BulkData& bulk_data_;  ///< The bulk data object
  stk::topology topology_;                ///< The topology of entities within this aggregate
  stk::topology::rank_t rank_;            ///< The rank of entities within this aggregate
};

/// \brief Aggregate to hold the data for a collection of sphere
///
/// 
///
/// Shared data is stored as a const ref to the original data and is therefore unmodifiable. This is to prevent
/// accidental non-thread-safe modifications to the shared data. If this is every limiting, let us know and we can
/// consider adding a Kokkos::View to the shared data.
///
/// The tags for the field types are used to control the type of data stored in the SphereData object.
///   - *_IS_FIELD: The data is a field pointer: stk::mesh::Field<Scalar>*
///   - *_IS_VECTOR_OF_FIELDS: The data is a vector of field pointers: std::vector<stk::mesh::Field<Scalar>*>
///   - *_IS_SHARED: A single shared value is stored : SharedType
///   - *_IS_VECTOR_OF_SHARED: A vector of shared values is stored: std::vector<SharedType>
///
/// Use \ref create_sphere_data to build an SphereData object with automatic template deduction.
template <typename Scalar,       //
          typename OurTopology,  //
          typename CenterDataTag = CENTER_DATA_IS_FIELD, //
typename RadiusDataTag = RADIUS_DATA_IS_FIELD>
class SphereData : public SphereDataBase {
  static_assert(OurTopology::value == NODE || OurTopology::value == PARTICLE,
"The topology of an sphere must be either NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using center_data_t = map_tag_to_data_type_t</* Tag */ CenterDataTag, //
                                           /* Scalar */ scalar_t, //
                                           /* Shared type */ mundy::math::Vector3<scalar_t>>;
using radius_data_t = map_tag_to_data_type_t</* Tag */ RadiusDataTag, //
                                           /* Scalar */ scalar_t, //
                                           /* Shared type */ scalar_t>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;
  static constexpr stk::topology::rank_t rank_t = stk::topology_detail::topology_data<OurTopology::value>::rank;

  /// \brief Constructor
  ///
  /// The type of each data object will vary depending on the tag. See the class discussion for more information.
  ///
  /// \param name The string name identifying an instance of the SphereData object
  /// \param bulk_data The bulk data object
  /// \param center_data Center of each sphere (NODE_RANK). (shared type: mundy::math::Vector3<scalar_t>)
/// \param radius_data Radius of each sphere (SAME_RANK_AS_TOPOLOGY). (shared type: scalar_t)
  SphereData(const std::string& name, const stk::mesh::BulkData& bulk_data, const stk::mesh::PartVector& parts,
                  center, radius)
      : SphereDataBase(name, bulk_data, topology_t, parts), center_data_(center_data_), radius_data_(radius_data_) {
    if constexpr (std::is_same_v<CenterDataTag, tag_type::FIELD>) {
  MUNDY_THROW_ASSERT(center_data_->entity_rank() == NODE_RANK, std::invalid_argument,
                     "The center_data data must be a field of NODE_RANK");
} else if constexpr (std::is_same_v<CenterDataTag, tag_type::VECTOR_OF_FIELDS>) {
  for (const auto center_data_field_ptr_ : center_data_) {
    MUNDY_THROW_ASSERT(center_data_field_ptr_->entity_rank() == NODE_RANK, std::invalid_argument,
                       "The center_data data must be a vector of fields of NODE_RANK");
  }
}

if constexpr (std::is_same_v<RadiusDataTag, tag_type::FIELD>) {
  MUNDY_THROW_ASSERT(radius_data_->entity_rank() == rank_t, std::invalid_argument,
                     "The radius_data data must be a field of rank_t");
} else if constexpr (std::is_same_v<RadiusDataTag, tag_type::VECTOR_OF_FIELDS>) {
  for (const auto radius_data_field_ptr_ : radius_data_) {
    MUNDY_THROW_ASSERT(radius_data_field_ptr_->entity_rank() == rank_t, std::invalid_argument,
                       "The radius_data data must be a vector of fields of rank_t");
  }
}
  }

  /// \brief Get the compile-time topology of entities within this aggregate
  static constexpr stk::topology::topology_t get_topology_t() {
    return topology_t;
  }

  /// \brief Get the compile-time rank of entities within this aggregate
  static constexpr stk::topology::rank_t get_rank_t() {
    return rank_t;
  }

  /// \brief Get the tag for the center_data
static constexpr CenterDataTag center_data_tag() {
  return CenterDataTag{};
}

/// \brief Get the tag for the radius_data
static constexpr RadiusDataTag radius_data_tag() {
  return RadiusDataTag{};
}


  /// \brief Get the center_data
const center_data_t& center_data() const {
  return center_data_;
}

/// \brief Get the radius_data
const radius_data_t& radius_data() const {
  return radius_data_;
}


  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto augment_data(Args&&... args) const {
    return NextAugment<SphereData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  auto get_entity_view(stk::mesh::Entity entity) {
    using our_t = SphereData<Scalar, OurTopology, CenterDataTag, RadiusDataTag>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(SphereDataBase::bulk_data(), entity)
        .template augment_view<SphereEntityView, our_t>(*this);
  }

  /// \brief Get the entity view for an entity within this aggregate
  const auto get_entity_view(stk::mesh::Entity entity) const {
    using our_t = SphereData<Scalar, OurTopology, CenterDataTag, RadiusDataTag>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(SphereDataBase::bulk_data(), entity)
        .template augment_view<SphereEntityView, our_t>(*this);
  }

  /// \brief Get the NGP-compatible version of this SphereData object
  auto get_updated_ngp_data() const {
    return create_ngp_ellipsoid_data<scalar_t, OurTopology>(
        stk::mesh::get_updated_ngp_mesh(SphereDataBase::bulk_data()),  //
        list_of_tagged_data_to_ngp_placeholder);
  }

 private:
  center_data_t center_data_;  ///< Center of each sphere (NODE_RANK).
radius_data_t radius_data_;  ///< Radius of each sphere (SAME_RANK_AS_TOPOLOGY).
};  // SphereData

/// \brief Base class for all NgpSphereData objects
///
/// This base class simply hold runtime information about the NgpSphereData object.
class NgpSphereDataBase {
 public:
  /// \brief Constructor
  ///
  /// \param ngp_mesh The NgpMesh object
  /// \param topology The topology of entities within this aggregate
  NgpSphereDataBase(stk::mesh::NgpMesh ngp_mesh, mundy::core::NgpVector<stk::mesh::Part*> parts,
                         stk::topology topology)
      : ngp_mesh_(ngp_mesh), selector_(selector), topology_(topology), rank_(topology.rank()) {
  }

  /// \brief Get the NgpMesh object
  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh& ngp_mesh() {
    return ngp_mesh_;
  }

  /// \brief Get the NgpMesh object
  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh& ngp_mesh() const {
    return ngp_mesh_;
  }

  /// \brief Get the vector of parts of entities within this aggregate
  KOKKOS_INLINE_FUNCTION
  const mundy::core::NgpVector<stk::mesh::Part*>& parts() const {
    return parts_;
  }

  /// \brief Get the topology of entities within this aggregate
  KOKKOS_INLINE_FUNCTION
  stk::topology topology() const {
    return topology_;
  }

  /// \brief Get the rank of entities within this aggregate
  KOKKOS_INLINE_FUNCTION
  stk::topology::rank_t rank() const {
    return rank_;
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;   ///< The NgpMesh object
  stk::mesh::Selector selector_;  ///< The selector for all entities in this aggregate
  stk::topology topology_;        ///< The topology of entities within this aggregate
  stk::topology::rank_t rank_;    ///< The rank of entities within this aggregate
};

/// \brief Aggregate to hold the data for a collection of NGP-compatible ellipsoids
/// See the discussion for SphereData for more information. Only difference is NgpFields over Fields.
///
/// One additional difference is that, we cannot store a reference to host memory, so we store a const copy of any
/// shared data.
template <typename Scalar,       //
          typename OurTopology,  //
          typename CenterDataTag = CENTER_DATA_IS_FIELD, //
typename RadiusDataTag = RADIUS_DATA_IS_FIELD>
class NgpSphereData {
  static_assert(OurTopology::value == NODE || OurTopology::value == PARTICLE,
"The topology of an sphere must be either NODE or PARTICLE");

 public:
  using scalar_t = Scalar;
  using center_data_t = map_tag_to_ngp_data_type_t</* Tag */ CenterDataTag, //
                                           /* Scalar */ scalar_t, //
                                           /* Shared type */ mundy::math::Vector3<scalar_t>>;
using radius_data_t = map_tag_to_ngp_data_type_t</* Tag */ RadiusDataTag, //
                                           /* Scalar */ scalar_t, //
                                           /* Shared type */ scalar_t>;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;
  static constexpr stk::topology::rank_t rank_t = stk::topology_detail::topology_data<OurTopology::value>::rank;

  /// \brief Constructor
  /// \param ngp_mesh The NgpMesh object
  /// \param selector The selector for all entities in this aggregate
  /// \param center_data Center of each sphere (NODE_RANK). (shared type: mundy::math::Vector3<scalar_t>)
/// \param radius_data Radius of each sphere (SAME_RANK_AS_TOPOLOGY). (shared type: scalar_t)
  NgpSphereData(stk::mesh::NgpMesh ngp_mesh, mundy::core::NgpVector<stk::mesh::Part*> parts,
                     center, radius)
      : NgpSphereDataBase(ngp_mesh, parts, topology_t), center_data_(center_data_), radius_data_(radius_data_) {
    if constexpr (std::is_same_v<CenterDataTag, tag_type::FIELD>) {
  MUNDY_THROW_ASSERT(center_data_->get_rank() == NODE_RANK, std::invalid_argument,
                     "The center_data data must be a field of NODE_RANK");
} else if constexpr (std::is_same_v<CenterDataTag, tag_type::VECTOR_OF_FIELDS>) {
  for (const auto center_data_field_ptr_ : center_data_) {
    MUNDY_THROW_ASSERT(center_data_field_ptr_->get_rank() == NODE_RANK, std::invalid_argument,
                       "The center_data data must be a vector of fields of NODE_RANK");
  }
}

if constexpr (std::is_same_v<RadiusDataTag, tag_type::FIELD>) {
  MUNDY_THROW_ASSERT(radius_data_->get_rank() == rank_t, std::invalid_argument,
                     "The radius_data data must be a field of rank_t");
} else if constexpr (std::is_same_v<RadiusDataTag, tag_type::VECTOR_OF_FIELDS>) {
  for (const auto radius_data_field_ptr_ : radius_data_) {
    MUNDY_THROW_ASSERT(radius_data_field_ptr_->get_rank() == rank_t, std::invalid_argument,
                       "The radius_data data must be a vector of fields of rank_t");
  }
}
  }

  /// \brief Get the compile-time topology of entities within this aggregate
  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t get_topology_t() {
    return topology_t;
  }

  /// \brief Get the compile-time rank of entities within this aggregate
  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::rank_t get_rank_t() {
    return rank_t;
  }

  /// \brief Get the tag for the center_data
KOKKOS_INLINE_FUNCTION
static constexpr CenterDataTag center_data_tag() {
  return CenterDataTag{};
}

/// \brief Get the tag for the radius_data
KOKKOS_INLINE_FUNCTION
static constexpr RadiusDataTag radius_data_tag() {
  return RadiusDataTag{};
}


  /// \brief Get the center_data
KOKKOS_INLINE_FUNCTION
const center_data_t& center_data() const {
  return center_data_;
}

/// \brief Get the radius_data
KOKKOS_INLINE_FUNCTION
const radius_data_t& radius_data() const {
  return radius_data_;
}


  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION auto augment_data(Args&&... args) const {
    return NextAugment<NgpSphereData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  auto get_entity_view(stk::mesh::FastMeshIndex entity_index) {
    using our_t = NgpSphereData<Scalar, OurTopology, CenterDataTag, RadiusDataTag>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(NgpSphereDataBase::ngp_mesh(),
                                                                               entity_index)
        .template augment_view<NgpSphereEntityView, our_t>(*this);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  const auto get_entity_view(stk::mesh::FastMeshIndex entity_index) const {
    using our_t = NgpSphereData<Scalar, OurTopology, CenterDataTag, RadiusDataTag>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(NgpSphereDataBase::ngp_mesh(),
                                                                               entity_index)
        .template augment_view<NgpSphereEntityView, our_t>(*this);
  }

 private:
  center_data_t center_data_;  ///< Center of each sphere (NODE_RANK).
radius_data_t radius_data_;  ///< Radius of each sphere (SAME_RANK_AS_TOPOLOGY).
};  // NgpSphereData

/// \brief A helper function to create an SphereData object
///
/// This function creates a SphereData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
///
/// Each of the provided pieces of data may be
///   - a field pointer: stk::mesh::Field<Scalar>*
///   - a vector of field pointers: std::vector<stk::mesh::Field<Scalar>*>
///   - a single shared value: SharedType
///   - a vector of shared values: std::vector<SharedType>
///
/// The actual SharedType for each of the data objects is stated in their doc strings.
///
/// \param bulk_data The bulk data object
/// \param parts The parts of entities within this aggregate
/// \param center_data Center of each sphere (NODE_RANK). (shared type: mundy::math::Vector3<scalar_t>)
/// \param radius_data Radius of each sphere (SAME_RANK_AS_TOPOLOGY). (shared type: scalar_t)
template <typename Scalar,                           // Must be provided
          stk::topology::topology_t OurTopology,     // Must be provided
          CenterDataType, RadiusDataType>  // deduced
auto create_sphere_data(const stk::mesh::BulkData& bulk_data, const stk::mesh::PartVector& parts,
                              const CenterDataType &center_data, const RadiusDataType &radius_data) {
  using CenterDataTag = map_data_type_to_tag_t<CenterDataType>;
using RadiusDataTag = map_data_type_to_tag_t<RadiusDataType>;
  return SphereData<Scalar, stk::topology_detail::topology_data<OurTopology>, CenterDataTag, RadiusDataTag>{
      bulk_data, parts, center, radius};
}

/// \brief A helper function to create an NgpSphereData object
/// See the discussion for create_sphere_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                           // Must be provided
          stk::topology::topology_t OurTopology,     // Must be provided
          CenterDataType, RadiusDataType>  // deduced
auto create_ngp_sphere_data(const stk::mesh::NgpMesh& ngp_mesh, const CenterDataType &center_data, const RadiusDataType &radius_data) {
  using CenterDataTag = map_ngp_data_type_to_tag_t<CenterDataType>;
using RadiusDataTag = map_ngp_data_type_to_tag_t<RadiusDataType>;
  return NgpSphereData<Scalar, stk::topology_detail::topology_data<OurTopology>, CenterDataTag, RadiusDataTag>{
      ngp_mesh, center, radius};
}

/// \brief A helper function to get an updated NgpSphereData object from a SphereData object
/// \param data The SphereData object to convert
template <typename Scalar, typename OurTopology, typename CenterDataTag, typename RadiusDataTag>
auto get_updated_ngp_data(const SphereData<Scalar, OurTopology, CenterDataTag, RadiusDataTag>& data) {
  return data.get_updated_ngp_data();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_SPHEREDATA_HPP_
