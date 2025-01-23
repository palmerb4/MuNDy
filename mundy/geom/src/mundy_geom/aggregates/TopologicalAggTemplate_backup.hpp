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

#ifndef MUNDY_GEOM_AGGREGATES_EXAMPLENAMEDATA_HPP_
#define MUNDY_GEOM_AGGREGATES_EXAMPLENAMEDATA_HPP_

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
#include <mundy_geom/aggregates/ExampleNameDataConcepts.hpp>  // for mundy::geom::ValidExampleNameDataType
#include <mundy_geom/aggregates/ExampleNameEntityView.hpp>    // for mundy::geom::ExampleNameEntityView
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

//! \name Aggregate traits
//@{

/// \brief Base class for all ExampleNameData objects
///
/// This base class simply hold runtime information about the ExampleNameData object.
class ExampleNameDataBase {
 public:
  /// \brief Constructor
  ///
  /// \param name The string name identifying an instance of the ExampleNameData object
  /// \param bulk_data The bulk data object
  /// \param topology The topology of entities within this aggregate
  ExampleNameDataBase(const std::string& name, const stk::mesh::BulkData& bulk_data, stk::topology topology,
                      const stk::mesh::PartVector& parts)
      : name_(name), bulk_data_(bulk_data), topology_(topology), rank_(topology_.rank()), parts_(parts) {
  }

  /// \brief Get the name of the ExampleNameData object
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
  const std::string name_;                ///< The name of the ExampleNameData object
  const stk::mesh::BulkData& bulk_data_;  ///< The bulk data object
  stk::topology topology_;                ///< The topology of entities within this aggregate
  stk::topology::rank_t rank_;            ///< The rank of entities within this aggregate
};

/// \brief Aggregate to hold the data for a collection of example_name
///
/// example_discussion_placeholder
///
/// Shared data is stored as a const ref to the original data and is therefore unmodifiable. This is to prevent
/// accidental non-thread-safe modifications to the shared data. If this is every limiting, let us know and we can
/// consider adding a Kokkos::View to the shared data.
///
/// The tags for the field types are used to control the type of data stored in the ExampleNameData object.
///   - HAS_*_FIELD: The data is a field pointer: stk::mesh::Field<Scalar>*
///   - HAS_*_VECTOR_OF_FIELDS: The data is a vector of field pointers: std::vector<stk::mesh::Field<Scalar>*>
///   - HAS_*_SHARED: A single shared value is stored : SharedType
///   - HAS_*_VECTOR_OF_SHARED: A vector of shared values is stored: std::vector<SharedType>
///
/// Use \ref create_example_name_data to build an ExampleNameData object with automatic template deduction.
template <typename Scalar,       //
          typename OurTopology,  //
          list_of_templated_tags_with_defaults_placeholder>
class ExampleNameData : public ExampleNameDataBase {
  static_assert_valid_topology_placeholder;

 public:
  using scalar_t = Scalar;
  data_type_aliases_placeholder;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;
  static constexpr stk::topology::rank_t rank_t = stk::topology_detail::topology_data<OurTopology::value>::rank;

  /// \brief Constructor
  ///
  /// The type of each data object will vary depending on the tag. See the class discussion for more information.
  ///
  /// \param name The string name identifying an instance of the ExampleNameData object
  /// \param bulk_data The bulk data object
  /// params_documentation_placeholder
  ExampleNameData(const std::string& name, const stk::mesh::BulkData& bulk_data, const stk::mesh::PartVector& parts,
                  list_of_input_data_placeholder)
      : ExampleNameDataBase(name, bulk_data, topology_t, parts), list_of_input_data_initialization_placeholder {
    assert_correct_rank_for_example_data1_placeholder;
    assert_correct_rank_for_example_data2_placeholder;
  }

  /// \brief Get the compile-time topology of entities within this aggregate
  static constexpr stk::topology::topology_t get_topology_t() {
    return topology_t;
  }

  /// \brief Get the compile-time rank of entities within this aggregate
  static constexpr stk::topology::rank_t get_rank_t() {
    return rank_t;
  }

  getters_for_tags_placeholder;

  getters_for_data_placeholder;

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent the boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto augment_data(Args&&... args) const {
    return NextAugment<ExampleNameData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  auto get_entity_view(stk::mesh::Entity entity) {
    using our_t = ExampleNameData<Scalar, OurTopology, list_of_tags_placeholder>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(ExampleNameDataBase::bulk_data(), entity)
        .template augment_view<ExampleNameEntityView, our_t>(*this);
  }

  /// \brief Get the entity view for an entity within this aggregate
  const auto get_entity_view(stk::mesh::Entity entity) const {
    using our_t = ExampleNameData<Scalar, OurTopology, list_of_tags_placeholder>;
    return mundy::geom::create_topological_entity_view<OurTopology::value>(ExampleNameDataBase::bulk_data(), entity)
        .template augment_view<ExampleNameEntityView, our_t>(*this);
  }

  /// \brief Get the NGP-compatible version of this ExampleNameData object
  auto get_updated_ngp_data() const {
    return create_ngp_ellipsoid_data<scalar_t, OurTopology>(
        stk::mesh::get_updated_ngp_mesh(ExampleNameDataBase::bulk_data()),  //
        list_of_tagged_data_to_ngp_placeholder);
  }

 private:
  list_of_internal_data_placeholder;
};  // ExampleNameData

/// \brief Base class for all NgpExampleNameData objects
///
/// This base class simply hold runtime information about the NgpExampleNameData object.
class NgpExampleNameDataBase {
 public:
  /// \brief Constructor
  ///
  /// \param ngp_mesh The NgpMesh object
  /// \param topology The topology of entities within this aggregate
  NgpExampleNameDataBase(stk::mesh::NgpMesh ngp_mesh, mundy::core::NgpVector<stk::mesh::Part*> parts,
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
/// See the discussion for ExampleNameData for more information. Only difference is NgpFields over Fields.
///
/// One additional difference is that, we cannot store a reference to host memory, so we store a const copy of any
/// shared data.
template <typename Scalar,       //
          typename OurTopology,  //
          list_of_templated_tags_with_defaults_placeholder>
class NgpExampleNameData {
  static_assert_valid_topology_placeholder;

 public:
  using scalar_t = Scalar;
  data_type_ngp_aliases_placeholder;
  static constexpr stk::topology::topology_t topology_t = OurTopology::value;
  static constexpr stk::topology::rank_t rank_t = stk::topology_detail::topology_data<OurTopology::value>::rank;

  /// \brief Constructor
  /// \param ngp_mesh The NgpMesh object
  /// \param selector The selector for all entities in this aggregate
  /// params_documentation_placeholder
  NgpExampleNameData(stk::mesh::NgpMesh ngp_mesh, mundy::core::NgpVector<stk::mesh::Part*> parts,
                     list_of_input_data_placeholder)
      : NgpExampleNameDataBase(ngp_mesh, parts, topology_t), list_of_input_data_initialization_placeholder {
    assert_correct_rank_for_input_ngp_data_placeholder;
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

  getters_for_ngp_tags_placeholder;

  getters_for_ngp_data_placeholder;

  /// \brief Chainable function to add augments to this aggregate
  ///
  /// \note Aggregates may ~not~ be templated by non-type template parameters. This is not overly limiting, as you
  ///  simply need to introduce a wrapper class to hold the non-type template parameters. For example, use
  ///  std::true_type and std::false_type to represent boolean template parameters.
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION auto augment_data(Args&&... args) const {
    return NextAugment<NgpExampleNameData, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  auto get_entity_view(stk::mesh::FastMeshIndex entity_index) {
    using our_t = NgpExampleNameData<Scalar, OurTopology, list_of_tags_placeholder>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(NgpExampleNameDataBase::ngp_mesh(),
                                                                               entity_index)
        .template augment_view<NgpExampleNameEntityView, our_t>(*this);
  }

  /// \brief Get the entity view for an entity within this aggregate
  KOKKOS_INLINE_FUNCTION
  const auto get_entity_view(stk::mesh::FastMeshIndex entity_index) const {
    using our_t = NgpExampleNameData<Scalar, OurTopology, list_of_tags_placeholder>;
    return mundy::geom::create_ngp_topological_entity_view<OurTopology::value>(NgpExampleNameDataBase::ngp_mesh(),
                                                                               entity_index)
        .template augment_view<NgpExampleNameEntityView, our_t>(*this);
  }

 private:
  list_of_internal_data_placeholder;
};  // NgpExampleNameData

/// \brief A helper function to create an ExampleNameData object
///
/// This function creates a ExampleNameData object given its rank and data (be they shared or field data)
/// and is used to automatically deduce the template parameters.
///
/// \param bulk_data The bulk data object
/// \param parts The parts of entities within this aggregate
///

template <typename Scalar,                           // Must be provided
          stk::topology::topology_t OurTopology,     // Must be provided
          list_of_templated_data_types_placeholder>  // deduced
auto create_ellipsoid_data(const stk::mesh::BulkData& bulk_data, const stk::mesh::PartVector& parts,
                           list_of_type_deduced_data_placeholder) {
  type_to_tag_deduction_placeholder;
  return ExampleNameData<Scalar, stk::topology_detail::topology_data<OurTopology>, list_of_tags_placeholder>{
      bulk_data, parts, list_of_data_placeholder};
}

/// \brief A helper function to create an NgpExampleNameData object
/// See the discussion for create_ellipsoid_data for more information. Only difference is NgpFields over Fields.
template <typename Scalar,                           // Must be provided
          stk::topology::topology_t OurTopology,     // Must be provided
          list_of_templated_data_types_placeholder>  // deduced
auto create_ngp_ellipsoid_data(const stk::mesh::NgpMesh& ngp_mesh, list_of_type_deduced_data_placeholder) {
  type_to_ngp_tag_deduction_placeholder;
  return NgpExampleNameData<Scalar, stk::topology_detail::topology_data<OurTopology>, list_of_tags_placeholder>{
      ngp_mesh, list_of_data_placeholder};
}

/// \brief A helper function to get an updated NgpExampleNameData object from a ExampleNameData object
/// \param data The ExampleNameData object to convert
template <typename Scalar, typename OurTopology, list_of_templated_tags_placeholder>
auto get_updated_ngp_data(const ExampleNameData<Scalar, OurTopology, list_of_tags_placeholder>& data) {
  return data.get_updated_ngp_data();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_EXAMPLENAMEDATA_HPP_
