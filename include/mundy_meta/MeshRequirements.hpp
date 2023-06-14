// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
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

#ifndef MUNDY_META_MESHREQUIREMENTS_HPP_
#define MUNDY_META_MESHREQUIREMENTS_HPP_

/// \file MeshRequirements.hpp
/// \brief Declaration of the MeshRequirements class

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part
#include <stk_mesh/base/Types.hpp>     // for EntityRank, etc
#include <stk_topology/topology.hpp>   // for stk::topology

// Boost libs
#include <boost/hana.hpp>

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements, mundy::meta::FieldRequirementsBase
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

//! \name Shorthand types
//@{

/// @brief Shorthand type for an empty heterogeneous map.
using EmptyMap = decltype(boost::hana::make_map());

/// @brief Shorthand type for an empty tuple.
using EmptyTuple = decltype(boost::hana::make_tuple());
//@}

/// \class MeshRequirements
/// \brief A set of requirements imposed upon the structure and contents of a MetaMesh.
///
/// Note, the templates for this class are an implementation detail and something that you need not specify directly;
/// instead, the easiest way to interact with this class is to chain function calls:
///     MeshRequirements reqs;
///     reqs.declare_part("parent")
///         .declare_part("child")
///         .declare_part_subset("parent", "child");
/// However, if you wish to break up construction into multiple steps, you'll need to introduce a new \c
/// MeshRequirements object by using "auto." It may feel tempting to write reqs = reqs.declare_part("parent"), but this
/// will throw an error since declare_part returns a \c MeshRequirements instance that is larger than the original reqs
/// object. Instead, use
///     MeshRequirements reqs;
///     auto reqs_new = reqs.declare_part("parent");
///     /* other stuff */
///     reqs_new.declare_part("child")
///             .declare_part_subset("parent", "child");
/// At first glance, one might also think that returning different a succession of ever-growing \c MeshRequirements
/// classes would lead to tons of undesirable copies. For example, one could think that an intermediate copy is made
/// after each call to declare_part:
///     MeshRequirements reqs;
///     reqs.declare_part("parent")                   /* copy here? */
///         .declare_part("child")                    /* copy here? */
///         .declare_part_subset("parent", "child");  /* copy here? */
/// Instead, Mundy uses move semantics with perfect forwarding to reduce unnecessary copies. As a result, the actual
/// behavior of \c MeshRequirements is:
///     MeshRequirements reqs;
///     reqs.declare_part("parent")  /* copy here to preserve reqs for later use */
///         .declare_part("child")   /* move here since the output from reqs.declare_part("parent") is never stored */
///         .declare_part_subset("parent", "child");  /* move here ... */
template <typename FieldIdMap = EmptyMap, typename PartIdMap = EmptyMap, typename AttributeIdMap = EmptyMap,
          typename IdToFieldMap = EmptyMap, typename IdToPartMap = EmptyMap, typename MeshAttributes = EmptyTuple>
class MeshRequirements {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed.
  /// Default construction corresponds to having no requirements and is perfectly valid.
  MeshRequirements() = default;

  /// \brief Copy constructor.
  MeshRequirements(const MeshRequirements<FieldIdMap, PartIdMap, AttributeIdMap, IdToFieldMap, IdToPartMap,
                                          MeshAttributes> &other_reqs)
      : field_id_map_(other_reqs.field_id_map_),
        part_id_map_(other_reqs.part_id_map_),
        att_id_map_(other_reqs.att_id_map_),
        id_to_field_map_(other_reqs.id_to_field_map_),
        id_to_part_map_(other_reqs.id_to_part_map_),
        mesh_atts_(other_reqs.mesh_atts_) {
  }

  /// \brief Move constructor.
  MeshRequirements(
      MeshRequirements<FieldIdMap, PartIdMap, AttributeIdMap, IdToFieldMap, IdToPartMap, MeshAttributes> &&other_reqs)
      : field_id_map_(std::move(other_reqs.field_id_map_)),
        part_id_map_(std::move(other_reqs.part_id_map_)),
        att_id_map_(std::move(other_reqs.att_id_map_)),
        id_to_field_map_(std::move(other_reqs.id_to_field_map_)),
        id_to_part_map_(std::move(other_reqs.id_to_part_map_)),
        mesh_atts_(std::move(other_reqs.mesh_atts_)) {
  }

  /// \brief Destructor.
  ~MeshRequirements() = default;
  //@}

  //! \name Actions
  //@{

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means:
  ///   - TODO(palmerb4): What are the mesh invariants set by STK?
  void check_if_valid() const {
  }

  /// \brief Declare a part with a given rank. It may explicitly contain any entity of lower rank with optional forced
  /// induction.
  ///
  /// Redeclaration of a previously declared part is perfectly valid given that they have the same rank and induction.
  /// If the existing part only has its name set, then redeclaration will set the rank and arg_no_force. Otherwise, we
  /// check for compatibility.
  ///
  /// \param part_name Name of the part.
  /// \param rank Maximum rank of entities in the part.
  /// \param arg_force_no_induce Flag specifying if sub-entities of part members should not become induced members.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  constexpr auto declare_part(const std::string_view &part_name, const stk::mesh::EntityRank &rank,
                              const bool &arg_force_no_induce = false) const & {
    // Check if the part already exists.
    if constexpr (boost::hana::contains(new_part_id_map, part_name)) {
      // The part exists, check for compatibility.
      auto existing_part = boost::hana::at_key(new_part_id_map, part_name);
      static_assert(existing_part.state == PartRequirements::NAME_AND_TOPOLOGY_SET,
                    "MeshRequirements: Attempting to redeclare a part with a given rank; \n"
                        << "however, the part was previously declared with a given topology.");
      static_assert(existing_part.state == PartRequirements::INVALID_STATE,
                    "MeshRequirements: Attempting to redeclare a part with a given rank; \n"
                        << "however, the previously declared part is currently invalid. \n"
                        << "Odd... please contact the development team.");
      static_assert(existing_part.arg_force_no_induce == arg_force_no_induce,
                    "MeshRequirements: Attempting to redeclare a part with a given induction flag ( "
                        << arg_force_no_induce << " ); \n"
                        << "however, the part was previously declared with a different induction flag ( "
                        << existing_part.arg_force_no_induce << ").");

      // Create the updated part requirements and replace the old one. 
      // Uses move semantics to avoid copying.
      PartRequirements pr{part_name, stk::topology::INVALID_TOPOLOGY, rank, arg_force_no_induce,
                          PartRequirements::NAME_AND_RANK_SET};

      // Plundering not allowed, creates a copy of the internal members of *this.
      auto new_id_to_part_map = boost::hana::erase_key(id_to_part_map_, part_id_map_[part_name]);

      MeshRequirements<FieldIdMap, NewPartIdMap, AttributeIdMap, IdToFieldMap, NewIdToPartMap, MeshAttributes>(
          field_id_map_, part_id_map, att_id_map_, id_to_field_map_, id_to_part_map, mesh_atts_);
      using new_tuple_type = decltype(new_tuple);
      return MapWrapper<new_tuple_type>{new_tuple};
    } else {

      auto new_part_id_map = boost::hana::append(part_id_map_, std::forward<T>(t));
    }
  }

  /// \brief Declare a part with a given rank. It may explicitly contain any entity of lower rank with optional forced
  /// induction.
  ///
  /// Redeclaration of a previously declared part is perfectly valid given that they have the same rank and induction.
  /// In this sense, redeclaration is a no-op with a compatibility check.
  ///
  /// \param part_name Name of the part.
  /// \param rank Maximum rank of entities in the part.
  /// \param arg_force_no_induce Flag specifying if sub-entities of part members should not become induced members.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  constexpr auto declare_part(const std::string_view &part_name, const stk::mesh::EntityRank &rank,
                              const bool &arg_force_no_induce = false) && {
  }

  /// \brief Declare a part with given topology. It may contain any element with the given topology with optional forced
  /// induction of downward connected entities.
  ///
  /// Redeclaration of a previously declared part is perfectly valid given that they have the same topology and
  /// induction. In this sense, redeclaration is a no-op with a compatibility check.
  ///
  /// \param part_name Name of the part.
  /// \param topology Topology of entities in the part.
  /// \param arg_force_no_induce Flag specifying if sub-entities of part members should not become induced members.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  constexpr auto declare_part_with_topology(const std::string_view &part_name,
                                            const stk::topology::topology_t &topology,
                                            const bool &arg_force_no_induce = false) const & {
  }

  /// \brief Declare a part with given topology. It may contain any element with the given topology with optional forced
  /// induction of downward connected entities.
  ///
  /// Redeclaration of a previously declared part is perfectly valid given that they have the same topology and
  /// induction. In this sense, redeclaration is a no-op with a compatibility check.
  ///
  /// \param part_name Name of the part.
  /// \param topology Topology of entities in the part.
  /// \param arg_force_no_induce Flag specifying if sub-entities of part members should not become induced members.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  constexpr auto declare_part_with_topology(const std::string_view &part_name,
                                            const stk::topology::topology_t &topology,
                                            const bool &arg_force_no_induce = false) &&;

  /// \brief Declare a part without rank or topology (these will need set later).
  ///
  /// Redeclaration of a previously declared part is perfectly valid. In this sense, redeclaration is a no-op with a
  /// compatibility check.
  ///
  /// \param part_name Name of the part.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  constexpr auto declare_part(const std::string_view &part_name) const &;

  /// \brief Declare a part without rank or topology (these will need set later).
  ///
  /// Redeclaration of a previously declared part is perfectly valid. In this sense, redeclaration is a no-op with a
  /// compatibility check.
  ///
  /// \param part_name Name of the part.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  constexpr auto declare_part(const std::string_view &part_name) &&;

  /// \brief Declare a subset relation between two parts.
  ///
  /// An important comment: If you do specify verifyFieldRestrictions = true, this check will be delayed until the
  /// entire mesh is constructed.
  ///
  /// Redeclaration of a previously declared subset relation is perfectly valid. There's no need to force two
  /// declarations to have the same verifyFieldRestrictions, so if any declaration sets this flag to true, then the
  /// check will be performed. In this sense, redeclaration will update verifyFieldRestrictions and check compatibility.
  ///
  /// \param superset_part_name Name of the parent/superset part.
  /// \param subset_part_name Name of the child/subset part.
  /// \param verifyFieldRestrictions Flag specifying if STK should validate the field restriction for the parts.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  constexpr auto declare_part_subset(const std::string_view &superset_part_name,
                                     const std::string_view &subset_part_name,
                                     const bool &verifyFieldRestrictions = true) const &;

  /// \brief Declare a subset relation between two parts.
  ///
  /// An important comment: If you do specify verifyFieldRestrictions = true, this check will be delayed until the
  /// entire mesh is constructed.
  ///
  /// Redeclaration of a previously declared subset relation is perfectly valid. There's no need to force two
  /// declarations to have the same verifyFieldRestrictions, so if any declaration sets this flag to true, then the
  /// check will be performed. In this sense, redeclaration will update verifyFieldRestrictions and check compatibility.
  ///
  /// \param superset_part_name Name of the parent/superset part.
  /// \param subset_part_name Name of the child/subset part.
  /// \param verifyFieldRestrictions Flag specifying if STK should validate the field restriction for the parts.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  constexpr auto declare_part_subset(const std::string_view &superset_part_name,
                                     const std::string_view &subset_part_name,
                                     const bool &verifyFieldRestrictions = true) &&;

  /// \brief Declare a part attribute with the given type.
  ///
  /// An important comment: Notice that we only pass a type to this interface and not an instance of the attribute. When
  /// the mesh is constructed, STK will generate an internal attribute with the given type, the instance of which can be
  /// fetched as an AttributeType pointer. This pointer will initially be a nullptr.
  ///
  /// Redeclaration of a previously declared AttributeType is perfectly valid and will simply perform a no-op.
  ///
  /// \tparam AttributeType The attribute type to store on the mesh.
  /// \param part_name Name of the part to store the attribute on.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  template <typename AttributeType>
  constexpr auto declare_attribute(const std::string_view &part_name) const & {
  }

  /// \brief Declare a part attribute with the given type.
  ///
  /// An important comment: Notice that we only pass a type to this interface and not an instance of the attribute. When
  /// the mesh is constructed, STK will generate an internal attribute with the given type, the instance of which can be
  /// fetched as an AttributeType pointer. This pointer will initially be a nullptr.
  ///
  /// Redeclaration of a previously declared AttributeType is perfectly valid and will simply perform a no-op.
  ///
  /// \tparam AttributeType The attribute type to store on the mesh.
  /// \param part_name Name of the part to store the attribute on.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  template <typename AttributeType>
  constexpr auto declare_attribute(const std::string_view &part_name) && {
  }

  /// \brief Declare a field attribute with the given type.
  ///
  /// An important comment: Notice that we only pass a type to this interface and not an instance of the attribute. When
  /// the mesh is constructed, STK will generate an internal attribute with the given type, the instance of which can be
  /// fetched as an AttributeType pointer. This pointer will initially be a nullptr.
  ///
  /// Redeclaration of a previously declared AttributeType is perfectly valid and will simply perform a no-op.
  ///
  /// \tparam AttributeType The attribute type to store on the mesh.
  /// \param field_name Name of the field to store the attribute on.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  template <typename AttributeType>
  constexpr auto declare_attribute(const std::string_view &field_name) const & {
  }

  /// \brief Declare a field attribute with the given type.
  ///
  /// An important comment: Notice that we only pass a type to this interface and not an instance of the attribute. When
  /// the mesh is constructed, STK will generate an internal attribute with the given type, the instance of which can be
  /// fetched as an AttributeType pointer. This pointer will initially be a nullptr.
  ///
  /// Redeclaration of a previously declared AttributeType is perfectly valid and will simply perform a no-op.
  ///
  /// \tparam AttributeType The attribute type to store on the mesh.
  /// \param field_name Name of the field to store the attribute on.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  template <typename AttributeType>
  constexpr auto declare_attribute(const std::string_view &field_name) && {
  }

  /// \brief Declare a mesh attribute with the given type.
  ///
  /// An important comment: Notice that we only pass a type to this interface and not an instance of the attribute. When
  /// the mesh is constructed, STK will generate an internal attribute with the given type, the instance of which can be
  /// fetched as an AttributeType pointer. This pointer will initially be a nullptr.
  ///
  /// Redeclaration of a previously declared AttributeType is perfectly valid and will simply perform a no-op.
  ///
  /// \tparam AttributeType The attribute type to store on the mesh.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  template <typename AttributeType>
  constexpr auto declare_attribute() const & {
  }

  /// \brief Declare a mesh attribute with the given type.
  ///
  /// An important comment: Notice that we only pass a type to this interface and not an instance of the attribute. When
  /// the mesh is constructed, STK will generate an internal attribute with the given type, the instance of which can be
  /// fetched as an AttributeType pointer. This pointer will initially be a nullptr.
  ///
  /// Redeclaration of a previously declared AttributeType is perfectly valid and will simply perform a no-op.
  ///
  /// \tparam AttributeType The attribute type to store on the mesh.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  template <typename AttributeType>
  constexpr auto declare_attribute() && {
  }

  /// \brief Declare a new field.
  ///
  /// Redeclaration of a previously declared field (that is, declaring a field with the same name and rank) is perfectly
  /// valid. Note, redeclaration with a larger min_number_of_states will overwrite the previous requested minimum number
  /// of states. This behavior is consistant with the definition of number of states being the last N values of the
  /// field. If one algorithm requires N states and another M > N, then storing M states won't effect the first
  /// algorithm.
  ///
  /// * There are certain limitations on what can or cannot be a field type.
  /// TODO(palmerb4): What are these are these requirements explicitly?
  ///
  /// \tparam FieldType The field type*
  /// \param entity_rank The rank of entities associated with this field.
  /// \param field_name The name of the field.
  /// \param dimension The dimension of the field. For example, a dimension of three would be a vector.
  /// \param min_number_of_states The minimum number of previous field values that STK should maintain.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  template <typename FieldType>
  constexpr auto declare_field(const stk::topology::rank_t &entity_rank, const std::string_view &field_name,
                               const unsigned dimension, const unsigned &min_number_of_states = 1) const & {
  }

  /// \brief Declare a new field.
  ///
  /// Redeclaration of a previously declared field (that is, declaring a field with the same name and rank) is perfectly
  /// valid. Note, redeclaration with a larger min_number_of_states will overwrite the previous requested minimum number
  /// of states. This behavior is consistant with the definition of number of states being the last N values of the
  /// field. If one algorithm requires N states and another M > N, then storing M states won't effect the first
  /// algorithm.
  ///
  /// * There are certain limitations on what can or cannot be a field type.
  /// TODO(palmerb4): What are these are these requirements explicitly?
  ///
  /// \tparam FieldType
  /// \param entity_rank The rank of entities associated with this field.
  /// \param field_name The name of the field.
  /// \param dimension The dimension of the field. For example, a dimension of three would be a vector.
  /// \param min_number_of_states The minimum number of previous field values that STK should maintain.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  template <typename FieldType>
  constexpr auto declare_field(const stk::topology::rank_t &entity_rank, const std::string_view &field_name,
                               const unsigned dimension, const unsigned &min_number_of_states = 1) && {
  }

  /// \brief Put an already-declared field on an already-declared part.
  ///
  /// Redeclaration of an existing field-part connection, is perfectly valid and will perform a no-op.
  ///
  /// \param field_name The name of an already-declared field.
  /// \param part_name The name of an already-declared part, which should contain said field.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  constexpr auto put_field_on_mesh(const std::string_view &field_name, const std::string_view &part_name) const & {
  }

  /// \brief Put an already-declared field on an already-declared part.
  ///
  /// Redeclaration of an existing field-part connection, is perfectly valid and will perform a no-op.
  ///
  /// \param field_name The name of an already-declared field.
  /// \param part_name The name of an already-declared part, which should contain said field.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  constexpr auto put_field_on_mesh(const std::string_view &field_name, const std::string_view &part_name) && {
  }

  /// \brief Merge the current requirements with another \c MeshRequirements.
  ///
  /// The merge process (effectively) involves the following:
  ///   - call declare_attribute for each field/part/mesh attribute in the other reqs.
  ///   - call declare_part for each part in the other reqs.
  ///   - call declare_part_subset for each subset relation in the other reqs.
  ///   - call declare_field for each field in the other reqs.
  ///
  /// As an example, consider the following example:
  ///     // Create and merge two sets of requirements.
  ///     MeshRequirements reqs1;
  ///     reqs1.declare_part("animal")
  ///         .declare_field<double>(stk::topology::ENTITY_RANK, "height", 1)
  ///         .put_field_on_mesh("height", "animal");
  ///     MeshRequirements reqs2;
  ///     reqs2.declare_part("animal")
  ///          .declare_field<double>(stk::topology::ENTITY_RANK, "height", 2)
  ///          .put_field_on_mesh("height", "animal")
  ///          .declare_part("cat")
  ///          .declare_part_subset("animal", "cat");
  ///     auto merged_reqs = reqs1.merge(reqs2);
  ///     // Create the equivalent requirements for comparison.
  ///     MeshRequirements reqs3;
  ///     reqs3.declare_part("animal")
  ///          .declare_field<double>(stk::topology::ENTITY_RANK, "height", 2)
  ///          .put_field_on_mesh("height", "animal")
  ///          .declare_part("cat")
  ///          .declare_part_subset("animal", "cat");
  ///     static_assert(merged_reqs == reqs3);
  ///
  /// \tparam OtherMeshRequirements The type of the other MeshRequirements.
  /// \param other_reqs [in] Some other \c MeshRequirements object to merge with the current object.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  template <typename OtherMeshRequirements>
  constexpr auto merge(OtherMeshRequirements &&other_reqs) const & {
  }

  /// \brief Merge the current requirements with another \c MeshRequirements.
  ///
  /// The merge process (effectively) involves the following:
  ///   - call declare_attribute for each field/part/mesh attribute in the other reqs.
  ///   - call declare_part for each part in the other reqs.
  ///   - call declare_part_subset for each subset relation in the other reqs.
  ///   - call declare_field for each field in the other reqs.
  ///
  /// As an example, consider the following example:
  ///     // Create and merge two sets of requirements.
  ///     MeshRequirements reqs1;
  ///     reqs1.declare_part("animal")
  ///         .declare_field<double>(stk::topology::ENTITY_RANK, "height", 1)
  ///         .put_field_on_mesh("height", "animal");
  ///     MeshRequirements reqs2;
  ///     reqs2.declare_part("animal")
  ///          .declare_field<double>(stk::topology::ENTITY_RANK, "height", 2)
  ///          .put_field_on_mesh("height", "animal")
  ///          .declare_part("cat")
  ///          .declare_part_subset("animal", "cat");
  ///     auto merged_reqs = reqs1.merge(reqs2);
  ///     // Create the equivalent requirements for comparison.
  ///     MeshRequirements reqs3;
  ///     reqs3.declare_part("animal")
  ///          .declare_field<double>(stk::topology::ENTITY_RANK, "height", 2)
  ///          .put_field_on_mesh("height", "animal")
  ///          .declare_part("cat")
  ///          .declare_part_subset("animal", "cat");
  ///     static_assert(merged_reqs == reqs3);
  ///
  /// \tparam OtherMeshRequirements The type of the other MeshRequirements.
  /// \param other_reqs [in] Some other \c MeshRequirements object to merge with the current object.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  template <typename OtherMeshRequirements>
  constexpr auto merge(OtherMeshRequirements &&other_reqs) && {
  }
  //@}

 private:
  //! \name Private constructors
  //@{

  /// @brief Constructor will full fill.
  /// @param field_id_map A map from field name and rank to field ordinal.
  /// @param part_id_map A map from part name part ordinal.
  /// @param att_id_map A map from attribute typeid attribute ordinal.
  /// @param id_to_field_map A map from field ordinal to the actual field requirements.
  /// @param id_to_part_map A map from part ordinal to the actual part requirements.
  /// @param mesh_atts A tuple of mesh attributes (stored as att ordinals).
  MeshRequirements(FieldIdMap field_id_map, PartIdMap part_id_map, AttributeIdMap att_id_map,
                   IdToFieldMap id_to_field_map, IdToPartMap id_to_part_map, MeshAttributes mesh_atts)
      : field_id_map_(field_id_map),
        part_id_map_(part_id_map),
        att_id_map_(att_id_map),
        id_to_field_map_(id_to_field_map),
        id_to_part_map_(id_to_part_map),
        mesh_atts_(mesh_atts) {
  }

  //@}

  //! \name Private helper classes
  //@{

  struct PartRequirements {
    enum state_t : int8_t { INVALID_STATE, NAME_SET, NAME_AND_RANK_SET, NAME_AND_TOPOLOGY_SET };
    std::string_view name;
    stk::topology_topology_t topology;
    stk::mesh::EntityRank rank;
    bool arg_force_no_induce;
    state_t state = INVALID_STATE;
  };  // PartRequirements

  template <typename FieldType>
  struct FieldRequirements {
    stk::topology::rank_t entity_rank;
    std::string_view field_name;
    unsigned dimension;
    unsigned min_number_of_states;
    using type = FieldType;
  };  // FieldRequirements
  //@}

  //! \name Internal members
  //@{

  /// \brief A map from field name and rank to field ordinal.
  FieldIdMap field_id_map_;

  /// \brief A map from part name part ordinal.
  PartIdMap part_id_map_;

  /// \brief A map from attribute typeid attribute ordinal.
  AttributeIdMap att_id_map_;

  /// \brief A map from field ordinal to the actual field requirements.
  IdToFieldMap id_to_field_map_;

  /// \brief A map from part ordinal to the actual part requirements.
  IdToPartMap id_to_part_map_;

  /// \brief A tuple of mesh attributes (stored as att ordinals).
  MeshAttributes mesh_atts_;
  //@}
};  // MeshRequirements

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_MESHREQUIREMENTS_HPP_
