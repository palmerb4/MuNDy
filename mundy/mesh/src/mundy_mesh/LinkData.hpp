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

#ifndef MUNDY_MESH_LINKDATA_HPP_
#define MUNDY_MESH_LINKDATA_HPP_

/// \file LinkData.hpp
/// \brief Declaration of the LinkData class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>        // for stk::mesh::Field
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/GetNgpMesh.hpp>   // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh
#include <stk_mesh/base/Part.hpp>         // stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>     // stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>        // for stk::mesh::EntityRank

// Mundy libs
#include <mundy_core/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>          // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>           // for mundy::mesh::field_copy
#include <mundy_mesh/impl/LinkedBucketConn.hpp>  // for mundy::mesh::impl::LinkedBucketConn

namespace mundy {

namespace mesh {

//! \name Forward declarations
//@{

class LinkData;
class NgpLinkData;
//@}

/// \class LinkData
/// \brief The main interface for interacting with the link data on the mesh.
///
/// # Links vs. Connectivity
/// Unlike STK's connectivity, links
///   - do not induce changes to part membership, ownership, or sharing
///   - have no restrictions on which entities they can connect (may link same rank entities)
///   - allow data to be stored on the links themselves in a bucket-aware manner
///   - allow either looping over the links or the entities to which they are connected
///   - enforce a weaker aura condition whereby every locally owned or shared link and every locally owned or shared
///   linked entity that it connects have at least ghosted access to one another
///   - allow for creation and destruction of links outside of a modification cycle
///
/// Links are not meant to be a replacement for connectivity, but rather a more flexible and dynamic alternative. They
/// are great for encoding neighbor relationships that require data to be stored on the link itself, such as reused
/// invariants. They also work well for composite relationships whereby an entity needs to "know about" the state of
/// another entity in a way that cannot be expressed topologically. For example, a quad face storing a set of nodes that
/// are "attached" to it at locations other than the corners.
///
/// From a user's perspective, links are Entities with a connected entity ids and connected entity rank field. You are
/// free to use links like regular entities by adding them to parts, using subset relations/selectors, and by adding
/// additional fields to them. This makes them fully compatible with our Aggregates. Importantly, they may even be
/// written out to the EXO file using standard STK_IO functionality with care to avoid accidentally loadbalancing them
/// if they don't have spacial information.
///
/// # LinkData
/// The LinkData class is the main interface for interacting with the link data on the mesh. It is meant to mirror
/// BulkData's connectivity interface while allowing multiple LinkData objects to be used on the same mesh, each with
/// separately managed data. Use LinkData to connect links to linked entities and to get the linked entities for a given
/// link. And similar to STK's for_each_entity_run, use non-member functions acting on the LinkData to loop over links
/// and linked entities.
///
/// ## Declaring/Connecting Links
/// Declaring links can be done via the standard declare_entity interface, however, connecting these links to their
/// linked entities must be mediated via the link data. For this reason, we provide only const access to the linked
/// entities field. Use it to ~view~ the connections, not to ~modify~ them. Instead, use the link data's
/// declare_relation(linker, linked_entity, link_ordinal) and delete_relation(linker, link_ordinal) to modify the
/// relationship between a link and its linked entities. These functions are thread-safe and may be called in parallel
/// so long as you do not call declare_relation(linker, *, link_ordinal) for the same linker and ordinal on two
/// different threads (something that would be weird to do anyway).
///
/// Some comments on requirements:
///  - Both declare_relation and delete_relation require that the linker be valid, but not necessarily the linked
///  entity.
///  - To maintain parallel consistency, we require that declare/delete_relation be performed consistently for each
/// process that locally owns or shares the given linker or linked entity.
///
/// \note Once a relationship between a link and a linked entity is declared or destroyed, the link data is marked as
/// "needs updated", as the changes to the linked data are only reflected by the get_linked_entity function and not some
/// of the other infrastructure needed to maintain parallel consistency. As such, you must call propagate_updates()
/// before entering a modification cycle or before using the for_each_linked_entity_run function.
///
/// ## Getting Linked Entities
/// In contrast to STK's connectivity, links are designed to be declared dynamically and to be created and destroyed at
/// any time. This must be done *outside* of a mesh modification cycle. Once a relation between a link and a linked
/// entity is declared, you may call get_linked_entity(linker, link_ordinal) to get the linked entity at the given
/// ordinal. If no relation exists, this will return an invalid entity and is valid so
/// long as link_ordinal is less than the linker dimensionality, otherwise, it will throw an exception (in debug).
/// This function is thread-safe and immediately reflects any changes made to the link data.
///
/// Note, we do not offer a get_linked_entities function, as this would require either dynamic memory allocation or
/// compile-time link dimensionality. Similarly, we do not offer a get_connected_links(entity) function. Instead, we
/// offer two for_each_entity_run-esk functions for looping over links and linked entities.
///
///  - for_each_link_run(link_data, link_subset_selector, functor) works the same as for_each_entity_run, but for links.
///  The functor must either have an
///     operator(const stk::mesh::BulkData& bulk_data, const stk::mesh::Entity& linker) or an
//      operator(const LinkData& link_data, const stk::mesh::Entity& linker).
///  This function is thread parallel over each link in the given link_data that falls within the given
///  link_subset_selector. Notice that for_each_link does not require this synchronization, meaning that
/// dynamic linker creation and destruction can be done in parallel within a for_each_link loop AND you can continue
/// to perform for_each_link loops while the link data is out-of_sync.
///
/// - for_each_linked_entity_run(link_data, linked_entity_selector, linker_subset_selector, functor)
/// The functor must either have an
///     operator(const stk::mesh::BulkData&, const stk::mesh::Entity&linked_entity, const stk::mesh::Entity&linker) or
///     an operator(const LinkData&, const stk::mesh::Entity& linked_entity, const stk::mesh::Entity& linker).
/// This functions is thread parallel over each linked entity in the given linked_entity_selector that falls within a
/// bucket that a linker in the given linker_subset_selector connects to. This means that the functor will be called in
/// serial for each link that connects to a linked entity. Importantly, this function requires infrastructure that
/// requires the link data to be "up-to-date" (i.e., you must call link_data.propagate_updates() before using it if you
/// have modified the link data).
///
/// You might think, why not just provide a get_connected_links(entity) function? The reason is that the links
/// themselves are heterogeneous and bucketized. As such, there is no practical way to provide contiguous access to the
/// set of links that an entity connects to while supporting subset selection of links and without dynamic memory
/// allocation.
///
/// \note To Devs: Hi! Welcome. If you want to better understand the LinkData or our links in general, I recommend
/// looking at it as maintaining two connectivity structures: a COO-like structure providing access from a linker to its
/// linked entities and a CRS-like structure providing access from a linked entity to its linkers. The COO is the
/// dynamic "driver" that is trivial to modify (even in parallel) and the CRS is a more heavy-weight sparse data
/// structure that is non-trivial to modify in parallel since it often requires memory allocation. The propagate_updates
/// function is responsible for mapping all of the modifications to the COO structure to the CRS structure. There are
/// some operations that fundamentally require a CRS-like structure such as maintaining parallel consistency as entities
/// are removed from the mesh or change parallel ownership/sharing or performing operations that require a serial loop
/// over each linker that connects to a given linked entity.
///
/// # NgpLinkData
/// Following STK's lead, we provide an NgpLinkData class, which is device compatible and a
/// get_updated_ngp_data(link_data)-> NgpLinkData for getting the NgpLinkData for a given LinkData. With the
/// NgpLinkData, you may perform declare/delete_relation and get_linked_entity on either the host or device. As with all
/// of our Ngp* classes, modifications need to be synchronized to and from the host via the modify_on_host/device and
/// and sync_to_host/device functions. We also offer an ngp versions of
/// for_each_link_run; although, we do not currently support for_each_linked_entity_run on the device. We hope to
/// add support for this in the future.

class LinkMetaData {
 public:
  //! \name Type aliases
  //@{

  using entity_rank_value_t = std::underlying_type_t<stk::mesh::EntityRank>;
  using entity_id_value_t = stk::mesh::EntityId;
  using linked_entity_ids_field_t = stk::mesh::Field<entity_id_value_t>;
  using linked_entity_ranks_field_t = stk::mesh::Field<entity_rank_value_t>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Destructor.
  virtual ~LinkMetaData() = default;

  /// \brief Default copy/move constructors/operators.
  LinkMetaData(const LinkMetaData &) = default;
  LinkMetaData(LinkMetaData &&) = default;
  LinkMetaData &operator=(const LinkMetaData &) = default;
  LinkMetaData &operator=(LinkMetaData &&) = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the name of this link data.
  const std::string &name() const {
    return our_name_;
  }

  /// \brief Fetch the link rank.
  stk::mesh::EntityRank link_rank() const {
    return link_rank_;
  }

  /// \brief Fetch the linked entity ids field.
  ///
  /// \note Users should not edit this field yourself. We expose it to you because it's how you'll interact with the
  /// linked entities when doing things like post-processing the output EXO file, but it should be seen as read-only.
  /// Use declare/delete_relation to modify it since they perform additional behind-the-scenes bookkeeping.
  const linked_entity_ids_field_t &linked_entity_ids_field() const {
    return linked_entity_ids_field_;
  }

  /// \brief Fetch the linked entity ranks field.
  ///
  /// Same comment as linked_entity_ids_field. Treat this field as read-only.
  const linked_entity_ranks_field_t &linked_entity_ranks_field() const {
    return linked_entity_ranks_field_;
  }

  /// \brief Fetch the universal link part.
  const stk::mesh::Part &universal_link_part() const {
    return universal_link_part_;
  }

  /// \brief Fetch the universal link part.
  stk::mesh::Part &universal_link_part() {
    return universal_link_part_;
  }

  /// \brief Fetch the mesh meta data manager for this bulk data manager.
  const MetaData &mesh_meta_data() const {
    return meta_data_;
  }

  /// \brief Fetch the mesh meta data manager for this bulk data manager.
  MetaData &mesh_meta_data() {
    return meta_data_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Declare a link-part (compatible with the current link data)
  ///
  /// \param part_name [in] The name of the part.
  /// \param link_dimensionality_for_this_part [in] The number of linked entities per link.
  stk::mesh::Part &declare_link_part(const std::string &part_name, unsigned link_dimensionality_for_this_part) {
    stk::mesh::Part &part = meta_data_.declare_part(part_name, link_rank_);
    add_link_support_to_part(part, link_dimensionality_for_this_part);
    return part;
  }

  /// \brief Declare a link assembly part (compatible with the current link data)
  ///
  /// \param part_name [in] The name of the part.
  stk::mesh::Part &declare_link_assembly_part(const std::string &part_name) {
    stk::mesh::Part &part = meta_data_.declare_part(part_name, link_rank_);
    add_link_support_to_assembly_part(part);
    return part;
  }

  /// \brief Make an existing part into a link-compatible part (compatible with the current link data)
  ///
  /// \param part_name [in] The name of the part.
  /// \param link_dimensionality_for_this_part [in] The number of linked entities per link.
  stk::mesh::Part &add_link_support_to_part(stk::mesh::Part &part, unsigned link_dimensionality_for_this_part) {
    meta_data_.declare_part_subset(universal_link_part_, part);
    put_link_fields_on_part(part, link_dimensionality_for_this_part);
    return part;
  }

  /// \brief Make an existing assembly part into a link-compatible part (compatible with the current link data)
  ///
  /// \param part_name [in] The name of the part.
  /// \param link_dimensionality_for_this_part [in] The number of linked entities per link.
  stk::mesh::Part &add_link_support_to_assembly_part(stk::mesh::Part &part) {
    meta_data_.declare_part_subset(universal_link_part_, part);
    return part;
  }
  //@}

 protected:
  //! \name Internal aliases
  //@{

  using entity_value_t = stk::mesh::Entity::entity_value_type;
  using linked_entities_field_t = stk::mesh::Field<entity_value_t>;
  using linked_entity_bucket_ids_field_t = stk::mesh::Field<unsigned>;
  using linked_entity_bucket_ords_field_t = stk::mesh::Field<unsigned>;
  using link_crs_needs_updated_field_t = stk::mesh::Field<int>;
  //@}

  //! \name Constructor
  //@{

  /// \brief No default constructor.
  LinkMetaData() = delete;

  /// \brief Construct and declare
  LinkMetaData(MetaData &meta_data, const std::string &our_name, stk::mesh::EntityRank link_rank)
      : our_name_(our_name),
        meta_data_(meta_data),
        link_rank_(link_rank),
        linked_entities_field_(meta_data.declare_field<entity_value_t>(link_rank_, "MUNDY_LINKED_ENTITIES")),
        linked_entities_crs_field_(meta_data.declare_field<entity_value_t>(link_rank_, "MUNDY_LINKED_ENTITIES_CRS")),
        linked_entity_ids_field_(meta_data.declare_field<entity_id_value_t>(link_rank_, "MUNDY_LINKED_ENTITY_IDS")),
        linked_entity_ranks_field_(
            meta_data.declare_field<entity_rank_value_t>(link_rank_, "MUNDY_LINKED_ENTITY_RANKS")),
        linked_entity_bucket_ids_field_(meta_data.declare_field<unsigned>(link_rank_, "MUNDY_LINKED_ENTITY_BUCKET_ID")),
        linked_entity_bucket_ords_field_(
            meta_data.declare_field<unsigned>(link_rank_, "MUNDY_LINKED_ENTITY_BUCKET_ORD")),
        link_crs_needs_updated_field_(meta_data.declare_field<int>(link_rank_, "MUNDY_LINK_CRS_NEEDS_UPDATED")),
        universal_link_part_(meta_data.declare_part(std::string("MUNDY_UNIVERSAL_") + our_name, link_rank_)) {
  }
  //@}

  //! \name Internal getters
  //@{

  /// \brief Fetch the linked entity ids field (non-const).
  linked_entity_ids_field_t &linked_entity_ids_field() {
    return linked_entity_ids_field_;
  }

  /// \brief Fetch the linked entity ranks field (non-const).
  linked_entity_ranks_field_t &linked_entity_ranks_field() {
    return linked_entity_ranks_field_;
  }

  /// \brief Fetch the linked entities field.
  const linked_entities_field_t &linked_entities_field() const {
    return linked_entities_field_;
  }
  linked_entities_field_t &linked_entities_field() {
    return linked_entities_field_;
  }

  /// \brief Fetch the linked entities field (as last seen by the CRS).
  const linked_entities_field_t &linked_entities_crs_field() const {
    return linked_entities_crs_field_;
  }
  linked_entities_field_t &linked_entities_crs_field() {
    return linked_entities_crs_field_;
  }

  /// \brief Fetch the linked entity bucket id field.
  const linked_entity_bucket_ids_field_t &linked_entity_bucket_ids_field() const {
    return linked_entity_bucket_ids_field_;
  }
  linked_entity_bucket_ids_field_t &linked_entity_bucket_ids_field() {
    return linked_entity_bucket_ids_field_;
  }

  /// \brief Fetch the linked entity bucket ord field.
  const linked_entity_bucket_ords_field_t &linked_entity_bucket_ords_field() const {
    return linked_entity_bucket_ords_field_;
  }
  linked_entity_bucket_ords_field_t &linked_entity_bucket_ords_field() {
    return linked_entity_bucket_ords_field_;
  }

  /// \brief Fetch the link crs needs updated field.
  const link_crs_needs_updated_field_t &link_crs_needs_updated_field() const {
    return link_crs_needs_updated_field_;
  }
  link_crs_needs_updated_field_t &link_crs_needs_updated_field() {
    return link_crs_needs_updated_field_;
  }
  //@}

  //! \name Helper functions
  //@{

  /// \brief Add the linked entities and keys field to the part with the given dimensionality
  void put_link_fields_on_part(stk::mesh::Part &part, unsigned link_dimensionality) {
    std::vector<entity_value_t> initial_linked_entities(link_dimensionality, stk::mesh::Entity().local_offset());
    std::vector<entity_id_value_t> initial_linked_entity_ids(link_dimensionality, stk::mesh::EntityId());
    std::vector<entity_rank_value_t> initial_linked_entity_ranks(
        link_dimensionality, static_cast<entity_rank_value_t>(stk::topology::INVALID_RANK));
    int initial_link_crs_needs_updated[1] = {true};
    stk::mesh::put_field_on_mesh(linked_entities_field_, part, link_dimensionality, initial_linked_entities.data());
    stk::mesh::put_field_on_mesh(linked_entities_crs_field_, part, link_dimensionality, initial_linked_entities.data());
    stk::mesh::put_field_on_mesh(linked_entity_ids_field_, part, link_dimensionality, initial_linked_entity_ids.data());
    stk::mesh::put_field_on_mesh(linked_entity_ranks_field_, part, link_dimensionality,
                                 initial_linked_entity_ranks.data());
    stk::mesh::put_field_on_mesh(linked_entity_bucket_ids_field_, part, link_dimensionality, nullptr);
    stk::mesh::put_field_on_mesh(linked_entity_bucket_ords_field_, part, link_dimensionality, nullptr);
    stk::mesh::put_field_on_mesh(link_crs_needs_updated_field_, part, 1, initial_link_crs_needs_updated);
  }
  //@}

 private:
  //! \name Friends <3
  //@{

  friend class LinkData;
  friend class NgpLinkData;
  friend LinkMetaData declare_link_meta_data(MetaData &meta_data, const std::string &our_name,
                                             stk::mesh::EntityRank link_rank);

  template <typename FunctionToRunPerLinkedEntity>
  friend void for_each_linked_entity_run(const LinkData &, const stk::mesh::Selector &, const stk::mesh::Selector &,
                                         const FunctionToRunPerLinkedEntity &);
  //@}

  //! \name Internal members
  //@{

  std::string our_name_;
  MetaData &meta_data_;
  stk::mesh::EntityRank link_rank_;
  linked_entities_field_t &linked_entities_field_;
  linked_entities_field_t &linked_entities_crs_field_;
  linked_entity_ids_field_t &linked_entity_ids_field_;
  linked_entity_ranks_field_t &linked_entity_ranks_field_;
  linked_entity_bucket_ids_field_t &linked_entity_bucket_ids_field_;
  linked_entity_bucket_ords_field_t &linked_entity_bucket_ords_field_;
  link_crs_needs_updated_field_t &link_crs_needs_updated_field_;
  stk::mesh::Part &universal_link_part_;
  //@}
};

LinkMetaData declare_link_meta_data(MetaData &meta_data, const std::string &our_name, stk::mesh::EntityRank link_rank) {
  return LinkMetaData(meta_data, our_name, link_rank);
}

class LinkData {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor.
  LinkData() = delete;

  /// \brief No copy or move constructors/operators.
  LinkData(const LinkData &) = delete;
  LinkData(LinkData &&) = delete;
  LinkData &operator=(const LinkData &) = delete;
  LinkData &operator=(LinkData &&) = delete;

  /// \brief Canonical constructor.
  /// \param bulk_data [in] The bulk data manager we extend.
  /// \param link_meta_data [in] Our meta data manager.
  LinkData(BulkData &bulk_data, LinkMetaData link_meta_data)
      : bulk_data_(bulk_data), mesh_meta_data_(bulk_data.mesh_meta_data()), link_meta_data_(link_meta_data) {
  }

  /// \brief Destructor.
  virtual ~LinkData() = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Fetch the bulk data's meta data manager
  const MetaData &mesh_meta_data() const {
    return mesh_meta_data_;
  }

  /// \brief Fetch the bulk data's meta data manager
  MetaData &mesh_meta_data() {
    return mesh_meta_data_;
  }

  /// \brief Fetch the link meta data manager
  const LinkMetaData &link_meta_data() const {
    return link_meta_data_;
  }

  /// \brief Fetch the link meta data manager
  LinkMetaData &link_meta_data() {
    return link_meta_data_;
  }

  /// \brief Fetch the bulk data manager we extend
  const BulkData &bulk_data() const {
    return bulk_data_;
  }

  /// \brief Fetch the bulk data manager we extend
  BulkData &bulk_data() {
    return bulk_data_;
  }
  //@}

  //!\name Actions
  //@{

  /// \brief Declare a relation between a linker and a linked entity.
  ///
  /// # To explain ordinals:
  /// If a linker has dimensionality 3 then it can have up to 3 linked entities. The first
  /// linked entity has ordinal 0, the second has ordinal 1, and so on.
  ///
  /// Importantly, the relationship between links and its linked entities is static with fixed size.
  /// If you fetch the linked entities and have only declared the first two, then the third will be invalid.
  /// This is a slight deviation from STK, which would return a set of two valid entities and provide access to their
  /// ordinals.
  ///
  /// # How does a link attain a certain dimensionality?
  /// A link's dimensionality is determined by the set of parts that it belongs to. When link parts are declared, they
  /// are assigned a dimensionality. If a link belongs to multiple link parts, then the maximum dimensionality of
  /// those parts is the link's dimensionality.
  ///
  /// TODO(palmerb4): Bounds check the link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param linked_entity [in] The linked entity (may be invalid).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline void declare_relation(const stk::mesh::Entity &linker, const stk::mesh::Entity &linked_entity,
                               unsigned link_ordinal) {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data_.linked_entities_field();
    auto &linked_e_ids_field = link_meta_data_.linked_entity_ids_field();
    auto &linked_e_ranks_field = link_meta_data_.linked_entity_ranks_field();
    auto &linked_e_bucket_ids_field = link_meta_data_.linked_entity_bucket_ids_field();
    auto &linked_e_bucket_ords_field = link_meta_data_.linked_entity_bucket_ords_field();
    auto &link_needs_updated_field = link_meta_data_.link_crs_needs_updated_field();

    stk::mesh::field_data(linked_es_field, linker)[link_ordinal] = linked_entity.local_offset();
    stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal] = bulk_data_.identifier(linked_entity);
    stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal] = bulk_data_.entity_rank(linked_entity);
    stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal] =
        bulk_data_.bucket(linked_entity).bucket_id();
    stk::mesh::field_data(linked_e_bucket_ords_field, linker)[link_ordinal] = bulk_data_.bucket_ordinal(linked_entity);
    stk::mesh::field_data(link_needs_updated_field, linker)[0] = true;
  }

  /// \brief Delete a relation between a linker and a linked entity.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  inline void delete_relation(const stk::mesh::Entity &linker, unsigned link_ordinal) {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data_.linked_entities_field();
    auto &linked_e_ids_field = link_meta_data_.linked_entity_ids_field();
    auto &linked_e_ranks_field = link_meta_data_.linked_entity_ranks_field();
    auto &linked_e_bucket_ids_field = link_meta_data_.linked_entity_bucket_ids_field();
    auto &linked_e_bucket_ords_field = link_meta_data_.linked_entity_bucket_ords_field();
    auto &link_needs_updated_field = link_meta_data_.link_crs_needs_updated_field();

    // Intentionally avoids updating the CRS linked entities field so that we can properly detect deletions.
    stk::mesh::field_data(linked_es_field, linker)[link_ordinal] = stk::mesh::Entity().local_offset();
    stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal] = stk::mesh::EntityId();
    stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal] =
        static_cast<LinkMetaData::entity_rank_value_t>(stk::topology::INVALID_RANK);
    stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal] = 0;
    stk::mesh::field_data(linked_e_bucket_ords_field, linker)[link_ordinal] = 0;
    stk::mesh::field_data(link_needs_updated_field, linker)[0] = true;
  }

  /// \brief Get the linked entity for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  /// \return The linked entity.
  inline stk::mesh::Entity get_linked_entity(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data_.linked_entities_field();
    return stk::mesh::Entity(stk::mesh::field_data(linked_es_field, linker)[link_ordinal]);
  }

  /// \brief Get the linked entity id for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  /// \return The linked entity id.
  inline stk::mesh::EntityId get_linked_entity_id(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_ids_field = link_meta_data_.linked_entity_ids_field();
    return stk::mesh::field_data(linked_e_ids_field, linker)[link_ordinal];
  }

  /// \brief Get the linked entity rank for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  /// \return The linked entity rank.
  inline stk::mesh::EntityRank get_linked_entity_rank(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_e_ranks_field = link_meta_data_.linked_entity_ranks_field();
    return static_cast<stk::mesh::EntityRank>(stk::mesh::field_data(linked_e_ranks_field, linker)[link_ordinal]);
  }

  /// \brief Propagate changes made via the declare/delete_relation functions to internal data structures.
  ///
  /// This function must be called before either entering a modification cycle or using the for_each_linked_entity_run
  /// function.
  void propagate_updates() {
    // 1. get each stk link bucket,
    // 2. get the key for that bucket's partition,
    // 3. insert a new partition conn into the map if one doesn't already exist,
    // 4. loop over each link in said stk partition
    // 5. if the link is out of sync, loop over each of the linked entities and its crs version
    // 6. if the linked entity and its crs version are the same, the connectivity is in sync, skip
    // 7. if the linked entity is valid add the linker to the end of the linked entity's connectivity
    // 8. set all entities to up-to-data and stash current crs entities.
    using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;

    stk::mesh::BucketVector link_buckets =
        bulk_data_.get_buckets(link_meta_data_.link_rank(), link_meta_data_.universal_link_part());
    for (stk::mesh::Bucket *link_bucket : link_buckets) {
      unsigned link_dim = get_linker_dimensionality(*link_bucket);

      PartitionKey partition_key = get_partition_key(link_bucket->supersets());
      LinkedPartitionConn &linked_partition_conn = partition_to_linked_conn[partition_key];
      for (const stk::mesh::Entity &linker : *link_bucket) {
        const auto &link_needs_updated = link_meta_data_.link_crs_needs_updated_field();
        const bool is_out_of_sync = stk::mesh::field_data(link_needs_updated, linker)[0];
        if (!is_out_of_sync) {
          continue;
        }

        const auto &linked_es_field = link_meta_data_.linked_entities_field();
        const auto &linked_es_field_crs = link_meta_data_.linked_entities_crs_field();
        for (unsigned i = 0; i < link_dim; ++i) {
          stk::mesh::Entity linked_entity = get_linked_entity(linker, i);
          stk::mesh::Entity old_linked_entity = get_linked_entity_crs(linker, i);

          if (linked_entity == old_linked_entity) {
            // Already up to date, skip
            continue;
          }

          // If the old_linked_entity is valid, remove the linker from its connectivity
          if (bulk_data_.is_valid(old_linked_entity)) {
            unsigned old_linked_entity_ord = bulk_data_.bucket_ordinal(old_linked_entity);
            impl::LinkedBucketConn &old_linked_bucket_conn =
                linked_partition_conn[bulk_data_.bucket_ptr(old_linked_entity)];
            MUNDY_THROW_ASSERT(old_linked_bucket_conn.bucket_size() > 0, std::logic_error,
                               "Bug: Old linked bucket conn is empty, but we are trying to remove a linker from it.");

            // Determine where in our connectivity the linker is located and remove it
            ConnectedEntities connected_links = old_linked_bucket_conn.get_connected_entities(old_linked_entity_ord);
            unsigned num_connected_links = connected_links.size();
            for (unsigned j = 0; j < num_connected_links; ++j) {
              if (connected_links[j] == linker) {
                old_linked_bucket_conn.remove_connectivity(old_linked_entity_ord, linker, j);
                break;
              }
            }
          }

          // If the new linked entity is valid, add the linker to the end of its connectivity
          // PLEASE NOTE: We assume that a link will never link to the same entity multiple times
          //  we enforce this in debug mode.
          if (bulk_data_.is_valid(linked_entity)) {
            unsigned linked_entity_ord = bulk_data_.bucket_ordinal(linked_entity);

            impl::LinkedBucketConn &linked_bucket_conn = linked_partition_conn[bulk_data_.bucket_ptr(linked_entity)];
            if (linked_bucket_conn.bucket_size() == 0) {
              // The bucket was not previously in the map, so we need to resize it to fit our bucket.
              linked_bucket_conn.resize(bulk_data_.bucket_ptr(linked_entity)->size());
            }

            // Check if the linker is already connected to the linked entity
            ConnectedEntities connected_links = linked_bucket_conn.get_connected_entities(linked_entity_ord);
            unsigned num_connected_links = connected_links.size();

#ifndef NDEBUG
            // If in debug, check if the linker is already connected to the linked entity.
            bool linker_already_connected = false;
            for (unsigned j = 0; j < num_connected_links; ++j) {
              if (connected_links[j] == linker) {
                linker_already_connected = true;
              }
            }
            MUNDY_THROW_ASSERT(
                !linker_already_connected, std::logic_error,
                "Attempting to add a linker to linked entity crs relation that somehow already exists???."
                "This is possible if the link somehow links the same entity multiple times.");
#endif
            // Assume that the linker is not already connected and add it to the end
            linked_bucket_conn.add_connectivity(linked_entity_ord, linker, num_connected_links);
          }
        }

        // At this point, the CRS and COO link connectivity for this link are in sync.
        // We can now reset the out-of-sync flag.
        stk::mesh::field_data(link_needs_updated, linker)[0] = false;
      }
    }

    // At this point, all links are in sync. We can copy the current linked entities field to the crs linked entities
    // field for the current set of links.
    using entity_value_t = stk::mesh::Entity::entity_value_type;
    field_copy<entity_value_t>(link_meta_data_.linked_entities_field(),      // source
                               link_meta_data_.linked_entities_crs_field(),  // target
                               link_meta_data_.universal_link_part(),        //
                               stk::ngp::HostExecSpace());
  }
  //@}

 protected:
  //! \name Internal getters
  //@{

  /// \brief Get the (CRS) linked entity for a given linker and link ordinal.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  /// \return The linked entity.
  inline stk::mesh::Entity get_linked_entity_crs(const stk::mesh::Entity &linker, unsigned link_ordinal) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    auto &linked_es_field = link_meta_data_.linked_entities_crs_field();
    return stk::mesh::Entity(stk::mesh::field_data(linked_es_field, linker)[link_ordinal]);
  }

  /// \brief Get the dimensionality of a linker
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \return The dimensionality of the linker.
  inline unsigned get_linker_dimensionality(const stk::mesh::Entity &linker) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == bulk_data_.entity_rank(linker), std::invalid_argument,
                       "Linker is not of the correct rank.");
    MUNDY_THROW_ASSERT(bulk_data_.is_valid(linker), std::invalid_argument, "Linker is not valid.");

    return stk::mesh::field_scalars_per_entity(link_meta_data_.linked_entities_field(), linker);
  }

  /// \brief Get the dimensionality of a linker bucket
  /// \param linker_bucket [in] The linker bucket (must be of the correct rank and a subset of our universal link part).
  /// \return The dimensionality of linkers in the bucket.
  inline unsigned get_linker_dimensionality(const stk::mesh::Bucket &linker_bucket) const {
    MUNDY_THROW_ASSERT(link_meta_data_.link_rank() == linker_bucket.entity_rank(), std::invalid_argument,
                       "Linker bucket is not of the correct rank.");
    MUNDY_THROW_ASSERT(linker_bucket.member(link_meta_data_.universal_link_part()), std::invalid_argument,
                       "Linker bucket is not a subset of our universal link part.");

    auto &linked_es_field = link_meta_data_.linked_entities_field();
    return stk::mesh::field_scalars_per_entity(linked_es_field, linker_bucket);
  }
  //@}

 private:
  //! \name Friends <3
  //@{

  friend class NgpLinkData;

  template <typename FunctionToRunPerLinkedEntity>
  friend void for_each_linked_entity_run(const LinkData &, const stk::mesh::Selector &, const stk::mesh::Selector &,
                                         const FunctionToRunPerLinkedEntity &);
  //@}

  //! \name Type aliases
  //@{

  using PartitionKey = stk::mesh::OrdinalVector;  // sorted list of part ordinals
  using LinkedPartitionConn = std::map<stk::mesh::Bucket *, impl::LinkedBucketConn>;
  //@}

  //! \name Private helpers
  //@{

  /// \brief Get the partition key for a given set of link parts (sorted vector of part ordinals)
  PartitionKey get_partition_key(const stk::mesh::PartVector &link_parts) {
    size_t num_parts = link_parts.size();
    stk::mesh::OrdinalVector link_part_ords(num_parts);
    for (size_t i = 0; i < num_parts; ++i) {
      link_part_ords[i] = link_parts[i]->mesh_meta_data_ordinal();
    }
    return get_partition_key(link_part_ords);
  }

  /// \brief Get the partition key for a given set of link parts (sorted vector of part ordinals)
  PartitionKey get_partition_key(stk::mesh::OrdinalVector link_part_ords) {
    std::sort(link_part_ords.begin(), link_part_ords.end());
    return link_part_ords;
  }
  //@}

  //! \name Internal members
  //@{

  BulkData &bulk_data_;
  MetaData &mesh_meta_data_;
  LinkMetaData link_meta_data_;
  std::map<PartitionKey, LinkedPartitionConn> partition_to_linked_conn;  ///< Our CRS connectivity per link partition
  //@}
};  // LinkData

LinkData declare_link_data(BulkData &bulk_data, LinkMetaData link_meta_data) {
  return LinkData(bulk_data, link_meta_data);
}

class NgpLinkData {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor.
  /// If you find that you need one, let us know and we can consider added one.
  /// The only real reason to not have one is that we want to store a reference to LinkData.
  /// We could replace this with a pointer and add setters + a concretize function.
  KOKKOS_FUNCTION
  NgpLinkData() = delete;

  /// \brief Default copy/move constructors/operators.
  NgpLinkData(const NgpLinkData &) = default;
  NgpLinkData(NgpLinkData &&) = default;
  NgpLinkData &operator=(const NgpLinkData &) = default;
  NgpLinkData &operator=(NgpLinkData &&) = default;

  /// \brief Construct from a LinkData.
  /// Either build this class directly or use get_updated_ngp_data(link_data).
  ///
  /// \param link_meta_data_ptr [in] A pointer to this link's meta data manager.
  /// \param bulk_data [in] The bulk data manager we extend.
  NgpLinkData(const LinkData &link_data)
      : link_data_(link_data),
        link_rank_(link_data_.link_meta_data().link_rank()),
        ngp_mesh_(stk::mesh::get_updated_ngp_mesh(link_data.bulk_data())),
        ngp_linked_entities_field_(                            //
            stk::mesh::get_updated_ngp_field<entity_value_t>(  //
                link_data.link_meta_data().linked_entities_field())),
        ngp_linked_entity_ids_field_(                                           //
            stk::mesh::get_updated_ngp_field<LinkMetaData::entity_id_value_t>(  //
                link_data.link_meta_data().linked_entity_ids_field())),
        ngp_linked_entity_ranks_field_(                                           //
            stk::mesh::get_updated_ngp_field<LinkMetaData::entity_rank_value_t>(  //
                link_data.link_meta_data().linked_entity_ranks_field())),
        ngp_linked_entity_bucket_ids_field_(             //
            stk::mesh::get_updated_ngp_field<unsigned>(  //
                link_data.link_meta_data().linked_entity_bucket_ids_field())),
        ngp_linked_entity_bucket_ords_field_(            //
            stk::mesh::get_updated_ngp_field<unsigned>(  //
                link_data.link_meta_data().linked_entity_bucket_ords_field())),
        ngp_link_crs_needs_updated_field_(          //
            stk::mesh::get_updated_ngp_field<int>(  //
                link_data.link_meta_data().link_crs_needs_updated_field())) {
  }

  /// \brief Destructor.
  KOKKOS_FUNCTION
  virtual ~NgpLinkData() = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Fetch the link data
  const LinkData &host_link_data() const {
    return link_data_;
  }

  /// \brief Fetch the ngp mesh
  KOKKOS_FUNCTION
  const stk::mesh::NgpMesh &ngp_mesh() const {
    return ngp_mesh_;
  }

  /// \brief Fetch the ngp mesh
  KOKKOS_FUNCTION
  stk::mesh::NgpMesh &ngp_mesh() {
    return ngp_mesh_;
  }

  /// \brief Fetch the link rank
  KOKKOS_FUNCTION
  stk::mesh::EntityRank link_rank() const {
    return link_rank_;
  }
  //@}

  //!\name Actions
  //@{

  /// \brief Declare a relation between a linker and a linked entity.
  ///
  /// # To explain ordinals:
  /// If a linker has dimensionality 3 then it can have up to 3 linked entities. The first
  /// linked entity has ordinal 0, the second has ordinal 1, and so on.
  ///
  /// Importantly, the relationship between links and its linked entities is static with fixed size.
  /// If you fetch the linked entities and have only declared the first two, then the third will be invalid.
  /// This is a slight deviation from STK, which would return a set of two valid entities and provide access to their
  /// ordinals.
  ///
  /// # How does a link attain a certain dimensionality?
  /// A link's dimensionality is determined by the set of parts that it belongs to. When link parts are declared, they
  /// are assigned a dimensionality. If a link belongs to multiple link parts, then the maximum dimensionality of
  /// those parts is the link's dimensionality.
  ///
  /// TODO(palmerb4): Bounds check the link ordinal using the link dimensionality.
  ///
  /// \param linker [in] The linker (must be valid and of the correct rank).
  /// \param linked_entity [in] The linked entity (may be invalid).
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  void declare_relation(const stk::mesh::FastMeshIndex &linker_index,         //
                        const stk::mesh::EntityRank &linked_entity_rank,      //
                        const stk::mesh::FastMeshIndex &linked_entity_index,  //
                        unsigned link_ordinal) const {
    stk::mesh::Entity linked_entity = ngp_mesh_.get_entity(linked_entity_rank, linked_entity_index);
    stk::mesh::EntityKey linked_entity_key = ngp_mesh_.entity_key(linked_entity);

    ngp_linked_entities_field_(linker_index, link_ordinal) = linked_entity.local_offset();
    ngp_linked_entity_ids_field_(linker_index, link_ordinal) = linked_entity_key.id();
    ngp_linked_entity_ranks_field_(linker_index, link_ordinal) = linked_entity_rank;
    ngp_linked_entity_bucket_ids_field_(linker_index, link_ordinal) = linked_entity_index.bucket_id;
    ngp_linked_entity_bucket_ords_field_(linker_index, link_ordinal) = linked_entity_index.bucket_ord;
    ngp_link_crs_needs_updated_field_(linker_index, 0) = true;
  }

  /// \brief Delete a relation between a linker and a linked entity.
  ///
  /// \param linker_index [in] The index of the linker.
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  void delete_relation(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    ngp_linked_entities_field_(linker_index, link_ordinal) = stk::mesh::Entity().local_offset();
    ngp_linked_entity_ids_field_(linker_index, link_ordinal) = stk::mesh::EntityId();
    ngp_linked_entity_ranks_field_(linker_index, link_ordinal) =
        static_cast<LinkMetaData::entity_rank_value_t>(stk::topology::INVALID_RANK);
    ngp_linked_entity_bucket_ids_field_(linker_index, link_ordinal) = 0;
    ngp_linked_entity_bucket_ords_field_(linker_index, link_ordinal) = 0;
    ngp_link_crs_needs_updated_field_(linker_index, 0) = true;
  }

  /// \brief Get the linked entity for a given linker and link ordinal.
  /// \param linker_index [in] The index of the linker.
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex get_linked_entity(const stk::mesh::FastMeshIndex &linker_index,
                                             unsigned link_ordinal) const {
    return stk::mesh::FastMeshIndex(ngp_linked_entity_bucket_ids_field_(linker_index, link_ordinal),
                                    ngp_linked_entity_bucket_ords_field_(linker_index, link_ordinal));
  }

  /// \brief Get the linked entity id for a given linker and link ordinal.
  /// \param linker_index [in] The index of the linker.
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityId get_linked_entity_id(const stk::mesh::FastMeshIndex &linker_index, unsigned link_ordinal) const {
    return ngp_linked_entity_ids_field_(linker_index, link_ordinal);
  }

  /// \brief Get the linked entity rank for a given linker and link ordinal.
  /// \param linker_index [in] The index of the linker.
  /// \param link_ordinal [in] The ordinal of the linked entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank get_linked_entity_rank(const stk::mesh::FastMeshIndex &linker_index,
                                               unsigned link_ordinal) const {
    return static_cast<stk::mesh::EntityRank>(ngp_linked_entity_ranks_field_(linker_index, link_ordinal));
  }

  void modify_on_host() {
    ngp_linked_entities_field_.modify_on_host();
    ngp_linked_entity_ids_field_.modify_on_host();
    ngp_linked_entity_ranks_field_.modify_on_host();
    ngp_linked_entity_bucket_ids_field_.modify_on_host();
    ngp_linked_entity_bucket_ords_field_.modify_on_host();
    ngp_link_crs_needs_updated_field_.modify_on_host();
  }

  void modify_on_device() {
    ngp_linked_entities_field_.modify_on_device();
    ngp_linked_entity_ids_field_.modify_on_device();
    ngp_linked_entity_ranks_field_.modify_on_device();
    ngp_linked_entity_bucket_ids_field_.modify_on_device();
    ngp_linked_entity_bucket_ords_field_.modify_on_device();
    ngp_link_crs_needs_updated_field_.modify_on_device();
  }

  void sync_to_host() {
    ngp_linked_entities_field_.sync_to_host();
    ngp_linked_entity_ids_field_.sync_to_host();
    ngp_linked_entity_ranks_field_.sync_to_host();
    ngp_linked_entity_bucket_ids_field_.sync_to_host();
    ngp_linked_entity_bucket_ords_field_.sync_to_host();
    ngp_link_crs_needs_updated_field_.sync_to_host();
  }

  void sync_to_device() {
    ngp_linked_entities_field_.sync_to_device();
    ngp_linked_entity_ids_field_.sync_to_device();
    ngp_linked_entity_ranks_field_.sync_to_device();
    ngp_linked_entity_bucket_ids_field_.sync_to_device();
    ngp_linked_entity_bucket_ords_field_.sync_to_device();
    ngp_link_crs_needs_updated_field_.sync_to_device();
  }
  //@}

 private:
  //! \name Internal aliases
  //@{

  using entity_value_t = stk::mesh::Entity::entity_value_type;
  using ngp_linked_entities_field_t = stk::mesh::NgpField<entity_value_t>;
  using ngp_linked_entity_ids_field_t = stk::mesh::NgpField<LinkMetaData::entity_id_value_t>;
  using ngp_linked_entity_ranks_field_t = stk::mesh::NgpField<LinkMetaData::entity_rank_value_t>;
  using ngp_linked_entity_bucket_ids_field_t = stk::mesh::NgpField<unsigned>;
  using ngp_linked_entity_bucket_ords_field_t = stk::mesh::NgpField<unsigned>;
  using ngp_link_crs_needs_updated_field_t = stk::mesh::NgpField<int>;
  //@}

  //! \name Internal members
  //@{

  const LinkData &link_data_;
  stk::mesh::EntityRank link_rank_;
  stk::mesh::NgpMesh ngp_mesh_;
  ngp_linked_entities_field_t ngp_linked_entities_field_;
  ngp_linked_entity_ids_field_t ngp_linked_entity_ids_field_;
  ngp_linked_entity_ranks_field_t ngp_linked_entity_ranks_field_;
  ngp_linked_entity_bucket_ids_field_t ngp_linked_entity_bucket_ids_field_;
  ngp_linked_entity_bucket_ords_field_t ngp_linked_entity_bucket_ords_field_;
  ngp_link_crs_needs_updated_field_t ngp_link_crs_needs_updated_field_;
  //@}
};  // NgpLinkData

inline NgpLinkData get_updated_ngp_data(const LinkData &link_data) {
  return NgpLinkData(link_data);
}

//! \name Link iteration
//@{

/// \brief Run a host function over each link in the link_data that falls in the given selector in parallel.
///
/// The functor must have one of the following signatures:
///   1. operator()(const BulkData &, const Entity &) -> void
///   2. operator()(const BulkData &, const MeshIndex &) -> void
///   3. operator()(const LinkData &, const Entity &) -> void
///   4. operator()(const LinkData &, const MeshIndex &) -> void
template <typename FunctionToRunPerLink>
void for_each_link_run(const LinkData &link_data, const stk::mesh::Selector &linker_subset_selector,
                       const FunctionToRunPerLink &functor) {
  if constexpr (std::is_invocable_v<FunctionToRunPerLink, const BulkData &, const stk::mesh::Entity &>) {
    // Just use the standard for_each_entity_run
    ::mundy::mesh::for_each_entity_run(link_data.bulk_data(), link_data.link_meta_data().link_rank(),
                                       linker_subset_selector & link_data.link_meta_data().universal_link_part(),
                                       functor);
  } else {
    // Use the link data version
    const stk::mesh::BucketVector &buckets =
        link_data.bulk_data().get_buckets(link_data.link_meta_data().link_rank(),
                                          linker_subset_selector & link_data.link_meta_data().universal_link_part());
    const size_t num_buckets = buckets.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t j = 0; j < num_buckets; j++) {
      stk::mesh::Bucket *bucket = buckets[j];
      const size_t bucket_size = bucket->size();
      for (size_t i = 0; i < bucket_size; i++) {
        if constexpr (std::is_invocable_v<FunctionToRunPerLink, const stk::mesh::BulkData &,
                                          const stk::mesh::MeshIndex &>) {
          functor(link_data, stk::mesh::MeshIndex({bucket, i}));
        } else {
          functor(link_data, (*bucket)[i]);
        }
      }
    }
  }
}

/// \brief Run a host function over each link in the link_data in parallel.
/// See for_each_link_run(const LinkData &, const stk::mesh::Selector &, const FunctionToRunPerLink &)
template <typename FunctionToRunPerLink>
void for_each_link_run(const LinkData &link_data, const FunctionToRunPerLink &functor) {
  for_each_link_run(link_data, link_data.link_meta_data().universal_link_part(), functor);
}

/// \brief Run an ngp-compatible function over each link in the ngp_link_data that falls in the given selector in
/// parallel.
///
/// Thread parallel over each link.
///
/// The functor must have the following signature:
///   operator()(const FastMeshIndex &) -> void
template <typename FunctionToRunPerLink>
void for_each_link_run(const NgpLinkData &ngp_link_data, const stk::mesh::Selector &linker_subset_selector,
                       const FunctionToRunPerLink &functor) {
  ::mundy::mesh::for_each_entity_run(
      ngp_link_data.ngp_mesh(), ngp_link_data.link_rank(),
      ngp_link_data.host_link_data().link_meta_data().universal_link_part() & linker_subset_selector, functor);
}

/// \brief Run an ngp-compatible function over each link in the ngp_link_data in parallel.
/// See for_each_link_run(const NgpLinkData &, const stk::mesh::Selector &, const FunctionToRunPerLink &)
template <typename FunctionToRunPerLink>
void for_each_link_run(const NgpLinkData &ngp_link_data, const FunctionToRunPerLink &functor) {
  for_each_link_run(ngp_link_data, ngp_link_data.host_link_data().link_meta_data().universal_link_part(), functor);
}
//@}

//! \name Linked entity iteration
//@{

/// \brief Run a host function over each linked entity in the given selector and each of its connected links in
/// parallel.
///
/// Thread parallel over each linked entity. Serial over each link.
///
/// Same function signature as for_each_link_run.
///
/// Must call link_data.propagate_updates() before using this function.
template <typename FunctionToRunPerLinkedEntity>
void for_each_linked_entity_run(const LinkData &link_data, const stk::mesh::Selector &linked_entity_selector,
                                const stk::mesh::Selector &linker_subset_selector,
                                const FunctionToRunPerLinkedEntity &functor) {
  // Procedure:
  //  1. If in debug, validate that all links are up to date
  //  2. Loop over each linker partition in serial and check if the partition is in the given selector
  //  3. Loop over each linked bucket in parallel
  //  4. For each entity in the bucket, loop over each of its linked entities in serial and evaluate the functor

#ifndef NDEBUG
  for_each_link_run(link_data, [](const LinkData &link_data, const stk::mesh::Entity &linker) {
    const auto &link_needs_updated = link_data.link_meta_data().link_crs_needs_updated_field();
    MUNDY_THROW_ASSERT(!stk::mesh::field_data(link_needs_updated, linker)[0], std::logic_error,
                       "Linker is out of sync with its linked entities. Make sure to call propagate_updates() before "
                       "using the for_each_linked_entity_run() function.");
  });
#endif

  for (auto &[partition_key, linked_partition_conn] : link_data.partition_to_linked_conn) {
    stk::mesh::PartVector link_parts(partition_key.size());
    for (size_t i = 0; i < partition_key.size(); ++i) {
      link_parts[i] = &link_data.mesh_meta_data().get_part(partition_key[i]);
    }

    stk::mesh::Selector partition_subset = stk::mesh::selectIntersection(link_parts) & linker_subset_selector;
    stk::mesh::BucketVector link_buckets_in_subset =
        link_data.bulk_data().get_buckets(link_data.link_meta_data().link_rank(), partition_subset);
    if (link_buckets_in_subset.empty()) {
      continue;  // No link buckets in this partition are in the given selector
    }

    // Get each bucket in the linked_partition_conn keys that is in the given selector.
    stk::mesh::BucketVector linked_buckets_in_subset;
    for (auto &[linked_bucket, _] : linked_partition_conn) {
      if (linked_entity_selector(linked_bucket)) {
        linked_buckets_in_subset.push_back(linked_bucket);
      }
    }

    // Loop over each linked bucket in parallel
    size_t num_linked_buckets = linked_buckets_in_subset.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < num_linked_buckets; ++i) {
      stk::mesh::Bucket *linked_bucket = linked_buckets_in_subset[i];

      auto it = linked_partition_conn.find(linked_bucket);  // can't use operator[] because it could insert a new entry
      MUNDY_THROW_ASSERT(it != linked_partition_conn.end(), std::logic_error,
                         "Bug: Linked bucket not found in linked partition conn map.");
      const impl::LinkedBucketConn &linked_bucket_conn = it->second;
      size_t linked_bucket_size = linked_bucket->size();
      for (size_t j = 0; j < linked_bucket_size; ++j) {
        stk::mesh::Entity linked_entity = (*linked_bucket)[j];
        using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
        ConnectedEntities connected_links = linked_bucket_conn.get_connected_entities(j);
        unsigned num_connected_links = connected_links.size();

        // Loop over each of the linked entities in serial and evaluate the functor
        for (unsigned k = 0; k < num_connected_links; ++k) {
          stk::mesh::Entity linker = connected_links[k];

          // Two options for functors. Accepts BulkData/LinkData.
          if constexpr (std::is_invocable_v<FunctionToRunPerLinkedEntity, const stk::mesh::BulkData &,
                                            const stk::mesh::Entity &, const stk::mesh::Entity &>) {
            functor(link_data.bulk_data(), linked_entity, linker);
          } else {
            functor(link_data, linked_entity, linker);
          }
        }
      }
    }
  }
}

/// \brief Run a host function over each linked entity in the link_data in parallel.
/// See for_each_linked_entity_run(const LinkData &, const stk::mesh::Selector &, const stk::mesh::Selector &,
/// const FunctionToRunPerLinkedEntity &)
template <typename FunctionToRunPerLinkedEntity>
void for_each_linked_entity_run(const LinkData &link_data, const FunctionToRunPerLinkedEntity &functor) {
  for_each_linked_entity_run(link_data, link_data.bulk_data().mesh_meta_data().universal_part(),  //
                             link_data.link_meta_data().universal_link_part(), functor);
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKDATA_HPP_
