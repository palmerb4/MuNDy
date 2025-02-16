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
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

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
/// From a user's perspective, links are Entities with a connected entities field. You are free to use links like
/// regular entities by adding them to parts, using subset relations/selectors, and by adding additional fields to them.
/// Importantly, they may even be written out to the EXO file using standard STK_IO functionality with care to avoid
/// accidentally loadbalancing them if they don't have a spacial information.
///
/// # LinkData
/// The LinkData class is the main interface for interacting with the link data on the mesh. It is meant to mirror
/// BulkData's connectivity interface while allowing multiple LinkData objects to be used on the same mesh, each with
/// separately managed data. Use it to connect links to linked entities and to get the linked entities for a given link.
/// And similarly, use non-member functions acting on the LinkData to loop over links and linked entities.
///
/// ## Declaring Links
/// Declaring links can be done via the standard declare_entity interface, however, connecting these links to their
/// linked entities must be mediated via the link data; although, this mediation may be delayed. For this reason,
/// we provide only const access to the linked entities field. Use it to view the connections, not to modify them.
/// Instead, use the link data's declare_relation(linker, linked_entity, link_ordinal). This requires that the linker
/// and linked entity be valid and must be performed consistently for each process that locally owns or shares the given
/// linker or linked entity. It does not, however, require that the mesh be within a modification cycle. Instead, it
/// flags the linker as "out-of_sync", requiring that the link_data.synchronize() be called before using
/// for_each_linked_entity or entering a modification cycle. Notice that for_each_link does not require this
/// synchronization, meaning that dynamic linker creation and destruction can be done in parallel within a for_each_link
/// loop AND you can continue to perform for_each_link loops while the link data is out-of_sync. Note, even if you plan
/// on never using for_each_linked_entity, we still need the linked data to be synchronized before entering a
/// modification cycle so we can properly catch mesh modifications that would invalidate the links.
///
/// ## Looping over Links and Linked Entities
/// Through details which are intentionally hidden, users can either loop over each link (thread parallel
/// over the link buckets) using for_each_link, or loop over the entities to which they are connected (thread parallel
/// over the buckets of the linked entities) using for_each_linked_entity. In general, for_each_link is more efficient,
/// as we are able to assign one team per linker bucket with threading over the entities in each bucket, whereas
/// for_each_linked_entity requires one team per linked entity bucket with threading over the entities in that bucket
/// and a non-parallel for loop over that entity's linkers.
///
/// In practice, we prefer to use a single LinkData object to store all links of a given rank and to use
/// selectors to filter the links and linked entities. So even if you have a set of surface links and a set of neighbor
/// links, they are best stored in the same LinkData object. This makes interacting with the link data feel more like
/// working with the bulk data.
///
///
/// Following STK's lead, we provide an NgpLinkData class, which is device compatible and a
/// get_updated_ngp_data(link_data)-> NgpLinkData for getting the NgpLinkData for a given LinkData. With the
/// NgpLinkData, you may perform declare_relation on either the host or device and we offer modify_on_host/device and
/// synchronize_on_host/device functions same as all of our Ngp* classes. We also offer an ngp versions of
/// for_each_link_run; although, we do not currently support for_each_linked_entity_run on the device.
///
/// Internal details: (To be deleted)
/// - All links will be managed by a single LinkData, which has a LinkMetaData. Both of their constructors will be
/// protected.
/// - Link parts may have any rank or dimensionality (num linked entities) and are declared via the LinkMetaData.
/// - No duplicate links per link partition
/// - LinkData stores a vector or view of LinkedPartitions, which we grow or shrink as needed. Because links are
/// declared in bulk,
///    we know which partitions to add them to.
/// - Link partitions store their contribution to the CRS connectivity by storing one LinkedBucket per bucket of an
/// entity that is linked by a link in said partition. The LinkedBucket will store a modified (thread-safe?)
/// DynamicConnectivity object.
/// - LinkData will offer a for_each_link (COO-like parallel over links) and a for_each_linked_entity (CRS-like parallel
/// over linked entities, but two entities connected to the same linker may be acted on simultaneously by different
/// threads).
///
/// LinkPartition -> LinkedBucket -> DynamicConnectivity -> get_connected_links(entity)
///
///
///
/// // Declare the link meta data pre-commit:
/// //   Declares an "overarching_name_universal" part, which will contain all linkers in this link data.
/// //   Declares a linked entities field of the given rank.
/// //   Has other internal effects related to setting up the link data.
/// LinkMetaData link_meta_data = mundy::mesh::declare_link_meta_data(meta_data, "overarching name", link_rank)
///
/// // Use it to declare a link-compatible part (compatible with the current link data):
/// //   Declares the part, adds the linked entities field to it with the given dimensionality, and makes the part a
/// subpart of
/// //   the universal link part.
/// Part& some_link_part = link_meta_data.declare_link_part("name", link_dimensionality_for_this_part);
///
/// // Make an existing part into a link-compatible part:
/// //   Adds the linked entities field to the part with the given dimensionality and
/// //   makes the part a subpart of the universal link part.
/// Part& some_existing_part = link_meta_data.add_link_info_to_part(some_existing_part,
/// link_dimensionality_for_this_part);
///
/// // Declare the link data (may be done pre- or post-commit):
/// //   As a shared_ptr:
/// std::shared_ptr<LinkData> link_data = mundy::mesh::declare_link_data(std::shared_ptr<bulk_data>, link_meta_data);
///
/// //   Directly:
/// LinkData link_data = mundy::mesh::declare_link_data(bulk_data, link_meta_data);
///
///
/// LinkMetaData has the following interface:
/// - link_part() -> Part&
/// - link_rank() -> stk::mesh::EntityRank
/// - linked_entities_field() -> const Field&
/// - universal_link_part() -> Part&
/// - mesh_meta_data() -> MetaData&
/// - link_dimension() -> unsigned (max number of linked entities per link)
///
/// LinkData has the following interface:
/// - mesh_meta_data() -> MetaData&
/// - link_meta_data() -> LinkMetaData&
/// - declare_relation(linker, linked_entity, link_ordinal) -> void (host only)
///
///
/// We also offer non-member for-each-entity-like functions:
/// - for_each_linked_entity(link_data, linked_entity_selector, linker_selector, [](const
///     stk::mesh::Entity&linked_entity, const stk::mesh::Entity& linker){...});
///
/// - for_each_link(link_data, linker_selector, [](const stk::mesh::Entity& linker){...});

class LinkData;     // Forward declaration
class NgpLinkData;  // Forward declaration

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
  /// \note Users should not exit this field yourself. We expose it to you because it's how you'll interact with the
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
  //@}

  //! \name Internal members
  //@{

  std::string our_name_;
  MetaData &meta_data_;
  stk::mesh::EntityRank link_rank_;
  linked_entities_field_t &linked_entities_field_;
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
    stk::mesh::field_data(linked_e_bucket_ids_field, linker)[link_ordinal] = bulk_data_.bucket(linked_entity).bucket_id();
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
  //@}

 protected:
  //! \name Constructor
  //@{

  //@}

 private:
  //! \name Friends <3
  //@{

  friend class NgpLinkData;
  //@}

  //! \name Internal members
  //@{

  BulkData &bulk_data_;
  MetaData &mesh_meta_data_;
  LinkMetaData link_meta_data_;
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
  /// TODO(palmerb4): Bounds check the link ordinal.
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

template <typename FunctionToRunPerLink>
void for_each_link_run(const LinkData &link_data, const stk::mesh::Selector &linker_subset_selector,
                       const FunctionToRunPerLink &functor) {
  ::mundy::mesh::for_each_entity_run(link_data.bulk_data(), link_data.link_meta_data().link_rank(),
                                     linker_subset_selector & link_data.link_meta_data().universal_link_part(),
                                     functor);
}

template <typename FunctionToRunPerLink>
void for_each_link_run(const LinkData &link_data, const FunctionToRunPerLink &functor) {
  ::mundy::mesh::for_each_entity_run(link_data.bulk_data(), link_data.link_meta_data().link_rank(),
                                     link_data.link_meta_data().universal_link_part(), functor);
}

template <typename FunctionToRunPerLink>
void for_each_link_run(const NgpLinkData &ngp_link_data, const stk::mesh::Selector &linker_subset_selector,
                       const FunctionToRunPerLink &functor) {
  ::mundy::mesh::for_each_entity_run(
      ngp_link_data.ngp_mesh(), ngp_link_data.link_rank(),
      ngp_link_data.host_link_data().link_meta_data().universal_link_part() & linker_subset_selector, functor);
}

template <typename FunctionToRunPerLink>
void for_each_link_run(const NgpLinkData &ngp_link_data, const FunctionToRunPerLink &functor) {
  ::mundy::mesh::for_each_entity_run(ngp_link_data.ngp_mesh(), ngp_link_data.link_rank(),
                                     ngp_link_data.host_link_data().link_meta_data().universal_link_part(), functor);
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_LINKDATA_HPP_
