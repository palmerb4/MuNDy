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
template <typename FieldDimMap = EmptyMap, typename FieldMinNumStatesMap = EmptyMap,
          typename FieldAttributesMap = EmptyMap, typename PartStateMap = EmptyMap, typename PartTopologyMap = EmptyMap,
          typename PartRankMap = EmptyMap, typename PartNoInductionMap = EmptyMap, typename PartFieldMap = EmptyMap,
          typename PartSubPartMap = EmptyMap, typename PartAttributesMap = EmptyMap,
          typename MeshAttributes = EmptyTuple>
class MeshRequirements {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed.
  /// Default construction corresponds to having no requirements and is perfectly valid.
  MeshRequirements() = default;

  /// \brief Copy constructor.
  MeshRequirements(const MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap,
                                          PartTopologyMap, PartRankMap, PartNoInductionMap, PartFieldMap,
                                          PartSubPartMap, PartAttributesMap, MeshAttributes> &other_reqs)
      : field_dim_map_(other_reqs.field_dim_map_),
        field_min_num_states_map_(other_reqs.field_min_num_states_map_),
        field_att_map_(other_reqs.field_att_map_),
        part_state_map_(other_reqs.part_state_map_),
        part_topology_map_(other_reqs.part_topology_map_),
        part_rank_map_(other_reqs.part_rank_map_),
        part_no_induce_map_(other_reqs.part_no_induce_map_),
        part_field_map_(other_reqs.part_field_map_),
        part_subpart_map_(other_reqs.part_subpart_map_),
        part_att_map_(other_reqs.part_att_map_),
        mesh_atts_(other_reqs.mesh_atts_) {
  }

  /// \brief Move constructor.
  MeshRequirements(MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap,
                                    PartTopologyMap, PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap,
                                    PartAttributesMap, MeshAttributes> &&other_reqs)
      : field_dim_map_(std::move(other_reqs.field_dim_map_)),
        field_min_num_states_map_(std::move(other_reqs.field_min_num_states_map_)),
        field_att_map_(std::move(other_reqs.field_att_map_)),
        part_state_map_(std::move(other_reqs.part_state_map_)),
        part_topology_map_(std::move(other_reqs.part_topology_map_)),
        part_rank_map_(std::move(other_reqs.part_rank_map_)),
        part_no_induce_map_(std::move(other_reqs.part_no_induce_map_)),
        part_field_map_(std::move(other_reqs.part_field_map_)),
        part_subpart_map_(std::move(other_reqs.part_subpart_map_)),
        part_att_map_(std::move(other_reqs.part_att_map_)),
        mesh_atts_(std::move(other_reqs.mesh_atts_)) {
  }

  /// \brief Destructor.
  ~MeshRequirements() = default;
  //@}

  //! \name Actions
  //@{

  /// \brief Declare a part with a given rank. It may explicitly contain any entity of lower rank with optional forced
  /// induction.
  ///
  /// Redeclaration of a previously declared part is perfectly valid given that they have the same rank and induction.
  /// If the existing part only has its name set, then redeclaration will set the rank and arg_no_force. Otherwise, we
  /// check for compatibility.
  ///
  /// \param part_name Name of the part.
  /// \param rank Maximum rank of entities in the part.
  /// \param force_no_induce Flag specifying if sub-entities of part members should not become induced members.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  constexpr auto declare_part(const std::string_view &part_name, const stk::mesh::EntityRank &rank,
                              const bool &force_no_induce = false) const & {
    // Check if the part already exists.
    if constexpr (boost::hana::contains(new_part_id_map, part_name)) {
      // The part exists, check for compatibility.
      static_assert(boost::hana::at_key(part_state_map_, part_name) == NAME_AND_TOPOLOGY_SET,
                    "MeshRequirements: Attempting to redeclare a part with a given rank; \n"
                        << "however, the part was previously declared with a given topology.");
      static_assert(boost::hana::at_key(part_state_map_, part_name) == INVALID_STATE,
                    "MeshRequirements: Attempting to redeclare a part with a given rank; \n"
                        << "however, the previously declared part is currently invalid. \n"
                        << "Odd... please contact the development team.");
      static_assert(boost::hana::at_key(part_no_induce_map_, part_name) == force_no_induce,
                    "MeshRequirements: Attempting to redeclare a part with a given induction flag ( "
                        << force_no_induce << " ) incompatible with the existing flag");

      // Plundering not allowed, uses copies of the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{
          field_dim_map_, field_min_num_states_map_, field_att_map_,  part_state_map_,   part_topology_map_,
          part_rank_map_, part_no_induce_map_,       part_field_map_, part_subpart_map_, part_att_map_,
          mesh_atts_};
    } else {
      // The part does not exist, expand the existing maps using copies of the contents of *this.
      constexpr auto new_part_state_map =
          boost::hana::insert(part_state_map_, hana::make_pair(part_name, NAME_AND_RANK_SET));
      constexpr auto new_part_rank_map = boost::hana::insert(part_rank_map_, hana::make_pair(part_name, rank));
      constexpr auto new_part_no_induce_map =
          boost::hana::insert(part_no_induce_map_, hana::make_pair(part_name, force_no_induce));

      using NewPartStateMap = decltype(new_part_state_map);
      using NewPartRankMap = decltype(new_part_rank_map);
      using NewPartNoInductionMap = decltype(new_part_no_induce_map);

      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, NewPartStateMap, PartTopologyMap,
                              NewPartRankMap, NewPartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{field_dim_map_,
                                              field_min_num_states_map_,
                                              field_att_map_,
                                              std::move(new_part_state_map),
                                              part_topology_map_,
                                              std::move(new_part_rank_map),
                                              std::move(new_part_no_induce_map),
                                              part_field_map_,
                                              part_subpart_map_,
                                              part_att_map_,
                                              mesh_atts_};
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
  /// \param force_no_induce Flag specifying if sub-entities of part members should not become induced members.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  constexpr auto declare_part(const std::string_view &part_name, const stk::mesh::EntityRank &rank,
                              const bool &force_no_induce = false) && {
    // Check if the part already exists.
    if constexpr (boost::hana::contains(new_part_id_map, part_name)) {
      // The part exists, check for compatibility.
      static_assert(boost::hana::at_key(part_state_map_, part_name) == NAME_AND_TOPOLOGY_SET,
                    "MeshRequirements: Attempting to redeclare a part with a given rank; \n"
                        << "however, the part was previously declared with a given topology.");
      static_assert(boost::hana::at_key(part_state_map_, part_name) == INVALID_STATE,
                    "MeshRequirements: Attempting to redeclare a part with a given rank; \n"
                        << "however, the previously declared part is currently invalid. \n"
                        << "Odd... please contact the development team.");
      static_assert(boost::hana::at_key(part_rank_map_, part_name) == rank,
                    "MeshRequirements: Attempting to redeclare a part with a given rank ( "
                        << rank << " ) different than the existing rank ("
                        << boost::hana::at_key(part_rank_map_, part_name) << ").");
      static_assert(boost::hana::at_key(part_no_induce_map_, part_name) == force_no_induce,
                    "MeshRequirements: Attempting to redeclare a part with a given induction flag ( "
                        << force_no_induce << " ) incompatible with the existing flag");

      // Plundering allowed, uses move semantics to steal the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                              std::move(field_att_map_),      std::move(part_state_map_),
                                              std::move(part_topology_map_),  std::move(part_rank_map_),
                                              std::move(part_no_induce_map_), std::move(part_field_map_),
                                              std::move(part_subpart_map_),   std::move(part_att_map_),
                                              std::move(mesh_atts_)};
    } else {
      // The part does not exist, expand the existing maps using the contents of *this.
      constexpr auto new_part_state_map =
          boost::hana::insert(std::move(part_state_map_), hana::make_pair(part_name, NAME_AND_RANK_SET));
      constexpr auto new_part_rank_map =
          boost::hana::insert(std::move(part_rank_map_), hana::make_pair(part_name, rank));
      constexpr auto new_part_no_induce_map =
          boost::hana::insert(std::move(part_no_induce_map_), hana::make_pair(part_name, force_no_induce));

      using NewPartStateMap = decltype(new_part_state_map);
      using NewPartRankMap = decltype(new_part_rank_map);
      using NewPartNoInductionMap = decltype(new_part_no_induce_map);

      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, NewPartStateMap, PartTopologyMap,
                              NewPartRankMap, NewPartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{std::move(field_dim_map_),
                                              std::move(field_min_num_states_map_),
                                              std::move(field_att_map_),
                                              std::move(new_part_state_map),
                                              std::move(part_topology_map_),
                                              std::move(new_part_rank_map),
                                              std::move(new_part_no_induce_map),
                                              std::move(part_field_map_),
                                              std::move(part_subpart_map_),
                                              std::move(part_att_map_),
                                              std::move(mesh_atts_)};
    }
  }

  /// \brief Declare a part with given topology. It may contain any element with the given topology with optional forced
  /// induction of downward connected entities.
  ///
  /// Redeclaration of a previously declared part is perfectly valid given that they have the same topology and
  /// induction. In this sense, redeclaration is a no-op with a compatibility check.
  ///
  /// \param part_name Name of the part.
  /// \param topology Topology of entities in the part.
  /// \param force_no_induce Flag specifying if sub-entities of part members should not become induced members.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  constexpr auto declare_part_with_topology(const std::string_view &part_name,
                                            const stk::topology::topology_t &topology,
                                            const bool &force_no_induce = false) const & {
    // Check if the part already exists.
    if constexpr (boost::hana::contains(new_part_id_map, part_name)) {
      // The part exists, check for compatibility.
      static_assert(boost::hana::at_key(part_state_map_, part_name) == NAME_AND_RANK_SET,
                    "MeshRequirements: Attempting to redeclare a part with a given topology; \n"
                        << "however, the part was previously declared with a given rank.");
      static_assert(boost::hana::at_key(part_state_map_, part_name) == INVALID_STATE,
                    "MeshRequirements: Attempting to redeclare a part with a given rank; \n"
                        << "however, the previously declared part is currently invalid. \n"
                        << "Odd... please contact the development team.");
      static_assert(boost::hana::at_key(part_topology_map_, part_name) == topology,
                    "MeshRequirements: Attempting to redeclare a part with a given topology ( "
                        << rank << " ) different than the existing topology ("
                        << boost::hana::at_key(part_rank_map_, part_name) << ").");
      static_assert(boost::hana::at_key(part_no_induce_map_, part_name) == force_no_induce,
                    "MeshRequirements: Attempting to redeclare a part with a given induction flag ( "
                        << force_no_induce << " ) incompatible with the existing flag");

      // Plundering not allowed, uses copies of the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{
          field_dim_map_, field_min_num_states_map_, field_att_map_,  part_state_map_,   part_topology_map_,
          part_rank_map_, part_no_induce_map_,       part_field_map_, part_subpart_map_, part_att_map_,
          mesh_atts_};
    } else {
      // The part does not exist, expand the existing maps using copies of the contents of *this.
      constexpr auto new_part_state_map =
          boost::hana::insert(part_state_map_, hana::make_pair(part_name, NAME_AND_RANK_SET));
      constexpr auto new_part_topology_map =
          boost::hana::insert(part_topology_map_, hana::make_pair(part_name, topology));
      constexpr auto new_part_no_induce_map =
          boost::hana::insert(part_no_induce_map_, hana::make_pair(part_name, force_no_induce));

      using NewPartStateMap = decltype(new_part_state_map);
      using NewPartTopologyMap = decltype(new_part_topology_map);
      using NewPartNoInductionMap = decltype(new_part_no_induce_map);

      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, NewPartStateMap,
                              NewPartTopologyMap, PartRankMap, NewPartNoInductionMap, PartFieldMap, PartSubPartMap,
                              PartAttributesMap, MeshAttributes>{field_dim_map_,
                                                                 field_min_num_states_map_,
                                                                 field_att_map_,
                                                                 std::move(new_part_state_map),
                                                                 std::move(new_part_topology_map_),
                                                                 part_rank_map_,
                                                                 std::move(new_part_no_induce_map),
                                                                 part_field_map_,
                                                                 part_subpart_map_,
                                                                 part_att_map_,
                                                                 mesh_atts_};
    }
  }

  /// \brief Declare a part with given topology. It may contain any element with the given topology with optional forced
  /// induction of downward connected entities.
  ///
  /// Redeclaration of a previously declared part is perfectly valid given that they have the same topology and
  /// induction. In this sense, redeclaration is a no-op with a compatibility check.
  ///
  /// \param part_name Name of the part.
  /// \param topology Topology of entities in the part.
  /// \param force_no_induce Flag specifying if sub-entities of part members should not become induced members.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  constexpr auto declare_part_with_topology(const std::string_view &part_name,
                                            const stk::topology::topology_t &topology,
                                            const bool &force_no_induce = false) && {
    // Check if the part already exists.
    if constexpr (boost::hana::contains(new_part_id_map, part_name)) {
      // The part exists, check for compatibility.
      static_assert(boost::hana::at_key(part_state_map_, part_name) == NAME_AND_RANK_SET,
                    "MeshRequirements: Attempting to redeclare a part with a given topology; \n"
                        << "however, the part was previously declared with a given rank.");
      static_assert(boost::hana::at_key(part_state_map_, part_name) == INVALID_STATE,
                    "MeshRequirements: Attempting to redeclare a part with a given rank; \n"
                        << "however, the previously declared part is currently invalid. \n"
                        << "Odd... please contact the development team.");
      static_assert(boost::hana::at_key(part_topology_map_, part_name) == topology,
                    "MeshRequirements: Attempting to redeclare a part with a given topology ( "
                        << rank << " ) different than the existing topology ("
                        << boost::hana::at_key(part_rank_map_, part_name) << ").");
      static_assert(boost::hana::at_key(part_no_induce_map_, part_name) == force_no_induce,
                    "MeshRequirements: Attempting to redeclare a part with a given induction flag ( "
                        << force_no_induce << " ) incompatible with the existing flag");

      // Plundering allowed, uses move semantics to steal the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                              std::move(field_att_map_),      std::move(part_state_map_),
                                              std::move(part_topology_map_),  std::move(part_rank_map_),
                                              std::move(part_no_induce_map_), std::move(part_field_map_),
                                              std::move(part_subpart_map_),   std::move(part_att_map_),
                                              std::move(mesh_atts_)};
    } else {
      // The part does not exist, expand the existing maps using the contents of *this.
      constexpr auto new_part_state_map =
          boost::hana::insert(std::move(part_state_map_), hana::make_pair(part_name, NAME_AND_RANK_SET));
      constexpr auto new_part_topology_map =
          boost::hana::insert(std::move(part_topology_map_), hana::make_pair(part_name, topology));
      constexpr auto new_part_no_induce_map =
          boost::hana::insert(std::move(part_no_induce_map_), hana::make_pair(part_name, force_no_induce));

      using NewPartStateMap = decltype(new_part_state_map);
      using NewPartTopologyMap = decltype(new_part_topology_map);
      using NewPartNoInductionMap = decltype(new_part_no_induce_map);

      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, NewPartStateMap,
                              NewPartTopologyMap, PartRankMap, NewPartNoInductionMap, PartFieldMap, PartSubPartMap,
                              PartAttributesMap, MeshAttributes>{std::move(field_dim_map_),
                                                                 std::move(field_min_num_states_map_),
                                                                 std::move(field_att_map_),
                                                                 std::move(new_part_state_map),
                                                                 std::move(new_part_topology_map),
                                                                 std::move(part_rank_map_),
                                                                 std::move(new_part_no_induce_map),
                                                                 std::move(part_field_map_),
                                                                 std::move(part_subpart_map_),
                                                                 std::move(part_att_map_),
                                                                 std::move(mesh_atts_)};
    }
  }

  /// \brief Declare a subset relation between two parts.
  ///
  /// An important comment: If you do specify verifyFieldRestrictions = true, this check will be delayed until the
  /// entire mesh is constructed.
  ///
  /// Redeclaration of a previously declared subset relation is perfectly valid and will simply perform a no-op.
  ///
  /// \param superset_part_name Name of the parent/superset part.
  /// \param subset_part_name Name of the child/subset part.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  constexpr auto declare_part_subset(const std::string_view &superset_part_name,
                                     const std::string_view &subset_part_name) const & {
    if constexpr (boost::hana::contains(part_subpart_map_, superset_part_name)) {
      // The given superset part has subset parts.
      if constexpr (boost::hana::contains(part_subpart_map_[superset_part_name], subset_part_name)) {
        // The given subset relation exists; do nothing.

        // Plundering not allowed, uses copies of the contents of *this.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                                MeshAttributes>{
            field_dim_map_, field_min_num_states_map_, field_att_map_,  part_state_map_,   part_topology_map_,
            part_rank_map_, part_no_induce_map_,       part_field_map_, part_subpart_map_, part_att_map_,
            mesh_atts_};
      } else {
        // The given subset relation doesn't exist; create it using a copy of the existing data.
        constexpr auto new_subpart_tuple = boost::hana::append(part_subpart_map_[superset_part_name], subset_part_name);

        // Get the updated map.
        constexpr auto tmp_map = boost::hana::erase_key(part_subpart_map_, superset_part_name);
        constexpr auto new_part_subpart_map =
            boost::hana::insert(std::move(tmp_map), boost::hana::make_pair(superset_part_name, new_subpart_tuple));

        using NewPartSubPartMap = decltype(new_part_subpart_map);

        // Plundering not allowed, uses copies of the contents of *this.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, NewPartSubPartMap, PartAttributesMap,
                                MeshAttributes>{field_dim_map_,
                                                field_min_num_states_map_,
                                                field_att_map_,
                                                part_state_map_,
                                                part_topology_map_,
                                                part_rank_map_,
                                                part_no_induce_map_,
                                                part_field_map_,
                                                std::move(new_part_subpart_map),
                                                part_att_map_,
                                                mesh_atts_};
      }
    } else {
      // The given superset part has no existing subset parts.

      // Insert the new relation into the graph.
      constexpr auto new_part_subpart_map = boost::hana::insert(
          part_subpart_map_, boost::hana::make_pair(superset_part_name, boost::hana::make_tuple(subset_part_name)));

      using NewPartSubPartMap = decltype(new_part_subpart_map);

      // Plundering not allowed, uses copies of the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, NewPartSubPartMap, PartAttributesMap,
                              MeshAttributes>{field_dim_map_,
                                              field_min_num_states_map_,
                                              field_att_map_,
                                              part_state_map_,
                                              part_topology_map_,
                                              part_rank_map_,
                                              part_no_induce_map_,
                                              part_field_map_,
                                              std::move(new_part_subpart_map),
                                              part_att_map_,
                                              mesh_atts_};
    }
  }

  /// \brief Declare a subset relation between two parts.
  ///
  /// An important comment: If you do specify verifyFieldRestrictions = true, this check will be delayed until the
  /// entire mesh is constructed.
  ///
  /// Redeclaration of a previously declared subset relation is perfectly valid and will simply perform a no-op.
  ///
  /// \param superset_part_name Name of the parent/superset part.
  /// \param subset_part_name Name of the child/subset part.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  constexpr auto declare_part_subset(const std::string_view &superset_part_name,
                                     const std::string_view &subset_part_name) && {
    if constexpr (boost::hana::contains(part_subpart_map_, superset_part_name)) {
      // The given superset part has subset parts.
      if constexpr (boost::hana::contains(part_subpart_map_[superset_part_name], subset_part_name)) {
        // The given subset relation exists; do nothing.

        // Plundering allowed, uses move semantics to steal the contents of *this.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                                MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                                std::move(field_att_map_),      std::move(part_state_map_),
                                                std::move(part_topology_map_),  std::move(part_rank_map_),
                                                std::move(part_no_induce_map_), std::move(part_field_map_),
                                                std::move(part_subpart_map_),   std::move(part_att_map_),
                                                std::move(mesh_atts_)};
      } else {
        // The given subset relation doesn't exist; create it using the existing data.
        constexpr auto new_subpart_tuple =
            boost::hana::append(std::move(part_subpart_map_[superset_part_name]), subset_part_name);

        // Get the updated map.
        constexpr auto tmp_map = boost::hana::erase_key(std::move(part_subpart_map_), superset_part_name);
        constexpr auto new_part_subpart_map = boost::hana::insert(
            std::move(tmp_map), boost::hana::make_pair(superset_part_name, std::move(new_subpart_tuple)));

        using NewPartSubPartMap = decltype(new_part_subpart_map);

        // Plundering allowed, uses move semantics to steal the contents of *this.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, NewPartSubPartMap, PartAttributesMap,
                                MeshAttributes>{std::move(field_dim_map_),
                                                std::move(field_min_num_states_map_),
                                                std::move(field_att_map_),
                                                std::move(part_state_map_),
                                                std::move(part_topology_map_),
                                                std::move(part_rank_map_),
                                                std::move(part_no_induce_map_),
                                                std::move(part_field_map_),
                                                std::move(new_part_subpart_map),
                                                std::move(part_att_map_),
                                                std::move(mesh_atts_)};
      }
    } else {
      // The given superset part has no existing subset parts.

      // Insert the new relation into the graph.
      constexpr auto new_part_subpart_map =
          boost::hana::insert(std::move(part_subpart_map_),
                              boost::hana::make_pair(superset_part_name, boost::hana::make_tuple(subset_part_name)));

      using NewPartSubPartMap = decltype(new_part_subpart_map);

      // Plundering allowed, uses move semantics to steal the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, NewPartSubPartMap, PartAttributesMap,
                              MeshAttributes>{std::move(field_dim_map_),
                                              std::move(field_min_num_states_map_),
                                              std::move(field_att_map_),
                                              std::move(part_state_map_),
                                              std::move(part_topology_map_),
                                              std::move(part_rank_map_),
                                              std::move(part_no_induce_map_),
                                              std::move(part_field_map_),
                                              std::move(new_part_subpart_map),
                                              std::move(part_att_map_),
                                              std::move(mesh_atts_)};
    }
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
    if constexpr (boost::hana::contains(field_att_map_, field_name)) {
      // The given field has attributes.
      if constexpr (boost::hana::contains(field_att_map_[field_name], boost::hana::type_c<AttributeType>)) {
        // The given attribute already exists; do nothing.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                                MeshAttributes>{
            field_dim_map_, field_min_num_states_map_, field_att_map_,  part_state_map_,   part_topology_map_,
            part_rank_map_, part_no_induce_map_,       part_field_map_, part_subpart_map_, part_att_map_,
            mesh_atts_};
      } else {
        // The given attribute doesn't exist; append it to a copy of the existing attributes.
        constexpr auto new_attribute_tuple =
            boost::hana::append(field_att_map_[field_name], boost::hana::type_c<AttributeType>);

        // Get the updated map.
        constexpr auto tmp_map = boost::hana::erase_key(field_att_map_, field_name);
        constexpr auto new_field_att_map =
            boost::hana::insert(std::move(tmp_map), boost::hana::make_pair(field_name, std::move(new_attribute_tuple)));

        using NewFieldAttributesMap = decltype(new_field_att_map);

        // Plundering not allowed, uses copies of the contents of *this.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, NewFieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                                MeshAttributes>{field_dim_map_,
                                                field_min_num_states_map_,
                                                std::move(new_field_att_map),
                                                part_state_map_,
                                                part_topology_map_,
                                                part_rank_map_,
                                                part_no_induce_map_,
                                                part_field_map_,
                                                part_subpart_map_,
                                                part_att_map_,
                                                mesh_atts_};
      }
    } else {
      // The given field lacks attributes.

      // Insert the new tuple into the graph.
      constexpr auto new_field_att_map = boost::hana::insert(
          field_att_map_,
          boost::hana::make_pair(field_name, boost::hana::make_tuple(boost::hana::type_c<AttributeType>)));

      using NewFieldAttributesMap = decltype(new_field_att_map);

      // Plundering not allowed, uses copies of the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, NewFieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{field_dim_map_,
                                              field_min_num_states_map_,
                                              std::move(new_field_att_map),
                                              part_state_map_,
                                              part_topology_map_,
                                              part_rank_map_,
                                              part_no_induce_map_,
                                              part_field_map_,
                                              part_subpart_map_,
                                              part_att_map_,
                                              mesh_atts_};
    }
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
    if constexpr (boost::hana::contains(field_att_map_, field_name)) {
      // The given field has attributes.
      if constexpr (boost::hana::contains(field_att_map_[field_name], boost::hana::type_c<AttributeType>)) {
        // The given attribute already exists; do nothing.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                                MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                                std::move(field_att_map_),      std::move(part_state_map_),
                                                std::move(part_topology_map_),  std::move(part_rank_map_),
                                                std::move(part_no_induce_map_), std::move(part_field_map_),
                                                std::move(part_subpart_map_),   std::move(part_att_map_),
                                                std::move(mesh_atts_)};
      } else {
        // The given attribute doesn't exist; append it to the existing attributes.
        constexpr auto new_attribute_tuple =
            boost::hana::append(std::move(field_att_map_[field_name]), boost::hana::type_c<AttributeType>);

        // Get the updated map.
        constexpr auto tmp_map = boost::hana::erase_key(std::move(field_att_map_), field_name);
        constexpr auto new_field_att_map =
            boost::hana::insert(std::move(tmp_map), boost::hana::make_pair(field_name, std::move(new_attribute_tuple)));

        using NewFieldAttributesMap = decltype(new_field_att_map);

        // Plundering allowed, uses move semantics to steal the contents of *this.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                                MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                                std::move(new_field_att_map),   std::move(part_state_map_),
                                                std::move(part_topology_map_),  std::move(part_rank_map_),
                                                std::move(part_no_induce_map_), std::move(part_field_map_),
                                                std::move(part_subpart_map_),   std::move(part_att_map_),
                                                std::move(mesh_atts_)};
      }
    } else {
      // The given field lacks attributes.

      // Insert the new tuple into the graph.
      constexpr auto new_field_att_map = boost::hana::insert(
          std::move(field_att_map_),
          boost::hana::make_pair(field_name, boost::hana::make_tuple(boost::hana::type_c<AttributeType>)));

      using NewFieldAttributesMap = decltype(new_field_att_map);

      // Plundering allowed, uses move semantics to steal the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, NewFieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                              std::move(new_field_att_map),   std::move(part_state_map_),
                                              std::move(part_topology_map_),  std::move(part_rank_map_),
                                              std::move(part_no_induce_map_), std::move(part_field_map_),
                                              std::move(part_subpart_map_),   std::move(part_att_map_),
                                              std::move(mesh_atts_)};
    }
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
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  template <typename AttributeType>
  constexpr auto declare_attribute(const std::string_view &part_name) const & {
    if constexpr (boost::hana::contains(part_att_map_, part_name)) {
      // The given field has attributes.
      if constexpr (boost::hana::contains(part_att_map_[part_name], boost::hana::type_c<AttributeType>)) {
        // The given attribute already exists; do nothing.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                                MeshAttributes>{
            field_dim_map_, field_min_num_states_map_, field_att_map_,  part_state_map_,   part_topology_map_,
            part_rank_map_, part_no_induce_map_,       part_field_map_, part_subpart_map_, part_att_map_,
            mesh_atts_};
      } else {
        // The given attribute doesn't exist; append it to a copy of the existing attributes.
        constexpr auto new_attribute_tuple =
            boost::hana::append(part_att_map_[part_name], boost::hana::type_c<AttributeType>);

        // Get the updated map.
        constexpr auto tmp_map = boost::hana::erase_key(part_att_map_, part_name);
        constexpr auto new_part_att_map =
            boost::hana::insert(std::move(tmp_map), boost::hana::make_pair(part_name, std::move(new_attribute_tuple)));

        using NewPartAttributesMap = decltype(new_part_att_map);

        // Plundering not allowed, uses copies of the contents of *this.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, NewPartAttributesMap,
                                MeshAttributes>{
            field_dim_map_, field_min_num_states_map_, field_att_map_,  part_state_map_,   part_topology_map_,
            part_rank_map_, part_no_induce_map_,       part_field_map_, part_subpart_map_, std::move(new_part_att_map),
            mesh_atts_};
      }
    } else {
      // The given field lacks attributes.

      // Insert the new tuple into the graph.
      constexpr auto new_part_att_map = boost::hana::insert(
          part_att_map_,
          boost::hana::make_pair(part_name, boost::hana::make_tuple(boost::hana::type_c<AttributeType>)));

      using NewPartAttributesMap = decltype(new_part_att_map);

      // Plundering not allowed, uses copies of the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, NewPartAttributesMap,
                              MeshAttributes>{
          field_dim_map_, field_min_num_states_map_, field_att_map_,  part_state_map_,   part_topology_map_,
          part_rank_map_, part_no_induce_map_,       part_field_map_, part_subpart_map_, std::move(new_part_att_map_),
          mesh_atts_};
    }
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
    if constexpr (boost::hana::contains(part_att_map_, part_name)) {
      // The given field has attributes.
      if constexpr (boost::hana::contains(part_att_map_[part_name], boost::hana::type_c<AttributeType>)) {
        // The given attribute already exists; do nothing.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                                MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                                std::move(field_att_map_),      std::move(part_state_map_),
                                                std::move(part_topology_map_),  std::move(part_rank_map_),
                                                std::move(part_no_induce_map_), std::move(part_field_map_),
                                                std::move(part_subpart_map_),   std::move(part_att_map_),
                                                std::move(mesh_atts_)};
      } else {
        // The given attribute doesn't exist; append it to the existing attributes.
        constexpr auto new_attribute_tuple =
            boost::hana::append(std::move(part_att_map_[part_name]), boost::hana::type_c<AttributeType>);

        // Get the updated map.
        constexpr auto tmp_map = boost::hana::erase_key(std::move(part_att_map_), part_name);
        constexpr auto new_part_att_map =
            boost::hana::insert(std::move(tmp_map), boost::hana::make_pair(part_name, std::move(new_attribute_tuple)));

        using NewPartAttributesMap = decltype(new_part_att_map);

        // Plundering allowed, uses move semantics to steal the contents of *this.
        return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                                PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, NewPartAttributesMap,
                                MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                                std::move(field_att_map_),      std::move(part_state_map_),
                                                std::move(part_topology_map_),  std::move(part_rank_map_),
                                                std::move(part_no_induce_map_), std::move(part_field_map_),
                                                std::move(part_subpart_map_),   std::move(new_part_att_map),
                                                std::move(mesh_atts_)};
      }
    } else {
      // The given field lacks attributes.

      // Insert the new tuple into the graph.
      constexpr auto new_part_att_map = boost::hana::insert(
          std::move(part_att_map_),
          boost::hana::make_pair(part_name, boost::hana::make_tuple(boost::hana::type_c<AttributeType>)));

      using NewPartAttributesMap = decltype(new_part_att_map);

      // Plundering allowed, uses move semantics to steal the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, NewPartAttributesMap,
                              MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                              std::move(field_att_map_),      std::move(part_state_map_),
                                              std::move(part_topology_map_),  std::move(part_rank_map_),
                                              std::move(part_no_induce_map_), std::move(part_field_map_),
                                              std::move(part_subpart_map_),   std::move(new_part_att_map),
                                              std::move(mesh_atts_)};
    }
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
    if constexpr (boost::hana::contains(mesh_atts_, boost::hana::type_c<AttributeType>)) {
      // The given attribute already exists; do nothing.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{
          field_dim_map_, field_min_num_states_map_, field_att_map_,  part_state_map_,   part_topology_map_,
          part_rank_map_, part_no_induce_map_,       part_field_map_, part_subpart_map_, part_att_map_,
          mesh_atts_};
    } else {
      // The given attribute doesn't exist; append it to a copy of the existing attributes.
      constexpr auto new_mesh_atts = boost::hana::append(part_att_map_[part_name], boost::hana::type_c<AttributeType>);

      using NewMeshAttributes = decltype(new_mesh_atts);

      // Plundering not allowed, uses copies of the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              NewMeshAttributes>{field_dim_map_,      field_min_num_states_map_, field_att_map_,
                                                 part_state_map_,     part_topology_map_,        part_rank_map_,
                                                 part_no_induce_map_, part_field_map_,           part_subpart_map_,
                                                 part_att_map_,       std::move(new_mesh_atts)};
    }
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
    if constexpr (boost::hana::contains(mesh_atts_, boost::hana::type_c<AttributeType>)) {
      // The given attribute already exists; do nothing.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                              std::move(field_att_map_),      std::move(part_state_map_),
                                              std::move(part_topology_map_),  std::move(part_rank_map_),
                                              std::move(part_no_induce_map_), std::move(part_field_map_),
                                              std::move(part_subpart_map_),   std::move(part_att_map_),
                                              std::move(mesh_atts_)};
    } else {
      // The given attribute doesn't exist; append it to the existing attributes.
      constexpr auto new_mesh_atts = boost::hana::append(std::move(mesh_atts_), boost::hana::type_c<AttributeType>);

      using NewMeshAttributes = decltype(new_mesh_atts);

      // Plundering not allowed, uses the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              NewMeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                                 std::move(field_att_map_),      std::move(part_state_map_),
                                                 std::move(part_topology_map_),  std::move(part_rank_map_),
                                                 std::move(part_no_induce_map_), std::move(part_field_map_),
                                                 std::move(part_subpart_map_),   std::move(part_att_map_),
                                                 std::move(new_mesh_atts)};
    }
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
    // Check if the field (with the given rank and name) already exists.
    constexpr auto key = boost::hana::make_pair(entity_rank, field_name);
    if constexpr (boost::hana::contains(field_dim_map_, key)) {
      // The field exists, check for compatibility.
      static_assert(boost::hana::at_key(field_dim_map_, key) == dimension,
                    "MeshRequirements: Attempting to redeclare a field with name '"
                        << field_name << "' and rank '" << entity_rank << "; \n"
                        << "however, the given dimension '" << dimension
                        << "' is incompatible with the previously declared dimension '"
                        << boost::hana::at_key(field_dim_map_, key) << "'.");

      constexpr auto new_field_min_num_states_map(field_min_num_states_map_);
      boost::hana::at_key(new_field_min_num_states_map, key) = min_number_of_states;

      // Plundering not allowed, uses copies of the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{field_dim_map_,      std::move(new_field_min_num_states_map),
                                              field_att_map_,      part_state_map_,
                                              part_topology_map_,  part_rank_map_,
                                              part_no_induce_map_, part_field_map_,
                                              part_subpart_map_,   part_att_map_,
                                              mesh_atts_};
    } else {
      // The field doesn't exists, expand the existing maps using copies of the contents of *this.
      constexpr auto new_field_dim_map = boost::hana::insert(field_dim_map_, hana::make_pair(key, dimension));
      constexpr auto new_field_min_num_states_map =
          boost::hana::insert(field_min_num_states_map_, hana::make_pair(key, min_number_of_states));

      using NewFieldDimMap = decltype(new_field_dim_map);
      using NewFieldMinNumStatesMap = decltype(new_field_min_num_states_map);

      // Plundering not allowed, uses copies of the contents of *this.
      return MeshRequirements<NewFieldDimMap, NewFieldMinNumStatesMap, FieldAttributesMap, PartStateMap,
                              PartTopologyMap, PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap,
                              PartAttributesMap, MeshAttributes>{std::move(new_field_dim_map),
                                                                 std::move(new_field_min_num_states_map),
                                                                 field_att_map_,
                                                                 part_state_map,
                                                                 part_topology_map_,
                                                                 part_rank_map_,
                                                                 part_no_induce_map,
                                                                 part_field_map_,
                                                                 part_subpart_map_,
                                                                 part_att_map_,
                                                                 mesh_atts_};
    }
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
    // Check if the field (with the given rank and name) already exists.
    constexpr auto key = boost::hana::make_pair(entity_rank, field_name);
    if constexpr (boost::hana::contains(field_dim_map_, key)) {
      // The field exists, check for compatibility.
      static_assert(boost::hana::at_key(field_dim_map_, key) == dimension,
                    "MeshRequirements: Attempting to redeclare a field with name '"
                        << field_name << "' and rank '" << entity_rank << "; \n"
                        << "however, the given dimension '" << dimension
                        << "' is incompatible with the previously declared dimension '"
                        << boost::hana::at_key(field_dim_map_, key) << "'.");

      // TODO(palmerb4): this is incompatible with constexpr. pop and replace.
      boost::hana::at_key(field_min_num_states_map_, key) = min_number_of_states;

      // Plundering allowed, uses move semantics to steal the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                              std::move(field_att_map_),      std::move(part_state_map_),
                                              std::move(part_topology_map_),  std::move(part_rank_map_),
                                              std::move(part_no_induce_map_), std::move(part_field_map_),
                                              std::move(part_subpart_map_),   std::move(part_att_map_),
                                              std::move(mesh_atts_)};
    } else {
      // The field doesn't exists, expand the existing maps using copies of the contents of *this.
      constexpr auto new_field_dim_map =
          boost::hana::insert(std::move(field_dim_map_), hana::make_pair(key, dimension));
      constexpr auto new_field_min_num_states_map =
          boost::hana::insert(std::move(field_min_num_states_map_), hana::make_pair(key, min_number_of_states));

      using NewFieldDimMap = decltype(new_field_dim_map);
      using NewFieldMinNumStatesMap = decltype(new_field_min_num_states_map);

      // Plundering allowed, uses move semantics to steal the contents of *this.
      return MeshRequirements<NewFieldDimMap, NewFieldMinNumStatesMap, FieldAttributesMap, PartStateMap,
                              PartTopologyMap, PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap,
                              PartAttributesMap, MeshAttributes>{
          std::move(new_field_dim_map),  std::move(new_field_min_num_states_map),
          std::move(field_att_map_),     std::move(part_state_map_),
          std::move(part_topology_map_), std::move(part_rank_map_),
          std::move(part_no_induce_map), std::move(part_field_map_),
          std::move(part_subpart_map_),  std::move(part_att_map_),
          std::move(mesh_atts_)};
    }
  }

  /// \brief Put an already-declared field on an already-declared part.
  ///
  /// Redeclaration of an existing field-part connection, is perfectly valid and will perform a no-op.
  ///
  /// \param field_name The name of an already-declared field.
  /// \param field_entity_rank The rank of entities associated with the field.
  /// \param part_name The name of an already-declared part, which should contain said field.
  /// \return The updated MeshRequirements with the newest modifications and a copy of the contents of *this.
  constexpr auto put_field_on_mesh(const std::string_view &field_name, const stk::topology::rank_t &field_entity_rank,
                                   const std::string_view &part_name) const & {
    // Check if the relation between part and field (with the given rank and name) already exists.
    if constexpr (boost::hana::contains(part_field_map_, part_name)) {
      // The field exists on the given part; do nothing.

      // Plundering allowed, uses move semantics to steal the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{
          field_dim_map_, field_min_num_states_map_, field_att_map_,  part_state_map_,   part_topology_map_,
          part_rank_map_, part_no_induce_map_,       part_field_map_, part_subpart_map_, part_att_map_,
          mesh_atts_};
    } else {
      // The field doesn't exist on the part, append it to the existing field tuple using the contents of *this.
      constexpr auto new_field_tuple =
          boost::hana::append(part_field_map_[part_name], boost::hana::make_pair(field_entity_rank, field_name));

      // Get the updated map.
      constexpr auto tmp_map = boost::hana::erase_key(part_field_map_, part_name);
      constexpr auto new_part_field_map =
          boost::hana::insert(std::move(tmp_map), boost::hana::make_pair(part_name, std::move(new_field_tuple)));

      using NewPartFieldMap = decltype(new_part_field_map);

      // Plundering not allowed, uses copies of the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, NewPartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{field_dim_map_,     field_min_num_states_map_,
                                              field_att_map_,     part_state_map_,
                                              part_topology_map_, part_rank_map_,
                                              part_no_induce_map, std::move(new_part_field_map),
                                              part_subpart_map_,  part_att_map_,
                                              mesh_atts_};
    }
  }

  /// \brief Put an already-declared field on an already-declared part.
  ///
  /// Redeclaration of an existing field-part connection, is perfectly valid and will perform a no-op.
  ///
  /// \param field_name The name of an already-declared field.
  /// \param part_name The name of an already-declared part, which should contain said field.
  /// \return The updated MeshRequirements with the newest modifications and perfect forwarding of *this.
  constexpr auto put_field_on_mesh(const std::string_view &field_name, const stk::topology::rank_t &field_entity_rank,
                                   const std::string_view &part_name) && {
    // Check if the relation between part and field (with the given rank and name) already exists.
    if constexpr (boost::hana::contains(part_field_map_, part_name)) {
      // The field exists on the given part; do nothing.

      // Plundering allowed, uses move semantics to steal the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, PartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{std::move(field_dim_map_),      std::move(field_min_num_states_map_),
                                              std::move(field_att_map_),      std::move(part_state_map_),
                                              std::move(part_topology_map_),  std::move(part_rank_map_),
                                              std::move(part_no_induce_map_), std::move(part_field_map_),
                                              std::move(part_subpart_map_),   std::move(part_att_map_),
                                              std::move(mesh_atts_)};
    } else {
      // The field doesn't exist on the part, append it to the existing field tuple using the contents of *this.
      constexpr auto new_field_tuple = boost::hana::append(std::move(part_field_map_[part_name]),
                                                           boost::hana::make_pair(field_entity_rank, field_name));

      // Get the updated map.
      constexpr auto tmp_map = boost::hana::erase_key(std::move(part_field_map_), part_name);
      constexpr auto new_part_field_map =
          boost::hana::insert(std::move(tmp_map), boost::hana::make_pair(part_name, std::move(new_field_tuple)));

      using NewPartFieldMap = decltype(new_part_field_map);

      // Plundering allowed, uses move semantics to steal the contents of *this.
      return MeshRequirements<FieldDimMap, FieldMinNumStatesMap, FieldAttributesMap, PartStateMap, PartTopologyMap,
                              PartRankMap, PartNoInductionMap, NewPartFieldMap, PartSubPartMap, PartAttributesMap,
                              MeshAttributes>{std::move(field_dim_map_),     std::move(field_min_num_states_map_),
                                              std::move(field_att_map_),     std::move(part_state_map_),
                                              std::move(part_topology_map_), std::move(part_rank_map_),
                                              std::move(part_no_induce_map), std::move(new_part_field_map),
                                              std::move(part_subpart_map_),  std::move(part_att_map_),
                                              std::move(mesh_atts_)};
    }
  }

  /// \brief Merge any number of \c MeshRequirements with the current requirements.
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
  /// \tparam FirstMeshRequirements The type of the first MeshRequirements.
  /// \tparam OtherMeshRequirements The type(s) of the other MeshRequirements.
  /// \param first_mesh_reqs [in] The first \c MeshRequirements object to merge with the current object.
  /// \param other_mesh_reqs [in] Any number of other \c MeshRequirements object to merge with the current object.
  /// \return The updated \c MeshRequirements with the newest modifications.
  template <typename FirstMeshRequirements, typename... OtherMeshRequirements>
  constexpr auto merge(FirstMeshRequirements &&first_mesh_reqs, OtherMeshRequirements... &&other_mesh_reqs) &const {
    // Methodology:
    // Merge *this with first_mesh_reqs and then merge the result with other_mesh_reqs.
    // Recurse until other_mesh_reqs is empty.

    // Preform the initial merge by merging each of our individual maps/tuples.
    // Note, the intersection of two maps in boost::hana returns key, value pairs in the first map for which key shows
    // up in the second map. Similarly, the symmetric_difference of two maps in boost::hana returns key, value pairs
    // from each map that do not show up in the intersection. We use this concept to efficiently check for conflicts.

    // TODO(palmerb4): use set instead of tuple when our types need to be unique. This will help with merging.
    // TODO(palmerb4): equality is not allowed for things to be constexpr instead, we need to pop and insert.
    // TODO(palmerb4): constexpr everything!

    // For field_dim_map_, all fields that intersect must have the same dimension.
    // TODO(palmerb4): The key for the fields is a rank name pair.
    constexpr auto new_field_dim_map_part1 = boost::hana::intersection(field_dim_map_, first_mesh_reqs.field_dim_map_);
    boost::hana::for_each(new_field_dim_map_part1, [&](auto pair) {
      static_assert(first_mesh_reqs.field_dim_map_[boost::hana::first(pair)] == boost::hana::second(pair),
                    "MeshRequirements: Invalid input.\n"
                    "One of the provided MeshRequirements has a field with an invalid dimension.\n"
                        << "Invalid field name: " << boost::hana::first(pair)
                        << ". Invalid field dimension: " << first_mesh_reqs.field_dim_map_[boost::hana::first(pair)]
                        << "\nExpected dimension: " << boost::hana::second(pair));
    });
    constexpr auto new_field_dim_map_part2 =
        boost::hana::symmetric_difference(field_dim_map_, first_mesh_reqs.field_dim_map_);
    constexpr auto new_field_dim_map =
        boost::hana::union_(std::move(new_field_dim_map_part1), std::move(new_field_dim_map_part2));

    // For field_min_num_states_map_, merge the maps, keeping the largest min num states for fields in the intersection.
    constexpr auto new_field_min_num_states_map_part1 =
        boost::hana::intersection(field_min_num_states_map_, first_mesh_reqs.field_min_num_states_map_);
    boost::hana::for_each(new_field_dim_map_part1, [&](auto pair) {
      // TODO(palmerb4): incompatible with constexpr.
      boost::hana::second(pair) =
          std::max(first_mesh_reqs.field_dim_map_[boost::hana::first(pair)], boost::hana::second(pair));
    });
    constexpr auto new_field_min_num_states_map_part2 =
        boost::hana::symmetric_difference(field_dim_map_, first_mesh_reqs.field_dim_map_);
    constexpr auto new_field_min_num_states_map =
        boost::hana::union_(std::move(new_field_dim_map_part1), std::move(new_field_dim_map_part2));

    // For field_att_map_, merge the maps keeping only the unique types in each tuple.
    // TODO(palmerb4): If we switch everything to tuples then this merge is trivial.

    // For part_state_map_, all parts in the intersection must be in the same state.
    constexpr auto new_part_state_map_part1 =
        boost::hana::intersection(part_state_map_, first_mesh_reqs.part_state_map_);
    boost::hana::for_each(new_part_state_map_part1, [&](auto pair) {
      static_assert(first_mesh_reqs.part_state_map_[boost::hana::first(pair)] == boost::hana::second(pair),
                    "MeshRequirements: Invalid input.\n"
                    "One of the provided MeshRequirements has a part with an invalid state.\n"
                        << "Invalid part name: " << boost::hana::first(pair)
                        << ". Invalid part state: " << first_mesh_reqs.part_state_map_[boost::hana::first(pair)]
                        << "\nExpected state: " << boost::hana::second(pair));
    });
    constexpr auto new_part_state_map_part2 =
        boost::hana::symmetric_difference(part_state_map_, first_mesh_reqs.part_state_map_);
    constexpr auto new_part_state_map =
        boost::hana::union_(std::move(new_part_state_map_part1), std::move(new_part_state_map_part2));

    // For part_topology_map_, all parts in the intersection must have the same topology.
    constexpr auto new_part_topology_map_part1 =
        boost::hana::intersection(part_topology_map_, first_mesh_reqs.part_topology_map_);
    boost::hana::for_each(new_part_topology_map_part1, [&](auto pair) {
      static_assert(first_mesh_reqs.part_topology_map_[boost::hana::first(pair)] == boost::hana::second(pair),
                    "MeshRequirements: Invalid input.\n"
                    "One of the provided MeshRequirements has a part with an invalid topology.\n"
                        << "Invalid part name: " << boost::hana::first(pair)
                        << ". Invalid part topology: " << first_mesh_reqs.part_topology_map_[boost::hana::first(pair)]
                        << "\nExpected topology: " << boost::hana::second(pair));
    });
    constexpr auto new_part_topology_map_part2 =
        boost::hana::symmetric_difference(part_topology_map_, first_mesh_reqs.part_topology_map_);
    constexpr auto new_part_topology_map =
        boost::hana::union_(std::move(new_part_topology_map_part1), std::move(new_part_topology_map_part2));

    // For part_rank_map_, all parts in the intersection must have the same rank.
    constexpr auto new_part_rank_map_part1 = boost::hana::intersection(part_rank_map_, first_mesh_reqs.part_rank_map_);
    boost::hana::for_each(new_part_rank_map_part1, [&](auto pair) {
      static_assert(first_mesh_reqs.part_rank_map_[boost::hana::first(pair)] == boost::hana::second(pair),
                    "MeshRequirements: Invalid input.\n"
                    "One of the provided MeshRequirements has a part with an invalid rank.\n"
                        << "Invalid part name: " << boost::hana::first(pair)
                        << ". Invalid part rank: " << first_mesh_reqs.part_rank_map_[boost::hana::first(pair)]
                        << "\nExpected rank: " << boost::hana::second(pair));
    });
    constexpr auto new_part_rank_map_part2 =
        boost::hana::symmetric_difference(part_rank_map_, first_mesh_reqs.part_rank_map_);
    constexpr auto new_part_rank_map =
        boost::hana::union_(std::move(new_part_rank_map_part1), std::move(new_part_rank_map_part2));

    // For part_no_induce_map_, all parts in the intersection must have the same induction flag.
    constexpr auto new_part_no_induce_map_part1 =
        boost::hana::intersection(part_no_induce_map_, first_mesh_reqs.part_no_induce_map_);
    boost::hana::for_each(new_part_no_induce_map_part1, [&](auto pair) {
      static_assert(first_mesh_reqs.part_no_induce_map_[boost::hana::first(pair)] == boost::hana::second(pair),
                    "MeshRequirements: Invalid input.\n"
                    "One of the provided MeshRequirements has a part with an invalid induction flag.\n"
                        << "Invalid part name: " << boost::hana::first(pair) << ". Invalid part induction flag: "
                        << first_mesh_reqs.part_no_induce_map_[boost::hana::first(pair)]
                        << "\nExpected flag: " << boost::hana::second(pair));
    });
    constexpr auto new_part_no_induce_map_part2 =
        boost::hana::symmetric_difference(part_no_induce_map_, first_mesh_reqs.part_no_induce_map_);
    constexpr auto new_part_no_induce_map =
        boost::hana::union_(std::move(new_part_no_induce_map_part1), std::move(new_part_no_induce_map_part2));

    // For part_field_map_, because the fields have already been merged, we can merge the unique elements of each tuple.

    // For part_subpart_map_, because the parts have already been merged, we can merge the unique elements of each
    // tuple.

    // For part_att_map_, merge the maps keeping only the unique types in each tuple. For mesh_atts_, keep only
    // the unique types in each tuple.

    if constexpr (sizeof...(other_mesh_reqs) > 1) {
      // Recurse!
      return SOMETHING.merge(std::forward<OtherMeshRequirements>(other_mesh_reqs));
    } else {
      // Recursion complete, return the requirements.
      return SOMETHING;
    }
  }
  //@}

 private:
  //! \name Private constructors
  //@{

  /// \brief Constructor will full fill.
  /// \param field_dim_map A map from field name to field dimension.
  /// \param field_min_num_states_map A map from field name to field min num states.
  /// \param field_att_map A map from field name to field attributes (stored as a tuple).
  /// \param part_state_map A map from part name to part state.
  /// \param part_topology_map A map from part name to part topology.
  /// \param part_rank_map A map from part name to part rank.
  /// \param part_no_induce_map A map from part name to part force no induction flag.
  /// \param part_field_map_ A map from part name to part field names (stored as a tuple).
  /// \param part_subpart_map A map from part name to part subpart names (stored as a tuple).
  /// \param part_att_map A map from part name to part attributes (stored as a tuple).
  /// \param mesh_atts A tuple of mesh attributes.
  MeshRequirements(FieldDimMap field_dim_map, FieldMinNumStatesMap field_min_num_states_map,
                   FieldAttributesMap field_att_map, PartStateMap part_state_map, PartTopologyMap part_topology_map,
                   PartRankMap part_rank_map, PartNoInductionMap part_no_induce_map, PartFieldMap part_field_map_,
                   PartSubPartMap part_subpart_map, PartAttributesMap part_att_map, MeshAttributes mesh_atts)
      : field_dim_map_(field_dim_map),
        field_min_num_states_map_(field_min_num_states_map),
        field_att_map_(field_att_map),
        part_state_map_(part_state_map),
        part_topology_map_(part_topology_map),
        part_rank_map_(part_rank_map),
        part_no_induce_map_(part_no_induce_map),
        part_field_map_(part_field_map),
        part_subpart_map_(part_subpart_map),
        part_att_map_(part_att_map),
        mesh_atts_(mesh_atts) {
  }
  //@}

  //! \name Helper enums
  //@{

  /// \brief An enum for determining the state of a part.
  enum part_state_t : int8_t { INVALID_STATE, NAME_AND_RANK_SET, NAME_AND_TOPOLOGY_SET };
  //@}

  //! \name Internal members
  //@{

  /// \brief A map from field name to field dimension.
  FieldDimMap field_dim_map_;

  /// \brief A map from field name to field min num states.
  FieldMinNumStatesMap field_min_num_states_map_;

  /// \brief A map from field name to field attributes (stored as a tuple).
  FieldAttributesMap field_att_map_;

  /// \brief A map from part name to part state.
  PartStateMap part_state_map_;

  /// \brief A map from part name to part topology.
  PartTopologyMap part_topology_map_;

  /// \brief A map from part name to part rank.
  PartRankMap part_rank_map_;

  /// \brief A map from part name to part force no induction flag.
  PartNoInductionMap part_no_induce_map_;

  /// \brief  A map from part name to part field names (stored as a tuple).
  PartFieldMap part_field_map_;

  /// \brief A map from part name to part subpart names (stored as a tuple).
  PartSubPartMap part_subpart_map_;

  /// \brief A map from part name to part attributes (stored as a tuple).
  PartAttributesMap part_att_map_;

  /// \brief A tuple of mesh attributes.
  MeshAttributes mesh_atts_;
  //@}

  //! \name Friends
  //@{

  /// \brief We're friends with every other MeshRequirements.
  /// TODO(palmerb4): Although this violates encapsulation, I can find no other way to have merge to work. Ideally, we
  /// would be friends with a non-member function (like tuple_cat), but I don't see how one could write the friend
  /// declaration for such a class.
  template <typename, typename, typename, typename, typename, typename, typename, typename, typename, typename,
            typename>
  friend class MeshRequirements;
  //@}
};  // MeshRequirements

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_MESHREQUIREMENTS_HPP_
