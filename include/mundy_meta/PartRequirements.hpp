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

#ifndef MUNDY_META_PARTREQUIREMENTS_HPP_
#define MUNDY_META_PARTREQUIREMENTS_HPP_

/// \file PartRequirements.hpp
/// \brief Declaration of the PartRequirements class

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements, mundy::meta::FieldRequirementsBase

namespace mundy {

namespace meta {

//! \name Helper functions
//@{

/// \brief Map a string with a valid topology name to the corresponding topology.
///
/// The set of valid topology names and their corresponding return type is
///  - No rank topologies
///     - INVALID_TOPOLOGY                       -> stk::topology::INVALID_TOPOLOGY
///  - Node rank topologies
///     - NODE                                   -> stk::topology::NODE
///  - Edge rank topologies
///     - LINE_2                                 -> stk::topology::LINE_2
///     - LINE_3                                 -> stk::topology::LINE_3
///  - Face rank topologies
///     - TRI_3 or TRIANGLE_3                    -> stk::topology::TRI_3
///     - TRI_4 or TRIANGLE_4                    -> stk::topology::TRI_4
///     - TRI_6 or TRIANGLE_6                    -> stk::topology::TRI_6
///     - QUAD_4 or QUADRILATERAL_4              -> stk::topology::QUAD_4
///     - QUAD_6 or QUADRILATERAL_6              -> stk::topology::QUAD_6
///     - QUAD_8 or QUADRILATERAL_8              -> stk::topology::QUAD_8
///     - QUAD_9 or QUADRILATERAL_9              -> stk::topology::QUAD_9
///   - Element rank topologies
///     - PARTICLE                               -> stk::topology::PARTICLE
///     - LINE_2_1D                              -> stk::topology::LINE_2_1D
///     - LINE_3_1D                              -> stk::topology::LINE_3_1D
///     - BEAM_2                                 -> stk::topology::BEAM_2
///     - BEAM_3                                 -> stk::topology::BEAM_3
///     - SHELL_LINE_2                           -> stk::topology::SHELL_LINE_2
///     - SHELL_LINE_3                           -> stk::topology::SHELL_LINE_3
///     - SPRING_2                               -> stk::topology::SPRING_2
///     - SPRING_3                               -> stk::topology::SPRING_3
///     - TRI_3_2D or TRIANGLE_3_2D              -> stk::topology::TRI_3_2D
///     - TRI_4_2D or TRIANGLE_4_2D              -> stk::topology::TRI_4_2D
///     - TRI_6_2D or TRIANGLE_6_2D              -> stk::topology::TRI_6_2D
///     - QUAD_4_2D or QUADRILATERAL_4_2D        -> stk::topology::QUAD_4_2D
///     - QUAD_8_2D or QUADRILATERAL_8_2D        -> stk::topology::QUAD_8_2D
///     - QUAD_9_2D or QUADRILATERAL_9_2D        -> stk::topology::QUAD_9_2D
///     - SHELL_TRI_3 or SHELL_TRIANGLE_3        -> stk::topology::SHELL_TRI_3
///     - SHELL_TRI_4 or SHELL_TRIANGLE_4        -> stk::topology::SHELL_TRI_4
///     - SHELL_TRI_6 or SHELL_TRIANGLE_6        -> stk::topology::SHELL_TRI_6
///     - SHELL_QUAD_4 or SHELL_QUADRILATERAL_4  -> stk::topology::SHELL_QUAD_4
///     - SHELL_QUAD_8 or SHELL_QUADRILATERAL_8  -> stk::topology::SHELL_QUAD_8
///     - SHELL_QUAD_9 or SHELL_QUADRILATERAL_9  -> stk::topology::SHELL_QUAD_9
///     - TET_4 or TETRAHEDRON_4                 -> stk::topology::TET_4
///     - TET_8 or TETRAHEDRON_8                 -> stk::topology::TET_8
///     - TET_10 or TETRAHEDRON_10               -> stk::topology::TET_10
///     - TET_11 or TETRAHEDRON_11               -> stk::topology::TET_11
///     - PYRAMID_5                              -> stk::topology::PYRAMID_5
///     - PYRAMID_13                             -> stk::topology::PYRAMID_13
///     - PYRAMID_14                             -> stk::topology::PYRAMID_14
///     - WEDGE_6                                -> stk::topology::WEDGE_6
///     - WEDGE_12                               -> stk::topology::WEDGE_12
///     - WEDGE_15                               -> stk::topology::WEDGE_15
///     - WEDGE_18                               -> stk::topology::WEDGE_18
///     - HEX_8 or HEXAHEDRON_8                  -> stk::topology::HEX_8
///     - HEX_20 or HEXAHEDRON_20                -> stk::topology::HEX_20
///     - HEX_27 or HEXAHEDRON_27                -> stk::topology::HEX_27
///   - Super topologies
///     - SUPEREDGE<N>                           -> create_superedge_topology(N)
///     - SUPERFACE<N>                           -> create_superface_topology(N)
///     - SUPERELEMENT<N>                        -> create_superelement_topology(N)
/// \param rank_string [in] String containing a valid rank name.
stk::topology map_string_to_topology(const std::string &topology_string);
//@}

/// \class PartRequirements
/// \brief A set requirements imposed upon a Part and its fields.
class PartRequirements {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements.
  PartRequirements() = default;

  /// \brief Constructor with partial requirements. Version 1.
  ///
  /// \param part_name [in] Name of the part.
  ///
  /// \param part_topology [in] Topology of entities within the part.
  PartRequirements(const std::string &part_name, const stk::topology::topology_t &part_topology);

  /// \brief Constructor with partial requirements. Version 2.
  ///
  /// \param part_name [in] Name of the part. If the name already exists for the given topology, the two parameter
  /// sets must be valid.
  ///
  /// \param part_rank [in] Maximum rank of entities within the part. Can contain any element of lower rank, regardless
  /// of topology.
  ///
  /// \param part_fields [in] Vector of field parameters for the fields defined on this part.
  PartRequirements(const std::string &part_name, const stk::topology::rank_t &part_rank);
  /// \brief Construct from a parameter list.
  ///
  /// \param parameter_list [in] Optional list of parameters for specifying the part requirements. The set of valid
  /// parameters is accessible via \c get_valid_params.
  explicit PartRequirements(const Teuchos::ParameterList &parameter_list);
  //@}

  //! \name Setters and Getters
  //@{

  /// \brief Set the required part name.
  /// \brief part_name [in] Required name of the part.
  void set_part_name(const std::string &part_name);

  /// \brief Set the required part topology.
  /// \brief part_topology [in] Required topology of the part.
  void set_part_topology(const stk::topology::topology_t &part_topology);

  /// \brief Set the required part topology using a valid topology string name.
  /// \brief part_topology_name [in] Required topology of the part.
  void set_part_topology(const std::string &part_topology_string);

  /// \brief Set the required part rank.
  /// \brief part_rank [in] Required rank of the part.
  void set_part_rank(const stk::topology::rank_t &part_rank);

  /// \brief Set the required part rank using a valid rank string name.
  /// \brief part_rank [in] Required rank of the part.
  void set_part_rank(const std::string &part_rank_string);

  /// \brief Get if the part name is constrained or not.
  bool constrains_part_name() const;

  /// \brief Get if the part topology is constrained or not.
  bool constrains_part_topology() const;

  /// \brief Get if the part rank is constrained or not.
  bool constrains_part_rank() const;

  /// \brief Return the part name.
  /// Will throw an error if the part name is not constrained.
  std::string get_part_name() const;

  /// \brief Return the part topology.
  /// Will throw an error if the part topology is not constrained.
  stk::topology::topology_t get_part_topology() const;

  /// \brief Return the part rank.
  /// Will throw an error if the part rank is not constrained.
  stk::topology::rank_t get_part_rank() const;

  /// \brief Return the part field map.
  /// \brief field_rank [in] Rank associated with the retrieved fields.
  std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> get_part_field_map();

  /// \brief Get the default parameters for this class.
  static Teuchos::ParameterList get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("name", "INVALID", "Name of the part.");
    default_parameter_list.set("topology", stk::topology::INVALID_TOPOLOGY, "Topology of the part.");
    default_parameter_list.set("rank", stk::topology::INVALID_RANK, "Rank of the part.");
    return default_parameter_list;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Declare the part that this class defines including any of its fields, its subparts, and their
  /// fields/subparts.
  ///
  /// This method can return three different types of parts based on the existing set of constraints.
  ///  - those with a predefined topology (if constrains_part_topology is true),
  ///  - those with a predefined rank that will (if constrains_part_topology is false but constrains_part_rank is true),
  ///  - those with no topology or rank (if neither constrains_part_topology nor constrains_part_rank are true).
  ///
  /// In each case, the part name must be set or an error will be thrown.
  ///
  /// \note Redeclaration of a previously declared part, will return the previous part.
  stk::mesh::Part &declare_part(stk::mesh::MetaData *const meta_data_ptr) const;

  /// \brief Delete the part name constraint (if it exists).
  void delete_part_name_constraint();

  /// \brief Delete the part topology constraint (if it exists).
  void delete_part_topology_constraint();

  /// \brief Delete the part rank constraint (if it exists).
  void delete_part_rank_constraint();

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means:
  ///   1. the rank of the fields does not exceed the rank of the part's topology.
  void check_if_valid() const;

  /// \brief Add the provided field to the part, given that it is valid and does not conflict with existing fields.
  ///
  /// \param field_reqs_ptr [in] Pointer to the field parameters to add to the part.
  void add_field_reqs(std::shared_ptr<FieldRequirementsBase> field_reqs_ptr);

  /// \brief Add the provided part as a subpart of this part, given that it is valid.
  ///
  /// TODO(palmerb4): Are there any restrictions on what can and cannot be a subpart? If so, encode them here.
  ///
  /// \param part_reqs_ptr [in] Pointer to the sub-part requirements to add to the part.
  void add_subpart_reqs(std::shared_ptr<PartRequirements> part_reqs_ptr);

  /// \brief Merge the current requirements with any number of other \c PartRequirements.
  ///
  /// Here, merging two a \c PartRequirements object with this object amounts to merging their fields. For this process
  /// to be valid, the given \c PartRequirements must have the same topology and rank. The name of the other part does
  /// not need to match the current name of this part.
  ///
  /// \param part_req_ptr [in] An \c PartRequirements object to merge with the current object.
  void merge(const std::shared_ptr<PartRequirements> &part_req_ptr);

  /// \brief Merge the current requirements with any number of other \c PartRequirements.
  ///
  /// Here, merging two a \c PartRequirements object with this object amounts to merging their fields. For this process
  /// to be valid, the given \c PartRequirements must have the same topology and rank. The name of the other part does
  /// not need to match the current name of this part.
  ///
  /// \param vector_of_part_req_ptrs [in] A vector of pointers to other \c PartRequirements objects to merge with the
  /// current object.
  void merge(const std::vector<std::shared_ptr<PartRequirements>> &vector_of_part_req_ptrs);
  //@}

 private:
  /// \brief Name of the part.
  std::string part_name_;

  /// \brief Topology of entities in the part.
  stk::topology::topology_t part_topology_;

  /// \brief Rank of the part.
  stk::topology::rank_t part_rank_;

  /// \brief If the name of the part is set or not.
  bool part_name_is_set_ = false;

  /// \brief If the topology of entities in the part is set or not.
  bool part_topology_is_set_ = false;

  /// \brief If the rank of the part is set or not.
  bool part_rank_is_set_ = false;

  /// \brief A set of maps from field name to field params for each rank.
  std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> part_ranked_field_maps_{
      stk::topology::NUM_RANKS};

  /// \brief A map from subpart name to the part params of each sub-part.
  std::map<std::string, std::shared_ptr<PartRequirements>> part_subpart_map_;
};  // PartRequirements

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_PARTREQUIREMENTS_HPP_
