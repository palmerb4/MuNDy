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
#include <sstream>      // for std::stringstream
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_mesh/MetaData.hpp>           // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements, mundy::meta::FieldRequirementsBase

namespace mundy {

namespace meta {

/// \class PartRequirements
/// \brief A set requirements imposed upon a Part and its fields.
class PartRequirements {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements.
  PartRequirements() = default;

  /// \brief Fully  Constructor with partial requirements. Version 1.
  ///
  /// \param part_name [in] Name of the part.
  explicit PartRequirements(const std::string &part_name);

  /// \brief Constructor with partial requirements. Version 2.
  ///
  /// \param part_name [in] Name of the part.
  ///
  /// \param part_topology [in] Topology of entities within the part.
  PartRequirements(const std::string &part_name, const stk::topology::topology_t &part_topology);

  /// \brief Constructor with partial requirements. Version 3.
  ///
  /// \param part_name [in] Name of the part. If the name already exists for the given topology, the two parameter
  /// sets must be valid.
  ///
  /// \param part_rank [in] Maximum rank of entities within the part. Can contain any element of lower rank, regardless
  /// of topology.
  ///
  /// \param part_fields [in] Vector of field parameters for the fields defined on this part.
  PartRequirements(const std::string &part_name, const stk::topology::rank_t &part_rank);
  //@}

  //! \name Setters
  //@{

  /// \brief Set the required part name.
  /// \param part_name [in] Required name of the part.
  void set_part_name(const std::string &part_name);

  /// \brief Set the required part topology.
  /// \param part_topology [in] Required topology of the part.
  void set_part_topology(const stk::topology::topology_t &part_topology);

  /// \brief Set the required part rank.
  /// \param part_rank [in] Required rank of the part.
  void set_part_rank(const stk::topology::rank_t &part_rank);

  /// \brief Delete the part name constraint (if it exists).
  void delete_part_name();

  /// \brief Delete the part topology constraint (if it exists).
  void delete_part_topology();

  /// \brief Delete the part rank constraint (if it exists).
  void delete_part_rank();

  /// \brief Add the provided field to the part, given that it is valid and does not conflict with existing fields.
  ///
  /// \param field_req_ptr [in] Pointer to the field parameters to add to the part.
  void add_field_reqs(std::shared_ptr<FieldRequirementsBase> field_req_ptr);

  /// \brief Add the provided part as a subpart of this part, given that it is valid.
  ///
  /// TODO(palmerb4): Are there any restrictions on what can and cannot be a subpart? If so, encode them here.
  ///
  /// \param part_req_ptr [in] Pointer to the sub-part requirements to add to the part.
  void add_subpart_reqs(std::shared_ptr<PartRequirements> part_req_ptr);

  /// \brief Require that an attribute with the given name be present on the part.
  ///
  /// \param attribute_name [in] The name of the attribute that must be present on the part.
  void add_part_attribute(const std::string &attribute_name);
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the part name is constrained or not.
  bool constrains_part_name() const;

  /// \brief Get if the part topology is constrained or not.
  bool constrains_part_topology() const;

  /// \brief Get if the part rank is constrained or not.
  bool constrains_part_rank() const;

  /// @brief Get if the part is fully specified.
  bool is_fully_specified() const;

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
  std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> get_part_ranked_field_map();

  /// \brief Return the part subpart map.
  std::map<std::string, std::shared_ptr<PartRequirements>> get_part_subpart_map();

  /// \brief Return the required part attribute names.
  std::vector<std::string> get_part_attribute_names();
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
  stk::mesh::Part &declare_part_on_mesh(mundy::mesh::MetaData *const meta_data_ptr) const;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means:
  ///   1. the rank of the fields does not exceed the rank of the part's topology.
  void check_if_valid() const;

  /// \brief Merge the current requirements with another \c PartRequirements.
  ///
  /// \param part_req_ptr [in] An \c PartRequirements object to merge with the current object.
  void merge(const std::shared_ptr<PartRequirements> &part_req_ptr);

  /// \brief Merge the current requirements with any number of other \c PartRequirements.
  ///
  /// \param vector_of_part_req_ptrs [in] A vector of pointers to other \c PartRequirements objects to merge with the
  /// current object.
  void merge(const std::vector<std::shared_ptr<PartRequirements>> &vector_of_part_req_ptrs);

  /// \brief Dump the contents of \c PartRequirements to the given stream (defaults to std::cout).
  void print_reqs(std::ostream &os = std::cout, int indent_level = 0) const;

  /// \brief Return a string representation of the current set of requirements.
  std::string get_reqs_as_a_string() const;
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

  /// \brief If the part is an io-compatible part or not.
  bool is_io_part_ = false;

  /// \brief A set of maps from field name to field params for each rank.
  std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> part_ranked_field_maps_{
      stk::topology::NUM_RANKS};

  /// \brief A map from subpart name to the part params of each sub-part.
  std::map<std::string, std::shared_ptr<PartRequirements>> part_subpart_map_;

  /// \brief A vector of required part attribute names.
  std::vector<std::string> required_part_attribute_names_;
};  // PartRequirements

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_PARTREQUIREMENTS_HPP_
