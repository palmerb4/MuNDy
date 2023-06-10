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
#include <tuple>        // for std::tuple
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements

namespace mundy {

namespace meta {

/// \class PartRequirements
/// \brief A set of requirements imposed upon a Part and its fields.
///
/// \note \c PartAttributeTypes is an explicit template whereas \c FieldRequirementTypes and \c SubPartRequirementTypes
/// can be deduced by the compiler upon construction, i.e., there is no need to specify \c FieldRequirementTypes or
/// \c SubPartRequirementTypes.
///
/// \tparam PartAttributeTypes A set of required field attribute types. Warning, types must be unique.
/// \tparam FieldRequirementTypes The types for each field requirements defined on this part.
/// \tparam SubPartRequirementTypes The types for each subpart requirements defined on this part.
template <typename... PartAttributeTypes, typename... FieldRequirementTypes, typename... SubPartRequirementTypes,
          std::enable_if_t<impl::are_types_unique<PartAttributeTypes>::value, bool> = true>
class PartRequirements {
 public:
  //! \name Typedefs
  //@{

  /// \param part_attribute_types The set of unique part attribute types. Set by the template parameter.
  using part_attribute_types = impl::unique_tuple<PartAttributeTypes...>::type;

  /// \param field_requirements_types Tuple of types for each field requirements defined on this part.
  using field_requirements_types = std::tuple<FieldRequirementTypes...>;

  /// \param subpart_requirements_types Tuple of types for each subpart requirements defined on this part.
  using subpart_requirement_types = std::tuple<SubPartRequirementTypes...>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor with partial requirements. Version 1: Fixed topology.
  ///
  /// \param name [in] Name of the part.
  /// \param topology [in] Topology of entities within the part.
  /// \param field_requirements [in] Tuple of field requirements for the fields defined on this part.
  /// \param subpart_requirements [in] Tuple of requirements for the subparts of this part.
  PartRequirements(const std::string &name, const stk::topology::topology_t &topology,
                   const std::tuple<FieldRequirementTypes...> &field_requirements = std::tuple<>(),
                   const std::tuple<SubPartRequirementTypes...> &subpart_requirements = std::tuple<>());

  /// \brief Constructor with partial requirements. Version 2. Fixed rank.
  ///
  /// \param name [in] Name of the part.
  /// \param rank [in] Maximum rank of entities within the part. Can contain any element of lower rank, regardless
  /// of topology.
  /// \param field_requirements [in] Tuple of field requirements for the fields defined on this part.
  /// \param subpart_requirements [in] Tuple of requirements for the subparts of this part.
  PartRequirements(const std::string &name, const stk::topology::rank_t &rank,
                   const std::tuple<FieldRequirementTypes...> field_requirements = std::tuple<>(),
                   const std::tuple<SubPartRequirementTypes...> &subpart_requirements = std::tuple<>());

  /// \brief Constructor with partial requirements. Version 3. Neither topology nor rank are fixed.
  ///
  /// \param name [in] Name of the part.
  /// \param field_requirements [in] Tuple of field requirements for the fields defined on this part.
  /// \param subpart_requirements [in] Tuple of requirements for the subparts of this part.
  PartRequirements(const std::string &name,
                   const std::tuple<FieldRequirementTypes...> field_requirements = std::tuple<>(),
                   const std::tuple<SubPartRequirementTypes...> &subpart_requirements = std::tuple<>());
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the part topology is constrained or not.
  bool constrains_part_topology() const;

  /// \brief Get if the part rank is constrained or not.
  bool constrains_part_rank() const;

  /// \brief Return the part name.
  std::string get_name() const;

  /// \brief Return the part topology.
  /// Will throw an error if the part topology is not constrained.
  stk::topology::topology_t get_topology() const;

  /// \brief Return the part rank.
  /// Will throw an error if the part rank is not constrained.
  stk::topology::rank_t get_rank() const;

  /// \brief Return the set of field requirements defined on this class.
  field_requirements_types get_field_requirements() const;

  /// \brief Return the set of subpart requirements defined on this class.
  subpart_requirement_types get_subpart_requirements() const;
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
  stk::mesh::Part &declare_part_on_mesh(stk::mesh::MetaData *const meta_data_ptr) const;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means:
  ///   1. the rank of the fields does not exceed the rank of the part's topology.
  void check_if_valid() const;
  //@}

 private:
  /// \brief Name of the part.
  std::string part_name_;

  /// \brief Topology of entities in the part.
  stk::topology::topology_t part_topology_;

  /// \brief Rank of the part.
  stk::topology::rank_t part_rank_;

  /// \brief If the topology of entities in the part is set or not.
  bool part_topology_is_set_ = false;

  /// \brief If the rank of the part is set or not.
  bool part_rank_is_set_ = false;

  /// \brief Tuple of field requirements for the fields defined on this part.
  field_requirements_types field_requirements_;

  /// \brief Tuple of requirements for the subparts of this part.
  subpart_requirement_types subpart_requirements_;

  /// \brief A set of maps from field name to index of the field_requirements_ tuple for each rank.
  std::vector<std::map<std::string, std::size_t>> part_ranked_field_maps_{stk::topology::NUM_RANKS};

  /// \brief A map from subpart name to index of the subpart_requirements_.
  std::map<std::string, std::size_t> part_subpart_map_;
};  // PartRequirements

//! \name Template implementations
//@{

// \name Constructors and destructor
//{

PartRequirements::PartRequirements(const std::string &part_name, const stk::topology::topology_t &part_topology) {
  this->set_part_name(part_name);
  this->set_part_topology(part_topology);
}

PartRequirements::PartRequirements(const std::string &part_name, const stk::topology::rank_t &part_rank) {
  this->set_part_name(part_name);
  this->set_part_rank(part_rank);
}

PartRequirements::PartRequirements(const Teuchos::ParameterList &parameter_list) {
  // Validate the input params. Throws an error if a parameter is defined but not in the valid params.
  // This helps catch misspellings.
  parameter_list.validateParameters(this->get_valid_params());

  // Store the given parameters.
  if (parameter_list.isParameter("name")) {
    const std::string part_name = parameter_list.get<std::string>("name");
    this->set_part_name(part_name);
  }
  if (parameter_list.isParameter("topology")) {
    const std::string part_topology_name = parameter_list.get<std::string>("topology");
    this->set_part_topology(part_topology_name);
  }
  if (parameter_list.isParameter("rank")) {
    const std::string part_rank_name = parameter_list.get<std::string>("rank");
    this->set_part_topology(part_rank_name);
  }

  // Store the field params.
  if (parameter_list.isSublist("fields")) {
    const Teuchos::ParameterList &fields_sublist = parameter_list.sublist("fields");
    const unsigned num_fields = fields_sublist.get<unsigned>("count");
    for (unsigned i = 0; i < num_fields; i++) {
      const Teuchos::ParameterList &field_i_sublist = parameter_list.sublist("field_" + std::to_string(i));
      const std::string field_type_string = field_i_sublist.get<std::string>("type");
      std::shared_ptr<FieldRequirementsBase> field_i =
          FieldRequirementsFactory::create_new_instance(field_type_string, field_i_sublist);
      this->add_field_req(field_i);
    }
  }

  // Store the sub-part params.
  if (parameter_list.isSublist("sub_parts")) {
    const Teuchos::ParameterList &subparts_sublist = parameter_list.sublist("sub_parts");
    const unsigned num_subparts = subparts_sublist.get<unsigned>("count");
    for (unsigned i = 0; i < num_subparts; i++) {
      const Teuchos::ParameterList &subpart_i_sublist = parameter_list.sublist("sub_part_" + std::to_string(i));
      std::shared_ptr<PartRequirements> subpart_i = std::make_shared<PartRequirements>(subpart_i_sublist);
      this->add_subpart_reqs(subpart_i);
    }
  }
}
//}

// \name Getters
//{

bool PartRequirements::constrains_part_topology() const {
  return part_name_is_set_;
}

bool PartRequirements::constrains_part_rank() const {
  return part_name_is_set_;
}

std::string PartRequirements::get_part_name() const {
  return part_name_;
}

stk::topology::topology_t PartRequirements::get_part_topology() const {
  TEUCHOS_TEST_FOR_EXCEPTION(!this->constrains_part_topology(), std::logic_error,
                             "PartRequirements: Attempting to access the part topology requirement even though part "
                             "topology is unconstrained.");

  return part_topology_;
}

stk::topology::rank_t PartRequirements::get_part_rank() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_part_rank(), std::logic_error,
      "PartRequirements: Attempting to access the part rank requirement even though part rank is unconstrained.");

  return part_rank_;
}
//}

// \name Actions
//{
stk::mesh::Part &PartRequirements::declare_part_on_mesh(stk::mesh::MetaData *const meta_data_ptr) const {
  TEUCHOS_TEST_FOR_EXCEPTION(meta_data_ptr == nullptr, std::invalid_argument,
                             "PartRequirements: MetaData pointer cannot be null).");

  // Declare the Part.
  stk::mesh::Part *part_ptr;
  if (this->constrains_part_topology()) {
    part_ptr = &meta_data_ptr->declare_part_with_topology(this->get_part_name(), this->get_part_topology());
  } else if (this->constrains_part_rank()) {
    part_ptr = &meta_data_ptr->declare_part(this->get_part_name(), this->get_part_rank());
  } else {
    part_ptr = &meta_data_ptr->declare_part(this->get_part_name());
  }

  // Declare the Part's fields and associate them with the Part.
  // Loop over each rank's field map.
  for (auto const &part_field_map : part_ranked_field_maps_) {
    // Loop over each field and attempt to merge it.
    for ([[maybe_unused]] auto const &[field_name, field_req_ptr] : part_field_map) {
      field_req_ptr->declare_field_on_part(meta_data_ptr, *part_ptr);
    }
  }

  // Declare the sub-parts and declare them as sub-parts.
  // Each sub-part will. in turn, declare their fields and subparts.
  for ([[maybe_unused]] auto const &[subpart_name, subpart_req_ptr] : part_subpart_map_) {
    stk::mesh::Part &subpart = subpart_req_ptr->declare_part(meta_data_ptr);
    meta_data_ptr->declare_part_subset(*part_ptr, subpart);
  }

  return *part_ptr;
}

void PartRequirements::check_if_valid() const {
  ThrowRequireMsg(false, "not implemented yet");
}
//}
//@}

//! \name Non-member functions
//@{

/// \brief Merge any number of \c PartRequirements together.
///
/// The resulting \c PartRequirements will have the same name as the provided \c PartRequirements and (depending on if
/// they are constrained or not) the same topology or rank. Of course, this means an error will be thrown if the one
/// part require constrains topology/rank and the others do not or if that topology/rank differs from the others. The
/// resulting \c PartRequirements will also contain the merged set of field requirements. In the process, any two fields
/// with the same name and rank will be merged. Finally, the resulting \c PartRequirements will contain the set union of
/// all the provided \c PartAttributeTypes.
///
/// \param first_part_req [in] A \c PartRequirements objects to merge with the other requirements.
/// \param other_part_req [in] Any number of other \c PartRequirements objects to merge with the first.
auto merge_part_reqs(const FirstPartReqType &first_part_req, const OtherPartReqTypes... &other_part_req)
    -> decltype(PartRequirements<unique_tuple<tuple_cat_t<FirstPartReqType::part_attribute_types,
                                             OtherPartReqTypes::part_attribute_types...>>::type>) const {
  for (const auto &part_req_ptr : vector_of_part_req_ptrs) {
    // Check if the provided parameters are valid.
    part_req_ptr->check_if_valid();

    // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
    if (part_req_ptr->constrains_part_name()) {
      if (this->constrains_part_name()) {
        TEUCHOS_TEST_FOR_EXCEPTION(
            this->get_part_name() == part_req_ptr->get_part_name(), std::invalid_argument,
            "PartRequirements: One of the inputs has incompatible name (" << part_req_ptr->get_part_name() << ").");
      } else {
        this->set_part_name(part_req_ptr->get_part_name());
      }
    }

    if (part_req_ptr->constrains_part_rank()) {
      if (this->constrains_part_rank()) {
        TEUCHOS_TEST_FOR_EXCEPTION(
            this->get_part_rank() == part_req_ptr->get_part_rank(), std::invalid_argument,
            "PartRequirements: One of the inputs has incompatible rank (" << part_req_ptr->get_part_rank() << ").");
      } else {
        this->set_part_rank(part_req_ptr->get_part_rank());
      }
    }

    if (part_req_ptr->constrains_part_topology()) {
      if (this->constrains_part_topology()) {
        TEUCHOS_TEST_FOR_EXCEPTION(this->get_part_topology() == part_req_ptr->get_part_topology(),
                                   std::invalid_argument,
                                   "PartRequirements: One of the inputs has incompatible topology ("
                                       << part_req_ptr->get_part_topology() << ").");
      } else {
        this->set_part_topology(part_req_ptr->get_part_topology());
      }
    }

    // Loop over each rank's field map.
    for (auto const &part_field_map : part_req_ptr->get_part_field_map()) {
      // Loop over each field and attempt to merge it.
      for ([[maybe_unused]] auto const &[field_name, field_req_ptr] : part_field_map) {
        this->add_field_req(field_req_ptr);
      }
    }
  }
}
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_PARTREQUIREMENTS_HPP_
