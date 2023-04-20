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

/// \file PartRequirements.cpp
/// \brief Definition of the PartRequirements class

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>       // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>    // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/MetaData.hpp>      // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>          // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements, mundy::meta::FieldRequirementsBase
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

// \name Constructors and destructor
//{

PartRequirements::PartRequirements(const std::string &part_name, const stk::topology &part_topology) {
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
    const unsigned num_fields = fields_sublist.get<std::string>("count");
    for (int i = 0; i < num_fields; i++) {
      const Teuchos::ParameterList &field_i_sublist = parameter_list.sublist("field_" + std::to_string(num_fields));
      const std::string field_type_string = field_i_sublist.get<std::string>("type");
      std::shared_ptr<FieldRequirementsBase> field_i =
          FieldRequirements::create_new_instance(field_type_string, field_i_sublist);
      this->add_field_reqs(field_i);
    }
  }

  // Store the sub-part params.
  if (parameter_list.isSublist("sub_parts")) {
    const Teuchos::ParameterList &subparts_sublist = parameter_list.sublist("sub_parts");
    const unsigned num_subparts = subparts_sublist.get<std::string>("count");
    for (int i = 0; i < num_subparts; i++) {
      const Teuchos::ParameterList &subpart_i_sublist =
          parameter_list.sublist("sub_part_" + std::to_string(num_subparts));
      std::shared_ptr<PartRequirements> subpart_i = std::make_shared<PartRequirements>(subpart_i_sublist);
      this->add_subpart_reqs(subpart_i);
    }
  }
}
//}

// \name Setters and Getters
//{

void PartRequirements::set_part_name(const std::string &part_name) {
  part_name_ = part_name;
  part_name_is_set_ = true;
  this->check_if_valid();
}

void PartRequirements::set_part_topology(const stk::topology &part_topology) {
  part_topology_ = part_topology_;
  part_topology_is_set_ = true;
  this->check_if_valid();
}

void PartRequirements::set_part_topology(const std::string &part_topology_string) {
  const stk::topology part_topology = mundy::meta::map_string_to_topology(part_topology_string);
  this->set_part_topology(part_topology);
}

void PartRequirements::set_part_rank(const stk::topology::rank_t &part_rank) {
  part_rank_ = part_rank;
  part_rank_is_set_ = true;
  this->check_if_valid();
}

void PartRequirements::set_part_rank(const std::string &part_rank_string) {
  const stk::topology::rank_t part_rank = mundy::meta::map_string_to_rank(part_rank_string);
  this->set_part_rank(part_rank);
}

bool PartRequirements::constrains_part_name() const {
  return part_name_is_set_;
}

bool PartRequirements::constrains_part_topology() const {
  return part_name_is_set_;
}

bool PartRequirements::constrains_part_rank() const {
  return part_name_is_set_;
}

std::string PartRequirements::get_part_name() const {
  TEUCHOS_TEST_FOR_EXCEPTION(!this->constrains_part_name(), std::logic_error,
                             "Attempting to access the part name requirement even though part name is unconstrained.");

  return part_name_;
}

stk::topology PartRequirements::get_part_topology() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_part_topology(), std::logic_error,
      "Attempting to access the part topology requirement even though part topology is unconstrained.");

  return part_topology_;
}

stk::topology::rank_t PartRequirements::get_part_rank() const {
  TEUCHOS_TEST_FOR_EXCEPTION(!this->constrains_part_rank(), std::logic_error,
                             "Attempting to access the part rank requirement even though part rank is unconstrained.");

  return part_rank_;
}

std::vector<std::map<std::string, std::shared_ptr<const FieldRequirementsBase>>> PartRequirements::get_part_field_map(
    const stk::topology::rank_t &field_rank) const {
  return part_ranked_field_maps_[field_rank];
}

static Teuchos::ParameterList PartRequirements::get_valid_params() const {
  static Teuchos::ParameterList default_parameter_list;
  default_parameter_list.set("name", "INVALID", "Name of the part.");
  default_parameter_list.set("topology", stk::topology::INVALID_TOPOLOGY, "Topology of the part.");
  default_parameter_list.set("rank", stk::topology::INVALID_RANK, "Rank of the part.");
  return default_parameter_list;
}
//}

// \name Actions
//{
stk::mesh::Part &PartRequirements::declare_part(const stk::mesh::MetaData *meta_data_ptr) const {
  TEUCHOS_TEST_FOR_EXCEPTION(meta_data_ptr == nullptr, std::invalid_argument,
                             "mundy::meta::PartRequirements: MetaData pointer cannot be null).");
  TEUCHOS_TEST_FOR_EXCEPTION(this->constrains_part_name(), std::logic_error,
                             "mundy::meta::PartRequirements: Part name must be set before calling declare_part.");

  // Declare the Part.
  if (this->constrains_part_topology()) {
    return meta_data_ptr->declare_part_with_topology(this->get_part_name(), this->get_part_topology());
  } else if (this->constrains_part_rank()) {
    return meta_data_ptr->declare_part(this->get_part_name(), this->get_part_rank());
  } else {
    return meta_data_ptr->declare_part(this->get_part_name());
  }

  // Declare the Part's fields and associate them with the Part.
  // Loop over each rank's field map.
  for (auto const &part_field_map : part_ranked_field_maps_) {
    // Loop over each field and attempt to merge it.
    for ([[maybe_unused]] auto const &[field_name, field_reqs_ptr] : part_field_map) {
      field_reqs_ptr->declare_field_on_part(meta_data_ptr, part);
    }
  }

  // Declare the sub-parts and declare them as sub-parts.
  // Each sub-part will. in turn, declare their fields and subparts.
  for ([[maybe_unused]] auto const &[subpart_name, subpart_reqs_ptr] : part_subpart_map_) {
    stk::mesh::Part &subpart = subpart_reqs_ptr->declare_part(meta_data_ptr);
    meta_data_ptr->declare_part_subset(part, subpart);
  }
}

void PartRequirements::delete_part_name_constraint() {
  part_name_is_set_ = false;
}

void PartRequirements::delete_part_topology_constraint() {
  part_topology_is_set_ = false;
}

void PartRequirements::delete_part_rank_constraint() {
  part_rank_is_set_ = false;
}

void PartRequirements::check_if_valid() const {
  ThrowRequireMsg(false, "not implemented yet");
}

void PartRequirements::dd_field_reqs(const std::shared_ptr<FieldRequirementsBase> &field_reqs) {
  // Check if the provided parameters are valid.
  field_reqs.check_if_valid();

  // If a field with the same name and rank exists, attempt to merge them.
  // Otherwise, create a new field entity.
  const std::string field_name = field_reqs.get_field_name();
  const unsigned field_rank = field_reqs.get_field_rank();

  auto part_field_map_ptr = part_ranked_field_maps_.data() + field_rank;
  const bool name_already_exists = (part_field_map_ptr->count(field_name) != 0);
  if (name_already_exists) {
    *part_field_map_ptr[field_name]->merge(field_reqs);
  } else {
    *part_field_map_ptr[field_name] = field_reqs;
  }
}

void PartRequirements::add_subpart_reqs(const std::shared_ptr<const PartRequirements> &part_reqs) {
  // Check if the provided parameters are valid.
  part_reqs.check_if_valid();

  // Check for conflicts?

  // Store the params.
  part_subpart_map_[part_reqs.get_part_name(), part_reqs];
}

template <class... ArgTypes>
void PartRequirements::merge(const ArgTypes &...list_of_part_reqs) {
  for (const auto &part_reqs : list_of_part_reqs) {
    // Check if the provided parameters are valid.
    part_reqs.check_if_valid();

    // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
    if (part_reqs.constrains_part_name()) {
      if (this->constrains_part_name()) {
        TEUCHOS_TEST_FOR_EXCEPTION(this->get_part_name() == part_reqs.get_part_name(), std::invalid_argument,
                                   "mundy::meta::PartRequirements: One of the inputs has incompatible name ("
                                       << part_reqs.get_part_name() << ").");
      } else {
        this->set_part_name(part_reqs.get_part_name());
      }
    }

    if (part_reqs.constrains_part_rank()) {
      if (this->constrains_part_rank()) {
        TEUCHOS_TEST_FOR_EXCEPTION(this->get_part_rank() == part_reqs.get_part_rank(), std::invalid_argument,
                                   "mundy::meta::PartRequirements: One of the inputs has incompatible rank ("
                                       << part_reqs.get_part_rank() << ").");
      } else {
        this->set_part_rank(part_reqs.get_part_rank());
      }
    }

    if (part_reqs.constrains_part_topology()) {
      if (this->constrains_part_topology()) {
        TEUCHOS_TEST_FOR_EXCEPTION(this->get_part_topology() == part_reqs.get_part_topology(), std::invalid_argument,
                                   "mundy::meta::PartRequirements: One of the inputs has incompatible topology ("
                                       << part_reqs.get_part_topology() << ").");
      } else {
        this->set_part_topology(part_reqs.get_part_topology());
      }
    }

    // Loop over each rank's field map.
    for (auto const &part_field_map : part_reqs.get_part_field_map()) {
      // Loop over each field and attempt to merge it.
      for ([[maybe_unused]] auto const &[field_name, field_reqs_ptr] : part_field_map) {
        this->add_field_reqs(field_reqs_ptr);
      }
    }
  }
}
//}

}  // namespace meta

}  // namespace mundy
