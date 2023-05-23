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
#include <regex>        // for std::regex_match
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string, std::stoi
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/MetaData.hpp>    // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part
#include <stk_topology/topology.hpp>     // for stk::topology

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements, mundy::meta::FieldRequirementsBase
#include <mundy_meta/FieldRequirementsFactory.hpp>  // for mundy::meta::FieldRequirementsFactory
#include <mundy_meta/PartRequirements.hpp>          // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

// \name Helper functions
//{

stk::topology map_string_to_topology(const std::string &topology_string) {
  if (topology_string == "INVALID_TOPOLOGY") {
    return stk::topology::INVALID_TOPOLOGY;
  } else if (topology_string == "NODE") {
    return stk::topology::NODE;
  } else if (topology_string == "LINE_2") {
    return stk::topology::LINE_2;
  } else if (topology_string == "LINE_3") {
    return stk::topology::LINE_3;
  } else if (topology_string == "TRI_3") {
    return stk::topology::TRI_3;
  } else if (topology_string == "TRI_4") {
    return stk::topology::TRI_4;
  } else if (topology_string == "TRI_6") {
    return stk::topology::TRI_6;
  } else if (topology_string == "QUAD_4") {
    return stk::topology::QUAD_4;
  } else if (topology_string == "QUAD_6") {
    return stk::topology::QUAD_6;
  } else if (topology_string == "QUAD_8") {
    return stk::topology::QUAD_8;
  } else if (topology_string == "QUAD_9") {
    return stk::topology::QUAD_9;
  } else if (topology_string == "PARTICLE") {
    return stk::topology::PARTICLE;
  } else if (topology_string == "LINE_2_1D") {
    return stk::topology::LINE_2_1D;
  } else if (topology_string == "LINE_3_1D") {
    return stk::topology::LINE_3_1D;
  } else if (topology_string == "BEAM_2") {
    return stk::topology::BEAM_2;
  } else if (topology_string == "BEAM_3") {
    return stk::topology::BEAM_3;
  } else if (topology_string == "SHELL_LINE_2") {
    return stk::topology::SHELL_LINE_2;
  } else if (topology_string == "SHELL_LINE_3") {
    return stk::topology::SHELL_LINE_3;
  } else if (topology_string == "SPRING_2") {
    return stk::topology::SPRING_2;
  } else if (topology_string == "SPRING_3") {
    return stk::topology::SPRING_3;
  } else if (topology_string == "TRI_3_2D") {
    return stk::topology::TRI_3_2D;
  } else if (topology_string == "TRI_4_2D") {
    return stk::topology::TRI_4_2D;
  } else if (topology_string == "TRI_6_2D") {
    return stk::topology::TRI_6_2D;
  } else if (topology_string == "QUAD_4_2D") {
    return stk::topology::QUAD_4_2D;
  } else if (topology_string == "QUAD_8_2D") {
    return stk::topology::QUAD_8_2D;
  } else if (topology_string == "QUAD_9_2D") {
    return stk::topology::QUAD_9_2D;
  } else if (topology_string == "SHELL_TRI_3") {
    return stk::topology::SHELL_TRI_3;
  } else if (topology_string == "SHELL_TRI_4") {
    return stk::topology::SHELL_TRI_4;
  } else if (topology_string == "SHELL_TRI_6") {
    return stk::topology::SHELL_TRI_6;
  } else if (topology_string == "SHELL_QUAD_4") {
    return stk::topology::SHELL_QUAD_4;
  } else if (topology_string == "SHELL_QUAD_8") {
    return stk::topology::SHELL_QUAD_8;
  } else if (topology_string == "SHELL_QUAD_9") {
    return stk::topology::SHELL_QUAD_9;
  } else if (topology_string == "TET_4") {
    return stk::topology::TET_4;
  } else if (topology_string == "TET_8") {
    return stk::topology::TET_8;
  } else if (topology_string == "TET_10") {
    return stk::topology::TET_10;
  } else if (topology_string == "TET_11") {
    return stk::topology::TET_11;
  } else if (topology_string == "PYRAMID_5") {
    return stk::topology::PYRAMID_5;
  } else if (topology_string == "PYRAMID_13") {
    return stk::topology::PYRAMID_13;
  } else if (topology_string == "PYRAMID_14") {
    return stk::topology::PYRAMID_14;
  } else if (topology_string == "WEDGE_6") {
    return stk::topology::WEDGE_6;
  } else if (topology_string == "WEDGE_12") {
    return stk::topology::WEDGE_12;
  } else if (topology_string == "WEDGE_15") {
    return stk::topology::WEDGE_15;
  } else if (topology_string == "WEDGE_18") {
    return stk::topology::WEDGE_18;
  } else if (topology_string == "HEX_8") {
    return stk::topology::HEX_8;
  } else if (topology_string == "HEX_20") {
    return stk::topology::HEX_20;
  } else if (topology_string == "HEX_27") {
    return stk::topology::HEX_27;
  } else if (std::regex_match(topology_string, std::regex("SUPEREDGE<\\d+>"))) {
    std::smatch base_match;
    std::regex_match(topology_string, base_match, std::regex("\\d+"));
    const int num_nodes = std::stoi(base_match[1].str());
    return stk::create_superedge_topology(num_nodes);
  } else if (std::regex_match(topology_string, std::regex("SUPERFACE<\\d+>"))) {
    std::smatch base_match;
    std::regex_match(topology_string, base_match, std::regex("\\d+"));
    const int num_nodes = std::stoi(base_match[1].str());
    return stk::create_superface_topology(num_nodes);
  } else if (std::regex_match(topology_string, std::regex("SUPERELEMENT<\\d+>"))) {
    std::smatch base_match;
    std::regex_match(topology_string, base_match, std::regex("\\d+"));
    const int num_nodes = std::stoi(base_match[1].str());
    return stk::create_superelement_topology(num_nodes);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::invalid_argument,
        "PartRequirements: The provided topology string " << topology_string << " is not valid.");
  }
}
//}

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
      this->add_field_reqs(field_i);
    }
  }

  // Store the sub-part params.
  if (parameter_list.isSublist("sub_parts")) {
    const Teuchos::ParameterList &subparts_sublist = parameter_list.sublist("sub_parts");
    const unsigned num_subparts = subparts_sublist.get<unsigned>("count");
    for (unsigned i = 0; i < num_subparts; i++) {
      const Teuchos::ParameterList &subpart_i_sublist =
          parameter_list.sublist("sub_part_" + std::to_string(i));
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

void PartRequirements::set_part_topology(const stk::topology::topology_t &part_topology) {
  part_topology_ = part_topology;
  part_topology_is_set_ = true;
  this->check_if_valid();
}

void PartRequirements::set_part_topology(const std::string &part_topology_string) {
  const stk::topology::topology_t part_topology = mundy::meta::map_string_to_topology(part_topology_string);
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
                             "PartRequirements: Attempting to access the part name requirement even though part name is unconstrained.");

  return part_name_;
}

stk::topology::topology_t PartRequirements::get_part_topology() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_part_topology(), std::logic_error,
      "PartRequirements: Attempting to access the part topology requirement even though part topology is unconstrained.");

  return part_topology_;
}

stk::topology::rank_t PartRequirements::get_part_rank() const {
  TEUCHOS_TEST_FOR_EXCEPTION(!this->constrains_part_rank(), std::logic_error,
                             "PartRequirements: Attempting to access the part rank requirement even though part rank is unconstrained.");

  return part_rank_;
}

std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> PartRequirements::get_part_field_map() {
  // TODO(palmerb4): This is such an ugly and incorrect way to give other access to our internal fields.
  return part_ranked_field_maps_;
}
//}

// \name Actions
//{
stk::mesh::Part &PartRequirements::declare_part(stk::mesh::MetaData *const meta_data_ptr) const {
  TEUCHOS_TEST_FOR_EXCEPTION(meta_data_ptr == nullptr, std::invalid_argument,
                             "PartRequirements: MetaData pointer cannot be null).");
  TEUCHOS_TEST_FOR_EXCEPTION(this->constrains_part_name(), std::logic_error,
                             "PartRequirements: Part name must be set before calling declare_part.");

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
    for ([[maybe_unused]] auto const &[field_name, field_reqs_ptr] : part_field_map) {
      field_reqs_ptr->declare_field_on_part(meta_data_ptr, *part_ptr);
    }
  }

  // Declare the sub-parts and declare them as sub-parts.
  // Each sub-part will. in turn, declare their fields and subparts.
  for ([[maybe_unused]] auto const &[subpart_name, subpart_reqs_ptr] : part_subpart_map_) {
    stk::mesh::Part &subpart = subpart_reqs_ptr->declare_part(meta_data_ptr);
    meta_data_ptr->declare_part_subset(*part_ptr, subpart);
  }

  return *part_ptr;
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

void PartRequirements::add_field_reqs(std::shared_ptr<FieldRequirementsBase> field_reqs_ptr) {
  // Check if the provided parameters are valid.
  field_reqs_ptr->check_if_valid();

  // If a field with the same name and rank exists, attempt to merge them.
  // Otherwise, create a new field entity.
  const std::string field_name = field_reqs_ptr->get_field_name();
  const unsigned field_rank = field_reqs_ptr->get_field_rank();

  auto &part_field_map = part_ranked_field_maps_[field_rank];
  const bool name_already_exists = (part_field_map.count(field_name) != 0);
  if (name_already_exists) {
    part_field_map[field_name]->merge({field_reqs_ptr});
  } else {
    part_field_map[field_name] = field_reqs_ptr;
  }
}

void PartRequirements::add_subpart_reqs(std::shared_ptr<PartRequirements> part_reqs_ptr) {
  // Check if the provided parameters are valid.
  part_reqs_ptr->check_if_valid();

  // Check for conflicts?

  // Store the params.
  part_subpart_map_[part_reqs_ptr->get_part_name()] = part_reqs_ptr;
}

void merge(const std::shared_ptr<PartRequirements> &part_req_ptr) {
  merge({part_req_ptr});
}

void PartRequirements::merge(const std::vector<std::shared_ptr<PartRequirements>> &vector_of_part_req_ptrs) {
  for (const auto &part_req_ptr : vector_of_part_req_ptrs) {
    // Check if the provided parameters are valid.
    part_req_ptr->check_if_valid();

    // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
    if (part_req_ptr->constrains_part_name()) {
      if (this->constrains_part_name()) {
        TEUCHOS_TEST_FOR_EXCEPTION(this->get_part_name() == part_req_ptr->get_part_name(), std::invalid_argument,
                                   "PartRequirements: One of the inputs has incompatible name ("
                                       << part_req_ptr->get_part_name() << ").");
      } else {
        this->set_part_name(part_req_ptr->get_part_name());
      }
    }

    if (part_req_ptr->constrains_part_rank()) {
      if (this->constrains_part_rank()) {
        TEUCHOS_TEST_FOR_EXCEPTION(this->get_part_rank() == part_req_ptr->get_part_rank(), std::invalid_argument,
                                   "PartRequirements: One of the inputs has incompatible rank ("
                                       << part_req_ptr->get_part_rank() << ").");
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
      for ([[maybe_unused]] auto const &[field_name, field_reqs_ptr] : part_field_map) {
        this->add_field_reqs(field_reqs_ptr);
      }
    }
  }
}
//}

}  // namespace meta

}  // namespace mundy
