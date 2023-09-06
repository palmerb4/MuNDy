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
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>            // for MUNDY_THROW_ASSERT
#include <mundy_mesh/MetaData.hpp>           // for mundy::mesh::MetaData
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
    MUNDY_THROW_ASSERT(false, std::invalid_argument,
                       "PartRequirements: The provided topology string " << topology_string << " is not valid.");
  }
}
//}

// \name Constructors and destructor
//{

PartRequirements::PartRequirements(const std::string &part_name) {
  this->set_part_name(part_name);
}

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
  Teuchos::ParameterList valid_params = parameter_list;
  validate_parameters_and_set_defaults(&valid_params);

  // Store the given parameters.
  this->set_part_name(valid_params.get<std::string>("name"));
  if (valid_params.isParameter("topology")) {
    if (valid_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("topology")) {
      this->set_part_topology(valid_params.get<std::string>("topology"));
    } else {
      this->set_part_topology(valid_params.get<stk::topology::topology_t>("topology"));
    }
  } else if (valid_params.isParameter("rank")) {
    if (valid_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("rank")) {
      this->set_part_rank(valid_params.get<std::string>("rank"));
    } else {
      this->set_part_rank(valid_params.get<stk::topology::rank_t>("rank"));
    }
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
      const Teuchos::ParameterList &subpart_i_sublist = parameter_list.sublist("sub_part_" + std::to_string(i));
      std::shared_ptr<PartRequirements> subpart_i = std::make_shared<PartRequirements>(subpart_i_sublist);
      this->add_subpart_reqs(subpart_i);
    }
  }
}
//}

// \name Setters
//{

void PartRequirements::set_part_name(const std::string &part_name) {
  part_name_ = part_name;
  part_name_is_set_ = true;
  this->check_if_valid();
}

void PartRequirements::set_part_topology(const stk::topology::topology_t &part_topology) {
  bool part_rank_already_set = this->constrains_part_rank();
  MUNDY_THROW_ASSERT(
      !part_rank_already_set, std::logic_error,
      "PartRequirements: Parts are designed to fall into three catagories: name set, name and topology set, "
          << "name and rank set. \n This part already sets the rank, so it's invalid to also set the topology.");
  part_topology_ = part_topology;
  part_topology_is_set_ = true;
  this->check_if_valid();
}

void PartRequirements::set_part_topology(const std::string &part_topology_string) {
  const stk::topology::topology_t part_topology = mundy::meta::map_string_to_topology(part_topology_string);
  this->set_part_topology(part_topology);
}

void PartRequirements::set_part_rank(const stk::topology::rank_t &part_rank) {
  bool part_topology_already_set = this->constrains_part_topology();
  MUNDY_THROW_ASSERT(
      !part_topology_already_set, std::logic_error,
      "PartRequirements: Parts are designed to fall into three catagories: name set, name and topology set, "
          << "name and rank set. \n This part already sets the topology, so it's invalid to also set the rank.");
  part_rank_ = part_rank;
  part_rank_is_set_ = true;
  this->check_if_valid();
}

void PartRequirements::set_part_rank(const std::string &part_rank_string) {
  const stk::topology::rank_t part_rank = mundy::meta::map_string_to_rank(part_rank_string);
  this->set_part_rank(part_rank);
}

void PartRequirements::delete_part_name() {
  part_name_is_set_ = false;
}

void PartRequirements::delete_part_topology() {
  part_topology_is_set_ = false;
}

void PartRequirements::delete_part_rank() {
  part_rank_is_set_ = false;
}

void PartRequirements::add_field_reqs(std::shared_ptr<FieldRequirementsBase> field_req_ptr) {
  MUNDY_THROW_ASSERT(field_req_ptr != nullptr, std::invalid_argument,
                     "MeshRequirements: The pointer passed to add_field_reqs cannot be a nullptr.");

  // Check if the provided parameters are valid.
  field_req_ptr->check_if_valid();

  // If a field with the same name and rank exists, attempt to merge them.
  // Otherwise, create a new field entity.
  const std::string field_name = field_req_ptr->get_field_name();
  const unsigned field_rank = field_req_ptr->get_field_rank();

  auto &part_field_map = part_ranked_field_maps_[field_rank];
  const bool name_already_exists = (part_field_map.count(field_name) != 0);
  if (name_already_exists) {
    part_field_map[field_name]->merge({field_req_ptr});
  } else {
    part_field_map.insert(std::make_pair(field_name, field_req_ptr));
  }
}

void PartRequirements::add_subpart_reqs(std::shared_ptr<PartRequirements> part_req_ptr) {
  MUNDY_THROW_ASSERT(part_req_ptr != nullptr, std::invalid_argument,
                     "MeshRequirements: The pointer passed to add_subpart_reqs cannot be a nullptr.");

  // Check if the provided parameters are valid.
  part_req_ptr->check_if_valid();

  // TODO(palmerb4): Check for conflicts?

  // Store the params.
  part_subpart_map_.insert(std::make_pair(part_req_ptr->get_part_name(), part_req_ptr));
}

void PartRequirements::add_part_attribute(const std::any &some_attribute) {
  std::type_index attribute_type_index = std::type_index(some_attribute.type());
  part_attributes_map_.insert(std::make_pair(attribute_type_index, some_attribute));
}

void PartRequirements::add_part_attribute(std::any &&some_attribute) {
  std::type_index attribute_type_index = std::type_index(some_attribute.type());
  part_attributes_map_.insert(std::make_pair(attribute_type_index, std::move(some_attribute)));
}

void PartRequirements::put_io_part_attribute() {
  is_io_part_ = true;
}
//@}

// \name Getters
//{

bool PartRequirements::constrains_part_name() const {
  return part_name_is_set_;
}

bool PartRequirements::constrains_part_topology() const {
  return part_topology_is_set_;
}

bool PartRequirements::constrains_part_rank() const {
  return part_rank_is_set_;
}

bool PartRequirements::is_fully_specified() const {
  return this->constrains_part_name();
}

std::string PartRequirements::get_part_name() const {
  MUNDY_THROW_ASSERT(
      this->constrains_part_name(), std::logic_error,
      "PartRequirements: Attempting to access the part name requirement even though part name is unconstrained.");

  return part_name_;
}

stk::topology::topology_t PartRequirements::get_part_topology() const {
  MUNDY_THROW_ASSERT(this->constrains_part_topology(), std::logic_error,
                     "PartRequirements: Attempting to access the part topology requirement even though part "
                     "topology is unconstrained.");

  return part_topology_;
}

stk::topology::rank_t PartRequirements::get_part_rank() const {
  MUNDY_THROW_ASSERT(
      this->constrains_part_rank(), std::logic_error,
      "PartRequirements: Attempting to access the part rank requirement even though part rank is unconstrained.");

  return part_rank_;
}

std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>>
PartRequirements::get_part_ranked_field_map() {
  // TODO(palmerb4): This is such an ugly and incorrect way to give others access to our internal fields.
  return part_ranked_field_maps_;
}

std::map<std::string, std::shared_ptr<PartRequirements>> PartRequirements::get_part_subpart_map() {
  return part_subpart_map_;
}

std::map<std::type_index, std::any> PartRequirements::get_part_attributes_map() {
  return part_attributes_map_;
}
//}

// \name Actions
//{
stk::mesh::Part &PartRequirements::declare_part_on_mesh(mundy::mesh::MetaData *const meta_data_ptr) const {
  MUNDY_THROW_ASSERT(meta_data_ptr != nullptr, std::invalid_argument,
                     "PartRequirements: MetaData pointer cannot be null).");
  MUNDY_THROW_ASSERT(this->constrains_part_name(), std::logic_error,
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

  MUNDY_THROW_ASSERT(this->get_part_name() == part_ptr->name(), std::logic_error,
                     "PartRequirements: Weird. The desired part name and actual part name differ.\n"
                         << "This should never happen. Please report this bug to the developers.\n"
                         << "  Desired part name: " << this->get_part_name() << "\n"
                         << "  Actual part name: " << part_ptr->name() << "\n");

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
    stk::mesh::Part &subpart = subpart_req_ptr->declare_part_on_mesh(meta_data_ptr);
    meta_data_ptr->declare_part_subset(*part_ptr, subpart);
  }

  // Declare the Part's attributes.
  for ([[maybe_unused]] auto const &[attribute_type_index, attribute] : part_attributes_map_) {
    meta_data_ptr->declare_attribute(*part_ptr, attribute);
  }

  return *part_ptr;
}

void PartRequirements::check_if_valid() const {
  // TODO(palmerb4): What are the requirements for validity?
}

void PartRequirements::merge(const std::shared_ptr<PartRequirements> &part_req_ptr) {
  // TODO(palmerb4): Move this to a friend non-member function.
  // TODO(palmerb4): Optimize this function for perfect forwarding.

  // Check if the provided pointer is valid.
  // If it is not, then there is nothing to merge.
  if (part_req_ptr == nullptr) {
    return;
  }

  // Check if the provided parameters are valid.
  part_req_ptr->check_if_valid();

  // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
  if (part_req_ptr->constrains_part_name()) {
    if (this->constrains_part_name()) {
      MUNDY_THROW_ASSERT(
          this->get_part_name() == part_req_ptr->get_part_name(), std::invalid_argument,
          "PartRequirements: One of the inputs has incompatible name (" << part_req_ptr->get_part_name() << ").");
    } else {
      this->set_part_name(part_req_ptr->get_part_name());
    }
  }

  if (part_req_ptr->constrains_part_rank()) {
    if (this->constrains_part_rank()) {
      MUNDY_THROW_ASSERT(
          this->get_part_rank() == part_req_ptr->get_part_rank(), std::invalid_argument,
          "PartRequirements: One of the inputs has incompatible rank (" << part_req_ptr->get_part_rank() << ").");
    } else {
      this->set_part_rank(part_req_ptr->get_part_rank());
    }
  }

  if (part_req_ptr->constrains_part_topology()) {
    if (this->constrains_part_topology()) {
      MUNDY_THROW_ASSERT(this->get_part_topology() == part_req_ptr->get_part_topology(), std::invalid_argument,
                         "PartRequirements: One of the inputs has incompatible topology ("
                             << part_req_ptr->get_part_topology() << ").");
    } else {
      this->set_part_topology(part_req_ptr->get_part_topology());
    }
  }

  // Loop over each rank's field map.
  for (auto const &part_field_map : part_req_ptr->get_part_ranked_field_map()) {
    // Loop over each field and attempt to merge it.
    for ([[maybe_unused]] auto const &[field_name, field_req_ptr] : part_field_map) {
      this->add_field_reqs(field_req_ptr);
    }
  }

  // Loop over the subpart map.
  for ([[maybe_unused]] auto const &[part_name, part_req_ptr] : part_req_ptr->get_part_subpart_map()) {
    this->add_subpart_reqs(part_req_ptr);
  }

  // Loop over the attribute map.
  for ([[maybe_unused]] auto const &[attribute_type_index, attribute] : part_req_ptr->get_part_attributes_map()) {
    this->add_part_attribute(attribute);
  }
}

void PartRequirements::merge(const std::vector<std::shared_ptr<PartRequirements>> &vector_of_part_req_ptrs) {
  for (const auto &part_req_ptr : vector_of_part_req_ptrs) {
    merge(part_req_ptr);
  }
}

void PartRequirements::dump_to_screen(int indent_level) const {
  std::string indent(indent_level * 2, ' ');

  std::cout << indent << "PartRequirements: " << std::endl;

  if (this->constrains_part_name()) {
    std::cout << indent << "  Part name is set." << std::endl;
    std::cout << indent << "  Part name: " << this->get_part_name() << std::endl;
  } else {
    std::cout << "  Part name is not set." << std::endl;
  }

  if (this->constrains_part_rank()) {
    std::cout << indent << "  Part rank is set." << std::endl;
    std::cout << indent << "  Part rank: " << this->get_part_rank() << std::endl;
  } else {
    std::cout << indent << "  Part rank is not set." << std::endl;
  }

  if (this->constrains_part_topology()) {
    std::cout << indent << "  Part topology is set." << std::endl;
    std::cout << indent << "  Part topology: " << this->get_part_topology() << std::endl;
  } else {
    std::cout << indent << "  Part topology is not set." << std::endl;
  }

  std::cout << indent << "  Part Fields: " << std::endl;
  int rank = 0;
  int field_count = 0;
  for (auto const &part_field_map : part_ranked_field_maps_) {
    for (auto const &[field_name, field_req_ptr] : part_field_map) {
      std::cout << indent << "  Part field " << field_count << " has name (" << field_name << "), rank (" << rank
                << "), and requirements" << std::endl;
      field_req_ptr->dump_to_screen(indent_level + 1);
      field_count++;
    }

    rank++;
  }

  std::cout << indent << "  Part Subparts: " << std::endl;
  int subpart_count = 0;
  for (auto const &[subpart_name, subpart_req_ptr] : part_subpart_map_) {
    std::cout << indent << "  Part subpart " << subpart_count << " has name (" << subpart_name << ") and requirements"
              << std::endl;
    subpart_req_ptr->dump_to_screen(indent_level + 1);
    subpart_count++;
  }

  std::cout << indent << "  Part Attributes: " << std::endl;
  int attribute_count = 0;
  for (auto const &[attribute_type_index, attribute] : part_attributes_map_) {
    std::cout << indent << "  Part attribute " << attribute_count << " has type (" << attribute_type_index.name() << ")"
              << std::endl;
    attribute_count++;
  }

  std::cout << indent << "End of PartRequirements" << std::endl;
}
//}

}  // namespace meta

}  // namespace mundy
