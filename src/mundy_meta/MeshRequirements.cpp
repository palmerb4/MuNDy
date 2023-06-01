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

/// \file MeshRequirements.cpp
/// \brief Definition of the MeshRequirements class

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
#include <mundy_meta/MeshRequirements.hpp>          // for mundy::meta::MeshRequirements

namespace mundy {

namespace meta {

// \name Constructors and destructor
//{

MeshRequirements::MeshRequirements(const Teuchos::ParameterList &parameter_list) {
  // Validate the input params. Throws an error if a parameter is defined but not in the valid params.
  // This helps catch misspellings.
  parameter_list.validateParameters(this->get_valid_params());

  // Store the given parameters (if they are specified)
  if (parameter_list.isParameter("spatial_dimension")) {
    const std::string spatial_dimension = parameter_list.get<std::string>("spatial_dimension");
    this->set_spatial_dimension(spatial_dimension);
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
      std::shared_ptr<MeshRequirements> subpart_i = std::make_shared<MeshRequirements>(subpart_i_sublist);
      this->add_subpart_reqs(subpart_i);
    }
  }
}
//}

// \name Setters and Getters
//{

void MeshRequirements::set_spatial_dimension(const unsigned spatial_dimension) {
  spatial_dimension_ = spatial_dimension;
  spatial_dimension_is_set_ = true;
  this->check_if_valid();
}

MeshBuilder &MeshRequirements::set_entity_rank_names(const std::vector<std::string> &entity_rank_names) {
  entity_rank_names_ = entity_rank_names;
  entity_rank_names_is_set_ = true;
  this->check_if_valid();
}

MeshBuilder &MeshRequirements::set_communicator(const stk::ParallelMachine &comm) {
  comm_ = comm;
  comm_is_set_ = true;
  this->check_if_valid();
}

MeshBuilder &MeshRequirements::set_aura_option(const stk::mesh::BulkData::AutomaticAuraOption &aura_option) {
  aura_option_ = aura_option;
  aura_option_is_set_ = true;
  this->check_if_valid();
}

MeshBuilder &MeshRequirements::set_field_data_manager(stk::mesh::FieldDataManager *const field_data_manager_ptr) {
  field_data_manager_ptr_ = field_data_manager_ptr;
  field_data_manager_ptr_is_set_ = true;
  this->check_if_valid();
}

MeshBuilder &MeshRequirements::set_bucket_capacity(const unsigned bucket_capacity) {
  bucket_capacity_ = bucket_capacity;
  bucket_capacity_is_set_ = true;
  this->check_if_valid();
}

MeshBuilder &MeshRequirements::set_upward_connectivity_flag(const bool enable_upward_connectivity) {
  enable_upward_connectivity_ = enable_upward_connectivity;
  enable_upward_connectivity_is_set_ = true;
  this->check_if_valid();
}

bool MeshRequirements::constrains_spatial_dimension() const {
  return spatial_dimension_is_set_;
}

bool MeshRequirements::constrains_entity_rank_names() const {
  return entity_rank_names_is_set_;
}

bool MeshRequirements::constrains_communicator() const {
  return comm_is_set_;
}

bool MeshRequirements::constrains_aura_option() const {
  return aura_option_is_set_;
}

bool MeshRequirements::constrains_field_data_manager() const {
  return field_data_manager_ptr_is_set_;
}

bool MeshRequirements::constrains_bucket_capacity() const {
  return bucket_capacity_is_set_;
}

bool MeshRequirements::constrains_upward_connectivity_flag() const {
  return enable_upward_connectivity_is_set_;
}

unsigned MeshRequirements::get_spatial_dimension() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_spatial_dimension(), std::logic_error,
      "MeshRequirements: Attempting to access the part name requirement even though part name is unconstrained.");
}

std::vector<std::string> MeshRequirements::get_entity_rank_names() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_entity_rank_names(), std::logic_error,
      "MeshRequirements: Attempting to access the part name requirement even though part name is unconstrained.");
}

stk::ParallelMachine MeshRequirements::get_communicator() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_communicator(), std::logic_error,
      "MeshRequirements: Attempting to access the part name requirement even though part name is unconstrained.");
}

stk::mesh::BulkData::AutomaticAuraOption MeshRequirements::get_aura_option() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_aura_option(), std::logic_error,
      "MeshRequirements: Attempting to access the part name requirement even though part name is unconstrained.");
}

stk::mesh::FieldDataManager *const MeshRequirements::get_field_data_manager() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_field_data_manager(), std::logic_error,
      "MeshRequirements: Attempting to access the part name requirement even though part name is unconstrained.");
}

unsigned MeshRequirements::get_bucket_capacity() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_bucket_capacity(), std::logic_error,
      "MeshRequirements: Attempting to access the part name requirement even though part name is unconstrained.");
}

bool MeshRequirements::get_upward_connectivity_flag() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_upward_connectivity_flag(), std::logic_error,
      "MeshRequirements: Attempting to access the part name requirement even though part name is unconstrained.");
}

std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> get_mesh_field_map() {
  // TODO(palmerb4): This is such an ugly and incorrect way to give others access to our internal fields.
  // This could be private and all other MeshRequirements made friends.
  return mesh_field_map_;
}

std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> get_mesh_part_map() {
  // TODO(palmerb4): This is such an ugly and incorrect way to give others access to our internal parts.
  // This could be private and all other MeshRequirements made friends.
  return mesh_part_map_;
}
//}

// \name Actions
//{
std::shared_ptr<stk::mesh::BulkData> &MeshRequirements::declare_mesh() const {
  TEUCHOS_TEST_FOR_EXCEPTION(this->constrains_communicator(), std::logic_error,
                             "MeshRequirements: The MPI communicator must be ste before calling declare_part.");

  // The mesh itself is generated using stk's MeshBuilder which we provide a wrapper for.
  // If any of our parameters are not constrained, we use the default value.
  MeshBuilder mesh_builder(this->get_communicator());

  if (this->constrains_spatial_dimension()) {
    mesh_builder.set_spatial_dimension(this->get_spatial_dimension());
  }
  if (this->constrains_entity_rank_names()) {
    mesh_builder.set_entity_rank_names(this->get_entity_rank_names());
  }
  if (this->constrains_communicator()) {
    mesh_builder.set_communicator(this->get_communicator());
  }
  if (this->constrains_field_data_manager()) {
    mesh_builder.set_field_data_manager(this->get_field_data_manager());
  }
  if (this->constrains_bucket_capacity()) {
    mesh_builder.set_bucket_capacity(this->get_bucket_capacity());
  }
  if (this->constrains_upward_connectivity_flag()) {
    mesh_builder.set_upward_connectivity_flag(this->get_upward_connectivity_flag());
  }

  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = mesh_builder.create_bulk_data();
  stk::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Declare the mesh's fields.
  // Loop over each rank's field map.
  for (auto const &mesh_field_map : mesh_ranked_field_maps_) {
    // Loop over each field and attempt to merge it.
    for ([[maybe_unused]] auto const &[field_name, field_req_ptr] : mesh_field_map) {
      field_req_ptr->declare_field_on_mesh(&meta_data);
    }
  }

  // Declare the mesh's parts.
  for (auto part_req_ptr : part_req_ptrs) {
    part_req_ptr->declare_part_on_mesh(&meta_data);
  }

  return bulk_data_ptr;
}

void MeshRequirements::constrains_spatial_dimension() {
  spatial_dimension_is_set_ = false;
}

void MeshRequirements::constrains_entity_rank_names() {
  entity_rank_names_is_set_ = false;
}

void MeshRequirements::constrains_communicator() {
  communicator_is_set_ = false;
}

void MeshRequirements::constrains_aura_option() {
  aura_option_is_set_ = false;
}

void MeshRequirements::constrains_field_data_manager() {
  field_data_manager_ptr_is_set_ = false;
}

void MeshRequirements::constrains_bucket_capacity() {
  bucket_capacity_is_set_ = false;
}

void MeshRequirements::constrains_upward_connectivity_flag() {
  enable_upward_connectivity_is_set_ = false;
}

void MeshRequirements::check_if_valid() const {
  ThrowRequireMsg(false, "not implemented yet");
}

void MeshRequirements::add_field_req(std::shared_ptr<FieldRequirementsBase> field_req_ptr) {
  // Check if the provided parameters are valid.
  field_req_ptr->check_if_valid();

  // If a field with the same name and rank exists, attempt to merge them.
  // Otherwise, create a new field entity.
  const std::string field_name = field_req_ptr->get_field_name();
  const unsigned field_rank = field_req_ptr->get_field_rank();

  auto &mesh_field_map = mesh_ranked_field_maps_[field_rank];
  const bool name_already_exists = (mesh_field_map.count(field_name) != 0);
  if (name_already_exists) {
    mesh_field_map[field_name]->merge(field_req_ptr);
  } else {
    mesh_field_map[field_name] = field_req_ptr;
  }
}

void MeshRequirements::add_part_req(std::shared_ptr<MeshRequirements> part_req_ptr) {
  // Check if the provided parameters are valid.
  part_req_ptr->check_if_valid();

  // TODO(palmerb4): Check for conflicts?

  // Store the params.
  mesh_part_map_[part_req_ptr->get_part_name()] = part_req_ptr;
}

void merge(const std::shared_ptr<MeshRequirements> &mesh_req_ptr) {
  // Check if the provided parameters are valid.
  mesh_req_ptr->check_if_valid();

  // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
  if (mesh_req_ptr->constrains_spatial_dimension()) {
    if (this->constrains_spatial_dimension()) {
      TEUCHOS_TEST_FOR_EXCEPTION(this->get_spatial_dimension() == part_req_ptr->get_spatial_dimension(),
                                 std::invalid_argument,
                                 "MeshRequirements: One of the inputs has incompatible spatial dimension ("
                                     << part_req_ptr->get_part_name() << ").");
    } else {
      this->set_spatial_dimension(part_req_ptr->get_spatial_dimension());
    }
  }

  if (part_req_ptr->constrains_entity_rank_names()) {
    if (this->constrains_entity_rank_names()) {
      TEUCHOS_TEST_FOR_EXCEPTION(this->get_entity_rank_names() == part_req_ptr->get_entity_rank_names(),
                                 std::invalid_argument,
                                 "MeshRequirements: One of the inputs has incompatible entity rank names ("
                                     << part_req_ptr->get_part_name() << ").");
    } else {
      this->set_entity_rank_names(part_req_ptr->get_entity_rank_names());
    }
  }

  if (part_req_ptr->constrains_communicator()) {
    if (this->constrains_communicator()) {
      TEUCHOS_TEST_FOR_EXCEPTION(this->get_communicator() == part_req_ptr->get_communicator(), std::invalid_argument,
                                 "MeshRequirements: One of the inputs has incompatible MPI communicator ("
                                     << part_req_ptr->get_part_name() << ").");
    } else {
      this->set_communicator(part_req_ptr->get_communicator());
    }
  }

  if (part_req_ptr->constrains_field_data_manager()) {
    if (this->constrains_field_data_manager()) {
      TEUCHOS_TEST_FOR_EXCEPTION(this->get_field_data_manager() == part_req_ptr->get_field_data_manager(),
                                 std::invalid_argument,
                                 "MeshRequirements: One of the inputs has incompatible field data manager ("
                                     << part_req_ptr->get_part_name() << ").");
    } else {
      this->set_field_data_manager(part_req_ptr->get_field_data_manager());
    }
  }

  if (part_req_ptr->constrains_bucket_capacity()) {
    if (this->constrains_bucket_capacity()) {
      TEUCHOS_TEST_FOR_EXCEPTION(this->get_bucket_capacity() == part_req_ptr->get_bucket_capacity(),
                                 std::invalid_argument,
                                 "MeshRequirements: One of the inputs has incompatible bucket capacity ("
                                     << part_req_ptr->get_part_name() << ").");
    } else {
      this->set_bucket_capacity(part_req_ptr->get_bucket_capacity());
    }
  }

  if (part_req_ptr->constrains_upward_connectivity_flag()) {
    if (this->constrains_upward_connectivity_flag()) {
      TEUCHOS_TEST_FOR_EXCEPTION(this->get_upward_connectivity_flag() == part_req_ptr->get_upward_connectivity_flag(),
                                 std::invalid_argument,
                                 "MeshRequirements: One of the inputs has incompatible connectivity flag ("
                                     << part_req_ptr->get_part_name() << ").");
    } else {
      this->set_upward_connectivity_flag(part_req_ptr->get_upward_connectivity_flag());
    }
  }

  // Loop over each rank's field map.
  for (auto const &mesh_field_map : mesh_req_ptr->get_part_field_map()) {
    // Loop over each field and attempt to merge it.
    for ([[maybe_unused]] auto const &[field_name, field_req_ptr] : mesh_field_map) {
      this->add_field_req(field_req_ptr);
    }
  }

  // Loop over the part map.
  for (auto const &part_ptr : mesh_req_ptr->get_mesh_part_map()) {
    this->add_part_req(part_ptr);
  }

  
}

void MeshRequirements::merge(const std::vector<std::shared_ptr<MeshRequirements>> &vector_of_mesh_req_ptrs) {
  for (const auto &mesh_req_ptr : vector_of_mesh_req_ptrs) {
    merge(mesh_req_ptr);
  }
}
//}

}  // namespace meta

}  // namespace mundy
