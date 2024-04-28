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

/// \file MeshReqs.cpp
/// \brief Definition of the MeshReqs class

// C++ core libs
#include <iostream>
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <regex>        // for std::regex_match
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string, std::stoi
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Part.hpp>          // for stk::mesh::Part
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>       // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>         // for mundy::meta::FieldReqs, mundy::meta::FieldReqsBase
#include <mundy_meta/FieldReqsFactory.hpp>  // for mundy::meta::FieldReqsFactory
#include <mundy_meta/MeshReqs.hpp>          // for mundy::meta::MeshReqs

namespace mundy {

namespace meta {

// \name Default parameters (those that can't be inlined)
//{

const unsigned MeshReqs::default_bucket_capacity_ = stk::mesh::get_default_bucket_capacity();
//}

// \name Constructors and destructor
//{
MeshReqs::MeshReqs(const stk::ParallelMachine &comm) {
  this->set_communicator(comm);
}
//}

// \name Setters and Getters
//{

MeshReqs &MeshReqs::set_spatial_dimension(const unsigned spatial_dimension) {
  spatial_dimension_ = spatial_dimension;
  spatial_dimension_is_set_ = true;
  this->check_if_valid();
  return *this;
}

MeshReqs &MeshReqs::set_node_coordinates_name(const std::string &node_coordinates_name) {
  node_coordinates_name_ = node_coordinates_name;
  node_coordinates_name_is_set_ = true;
  this->check_if_valid();
  return *this;
}

MeshReqs &MeshReqs::set_entity_rank_names(const std::vector<std::string> &entity_rank_names) {
  entity_rank_names_ = entity_rank_names;
  entity_rank_names_is_set_ = true;
  this->check_if_valid();
  return *this;
}

MeshReqs &MeshReqs::set_communicator(const stk::ParallelMachine &communicator) {
  communicator_ = communicator;
  communicator_is_set_ = true;
  this->check_if_valid();
  return *this;
}

MeshReqs &MeshReqs::set_aura_option(const mundy::mesh::BulkData::AutomaticAuraOption &aura_option) {
  aura_option_ = aura_option;
  aura_option_is_set_ = true;
  this->check_if_valid();
  return *this;
}

MeshReqs &MeshReqs::set_field_data_manager(stk::mesh::FieldDataManager *const field_data_manager_ptr) {
  field_data_manager_ptr_ = field_data_manager_ptr;
  field_data_manager_ptr_is_set_ = true;
  this->check_if_valid();
  return *this;
}

MeshReqs &MeshReqs::set_bucket_capacity(const unsigned bucket_capacity) {
  bucket_capacity_ = bucket_capacity;
  bucket_capacity_is_set_ = true;
  this->check_if_valid();
  return *this;
}

MeshReqs &MeshReqs::set_upward_connectivity_flag(const bool upward_connectivity_flag) {
  upward_connectivity_flag_ = upward_connectivity_flag;
  upward_connectivity_flag_is_set_ = true;
  this->check_if_valid();
  return *this;
}

bool MeshReqs::constrains_spatial_dimension() const {
  return spatial_dimension_is_set_;
}

bool MeshReqs::constrains_node_coordinates_name() const {
  return node_coordinates_name_is_set_;
}

bool MeshReqs::constrains_entity_rank_names() const {
  return entity_rank_names_is_set_;
}

bool MeshReqs::constrains_communicator() const {
  return communicator_is_set_;
}

bool MeshReqs::constrains_aura_option() const {
  return aura_option_is_set_;
}

bool MeshReqs::constrains_field_data_manager() const {
  return field_data_manager_ptr_is_set_;
}

bool MeshReqs::constrains_bucket_capacity() const {
  return bucket_capacity_is_set_;
}

bool MeshReqs::constrains_upward_connectivity_flag() const {
  return upward_connectivity_flag_is_set_;
}

bool MeshReqs::is_fully_specified() const {
  return this->constrains_communicator();
}

unsigned MeshReqs::get_spatial_dimension() const {
  MUNDY_THROW_ASSERT(this->constrains_spatial_dimension(), std::logic_error,
                     "MeshReqs: Attempting to access a requirement though it is unconstrained.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  return spatial_dimension_;
}

std::string MeshReqs::get_node_coordinates_name() const {
  MUNDY_THROW_ASSERT(this->constrains_node_coordinates_name(), std::logic_error,
                     "MeshReqs: Attempting to access a requirement though it is unconstrained.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  return node_coordinates_name_;
}

std::vector<std::string> MeshReqs::get_entity_rank_names() const {
  MUNDY_THROW_ASSERT(this->constrains_entity_rank_names(), std::logic_error,
                     "MeshReqs: Attempting to access a requirement though it is unconstrained.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  return entity_rank_names_;
}

stk::ParallelMachine MeshReqs::get_communicator() const {
  MUNDY_THROW_ASSERT(this->constrains_communicator(), std::logic_error,
                     "MeshReqs: Attempting to access a requirement though it is unconstrained.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  return communicator_;
}

mundy::mesh::BulkData::AutomaticAuraOption MeshReqs::get_aura_option() const {
  MUNDY_THROW_ASSERT(
      this->constrains_aura_option(), std::logic_error,
      "MeshReqs: Attempting to access the part name requirement even though part name is unconstrained.\n"
          << "The current set of requirements is:\n"
          << get_reqs_as_a_string());
  return aura_option_;
}

stk::mesh::FieldDataManager *MeshReqs::get_field_data_manager() const {
  MUNDY_THROW_ASSERT(this->constrains_field_data_manager(), std::logic_error,
                     "MeshReqs: Attempting to access a requirement though it is unconstrained.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  return field_data_manager_ptr_;
}

unsigned MeshReqs::get_bucket_capacity() const {
  MUNDY_THROW_ASSERT(this->constrains_bucket_capacity(), std::logic_error,
                     "MeshReqs: Attempting to access a requirement though it is unconstrained.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  return bucket_capacity_;
}

bool MeshReqs::get_upward_connectivity_flag() const {
  MUNDY_THROW_ASSERT(this->constrains_upward_connectivity_flag(), std::logic_error,
                     "MeshReqs: Attempting to access a requirement though it is unconstrained.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  return upward_connectivity_flag_;
}

std::vector<std::map<std::string, std::shared_ptr<FieldReqsBase>>> MeshReqs::get_mesh_ranked_field_map() {
  // TODO(palmerb4): This is such an ugly and incorrect way to give others access to our internal fields.
  // This should be private and all other MeshReqs made friends. Better yet, sync should be a field function.
  return mesh_ranked_field_maps_;
}

std::map<std::string, std::shared_ptr<PartReqs>> MeshReqs::get_mesh_part_map() {
  // TODO(palmerb4): This is such an ugly and incorrect way to give others access to our internal parts.
  // This should be private and all other MeshReqs made friends.
  return mesh_part_map_;
}

std::vector<std::string> MeshReqs::get_mesh_attribute_names() {
  // TODO(palmerb4): This is such an ugly and incorrect way to give others access to our internal parts.
  // This should be private and all other MeshReqs made friends.
  return required_mesh_attribute_names_;
}
//}

// \name Actions
//{
std::shared_ptr<mundy::mesh::BulkData> MeshReqs::declare_mesh() {
  MUNDY_THROW_ASSERT(this->constrains_communicator(), std::logic_error,
                     "MeshReqs: The MPI communicator must be set before calling declare_mesh.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());

  // The mesh itself is generated using stk's MeshBuilder which we provide a wrapper for.
  // If any of our parameters are not constrained, we use the default value.
  mundy::mesh::MeshBuilder mesh_builder(this->get_communicator());

  if (this->constrains_spatial_dimension()) {
    mesh_builder.set_spatial_dimension(this->get_spatial_dimension());
  } else {
    mesh_builder.set_spatial_dimension(default_spatial_dimension_);
  }
  if (this->constrains_entity_rank_names()) {
    mesh_builder.set_entity_rank_names(this->get_entity_rank_names());
  } else {
    mesh_builder.set_entity_rank_names(default_entity_rank_names_);
  }
  if (this->constrains_aura_option()) {
    mesh_builder.set_auto_aura_option(this->get_aura_option());
  } else {
    mesh_builder.set_auto_aura_option(default_aura_option_);
  }
  if (this->constrains_field_data_manager()) {
    mesh_builder.set_field_data_manager(this->get_field_data_manager());
  } else {
    mesh_builder.set_field_data_manager(default_field_data_manager_ptr_);
  }
  if (this->constrains_bucket_capacity()) {
    mesh_builder.set_bucket_capacity(this->get_bucket_capacity());
  } else {
    mesh_builder.set_bucket_capacity(default_bucket_capacity_);
  }
  if (this->constrains_upward_connectivity_flag()) {
    mesh_builder.set_upward_connectivity_flag(this->get_upward_connectivity_flag());
  } else {
    mesh_builder.set_upward_connectivity_flag(default_upward_connectivity_flag_);
  }

  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Add the node coordinates field to our mesh.
  const std::string node_coordinates_field_name = this->constrains_node_coordinates_name()
                                                      ? this->get_node_coordinates_name()
                                                      : std::string(default_node_coordinates_name_);
  const unsigned spatial_dimension =
      this->constrains_spatial_dimension() ? this->get_spatial_dimension() : default_spatial_dimension_;
  const unsigned num_states = 1;
  this->add_field_reqs<double>(node_coordinates_field_name, stk::topology::NODE_RANK, spatial_dimension, num_states);

  // Declare the mesh's fields.
  // Loop over each rank's field map.
  for (auto const &mesh_field_map : mesh_ranked_field_maps_) {
    // Loop over each field and attempt to declare it.
    for ([[maybe_unused]] auto const &[field_name, field_req_ptr] : mesh_field_map) {
      field_req_ptr->declare_field_on_entire_mesh(&meta_data);
    }
  }

  // Declare the mesh's parts.
  for ([[maybe_unused]] auto const &[part_name, part_req_ptr] : mesh_part_map_) {
    part_req_ptr->declare_part_on_mesh(&meta_data);
  }

  // Declare the mesh's attributes.
  for (const std::string &attribute_name : required_mesh_attribute_names_) {
    std::any empty_attribute;
    meta_data.declare_attribute(attribute_name, empty_attribute);
  }

  return bulk_data_ptr;
}

MeshReqs &MeshReqs::delete_spatial_dimension() {
  spatial_dimension_is_set_ = false;
  return *this;
}

MeshReqs &MeshReqs::delete_entity_rank_names() {
  entity_rank_names_is_set_ = false;
  return *this;
}

MeshReqs &MeshReqs::delete_communicator() {
  communicator_is_set_ = false;
  return *this;
}

MeshReqs &MeshReqs::delete_aura_option() {
  aura_option_is_set_ = false;
  return *this;
}

MeshReqs &MeshReqs::delete_field_data_manager() {
  field_data_manager_ptr_is_set_ = false;
  return *this;
}

MeshReqs &MeshReqs::delete_bucket_capacity() {
  bucket_capacity_is_set_ = false;
  return *this;
}

MeshReqs &MeshReqs::delete_upward_connectivity_flag() {
  upward_connectivity_flag_is_set_ = false;
  return *this;
}

MeshReqs &MeshReqs::check_if_valid() {
  return *this;
}

MeshReqs &MeshReqs::add_and_sync_field_reqs(std::shared_ptr<FieldReqsBase> field_req_ptr) {
  MUNDY_THROW_ASSERT(field_req_ptr != nullptr, std::invalid_argument,
                     "MeshReqs: The pointer passed to add_and_sync_field_reqs cannot be a nullptr.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());

  // Check if the provided parameters are valid.
  field_req_ptr->check_if_valid();

  // If a field with the same name and rank exists, attempt to sync them.
  // Otherwise, create a new field entity.
  const std::string field_name = field_req_ptr->get_field_name();
  const unsigned field_rank = field_req_ptr->get_field_rank();

  auto &mesh_field_map = mesh_ranked_field_maps_[field_rank];
  const bool name_already_exists = (mesh_field_map.count(field_name) != 0);
  if (name_already_exists) {
    // Note, all syncs are two-way.
    mesh_field_map[field_name]->sync(field_req_ptr);
  } else {
    mesh_field_map.insert(std::make_pair(field_name, field_req_ptr));
  }

  // Add the field to all of our parts.
  for ([[maybe_unused]] auto &[part_name, part_req_ptr] : mesh_part_map_) {
    part_req_ptr->add_and_sync_field_reqs(field_req_ptr);
  }

  return *this;
}

MeshReqs &MeshReqs::add_and_sync_part_reqs(std::shared_ptr<PartReqs> part_req_ptr) {
  MUNDY_THROW_ASSERT(part_req_ptr != nullptr, std::invalid_argument,
                     "MeshReqs: The pointer passed to add_and_sync_part_reqs cannot be a nullptr.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());

  // Check if the provided parameters are valid.
  part_req_ptr->check_if_valid();

  // If a part with the same name and rank exists, attempt to sync them.
  // Otherwise, create a new part.
  const auto part_name = part_req_ptr->get_part_name();
  const bool name_already_exists = (mesh_part_map_.count(part_name) != 0);
  if (name_already_exists) {
    // Note, all syncs are two-way.
    mesh_part_map_[part_name]->sync(part_req_ptr);
  } else {
    mesh_part_map_.insert(std::make_pair(part_name, part_req_ptr));
  }

  // Add all of our fields to the part.
  // If a field with the same name and rank exists, adding a field will sync them.
  auto part_ranked_field_maps = part_req_ptr->get_part_ranked_field_map();
  for (auto const &mesh_field_map : mesh_ranked_field_maps_) {
    for ([[maybe_unused]] auto const &[field_name, field_req_ptr] : mesh_field_map) {
      // Either way, add our field to the part.
      part_req_ptr->add_and_sync_field_reqs(field_req_ptr);
    }
  }

  return *this;
}

MeshReqs &MeshReqs::add_mesh_attribute(const std::string &attribute_name) {
  // Adding an existing attribute is perfectly fine. It's a no-op. This merely adds more responsibility to
  // the user to ensure that an they don't unintentionally edit an attribute that is used by another method.
  const bool attribute_exists =
      std::count(required_mesh_attribute_names_.begin(), required_mesh_attribute_names_.end(), attribute_name) > 0;
  if (!attribute_exists) {
    required_mesh_attribute_names_.push_back(attribute_name);
  }
  return *this;
}

MeshReqs &MeshReqs::sync(std::shared_ptr<MeshReqs> mesh_req_ptr) {
  // TODO(palmerb4): Move this to a friend non-member function.
  // TODO(palmerb4): Optimize this function for perfect forwarding.

  // Check if the provided pointer is valid.
  // If it is not, then there is nothing to sync.
  if (mesh_req_ptr == nullptr) {
    return *this;
  }

  // Check if the provided parameters are valid.
  mesh_req_ptr->check_if_valid();

  // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
  const bool we_constrain_spatial_dimension = this->constrains_spatial_dimension();
  const bool they_constrain_spatial_dimension = mesh_req_ptr->constrains_spatial_dimension();
  if (we_constrain_spatial_dimension && they_constrain_spatial_dimension) {
    MUNDY_THROW_ASSERT(this->get_spatial_dimension() == mesh_req_ptr->get_spatial_dimension(), std::invalid_argument,
                       "MeshReqs: One of the inputs has incompatible spatial dimension ("
                           << mesh_req_ptr->get_spatial_dimension() << ").\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
  } else if (we_constrain_spatial_dimension) {
    mesh_req_ptr->set_spatial_dimension(this->get_spatial_dimension());
  } else if (they_constrain_spatial_dimension) {
    this->set_spatial_dimension(mesh_req_ptr->get_spatial_dimension());
  }

  const bool we_constrain_entity_rank_names = this->constrains_entity_rank_names();
  const bool they_constrain_entity_rank_names = mesh_req_ptr->constrains_entity_rank_names();
  if (we_constrain_entity_rank_names && they_constrain_entity_rank_names) {
    MUNDY_THROW_ASSERT(this->get_entity_rank_names() == mesh_req_ptr->get_entity_rank_names(), std::invalid_argument,
                       "MeshReqs: One of the inputs has incompatible entity rank names.\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
  } else if (we_constrain_entity_rank_names) {
    mesh_req_ptr->set_entity_rank_names(this->get_entity_rank_names());
  } else if (they_constrain_entity_rank_names) {
    this->set_entity_rank_names(mesh_req_ptr->get_entity_rank_names());
  }

  const bool we_constrain_communicator = this->constrains_communicator();
  const bool they_constrain_communicator = mesh_req_ptr->constrains_communicator();
  if (we_constrain_communicator && they_constrain_communicator) {
    MUNDY_THROW_ASSERT(this->get_communicator() == mesh_req_ptr->get_communicator(), std::invalid_argument,
                       "MeshReqs: One of the inputs has incompatible MPI communicator.\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
  } else if (we_constrain_communicator) {
    mesh_req_ptr->set_communicator(this->get_communicator());
  } else if (they_constrain_communicator) {
    this->set_communicator(mesh_req_ptr->get_communicator());
  }

  const bool we_constrain_aura_option = this->constrains_aura_option();
  const bool they_constrain_aura_option = mesh_req_ptr->constrains_aura_option();
  if (we_constrain_aura_option && they_constrain_aura_option) {
    MUNDY_THROW_ASSERT(this->get_aura_option() == mesh_req_ptr->get_aura_option(), std::invalid_argument,
                       "MeshReqs: One of the inputs has incompatible aura option.\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
  } else if (we_constrain_aura_option) {
    mesh_req_ptr->set_aura_option(this->get_aura_option());
  } else if (they_constrain_aura_option) {
    this->set_aura_option(mesh_req_ptr->get_aura_option());
  }

  const bool we_constrain_field_data_manager = this->constrains_field_data_manager();
  const bool they_constrain_field_data_manager = mesh_req_ptr->constrains_field_data_manager();
  if (we_constrain_field_data_manager && they_constrain_field_data_manager) {
    MUNDY_THROW_ASSERT(this->get_field_data_manager() == mesh_req_ptr->get_field_data_manager(), std::invalid_argument,
                       "MeshReqs: One of the inputs has incompatible field data manager.\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
  } else if (we_constrain_field_data_manager) {
    mesh_req_ptr->set_field_data_manager(this->get_field_data_manager());
  } else if (they_constrain_field_data_manager) {
    this->set_field_data_manager(mesh_req_ptr->get_field_data_manager());
  }

  const bool we_constrain_bucket_capacity = this->constrains_bucket_capacity();
  const bool they_constrain_bucket_capacity = mesh_req_ptr->constrains_bucket_capacity();
  if (we_constrain_bucket_capacity && they_constrain_bucket_capacity) {
    MUNDY_THROW_ASSERT(this->get_bucket_capacity() == mesh_req_ptr->get_bucket_capacity(), std::invalid_argument,
                       "MeshReqs: One of the inputs has incompatible bucket capacity ("
                           << mesh_req_ptr->get_bucket_capacity() << ").\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
  } else if (we_constrain_bucket_capacity) {
    mesh_req_ptr->set_bucket_capacity(this->get_bucket_capacity());
  } else if (they_constrain_bucket_capacity) {
    this->set_bucket_capacity(mesh_req_ptr->get_bucket_capacity());
  }

  const bool we_constrain_upward_connectivity_flag = this->constrains_upward_connectivity_flag();
  const bool they_constrain_upward_connectivity_flag = mesh_req_ptr->constrains_upward_connectivity_flag();
  if (we_constrain_upward_connectivity_flag && they_constrain_upward_connectivity_flag) {
    MUNDY_THROW_ASSERT(this->get_upward_connectivity_flag() == mesh_req_ptr->get_upward_connectivity_flag(),
                       std::invalid_argument,
                       "MeshReqs: One of the inputs has incompatible connectivity flag ("
                           << mesh_req_ptr->get_upward_connectivity_flag() << ").\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
  } else if (we_constrain_upward_connectivity_flag) {
    mesh_req_ptr->set_upward_connectivity_flag(this->get_upward_connectivity_flag());
  } else if (they_constrain_upward_connectivity_flag) {
    this->set_upward_connectivity_flag(mesh_req_ptr->get_upward_connectivity_flag());
  }

  // Loop over each rank's field map.
  for (auto &mesh_field_map : mesh_req_ptr->get_mesh_ranked_field_map()) {
    // Loop over each field and attempt to sync it.
    for ([[maybe_unused]] auto &[field_name, field_req_ptr] : mesh_field_map) {
      this->add_and_sync_field_reqs(field_req_ptr);
    }
  }

  // Loop over the part map.
  for ([[maybe_unused]] auto &[part_name, part_req_ptr] : mesh_req_ptr->get_mesh_part_map()) {
    this->add_and_sync_part_reqs(part_req_ptr);
  }

  // Loop over the attribute names.
  for (const std::string &attribute_name : mesh_req_ptr->get_mesh_attribute_names()) {
    this->add_mesh_attribute(attribute_name);
  }

  return *this;
}

void MeshReqs::print(std::ostream &os, int indent_level) const {
  std::string indent(indent_level * 2, ' ');

  os << indent << "MeshReqs: " << std::endl;

  if (this->constrains_spatial_dimension()) {
    os << indent << "  Spatial dimension is set." << std::endl;
    os << indent << "  Spatial dimension: " << this->get_spatial_dimension() << std::endl;
  } else {
    os << indent << "  Spatial dimension is not set." << std::endl;
  }

  if (this->constrains_entity_rank_names()) {
    os << indent << "  Entity rank names are set." << std::endl;
    os << indent << "  Entity rank names: ";
    for (const auto &entity_rank_name : this->get_entity_rank_names()) {
      os << indent << entity_rank_name << " ";
    }
    os << indent << std::endl;
  } else {
    os << indent << "  Entity rank names are not set." << std::endl;
  }

  if (this->constrains_communicator()) {
    os << indent << "  MPI communicator is set." << std::endl;
  } else {
    os << indent << "  MPI communicator is not set." << std::endl;
  }

  if (this->constrains_aura_option()) {
    os << indent << "  Aura option is set." << std::endl;
    os << indent << "  Aura option: " << this->get_aura_option() << std::endl;
  } else {
    os << indent << "  Aura option is not set." << std::endl;
  }

  if (this->constrains_field_data_manager()) {
    os << indent << "  Field data manager is set." << std::endl;
  } else {
    os << indent << "  Field data manager is not set." << std::endl;
  }

  if (this->constrains_bucket_capacity()) {
    os << indent << "  Bucket capacity is set." << std::endl;
    os << indent << "  Bucket capacity: " << this->get_bucket_capacity() << std::endl;
  } else {
    os << indent << "  Bucket capacity is not set." << std::endl;
  }

  if (this->constrains_upward_connectivity_flag()) {
    os << indent << "  Upward connectivity flag is set." << std::endl;
    os << indent << "  Upward connectivity flag: " << this->get_upward_connectivity_flag() << std::endl;
  } else {
    os << indent << "  Upward connectivity flag is not set." << std::endl;
  }

  os << indent << "  Mesh attributes: " << std::endl;
  int attribute_count = 0;
  for (const std::string &attribute_name : required_mesh_attribute_names_) {
    os << indent << "  Mesh attribute " << attribute_count << " has name (" << attribute_name << ")" << std::endl;
    attribute_count++;
  }

  os << indent << "  Mesh Fields: " << std::endl;
  int rank = 0;
  int field_count = 0;
  for (auto const &mesh_field_map : mesh_ranked_field_maps_) {
    for (auto const &[field_name, field_req_ptr] : mesh_field_map) {
      os << indent << "  Mesh field " << field_count << " has name (" << field_name << "), rank (" << rank
         << "), and requirements" << std::endl;
      field_req_ptr->print(os, indent_level + 1);
      field_count++;
    }

    rank++;
  }

  os << indent << "  Mesh Parts: " << std::endl;
  int part_count = 0;
  for (auto const &[part_name, part_req_ptr] : mesh_part_map_) {
    os << "  Mesh part " << part_count << " has name (" << part_name << ") and requirements" << std::endl;
    part_req_ptr->print(os, indent_level + 1);
    part_count++;
  }

  os << indent << "End of MeshReqs" << std::endl;
}

std::string MeshReqs::get_reqs_as_a_string() const {
  std::stringstream ss;
  this->print(ss);
  return ss.str();
}
//}

}  // namespace meta

}  // namespace mundy
