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

/// \file PartReqs.cpp
/// \brief Definition of the PartReqs class

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <regex>        // for std::regex_match
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string, std::stoi
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>         // for mundy::meta::FieldReqs, mundy::meta::FieldReqsBase
#include <mundy_meta/FieldReqsFactory.hpp>  // for mundy::meta::FieldReqsFactory
#include <mundy_meta/PartReqs.hpp>          // for mundy::meta::PartReqs

namespace mundy {

namespace meta {

// \name Constructors and destructor
//{

PartReqs::PartReqs(const std::string &part_name) {
  this->set_part_name(part_name);
}

PartReqs::PartReqs(const std::string &part_name, const stk::topology::topology_t &part_topology) {
  this->set_part_name(part_name);
  this->set_part_topology(part_topology);
}

PartReqs::PartReqs(const std::string &part_name, const stk::topology::rank_t &part_rank) {
  this->set_part_name(part_name);
  this->set_part_rank(part_rank);
}
//}

// \name Setters
//{

PartReqs &PartReqs::set_part_name(const std::string &part_name) {
  if (has_master_part_reqs_) {
    master_part_req_ptr_->set_part_name(part_name);
  } else {
    part_name_ = part_name;
    part_name_is_set_ = true;
    this->check_if_valid();
  }
  return *this;
}

PartReqs &PartReqs::set_part_topology(const stk::topology::topology_t &part_topology) {
  if (has_master_part_reqs_) {
    master_part_req_ptr_->set_part_topology(part_topology);
  } else {
    bool part_rank_already_set = this->constrains_part_rank();
    MUNDY_THROW_ASSERT(
        !part_rank_already_set, std::logic_error,
        "PartReqs: Parts are designed to fall into three catagories: name set, name and topology set, "
            << "name and rank set. \n This part already sets the rank, so it's invalid to also set the topology.\n"
            << "The current set of requirements is:\n"
            << get_reqs_as_a_string());
    part_topology_ = part_topology;
    part_topology_is_set_ = true;
    this->check_if_valid();
  }
  return *this;
}

PartReqs &PartReqs::set_part_rank(const stk::topology::rank_t &part_rank) {
  if (has_master_part_reqs_) {
    master_part_req_ptr_->set_part_rank(part_rank);
  } else {
    bool part_topology_already_set = this->constrains_part_topology();
    MUNDY_THROW_ASSERT(
        !part_topology_already_set, std::logic_error,
        "PartReqs: Parts are designed to fall into three catagories: name set, name and topology set, "
            << "name and rank set. \n This part already sets the topology, so it's invalid to also set the rank.\n"
            << "The current set of requirements is:\n"
            << get_reqs_as_a_string());
    part_rank_ = part_rank;
    part_rank_is_set_ = true;
    this->check_if_valid();
  }
  return *this;
}

PartReqs &PartReqs::delete_part_name() {
  if (has_master_part_reqs_) {
    master_part_req_ptr_->delete_part_name();
  } else {
    part_name_is_set_ = false;
  }
  return *this;
}

PartReqs &PartReqs::delete_part_topology() {
  if (has_master_part_reqs_) {
    master_part_req_ptr_->delete_part_topology();
  } else {
    part_topology_is_set_ = false;
  }
  return *this;
}

PartReqs &PartReqs::delete_part_rank() {
  part_rank_is_set_ = false;
  return *this;
}

PartReqs &PartReqs::add_and_sync_field_reqs(std::shared_ptr<FieldReqsBase> field_req_ptr) {
  if (has_master_part_reqs_) {
    master_part_req_ptr_->add_and_sync_field_reqs(field_req_ptr);
  } else {
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

    auto &part_field_map = this->get_part_ranked_field_map()[field_rank];
    const bool name_already_exists = (part_field_map.count(field_name) != 0);
    if (name_already_exists) {
      part_field_map[field_name]->sync(field_req_ptr);
    } else {
      part_field_map.insert(std::make_pair(field_name, field_req_ptr));
    }

    // Add the field to all of our subparts.
    for (auto &[subpart_name, subpart_req_ptr] : this->get_part_subpart_map()) {
      subpart_req_ptr->add_and_sync_field_reqs(field_req_ptr);
    }
  }
  return *this;
}

PartReqs &PartReqs::add_and_sync_subpart_reqs(std::shared_ptr<PartReqs> part_req_ptr) {
  if (has_master_part_reqs_) {
    master_part_req_ptr_->add_and_sync_subpart_reqs(part_req_ptr);
  } else {
    MUNDY_THROW_ASSERT(part_req_ptr != nullptr, std::invalid_argument,
                       "MeshReqs: The pointer passed to add_and_sync_subpart_reqs cannot be a nullptr.\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());

    // Check if the provided parameters are valid.
    part_req_ptr->check_if_valid();

    // If a subpart with the same name exists, attempt to sync them.
    // Otherwise, create a new subpart entity.
    const bool name_already_exists = (part_subpart_map_.count(part_req_ptr->get_part_name()) != 0);
    if (name_already_exists) {
      part_subpart_map_[part_req_ptr->get_part_name()]->sync(part_req_ptr);
    } else {
      part_subpart_map_.insert(std::make_pair(part_req_ptr->get_part_name(), part_req_ptr));

      // Add all of our fields to the new subpart.
      for (auto const &part_field_map : this->get_part_ranked_field_map()) {
        for (auto const &[field_name, field_req_ptr] : part_field_map) {
          part_req_ptr->add_and_sync_field_reqs(field_req_ptr);
        }
      }

      // Add all of our part attributes to the new subpart.
      for (const std::string &attribute_name : this->get_part_attribute_names()) {
        part_req_ptr->add_part_attribute(attribute_name);
      }
    }
  }
  return *this;
}

PartReqs &PartReqs::add_part_attribute(const std::string &attribute_name) {
  if (has_master_part_reqs_) {
    master_part_req_ptr_->add_part_attribute(attribute_name);
  } else {
    // Adding an existing attribute is perfectly fine. It's a no-op. This merely adds more responsibility to
    // the user to ensure that an they don't unintentionally edit an attribute that is used by another method.
    const bool attribute_does_not_exist =
        std::count(required_part_attribute_names_.begin(), required_part_attribute_names_.end(), attribute_name) == 0;
    if (attribute_does_not_exist) {
      required_part_attribute_names_.push_back(attribute_name);

      // Add the attribute to all of our subparts.
      for (auto &[subpart_name, subpart_req_ptr] : part_subpart_map_) {
        subpart_req_ptr->add_part_attribute(attribute_name);
      }
    }
  }
  return *this;
}
//@}

// \name Getters
//{

bool PartReqs::constrains_part_name() const {
  if (has_master_part_reqs_) {
    return master_part_req_ptr_->constrains_part_name();
  } else {
    return part_name_is_set_;
  }
}

bool PartReqs::constrains_part_topology() const {
  if (has_master_part_reqs_) {
    return master_part_req_ptr_->constrains_part_topology();
  } else {
    return part_topology_is_set_;
  }
}

bool PartReqs::constrains_part_rank() const {
  if (has_master_part_reqs_) {
    return master_part_req_ptr_->constrains_part_rank();
  } else {
    return part_rank_is_set_;
  }
}

bool PartReqs::is_fully_specified() const {
  if (has_master_part_reqs_) {
    return master_part_req_ptr_->is_fully_specified();
  } else {
    return this->constrains_part_name();
  }
}

std::string PartReqs::get_part_name() const {
  if (has_master_part_reqs_) {
    return master_part_req_ptr_->get_part_name();
  } else {
    MUNDY_THROW_ASSERT(
        this->constrains_part_name(), std::logic_error,
        "PartReqs: Attempting to access the part name requirement even though part name is unconstrained.\n"
            << "The current set of requirements is:\n"
            << get_reqs_as_a_string());

    return part_name_;
  }
}

stk::topology::topology_t PartReqs::get_part_topology() const {
  if (has_master_part_reqs_) {
    return master_part_req_ptr_->get_part_topology();
  } else {
    MUNDY_THROW_ASSERT(this->constrains_part_topology(), std::logic_error,
                       "PartReqs: Attempting to access the part topology requirement even though part "
                       "topology is unconstrained.\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());

    return part_topology_;
  }
}

stk::topology::rank_t PartReqs::get_part_rank() const {
  if (has_master_part_reqs_) {
    return master_part_req_ptr_->get_part_rank();
  } else {
    MUNDY_THROW_ASSERT(
        this->constrains_part_rank(), std::logic_error,
        "PartReqs: Attempting to access the part rank requirement even though part rank is unconstrained.\n"
            << "The current set of requirements is:\n"
            << get_reqs_as_a_string());

    return part_rank_;
  }
}

std::vector<std::map<std::string, std::shared_ptr<FieldReqsBase>>> &PartReqs::get_part_ranked_field_map() {
  if (has_master_part_reqs_) {
    return master_part_req_ptr_->get_part_ranked_field_map();
  } else {
    // TODO(palmerb4): This is such an ugly and incorrect way to give others access to our internal fields.
    return part_ranked_field_maps_;
  }
}

std::map<std::string, std::shared_ptr<PartReqs>> &PartReqs::get_part_subpart_map() {
  if (has_master_part_reqs_) {
    return master_part_req_ptr_->get_part_subpart_map();
  } else {
    return part_subpart_map_;
  }
}

std::vector<std::string> &PartReqs::get_part_attribute_names() {
  if (has_master_part_reqs_) {
    return master_part_req_ptr_->get_part_attribute_names();
  } else {
    return required_part_attribute_names_;
  }
}
//}

// \name Private member functions
//{
PartReqs &PartReqs::set_master_part_reqs(std::shared_ptr<PartReqs> master_part_req_ptr) {
  MUNDY_THROW_ASSERT(
      !has_master_part_reqs_, std::logic_error,
      "PartReqs: The master part requirements have already been set. Overriding it could lead to undefined behavior.");
  master_part_req_ptr_ = std::move(master_part_req_ptr);
  has_master_part_reqs_ = true;
  return *this;
}

std::shared_ptr<PartReqs> PartReqs::get_master_part_reqs() {
  MUNDY_THROW_ASSERT(has_master_part_reqs_, std::logic_error,
                     "PartReqs: The master part requirements have not been set. Cannot return a null pointer.");
  return master_part_req_ptr_;
}

bool PartReqs::has_master_part_reqs() const {
  return has_master_part_reqs_;
}
//}

// \name Actions
//{
stk::mesh::Part &PartReqs::declare_part_on_mesh(mundy::mesh::MetaData *const meta_data_ptr) const {
  MUNDY_THROW_ASSERT(meta_data_ptr != nullptr, std::invalid_argument, "PartReqs: MetaData pointer cannot be null).");
  MUNDY_THROW_ASSERT(this->constrains_part_name(), std::logic_error,
                     "PartReqs: Part name must be set before calling declare_part.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());

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
                     "PartReqs: Weird. The desired part name and actual part name differ.\n"
                         << "This should never happen. Please report this bug to the developers.\n"
                         << "  Desired part name: " << this->get_part_name() << "\n"
                         << "  Actual part name: " << part_ptr->name() << ".\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());

  // Declare the Part's fields and associate them with the Part.
  // Loop over each rank's field map.
  for (auto const &part_field_map : const_cast<PartReqs *>(this)->get_part_ranked_field_map()) {
    // Loop over each field and attempt to sync it.
    for ([[maybe_unused]] auto const &[field_name, field_req_ptr] : part_field_map) {
      field_req_ptr->declare_field_on_part(meta_data_ptr, *part_ptr);
    }
  }

  // Declare the sub-parts and declare them as sub-parts.
  // Each sub-part will. in turn, declare their fields and subparts.
  for ([[maybe_unused]] auto const &[subpart_name, subpart_req_ptr] :
       const_cast<PartReqs *>(this)->get_part_subpart_map()) {
    stk::mesh::Part &subpart = subpart_req_ptr->declare_part_on_mesh(meta_data_ptr);
    meta_data_ptr->declare_part_subset(*part_ptr, subpart);
  }

  // Declare the Part's attributes.
  for (const std::string &attribute_name : const_cast<PartReqs *>(this)->get_part_attribute_names()) {
    std::any empty_attribute;
    meta_data_ptr->declare_attribute(*part_ptr, attribute_name, empty_attribute);
  }

  return *part_ptr;
}

PartReqs &PartReqs::check_if_valid() {
  // TODO(palmerb4): What are the other requirements for validity?

  // One invalid state is if we have a master part reqs object but master_part_req_ptr_ is null.
  if (has_master_part_reqs_) {
    MUNDY_THROW_ASSERT(master_part_req_ptr_ != nullptr, std::logic_error,
                       "FieldReqs: We have a master field reqs object but master_part_req_ptr_ is null.\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
    master_part_req_ptr_->check_if_valid();
  }

  return *this;
}

PartReqs &PartReqs::sync(std::shared_ptr<PartReqs> part_req_ptr) {
  // TODO(palmerb4): Move this to a friend non-member function.
  // TODO(palmerb4): Optimize this function for perfect forwarding.

  // Check if the provided pointer is valid. Throw an error if it is not.
  MUNDY_THROW_ASSERT(part_req_ptr != nullptr, std::invalid_argument,
                     "PartReqs: The given PartReqs pointer cannot be null.");

  // To prevent circular dependencies, we will check if the given FieldReqs pointer points to us. If it does, then
  // their's nothing to do.
  bool does_part_req_ptr_point_to_us = part_req_ptr.get() == this;
  if (!does_part_req_ptr_point_to_us) {
    const bool we_have_master_part_reqs = this->has_master_part_reqs();
    const bool they_have_master_part_reqs = part_req_ptr->has_master_part_reqs();

    if (we_have_master_part_reqs && they_have_master_part_reqs) {
      // If both have master reqs, then we synchronize the masters (potentially leading to an upward tree traversal).
      this->get_master_part_reqs()->sync(part_req_ptr->get_master_part_reqs());
    } else if (we_have_master_part_reqs && !they_have_master_part_reqs) {
      // If we have a master and they don't, then we set their master to be our master.
      part_req_ptr->set_master_part_reqs(this->get_master_part_reqs());
    } else if (!we_have_master_part_reqs && they_have_master_part_reqs) {
      // If they have a master and we don't, then we set our master to be their master.
      this->set_master_part_reqs(part_req_ptr->get_master_part_reqs());
    } else {
      // If neither has a master, then we will create a shared master FieldReqs object from our merged requirements.
      // Check if the provided parameters are valid.
      part_req_ptr->check_if_valid();

      // Use the contents of us and them to create a shared master FieldReqs object.
      auto shared_master_part_req_ptr = std::make_shared<PartReqs>();

      // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
      const bool we_constrain_part_name = this->constrains_part_name();
      const bool they_constrain_part_name = part_req_ptr->constrains_part_name();
      if (we_constrain_part_name && they_constrain_part_name) {
        MUNDY_THROW_ASSERT(this->get_part_name() == part_req_ptr->get_part_name(), std::invalid_argument,
                           "PartReqs: One of the inputs has incompatible name ("
                               << part_req_ptr->get_part_name() << ").\n"
                               << "The current set of requirements is:\n"
                               << get_reqs_as_a_string());
        shared_master_part_req_ptr->set_part_name(this->get_part_name());
      } else if (we_constrain_part_name) {
        shared_master_part_req_ptr->set_part_name(this->get_part_name());
      } else if (they_constrain_part_name) {
        shared_master_part_req_ptr->set_part_name(part_req_ptr->get_part_name());
      }

      const bool we_constrain_part_rank = this->constrains_part_rank();
      const bool they_constrain_part_rank = part_req_ptr->constrains_part_rank();
      if (we_constrain_part_rank && they_constrain_part_rank) {
        MUNDY_THROW_ASSERT(this->get_part_rank() == part_req_ptr->get_part_rank(), std::invalid_argument,
                           "PartReqs: One of the inputs has incompatible rank ("
                               << part_req_ptr->get_part_rank() << ").\n"
                               << "The current set of requirements is:\n"
                               << get_reqs_as_a_string());
        shared_master_part_req_ptr->set_part_rank(this->get_part_rank());
      } else if (we_constrain_part_rank) {
        shared_master_part_req_ptr->set_part_rank(this->get_part_rank());
      } else if (they_constrain_part_rank) {
        shared_master_part_req_ptr->set_part_rank(part_req_ptr->get_part_rank());
      }

      const bool we_constrain_part_topology = this->constrains_part_topology();
      const bool they_constrain_part_topology = part_req_ptr->constrains_part_topology();
      if (we_constrain_part_topology && they_constrain_part_topology) {
        MUNDY_THROW_ASSERT(this->get_part_topology() == part_req_ptr->get_part_topology(), std::invalid_argument,
                           "PartReqs: One of the inputs has incompatible topology ("
                               << part_req_ptr->get_part_topology() << ").\n"
                               << "The current set of requirements is:\n"
                               << get_reqs_as_a_string());
        shared_master_part_req_ptr->set_part_topology(this->get_part_topology());
      } else if (we_constrain_part_topology) {
        shared_master_part_req_ptr->set_part_topology(this->get_part_topology());
      } else if (they_constrain_part_topology) {
        shared_master_part_req_ptr->set_part_topology(part_req_ptr->get_part_topology());
      }

      // Add our/their field requirements to the given FieldReqs object.
      for (auto &part_field_map : this->get_part_ranked_field_map()) {
        for ([[maybe_unused]] auto &[field_name, field_req_ptr] : part_field_map) {
          shared_master_part_req_ptr->add_and_sync_field_reqs(field_req_ptr);
        }
      }
      for (auto &part_field_map : part_req_ptr->get_part_ranked_field_map()) {
        for ([[maybe_unused]] auto &[field_name, field_req_ptr] : part_field_map) {
          shared_master_part_req_ptr->add_and_sync_field_reqs(field_req_ptr);
        }
      }

      // Add our/their subpart requirements to the given FieldReqs object.
      for ([[maybe_unused]] auto &[part_name, subpart_req_ptr] : this->get_part_subpart_map()) {
        shared_master_part_req_ptr->add_and_sync_subpart_reqs(subpart_req_ptr);
      }
      for ([[maybe_unused]] auto &[part_name, subpart_req_ptr] : part_req_ptr->get_part_subpart_map()) {
        shared_master_part_req_ptr->add_and_sync_subpart_reqs(subpart_req_ptr);
      }

      // Add our/their attribute names to the given FieldReqs object.
      for (const std::string &attribute_name : this->get_part_attribute_names()) {
        shared_master_part_req_ptr->add_part_attribute(attribute_name);
      }
      for (const std::string &attribute_name : part_req_ptr->get_part_attribute_names()) {
        shared_master_part_req_ptr->add_part_attribute(attribute_name);
      }

      // The given shared_master_part_req_ptr now contains the merged requirements. Set it as the master for both.
      this->set_master_part_reqs(shared_master_part_req_ptr);
      part_req_ptr->set_master_part_reqs(shared_master_part_req_ptr);
    }
  }
  return *this;
}

void PartReqs::print(std::ostream &os, int indent_level) const {
  std::string indent(indent_level * 2, ' ');

  os << indent << "PartReqs: " << std::endl;

  if (this->constrains_part_name()) {
    os << indent << "  Part name is set." << std::endl;
    os << indent << "  Part name: " << this->get_part_name() << std::endl;
  } else {
    os << "  Part name is not set." << std::endl;
  }

  if (this->constrains_part_rank()) {
    os << indent << "  Part rank is set." << std::endl;
    os << indent << "  Part rank: " << this->get_part_rank() << std::endl;
  } else {
    os << indent << "  Part rank is not set." << std::endl;
  }

  if (this->constrains_part_topology()) {
    os << indent << "  Part topology is set." << std::endl;
    os << indent << "  Part topology: " << this->get_part_topology() << std::endl;
  } else {
    os << indent << "  Part topology is not set." << std::endl;
  }

  os << indent << "  Part Fields: " << std::endl;
  int rank = 0;
  int field_count = 0;
  for (auto const &part_field_map : const_cast<PartReqs *>(this)->get_part_ranked_field_map()) {
    for (auto const &[field_name, field_req_ptr] : part_field_map) {
      os << indent << "  Part field " << field_count << " has name (" << field_name << "), rank (" << rank
         << "), and requirements" << std::endl;
      field_req_ptr->print(os, indent_level + 1);
      field_count++;
    }

    rank++;
  }

  os << indent << "  Part Subparts: " << std::endl;
  int subpart_count = 0;
  for (auto const &[subpart_name, subpart_req_ptr] : const_cast<PartReqs *>(this)->get_part_subpart_map()) {
    os << indent << "  Part subpart " << subpart_count << " has name (" << subpart_name << ") and requirements"
       << std::endl;
    subpart_req_ptr->print(os, indent_level + 1);
    subpart_count++;
  }

  os << indent << "  Part attributes: " << std::endl;
  int attribute_count = 0;
  for (const std::string &attribute_name : const_cast<PartReqs *>(this)->get_part_attribute_names()) {
    os << indent << "  Part attribute " << attribute_count << " has name (" << attribute_name << ")" << std::endl;
    attribute_count++;
  }

  os << indent << "End of PartReqs" << std::endl;
}

std::string PartReqs::get_reqs_as_a_string() const {
  std::stringstream ss;
  this->print(ss);
  return ss.str();
}
//}

}  // namespace meta

}  // namespace mundy
