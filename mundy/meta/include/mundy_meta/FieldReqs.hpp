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

#ifndef MUNDY_META_FIELDREQS_HPP_
#define MUNDY_META_FIELDREQS_HPP_

/// \file FieldReqs.hpp
/// \brief Declaration of the FieldReqs class

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqsBase.hpp>  // for mundy::meta::FieldReqsBase

namespace mundy {

namespace meta {

/// \class FieldReqs
/// \brief A set of necessary parameters for declaring a new field.
///
/// \tparam FieldType_t Type for elements in the field.
template <typename FieldType_t>
class FieldReqs : public FieldReqsBase {
 public:
  //! \name Typedefs
  //@{

  /// \tparam FieldType Type for elements in the field. Set by the template parameter.
  using FieldType = FieldType_t;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements (expect for the field type).
  FieldReqs() = default;

  /// \brief Constructor with full fill.
  ///
  /// \param field_name [in] Name of the field.
  ///
  /// \param field_rank [in] Rank that the field will be assigned to.
  ///
  /// \param field_dimension [in] Dimension of the field. For example, a dimension of three would be a vector.
  ///
  /// \param field_min_number_of_states [in] Minimum number of rotating states that this field will have.
  FieldReqs(const std::string &field_name, const stk::topology::rank_t &field_rank,
                    const unsigned field_dimension, const unsigned field_min_number_of_states);
  //@}

  //! \name Setters and Getters
  //@{

  /// \brief Set the required field name.
  /// \param field_name [in] Required name of the field.
  FieldReqs<FieldType_t> &set_field_name(const std::string &field_name) final;

  /// \brief Set the required field rank.
  /// \param field_rank [in] Required rank of the field.
  FieldReqs<FieldType_t> &set_field_rank(const stk::topology::rank_t &field_rank) final;

  /// \brief Set the required field dimension.
  /// \param field_dimension [in] Required dimension of the field.
  FieldReqs<FieldType_t> &set_field_dimension(const unsigned field_dimension) final;

  /// \brief Set the minimum required number of field states.
  /// \param field_min_number_of_states [in] Minimum required number of states of the field.
  FieldReqs<FieldType_t> &set_field_min_number_of_states(const unsigned field_min_number_of_states) final;

  /// \brief Set the minimum required number of field states UNLESS the current minimum number of states is larger.
  /// \param field_min_number_of_states [in] Minimum required number of states of the field.
  FieldReqs<FieldType_t> &set_field_min_number_of_states_if_larger(
      const unsigned field_min_number_of_states) final;

  /// \brief Get if the field name is constrained or not.
  bool constrains_field_name() const final;

  /// \brief Get if the field rank is constrained or not.
  bool constrains_field_rank() const final;

  /// \brief Get if the field dimension is constrained or not.
  bool constrains_field_dimension() const final;

  /// \brief Get if the field minimum number of states is constrained or not.
  bool constrains_field_min_number_of_states() const final;

  /// @brief Get if the field is fully specified.
  bool is_fully_specified() const final;

  /// \brief Return the field name.
  /// Will throw an error if the field name is not constrained.
  std::string get_field_name() const final;

  /// \brief Return the field rank.
  /// Will throw an error if the field rank is not constrained.
  stk::topology::rank_t get_field_rank() const final;

  /// \brief Return the field dimension.
  /// Will throw an error if the field dimension is not constrained.
  unsigned get_field_dimension() const final;

  /// \brief Return the minimum number of field states.
  /// Will throw an error if the minimum number of field states.
  unsigned get_field_min_num_states() const final;

  /// \brief Return the typeinfo related to the field's type.
  const std::type_info &get_field_type_info() const final;

  /// \brief Return the required field attribute names.
  std::vector<std::string> get_field_attribute_names() final;
  //@}

  //! \name Actions
  //@{

  /// \brief Declare/create the field that this class defines and assign it to a part.
  stk::mesh::Field<FieldType_t> &declare_field_on_part(mundy::mesh::MetaData *const meta_data_ptr,
                                                       const stk::mesh::Part &part) const final;

  /// \brief Declare/create the field that this class defines and assign it to the entire mesh.
  stk::mesh::Field<FieldType_t> &declare_field_on_entire_mesh(mundy::mesh::MetaData *const meta_data_ptr) const final;

  /// \brief Delete the field name constraint (if it exists).
  FieldReqs<FieldType_t> &delete_field_name() final;

  /// \brief Delete the field rank constraint (if it exists).
  FieldReqs<FieldType_t> &delete_field_rank() final;

  /// \brief Delete the field dimension constraint (if it exists).
  FieldReqs<FieldType_t> &delete_field_dimension() final;

  /// \brief Delete the field minimum number of states constraint (if it exists).
  FieldReqs<FieldType_t> &delete_field_min_number_of_states() final;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means that the rank, dimension, and number of states are > 0, but as unsigned ints, this is always the
  /// case. We will however, leave this checker incase the class grows and the set of requirements is no longer
  /// automatically satisfied.
  FieldReqs<FieldType_t> &check_if_valid() final;

  /// \brief Require that an attribute with the given name be present on the field.
  ///
  /// \param attribute_name [in] The name of the attribute that must be present on the field.
  FieldReqs<FieldType_t> &add_field_attribute(const std::string &attribute_name) final;

  /// \brief Synchronize (merge and rectify differences) the current parameters with any number of other \c FieldReqs.
  ///
  /// Here, syncing two \c FieldReqs object amounts to setting their number of states to be the
  /// maximum of their set min num states. For this process to be valid, the given
  /// \c FieldReqs must have the same rank, type, and dimension. It also syncs their attributes.
  ///
  /// \param field_req_ptr [in] A \c FieldReqs objects to sync with the current object.
  FieldReqs<FieldType_t> &sync(std::shared_ptr<FieldReqsBase> field_req_ptr) final;

  /// \brief Dump the contents of \c FieldReqs to the given stream (defaults to std::cout).
  void print_reqs(std::ostream &os = std::cout, int indent_level = 0) const final;

  /// \brief Return a string representation of the current set of requirements.
  std::string get_reqs_as_a_string() const final;
  //@}

 private:
  /// \brief Name of the field.
  std::string field_name_;

  /// \brief Typeinfo related to the field's type.
  static const inline std::type_info &field_type_info_ = typeid(FieldType);

  /// \brief Rank that the field will be assigned to.
  stk::topology::rank_t field_rank_;

  /// \brief Dimension of the field. For example, a dimension of three would be a vector.
  unsigned field_dimension_;

  /// \brief Minimum number of rotating states that this field will have.
  unsigned field_min_number_of_states_;

  /// \brief If the name of the field is set or not.
  bool field_name_is_set_;

  /// \brief If the rank that the field will be assigned to is set or not.
  bool field_rank_is_set_;

  /// \brief If the dimension of the field. For example, a dimension of three would be a vector is set or not.
  bool field_dimension_is_set_;

  /// \brief If the minimum number of rotating states that this field will have is set or not.
  bool field_min_number_of_states_is_set_;

  /// \brief A vector of required field attribute names.
  std::vector<std::string> required_field_attribute_names_;
};  // FieldReqs

//! \name template implementations
//@{

// \name Constructors and destructor
//{
template <typename FieldType>
FieldReqs<FieldType>::FieldReqs(const std::string &field_name, const stk::topology::rank_t &field_rank,
                                                const unsigned field_dimension,
                                                const unsigned field_min_number_of_states) {
  this->set_field_name(field_name);
  this->set_field_rank(field_rank);
  this->set_field_dimension(field_dimension);
  this->set_field_min_number_of_states(field_min_number_of_states);
}
//}

// \name Setters and Getters
//{

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_field_name(const std::string &field_name) {
  field_name_ = field_name;
  field_name_is_set_ = true;
  this->check_if_valid();
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_field_rank(const stk::topology::rank_t &field_rank) {
  field_rank_ = field_rank;
  field_rank_is_set_ = true;
  this->check_if_valid();
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_field_dimension(const unsigned field_dimension) {
  field_dimension_ = field_dimension;
  field_dimension_is_set_ = true;
  this->check_if_valid();
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_field_min_number_of_states(
    const unsigned field_min_number_of_states) {
  field_min_number_of_states_ = field_min_number_of_states;
  field_min_number_of_states_is_set_ = true;
  this->check_if_valid();
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_field_min_number_of_states_if_larger(
    const unsigned field_min_number_of_states) {
  if (this->constrains_field_min_number_of_states()) {
    field_min_number_of_states_ = std::max(field_min_number_of_states, field_min_number_of_states_);
  } else {
    field_min_number_of_states_ = field_min_number_of_states;
  }
  field_min_number_of_states_is_set_ = true;
  this->check_if_valid();
  return *this;
}

template <typename FieldType>
bool FieldReqs<FieldType>::constrains_field_name() const {
  return field_name_is_set_;
}

template <typename FieldType>
bool FieldReqs<FieldType>::constrains_field_rank() const {
  return field_rank_is_set_;
}

template <typename FieldType>
bool FieldReqs<FieldType>::constrains_field_dimension() const {
  return field_dimension_is_set_;
}

template <typename FieldType>
bool FieldReqs<FieldType>::constrains_field_min_number_of_states() const {
  return field_min_number_of_states_is_set_;
}

template <typename FieldType>
bool FieldReqs<FieldType>::is_fully_specified() const {
  return this->constrains_field_name() && this->constrains_field_rank() && this->constrains_field_dimension() &&
         this->constrains_field_min_number_of_states();
}

template <typename FieldType>
std::string FieldReqs<FieldType>::get_field_name() const {
  MUNDY_THROW_ASSERT(
      this->constrains_field_name(), std::logic_error,
      "FieldReqs: Attempting to access the field name requirement even though field name is unconstrained.\n"
          << "The current set of requirements is:\n"
          << get_reqs_as_a_string());

  return field_name_;
}

template <typename FieldType>
stk::topology::rank_t FieldReqs<FieldType>::get_field_rank() const {
  MUNDY_THROW_ASSERT(
      this->constrains_field_rank(), std::logic_error,
      "FieldReqs: Attempting to access the field rank requirement even though field rank is unconstrained.\n"
          << "The current set of requirements is:\n"
          << get_reqs_as_a_string());

  return field_rank_;
}

template <typename FieldType>
unsigned FieldReqs<FieldType>::get_field_dimension() const {
  MUNDY_THROW_ASSERT(this->constrains_field_dimension(), std::logic_error,
                     "FieldReqs: Attempting to access the field dimension requirement even though "
                     "field dimension is unconstrained.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());

  return field_dimension_;
}

template <typename FieldType>
unsigned FieldReqs<FieldType>::get_field_min_num_states() const {
  MUNDY_THROW_ASSERT(
      this->constrains_field_min_number_of_states(), std::logic_error,
      "FieldReqs: Attempting to access the field minimum number of states requirement even though field "
      "min_number_of_states is unconstrained.\n"
          << "The current set of requirements is:\n"
          << get_reqs_as_a_string());

  return field_min_number_of_states_;
}

template <typename FieldType>
const std::type_info &FieldReqs<FieldType>::get_field_type_info() const {
  return field_type_info_;
}

template <typename FieldType>
std::vector<std::string> FieldReqs<FieldType>::get_field_attribute_names() {
  return required_field_attribute_names_;
}
//}

// \name Actions
//{

template <typename FieldType>
stk::mesh::Field<FieldType> &FieldReqs<FieldType>::declare_field_on_part(
    mundy::mesh::MetaData *const meta_data_ptr, const stk::mesh::Part &part) const {
  MUNDY_THROW_ASSERT(meta_data_ptr != nullptr, std::invalid_argument,
                     "FieldReqs: MetaData pointer cannot be null).");

  MUNDY_THROW_ASSERT(this->constrains_field_name(), std::logic_error,
                     "FieldReqs: Field name must be set before calling declare_field.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  MUNDY_THROW_ASSERT(this->constrains_field_rank(), std::logic_error,
                     "FieldReqs: Field rank must be set before calling declare_field.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  MUNDY_THROW_ASSERT(this->constrains_field_dimension(), std::logic_error,
                     "FieldReqs: Field dimension must be set before calling declare_field.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  MUNDY_THROW_ASSERT(this->constrains_field_min_number_of_states(), std::logic_error,
                     "FieldReqs: Field minimum number of states must be set before calling declare_field.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());

  // Declare the field and assign it to the given part.
  stk::mesh::Field<FieldType> &field = meta_data_ptr->declare_field<FieldType>(
      this->get_field_rank(), this->get_field_name(), this->get_field_min_num_states());
  stk::mesh::put_field_on_mesh(field, part, this->get_field_dimension(), nullptr);

  // Set the field attributes.
  for (auto const &attribute_name : required_field_attribute_names_) {
    std::any empty_attribute;
    meta_data_ptr->declare_attribute(field, attribute_name, empty_attribute);
  }
  return field;
}

template <typename FieldType>
stk::mesh::Field<FieldType> &FieldReqs<FieldType>::declare_field_on_entire_mesh(
    mundy::mesh::MetaData *const meta_data_ptr) const {
  MUNDY_THROW_ASSERT(meta_data_ptr != nullptr, std::invalid_argument,
                     "FieldReqs: MetaData pointer cannot be null).");

  MUNDY_THROW_ASSERT(this->constrains_field_name(), std::logic_error,
                     "FieldReqs: Field name must be set before calling declare_field.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  MUNDY_THROW_ASSERT(this->constrains_field_rank(), std::logic_error,
                     "FieldReqs: Field rank must be set before calling declare_field.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  MUNDY_THROW_ASSERT(this->constrains_field_dimension(), std::logic_error,
                     "FieldReqs: Field dimension must be set before calling declare_field.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());
  MUNDY_THROW_ASSERT(this->constrains_field_min_number_of_states(), std::logic_error,
                     "FieldReqs: Field minimum number of states must be set before calling declare_field.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());

  // Declare the field and assign it to the given part.
  stk::mesh::Field<FieldType> &field = meta_data_ptr->declare_field<FieldType>(
      this->get_field_rank(), this->get_field_name(), this->get_field_min_num_states());
  stk::mesh::put_field_on_entire_mesh(field, this->get_field_dimension());

  // Set the field attributes.
  for (auto const &attribute_name : required_field_attribute_names_) {
    std::any empty_attribute;
    meta_data_ptr->declare_attribute(field, attribute_name, empty_attribute);
  }
  return field;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::delete_field_name() {
  field_name_is_set_ = false;
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::delete_field_rank() {
  field_rank_is_set_ = false;
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::delete_field_dimension() {
  field_dimension_is_set_ = false;
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::delete_field_min_number_of_states() {
  field_min_number_of_states_is_set_ = false;
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::check_if_valid() {
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::add_field_attribute(const std::string &attribute_name) {
  // Adding an existing attribute is perfectly fine. It's a no-op. This merely adds more responsibility to
  // the user to ensure that an they don't unintentionally edit an attribute that is used by another method.
  const bool attribute_exists =
      std::count(required_field_attribute_names_.begin(), required_field_attribute_names_.end(), attribute_name) > 0;
  if (!attribute_exists) {
    required_field_attribute_names_.push_back(attribute_name);
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::sync(
    std::shared_ptr<FieldReqsBase> field_req_ptr) {
  // TODO(palmerb4): Move this to a friend non-member function.
  // TODO(palmerb4): Optimize this function for perfect forwarding.

  // Check if the provided pointer is valid.
  // If it is not, then there is nothing to sync.
  if (field_req_ptr == nullptr) {
    return *this;
  }

  // Check if the provided parameters are valid.
  field_req_ptr->check_if_valid();

  // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
  MUNDY_THROW_ASSERT(this->get_field_type_info() == field_req_ptr->get_field_type_info(), std::invalid_argument,
                     "FieldReqs: Field type mismatch between our field type and the given requirements.\n"
                         << "The current set of requirements is:\n"
                         << get_reqs_as_a_string());


  const bool we_constrain_field_name = this->constrains_field_name();
  const bool they_constrain_field_name = field_req_ptr->constrains_field_name();
  if (we_constrain_field_name && they_constrain_field_name) {
    MUNDY_THROW_ASSERT(this->get_field_name() == field_req_ptr->get_field_name(), std::invalid_argument,
                       "FieldReqs: One of the inputs has incompatible name ("
                           << field_req_ptr->get_field_name() << ").\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
  } else if (we_constrain_field_name) {
    field_req_ptr->set_field_name(this->get_field_name());
  } else if (they_constrain_field_name) {
    this->set_field_name(field_req_ptr->get_field_name());
  }

  const bool we_constrain_field_rank = this->constrains_field_rank();
  const bool they_constrain_field_rank = field_req_ptr->constrains_field_rank();
  if (we_constrain_field_rank && they_constrain_field_rank) {
    MUNDY_THROW_ASSERT(this->get_field_rank() == field_req_ptr->get_field_rank(), std::invalid_argument,
                       "FieldReqs: One of the inputs has incompatible rank ("
                           << field_req_ptr->get_field_rank() << ").\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
  } else if (we_constrain_field_rank) {
    field_req_ptr->set_field_rank(this->get_field_rank());
  } else if (they_constrain_field_rank) {
    this->set_field_rank(field_req_ptr->get_field_rank());
  }

  const bool we_constrain_field_dimension = this->constrains_field_dimension();
  const bool they_constrain_field_dimension = field_req_ptr->constrains_field_dimension();
  if (we_constrain_field_dimension && they_constrain_field_dimension) {
    MUNDY_THROW_ASSERT(this->get_field_dimension() == field_req_ptr->get_field_dimension(), std::invalid_argument,
                       "FieldReqs: One of the inputs has incompatible dimension ("
                           << field_req_ptr->get_field_dimension() << ").\n"
                           << "The current set of requirements is:\n"
                           << get_reqs_as_a_string());
  } else if (we_constrain_field_dimension) {
    field_req_ptr->set_field_dimension(this->get_field_dimension());
  } else if (they_constrain_field_dimension) {
    this->set_field_dimension(field_req_ptr->get_field_dimension());
  }

  const bool we_constrain_field_min_number_of_states = this->constrains_field_min_number_of_states();
  const bool they_constrain_field_min_number_of_states = field_req_ptr->constrains_field_min_number_of_states();
  if (we_constrain_field_min_number_of_states && they_constrain_field_min_number_of_states) {
    // Min num states is special. We take the max of the two.
    const unsigned max_num_states = std::max(this->get_field_min_num_states(), field_req_ptr->get_field_min_num_states());
    field_req_ptr->set_field_min_number_of_states(max_num_states);
    this->set_field_min_number_of_states(max_num_states);
  } else if (we_constrain_field_min_number_of_states) {
    field_req_ptr->set_field_min_number_of_states(this->get_field_min_num_states());
  } else if (they_constrain_field_min_number_of_states) {
    this->set_field_min_number_of_states(field_req_ptr->get_field_min_num_states());
  }

  // Loop over the attribute map.
  for (const std::string &attribute_name : field_req_ptr->get_field_attribute_names()) {
    this->add_field_attribute(attribute_name);
  }
  return *this;
}

template <typename FieldType>
void FieldReqs<FieldType>::print_reqs(std::ostream &os, int indent_level) const {
  std::string indent(indent_level * 2, ' ');

  os << indent << "FieldReqs: " << std::endl;

  if (this->constrains_field_name()) {
    os << indent << "  Field name is set." << std::endl;
    os << indent << "  Field name: " << this->get_field_name() << std::endl;
  } else {
    os << indent << "  Field name is not set." << std::endl;
  }

  if (this->constrains_field_rank()) {
    os << indent << "  Field rank is set." << std::endl;
    os << indent << "  Field rank: " << this->get_field_rank() << std::endl;
  } else {
    os << indent << "  Field rank is not set." << std::endl;
  }

  if (this->constrains_field_dimension()) {
    os << indent << "  Field dimension is set." << std::endl;
    os << indent << "  Field dimension: " << this->get_field_dimension() << std::endl;
  } else {
    os << indent << "  Field dimension is not set." << std::endl;
  }

  if (this->constrains_field_min_number_of_states()) {
    os << indent << "  Field min number of states is set." << std::endl;
    os << indent << "  Field min number of states: " << this->get_field_min_num_states() << std::endl;
  } else {
    os << indent << "  Field min number of states is not set." << std::endl;
  }

  os << indent << "  Field type info: " << field_type_info_.name() << std::endl;

  os << indent << "  Field attributes: " << std::endl;
  int attribute_count = 0;
  for (const std::string &attribute_name : required_field_attribute_names_) {
    os << indent << "  Field attribute " << attribute_count << " has name (" << attribute_name << ")" << std::endl;
    attribute_count++;
  }

  os << indent << "End of FieldReqs" << std::endl;
}

template <typename FieldType>
std::string FieldReqs<FieldType>::get_reqs_as_a_string() const {
  std::stringstream ss;
  this->print_reqs(ss);
  return ss.str();
}
//}

//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQS_HPP_
