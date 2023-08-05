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

#ifndef MUNDY_META_FIELDREQUIREMENTS_HPP_
#define MUNDY_META_FIELDREQUIREMENTS_HPP_

/// \file FieldRequirements.hpp
/// \brief Declaration of the FieldRequirements class

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
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology
#include <stk_util/util/CSet.hpp>     // for stk::CSet

// Mundy libs
#include <mundy/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirementsBase.hpp>  // for mundy::meta::FieldRequirementsBase

namespace mundy {

namespace meta {

//! \name Helper functions
//@{

/// \brief Map a string with a valid rank name to the corresponding rank.
///
/// The set of valid rank names and their corresponding type is
///  - NODE_RANK        -> stk::topology::NODE_RANK
///  - EDGE_RANK        -> stk::topology::EDGE_RANK
///  - FACE_RANK        -> stk::topology::FACE_RANK
///  - ELEMENT_RANK     -> stk::topology::ELEMENT_RANK
///  - CONSTRAINT_RANK  -> stk::topology::CONSTRAINT_RANK
///  - INVALID_RANK     -> stk::topology::INVALID_RANK
///
/// \param rank_string [in] String containing a valid rank name.
stk::topology::rank_t map_string_to_rank(const std::string &rank_string);
//@}

/// \class FieldRequirements
/// \brief A set of necessary parameters for declaring a new field.
///
/// \tparam FieldType_t Type for elements in the field.
template <typename FieldType_t>
class FieldRequirements : public FieldRequirementsBase {
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
  FieldRequirements() = default;

  /// \brief Constructor with full fill.
  ///
  /// \param field_name [in] Name of the field.
  ///
  /// \param field_rank [in] Rank that the field will be assigned to.
  ///
  /// \param field_dimension [in] Dimension of the field. For example, a dimension of three would be a vector.
  ///
  /// \param field_min_number_of_states [in] Minimum number of rotating states that this field will have.
  FieldRequirements(const std::string &field_name, const stk::topology::rank_t &field_rank,
                    const unsigned field_dimension, const unsigned field_min_number_of_states);

  /// \brief Construct from a parameter list.
  ///
  /// \param parameter_list [in] Optional list of parameters for specifying the part requirements. These parameters must
  /// be valid. That is, validate_parameters_and_set_defaults(&parameter_list) must run without error.
  explicit FieldRequirements(const Teuchos::ParameterList &parameter_list);
  //@}

  //! \name Setters and Getters
  //@{

  /// \brief Set the required field name.
  /// \param field_name [in] Required name of the field.
  void set_field_name(const std::string &field_name) final;

  /// \brief Set the required field rank.
  /// \param field_rank [in] Required rank of the field.
  void set_field_rank(const stk::topology::rank_t &field_rank) final;

  /// \brief Set the required field rank.
  /// \param field_rank [in] Required rank of the field.
  void set_field_rank(const std::string &field_rank_string) final;

  /// \brief Set the required field dimension.
  /// \param field_dimension [in] Required dimension of the field.
  void set_field_dimension(const unsigned field_dimension) final;

  /// \brief Set the minimum required number of field states.
  /// \param field_min_number_of_states [in] Minimum required number of states of the field.
  void set_field_min_number_of_states(const unsigned field_min_number_of_states) final;

  /// \brief Set the minimum required number of field states UNLESS the current minimum number of states is larger.
  /// \param field_min_number_of_states [in] Minimum required number of states of the field.
  void set_field_min_number_of_states_if_larger(const unsigned field_min_number_of_states) final;

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

  /// \brief Return the map from typeindex to field attribute.
  std::map<std::type_index, std::any> get_field_attributes_map() final;
  //@}

  //! \name Actions
  //@{

  /// \brief Declare/create the field that this class defines and assign it to a part.
  void declare_field_on_part(mundy::mesh::MetaData *const meta_data_ptr, const stk::mesh::Part &part) const final;

  /// \brief Declare/create the field that this class defines and assign it to the entire mesh.
  void declare_field_on_entire_mesh(mundy::mesh::MetaData *const meta_data_ptr) const final;

  /// \brief Delete the field name constraint (if it exists).
  void delete_field_name() final;

  /// \brief Delete the field rank constraint (if it exists).
  void delete_field_rank() final;

  /// \brief Delete the field dimension constraint (if it exists).
  void delete_field_dimension() final;

  /// \brief Delete the field minimum number of states constraint (if it exists).
  void delete_field_min_number_of_states() final;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means that the rank, dimension, and number of states are > 0, but as unsigned ints, this is always the
  /// case. We will however, leave this checker incase the class grows and the set of requirements is no longer
  /// automatically satisfied.
  void check_if_valid() const final;

  /// \brief Store a copy of an attribute on this field.
  ///
  /// Attributes are fetched from an mundy::mesh::MetaData via the get_attribute<T> routine. As a result, the
  /// identifying feature of an attribute is its type. If you attempt to add a new attribute requirement when an
  /// attribute of that type already exists, then the contents of the two attributes must match.
  ///
  /// Note, in all-too-common case where one knows the type of the desired attribute but wants to specify the value
  /// post-mesh construction, we suggest that you set store a void shared or unique pointer inside of some_attribute.
  ///
  /// \param some_attribute Any attribute that you wish to store on this field.
  void add_field_attribute(const std::any &some_attribute) final;

  /// \brief Store an attribute on this field.
  ///
  /// Attributes are fetched from an mundy::mesh::MetaData via the get_attribute<T> routine. As a result, the
  /// identifying feature of an attribute is its type. If you attempt to add a new attribute requirement when an
  /// attribute of that type already exists, then the contents of the two attributes must match.
  ///
  /// Note, in all-too-common case where one knows the type of the desired attribute but wants to specify the value
  /// post-mesh construction, we suggest that you set store a void shared or unique pointer inside of some_attribute.
  ///
  /// \param some_attribute Any attribute that you wish to store on this field.
  void add_field_attribute(std::any &&some_attribute) final;

  /// \brief Merge the current parameters with any number of other \c FieldRequirements.
  ///
  /// Here, merging two a \c FieldRequirements object with this object amounts to setting the number of states to be the
  /// maximum over all the number of states over all the \c FieldRequirements. For this process to be valid, the given
  /// \c FieldRequirements must have the same rank, type, and dimension.
  ///
  /// \param field_req_ptr [in] A \c FieldRequirements objects to merge with the current object.
  void merge(const std::shared_ptr<FieldRequirementsBase> &field_req_ptr) final;

  /// \brief Merge the current parameters with any number of other \c FieldRequirements.
  ///
  /// Here, merging two a \c FieldRequirements object with this object amounts to setting the number of states to be the
  /// maximum over all the number of states over all the \c FieldRequirements. For this process to be valid, the given
  /// \c FieldRequirements must have the same rank, type, and dimension.
  ///
  /// \param vector_of_field_req_ptrs [in] A vector of pointers to other \c FieldRequirements objects to merge with the
  /// current object.
  void merge(const std::vector<std::shared_ptr<FieldRequirementsBase>> &vector_of_field_req_ptrs) final;

  /// \brief Generate new instance of this class, constructed using the given parameter list.
  std::shared_ptr<FieldRequirementsBase> create_new_instance(const Teuchos::ParameterList &parameter_list) const final;

  /// \brief Generate new instance of this class, constructed using the given parameter list.
  static std::shared_ptr<FieldRequirementsBase> static_create_new_instance(
      const Teuchos::ParameterList &parameter_list) {
    return std::make_shared<FieldRequirements<FieldType>>(parameter_list);
  }

  /// \brief Dump the contents of \c FieldRequirements to the screen.
  void dump_to_screen(int indent_level = 0) const final;
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

  /// \brief A map from attribute type to this field's attributes.
  std::map<std::type_index, std::any> field_attributes_map_;
};  // FieldRequirements

//! \name template implementations
//@{

// \name Constructors and destructor
//{
template <typename FieldType>
FieldRequirements<FieldType>::FieldRequirements(const std::string &field_name, const stk::topology::rank_t &field_rank,
                                                const unsigned field_dimension,
                                                const unsigned field_min_number_of_states) {
  this->set_field_name(field_name);
  this->set_field_rank(field_rank);
  this->set_field_dimension(field_dimension);
  this->set_field_min_number_of_states(field_min_number_of_states);
}

template <typename FieldType>
FieldRequirements<FieldType>::FieldRequirements(const Teuchos::ParameterList &parameter_list) {
  // Validate the input params. Throws an error if a parameter is defined but not in the valid params.
  // This helps catch misspellings.
  Teuchos::ParameterList valid_params = parameter_list;
  validate_parameters_and_set_defaults(&valid_params);

  // Store the given parameters.
  this->set_field_name(valid_params.get<std::string>("name"));
  this->set_field_rank(valid_params.get<std::string>("rank"));
  this->set_field_dimension(valid_params.get<unsigned>("dimension"));
  this->set_field_min_number_of_states(valid_params.get<unsigned>("min_number_of_states"));
}
//}

// \name Setters and Getters
//{

template <typename FieldType>
void FieldRequirements<FieldType>::set_field_name(const std::string &field_name) {
  field_name_ = field_name;
  field_name_is_set_ = true;
  this->check_if_valid();
}

template <typename FieldType>
void FieldRequirements<FieldType>::set_field_rank(const stk::topology::rank_t &field_rank) {
  field_rank_ = field_rank;
  field_rank_is_set_ = true;
  this->check_if_valid();
}

template <typename FieldType>
void FieldRequirements<FieldType>::set_field_rank(const std::string &field_rank_string) {
  const stk::topology::rank_t field_rank = mundy::meta::map_string_to_rank(field_rank_string);
  this->set_field_rank(field_rank);
}

template <typename FieldType>
void FieldRequirements<FieldType>::set_field_dimension(const unsigned field_dimension) {
  field_dimension_ = field_dimension;
  field_dimension_is_set_ = true;
  this->check_if_valid();
}

template <typename FieldType>
void FieldRequirements<FieldType>::set_field_min_number_of_states(const unsigned field_min_number_of_states) {
  field_min_number_of_states_ = field_min_number_of_states;
  field_min_number_of_states_is_set_ = true;
  this->check_if_valid();
}

template <typename FieldType>
void FieldRequirements<FieldType>::set_field_min_number_of_states_if_larger(const unsigned field_min_number_of_states) {
  if (this->constrains_field_min_number_of_states()) {
    field_min_number_of_states_ = std::max(field_min_number_of_states, field_min_number_of_states_);
  } else {
    field_min_number_of_states_ = field_min_number_of_states;
  }
  field_min_number_of_states_is_set_ = true;
  this->check_if_valid();
}

template <typename FieldType>
bool FieldRequirements<FieldType>::constrains_field_name() const {
  return field_name_is_set_;
}

template <typename FieldType>
bool FieldRequirements<FieldType>::constrains_field_rank() const {
  return field_rank_is_set_;
}

template <typename FieldType>
bool FieldRequirements<FieldType>::constrains_field_dimension() const {
  return field_dimension_is_set_;
}

template <typename FieldType>
bool FieldRequirements<FieldType>::constrains_field_min_number_of_states() const {
  return field_min_number_of_states_is_set_;
}

template <typename FieldType>
bool FieldRequirements<FieldType>::is_fully_specified() const {
  return this->constrains_field_name() && this->constrains_field_rank() && this->constrains_field_dimension() &&
         this->constrains_field_min_number_of_states();
}

template <typename FieldType>
std::string FieldRequirements<FieldType>::get_field_name() const {
  MUNDY_THROW_ASSERT(
      this->constrains_field_name(), std::logic_error,
      "FieldRequirements: Attempting to access the field name requirement even though field name is unconstrained.");

  return field_name_;
}

template <typename FieldType>
stk::topology::rank_t FieldRequirements<FieldType>::get_field_rank() const {
  MUNDY_THROW_ASSERT(
      this->constrains_field_rank(), std::logic_error,
      "FieldRequirements: Attempting to access the field rank requirement even though field rank is unconstrained.");

  return field_rank_;
}

template <typename FieldType>
unsigned FieldRequirements<FieldType>::get_field_dimension() const {
  MUNDY_THROW_ASSERT(this->constrains_field_dimension(), std::logic_error,
                     "FieldRequirements: Attempting to access the field dimension requirement even though "
                     "field dimension is unconstrained.");

  return field_dimension_;
}

template <typename FieldType>
unsigned FieldRequirements<FieldType>::get_field_min_num_states() const {
  MUNDY_THROW_ASSERT(
      this->constrains_field_min_number_of_states(), std::logic_error,
      "FieldRequirements: Attempting to access the field minimum number of states requirement even though field "
      "min_number_of_states is unconstrained.");

  return field_min_number_of_states_;
}

template <typename FieldType>
const std::type_info &FieldRequirements<FieldType>::get_field_type_info() const {
  return field_type_info_;
}

template <typename FieldType>
std::map<std::type_index, std::any> FieldRequirements<FieldType>::get_field_attributes_map() {
  return field_attributes_map_;
}
//}

// \name Actions
//{

template <typename FieldType>
void FieldRequirements<FieldType>::declare_field_on_part(mundy::mesh::MetaData *const meta_data_ptr,
                                                         const stk::mesh::Part &part) const {
  MUNDY_THROW_ASSERT(meta_data_ptr != nullptr, std::invalid_argument,
                     "FieldRequirements: MetaData pointer cannot be null).");

  MUNDY_THROW_ASSERT(this->constrains_field_name(), std::logic_error,
                     "FieldRequirements: Field name must be set before calling declare_field.");
  MUNDY_THROW_ASSERT(this->constrains_field_rank(), std::logic_error,
                     "FieldRequirements: Field rank must be set before calling declare_field.");
  MUNDY_THROW_ASSERT(this->constrains_field_dimension(), std::logic_error,
                     "FieldRequirements: Field dimension must be set before calling declare_field.");
  MUNDY_THROW_ASSERT(this->constrains_field_min_number_of_states(), std::logic_error,
                     "FieldRequirements: Field minimum number of states must be set before calling declare_field.");

  // Declare the field and assign it to the given part.
  stk::mesh::Field<FieldType> &field = meta_data_ptr->declare_field<FieldType>(
      this->get_field_rank(), this->get_field_name(), this->get_field_min_num_states());
  stk::mesh::put_field_on_mesh(field, part, this->get_field_dimension(), nullptr);

  // Set the field attributes.
  for ([[maybe_unused]] auto const &[attribute_type_index, attribute] : field_attributes_map_) {
    meta_data_ptr->declare_attribute(field, attribute);
  }
}

template <typename FieldType>
void FieldRequirements<FieldType>::declare_field_on_entire_mesh(mundy::mesh::MetaData *const meta_data_ptr) const {
  MUNDY_THROW_ASSERT(meta_data_ptr != nullptr, std::invalid_argument,
                     "FieldRequirements: MetaData pointer cannot be null).");

  MUNDY_THROW_ASSERT(this->constrains_field_name(), std::logic_error,
                     "FieldRequirements: Field name must be set before calling declare_field.");
  MUNDY_THROW_ASSERT(this->constrains_field_rank(), std::logic_error,
                     "FieldRequirements: Field rank must be set before calling declare_field.");
  MUNDY_THROW_ASSERT(this->constrains_field_dimension(), std::logic_error,
                     "FieldRequirements: Field dimension must be set before calling declare_field.");
  MUNDY_THROW_ASSERT(this->constrains_field_min_number_of_states(), std::logic_error,
                     "FieldRequirements: Field minimum number of states must be set before calling declare_field.");

  // Declare the field and assign it to the given part.
  stk::mesh::Field<FieldType> &field = meta_data_ptr->declare_field<FieldType>(
      this->get_field_rank(), this->get_field_name(), this->get_field_min_num_states());
  stk::mesh::put_field_on_entire_mesh(field, this->get_field_dimension());

  // Set the field attributes.
  for ([[maybe_unused]] auto const &[attribute_type_index, attribute] : field_attributes_map_) {
    meta_data_ptr->declare_attribute(field, attribute);
  }
}

template <typename FieldType>
void FieldRequirements<FieldType>::delete_field_name() {
  field_name_is_set_ = false;
}

template <typename FieldType>
void FieldRequirements<FieldType>::delete_field_rank() {
  field_rank_is_set_ = false;
}

template <typename FieldType>
void FieldRequirements<FieldType>::delete_field_dimension() {
  field_dimension_is_set_ = false;
}

template <typename FieldType>
void FieldRequirements<FieldType>::delete_field_min_number_of_states() {
  field_min_number_of_states_is_set_ = false;
}

template <typename FieldType>
void FieldRequirements<FieldType>::check_if_valid() const {
}

template <typename FieldType>
void FieldRequirements<FieldType>::add_field_attribute(const std::any &some_attribute) {
  std::type_index attribute_type_index = std::type_index(some_attribute.type());
  field_attributes_map_.insert(std::make_pair(attribute_type_index, some_attribute));
}

template <typename FieldType>
void FieldRequirements<FieldType>::add_field_attribute(std::any &&some_attribute) {
  std::type_index attribute_type_index = std::type_index(some_attribute.type());
  field_attributes_map_.insert(std::make_pair(attribute_type_index, std::move(some_attribute)));
}

template <typename FieldType>
void FieldRequirements<FieldType>::merge(const std::shared_ptr<FieldRequirementsBase> &field_req_ptr) {
  // TODO(palmerb4): Move this to a friend non-member function.
  // TODO(palmerb4): Optimize this function for perfect forwarding.

  // Check if the provided pointer is valid.
  // If it is not, then there is nothing to merge.
  if (field_req_ptr == nullptr) {
    return;
  }

  // Check if the provided parameters are valid.
  field_req_ptr->check_if_valid();

  // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
  MUNDY_THROW_ASSERT(this->get_field_type_info() == field_req_ptr->get_field_type_info(), std::invalid_argument,
                     "FieldRequirements: Field type mismatch between our field type and the given requirements.");

  if (field_req_ptr->constrains_field_name()) {
    if (this->constrains_field_name()) {
      MUNDY_THROW_ASSERT(
          this->get_field_name() == field_req_ptr->get_field_name(), std::invalid_argument,
          "FieldRequirements: One of the inputs has incompatible name (" << field_req_ptr->get_field_name() << ").");
    } else {
      this->set_field_name(field_req_ptr->get_field_name());
    }
  }

  if (field_req_ptr->constrains_field_rank()) {
    if (this->constrains_field_rank()) {
      MUNDY_THROW_ASSERT(
          this->get_field_rank() == field_req_ptr->get_field_rank(), std::invalid_argument,
          "FieldRequirements: One of the inputs has incompatible rank (" << field_req_ptr->get_field_rank() << ").");
    } else {
      this->set_field_rank(field_req_ptr->get_field_rank());
    }
  }

  if (field_req_ptr->constrains_field_dimension()) {
    if (this->constrains_field_dimension()) {
      MUNDY_THROW_ASSERT(this->get_field_dimension() == field_req_ptr->get_field_dimension(), std::invalid_argument,
                         "FieldRequirements: One of the inputs has incompatible dimension ("
                             << field_req_ptr->get_field_dimension() << ").");
    } else {
      this->set_field_dimension(field_req_ptr->get_field_dimension());
    }
  }

  if (field_req_ptr->constrains_field_min_number_of_states()) {
    this->set_field_min_number_of_states_if_larger(field_req_ptr->get_field_min_num_states());
  }

  // Loop over the attribute map.
  for ([[maybe_unused]] auto const &[attribute_type_index, attribute] : field_req_ptr->get_field_attributes_map()) {
    this->add_field_attribute(attribute);
  }
}

template <typename FieldType>
void FieldRequirements<FieldType>::merge(
    const std::vector<std::shared_ptr<FieldRequirementsBase>> &vector_of_field_req_ptrs) {
  for (const auto &field_req_ptr : vector_of_field_req_ptrs) {
    merge(field_req_ptr);
  }
}

template <typename FieldType>
std::shared_ptr<FieldRequirementsBase> FieldRequirements<FieldType>::create_new_instance(
    const Teuchos::ParameterList &parameter_list) const {
  return create_new_instance(parameter_list);
}

template <typename FieldType>
void FieldRequirements<FieldType>::dump_to_screen(int indent_level) const {
  std::string indent(indent_level * 2, ' ');

  std::cout << indent << "FieldRequirements: " << std::endl;

  if (this->constrains_field_name()) {
    std::cout << indent << "  Field name is set." << std::endl;
    std::cout << indent << "  Field name: " << this->get_field_name() << std::endl;
  } else {
    std::cout << indent << "  Field name is not set." << std::endl;
  }

  if (this->constrains_field_rank()) {
    std::cout << indent << "  Field rank is set." << std::endl;
    std::cout << indent << "  Field rank: " << this->get_field_rank() << std::endl;
  } else {
    std::cout << indent << "  Field rank is not set." << std::endl;
  }

  if (this->constrains_field_dimension()) {
    std::cout << indent << "  Field dimension is set." << std::endl;
    std::cout << indent << "  Field dimension: " << this->get_field_dimension() << std::endl;
  } else {
    std::cout << indent << "  Field dimension is not set." << std::endl;
  }

  if (this->constrains_field_min_number_of_states()) {
    std::cout << indent << "  Field min number of states is set." << std::endl;
    std::cout << indent << "  Field min number of states: " << this->get_field_min_num_states() << std::endl;
  } else {
    std::cout << indent << "  Field min number of states is not set." << std::endl;
  }

  std::cout << indent << "  Field type info: " << field_type_info_.name() << std::endl;

  std::cout << indent << "  Field attributes map: " << std::endl;
  int attribute_count = 0;
  for (auto const &[attribute_type_index, attribute] : field_attributes_map_) {
    std::cout << indent << "  Field attribute " << attribute_count << " has type (" << attribute_type_index.name()
              << ")" << std::endl;
    attribute_count++;
  }

  std::cout << indent << "End of FieldRequirements" << std::endl;
}
//}

//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQUIREMENTS_HPP_
