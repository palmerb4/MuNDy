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
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector
// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/Field.hpp>       // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>    // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part
#include <stk_topology/topology.hpp>     // for stk::topology

// Mundy libs
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
/// \tparam FieldType Type for elements in the field.
template <typename FieldType>
class FieldRequirements : public FieldRequirementsBase {
 public:
  //! \name Typedefs
  //@{

  /// \tparam field_type Type for elements in the field. Set by the template parameter.
  typedef FieldType field_type;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements.
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
  /// \param parameter_list [in] Optional list of parameters for specifying the part requirements. The set of valid
  /// parameters is accessible via \c get_valid_params.
  explicit FieldRequirements(const Teuchos::ParameterList &parameter_list);
  //@}

  //! \name Setters and Getters
  //@{

  /// \brief Set the required field name.
  /// \brief field_name [in] Required name of the field.
  void set_field_name(const std::string &field_name) final;

  /// \brief Set the required field rank.
  /// \brief field_rank [in] Required rank of the field.
  void set_field_rank(const stk::topology::rank_t &field_rank) final;

  /// \brief Set the required field rank.
  /// \brief field_rank [in] Required rank of the field.
  void set_field_rank(const std::string &field_rank_string) final;

  /// \brief Set the required field dimension.
  /// \brief field_dimension [in] Required dimension of the field.
  void set_field_dimension(const unsigned field_dimension) final;

  /// \brief Set the minimum required number of field states UNLESS the current minimum number of states is larger.
  /// \brief field_min_number_of_states [in] Minimum required number of states of the field.
  void set_field_min_number_of_states_if_larger(const unsigned field_min_number_of_states) final;

  /// \brief Get if the field name is constrained or not.
  bool constrains_field_name() const final;

  /// \brief Get if the field rank is constrained or not.
  bool constrains_field_rank() const final;

  /// \brief Get if the field dimension is constrained or not.
  bool constrains_field_dimension() const final;

  /// \brief Get if the field minimum number of states is constrained or not.
  bool constrains_field_min_number_of_states() const final;

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
  unsigned get_field_min_number_of_states() const final;

  /// \brief Get the default transient parameters for this class (those that do not impact the part requirements).
  Teuchos::ParameterList get_valid_params() const final;

  /// \brief Get the default transient parameters for this class (those that do not impact the part requirements).
  static Teuchos::ParameterList static_get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("name", "INVALID", "Name of the field.");
    default_parameter_list.set("rank", stk::topology::INVALID_RANK, "Rank of the field.");
    default_parameter_list.set("dimension", 0, "Dimension of the part.");
    default_parameter_list.set("min_number_of_states", 1,
                               "Minimum number of rotating states that this field will have.");
    return default_parameter_list;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Declare/create the field that this class defines.
  void declare_field_on_part(stk::mesh::MetaData *const meta_data_ptr, const stk::mesh::Part &part) const final;

  /// \brief Delete the field name constraint (if it exists).
  void delete_field_name_constraint() final;

  /// \brief Delete the field rank constraint (if it exists).
  void delete_field_rank_constraint() final;

  /// \brief Delete the field dimension constraint (if it exists).
  void delete_field_dimension_constraint() final;

  /// \brief Delete the field minimum number of states constraint (if it exists).
  void delete_field_min_number_of_states_constraint() final;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means that the rank, dimension, and number of states are > 0, but as unsigned ints, this is always the
  /// case. We will however, leave this checker incase the class grows and the set of requirements is no longer
  /// automatically satisfied.
  void check_if_valid() const final;

  /// \brief Merge the current parameters with any number of other \c FieldRequirements.
  ///
  /// Here, merging two a \c FieldRequirements object with this object amounts to setting the number of states to be the
  /// maximum over all the number of states over all the \c FieldRequirements. For this process to be valid, the given
  /// \c FieldRequirements must have the same rank, type, and dimension. The name of the other fields does not need to
  /// match the current name of this field.
  ///
  /// \param field_req_ptr [in] A \c FieldRequirements objects to merge with the current object.
  void merge(const std::shared_ptr<FieldRequirementsBase> &field_req_ptr) final;

  /// \brief Merge the current parameters with any number of other \c FieldRequirements.
  ///
  /// Here, merging two a \c FieldRequirements object with this object amounts to setting the number of states to be the
  /// maximum over all the number of states over all the \c FieldRequirements. For this process to be valid, the given
  /// \c FieldRequirements must have the same rank, type, and dimension. The name of the other fields does not need to
  /// match the current name of this field.
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
  //@}

 private:
  /// \brief Name of the field.
  std::string field_name_;

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
  this->set_field_min_number_of_states_if_larger(field_min_number_of_states);
}

template <typename FieldType>
FieldRequirements<FieldType>::FieldRequirements(const Teuchos::ParameterList &parameter_list) {
  // Validate the input params. Throws an error if a parameter is defined but not in the valid params.
  // This helps catch misspellings.
  parameter_list.validateParameters(this->get_valid_params());

  // Store the given parameters.
  if (parameter_list.isParameter("name")) {
    const std::string field_name = parameter_list.get<std::string>("name");
    this->set_field_name(field_name);
  }
  if (parameter_list.isParameter("rank")) {
    const std::string field_rank = parameter_list.get<std::string>("rank");
    this->set_field_rank(field_rank);
  }
  if (parameter_list.isParameter("dimension")) {
    const unsigned field_dimension = parameter_list.get<unsigned>("dimension");
    this->set_field_dimension(field_dimension);
  }
  if (parameter_list.isParameter("min_number_of_states")) {
    const unsigned field_min_number_of_states = parameter_list.get<unsigned>("min_number_of_states");
    this->set_field_min_number_of_states_if_larger(field_min_number_of_states);
  }
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
std::string FieldRequirements<FieldType>::get_field_name() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_field_name(), std::logic_error,
      "FieldRequirements: Attempting to access the field name requirement even though field name is unconstrained.");

  return field_name_;
}

template <typename FieldType>
stk::topology::rank_t FieldRequirements<FieldType>::get_field_rank() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_field_rank(), std::logic_error,
      "FieldRequirements: Attempting to access the field rank requirement even though field rank is unconstrained.");

  return field_rank_;
}

template <typename FieldType>
unsigned FieldRequirements<FieldType>::get_field_dimension() const {
  TEUCHOS_TEST_FOR_EXCEPTION(!this->constrains_field_dimension(), std::logic_error,
                             "FieldRequirements: Attempting to access the field dimension requirement even though "
                             "field dimension is unconstrained.");

  return field_dimension_;
}

template <typename FieldType>
unsigned FieldRequirements<FieldType>::get_field_min_number_of_states() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !this->constrains_field_min_number_of_states(), std::logic_error,
      "FieldRequirements: Attempting to access the field minimum number of states requirement even though field "
      "min_number_of_states is unconstrained.");

  return field_min_number_of_states_;
}

template <typename FieldType>
Teuchos::ParameterList FieldRequirements<FieldType>::get_valid_params() const {
  return static_get_valid_params();
}
//}

// \name Actions
//{

template <typename FieldType>
void FieldRequirements<FieldType>::declare_field_on_part(stk::mesh::MetaData *const meta_data_ptr,
                                                         const stk::mesh::Part &part) const {
  TEUCHOS_TEST_FOR_EXCEPTION(meta_data_ptr == nullptr, std::invalid_argument,
                             "FieldRequirements: MetaData pointer cannot be null).");

  TEUCHOS_TEST_FOR_EXCEPTION(this->constrains_field_name(), std::logic_error,
                             "FieldRequirements: Field name must be set before calling declare_field.");
  TEUCHOS_TEST_FOR_EXCEPTION(this->constrains_field_rank(), std::logic_error,
                             "FieldRequirements: Field rank must be set before calling declare_field.");
  TEUCHOS_TEST_FOR_EXCEPTION(this->constrains_field_dimension(), std::logic_error,
                             "FieldRequirements: Field dimension must be set before calling declare_field.");
  TEUCHOS_TEST_FOR_EXCEPTION(
      this->constrains_field_min_number_of_states(), std::logic_error,
      "FieldRequirements: Field minimum number of states must be set before calling declare_field.");

  // Declare the field and assign it to the given part
  stk::mesh::Field<FieldType> &field =
      meta_data_ptr->declare_field<FieldType>(this->get_field_rank(), this->get_field_name());
  stk::mesh::put_field_on_mesh(field, part, nullptr);
}

template <typename FieldType>
void FieldRequirements<FieldType>::delete_field_name_constraint() {
  field_name_is_set_ = false;
}

template <typename FieldType>
void FieldRequirements<FieldType>::delete_field_rank_constraint() {
  field_rank_is_set_ = false;
}

template <typename FieldType>
void FieldRequirements<FieldType>::delete_field_dimension_constraint() {
  field_dimension_is_set_ = false;
}

template <typename FieldType>
void FieldRequirements<FieldType>::delete_field_min_number_of_states_constraint() {
  field_min_number_of_states_is_set_ = false;
}

template <typename FieldType>
void FieldRequirements<FieldType>::check_if_valid() const {
}

template <typename FieldType>
void FieldRequirements<FieldType>::merge(const std::shared_ptr<FieldRequirementsBase> &field_req_ptr) {
  merge({field_req_ptr});
}

template <typename FieldType>
void FieldRequirements<FieldType>::merge(
    const std::vector<std::shared_ptr<FieldRequirementsBase>> vector_of_field_req_ptrs) {
  for (const auto &field_req_ptr : vector_of_field_req_ptrs) {
    // Check if the provided parameters are valid.
    field_req_ptr->check_if_valid();

    // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
    if (field_req_ptr->constrains_field_name()) {
      if (this->constrains_field_name()) {
        TEUCHOS_TEST_FOR_EXCEPTION(
            this->get_field_name() == field_req_ptr->get_field_name(), std::invalid_argument,
            "FieldRequirements: One of the inputs has incompatible name (" << field_req_ptr->get_field_name() << ").");
      } else {
        this->set_field_name(field_req_ptr->get_field_name());
      }
    }

    if (field_req_ptr->constrains_field_rank()) {
      if (this->constrains_field_rank()) {
        TEUCHOS_TEST_FOR_EXCEPTION(
            this->get_field_rank() == field_req_ptr->get_field_rank(), std::invalid_argument,
            "FieldRequirements: One of the inputs has incompatible rank (" << field_req_ptr->get_field_rank() << ").");
      } else {
        this->set_field_rank(field_req_ptr->get_field_rank());
      }
    }

    if (field_req_ptr->constrains_field_dimension()) {
      if (this->constrains_field_dimension()) {
        TEUCHOS_TEST_FOR_EXCEPTION(this->get_field_dimension() == field_req_ptr->get_field_dimension(),
                                   std::invalid_argument,
                                   "FieldRequirements: One of the inputs has incompatible dimension ("
                                       << field_req_ptr->get_field_dimension() << ").");
      } else {
        this->set_field_dimension(field_req_ptr->get_field_dimension());
      }
    }

    if (field_req_ptr->constrains_field_min_number_of_states()) {
      this->set_field_min_number_of_states_if_larger(field_req_ptr->get_field_min_number_of_states());
    }
  }
}

template <typename FieldType>
std::shared_ptr<FieldRequirementsBase> FieldRequirements<FieldType>::create_new_instance(
    const Teuchos::ParameterList &parameter_list) const {
  return static_create_new_instance(parameter_list);
}
//}

//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQUIREMENTS_HPP_
