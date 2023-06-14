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
#include <tuple>        // for std::tuple
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::forward
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/Field.hpp>       // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>    // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part
#include <stk_topology/topology.hpp>     // for stk::topology
#include <stk_util/util/CSet.hpp>        // for stk::CSet

namespace mundy {

namespace meta {

/// \class FieldRequirements
/// \brief A set of necessary parameters for declaring a new field.
///
/// \c FieldRequirements is designed to take in a set of field attribute types that are
/// required to exist on the specified field as well as the type of the field. Neither the value of the attributes nor
/// the initial value of the field can be specified, since doing so would lead to fragile code. Instead, the initial
/// value and STK attributes will be initialized as a null shared pointer.
///
/// \tparam FieldType Type for elements in the field.
/// \tparam FieldAttributeTypes A set of required field attribute types. Warning, types must be unique.
template <typename FieldType>
class FieldRequirements {
 public:
  //! \name Typedefs
  //@{

  /// \param field_type Type for elements in the field. Set by the template parameter.
  using field_type = FieldType;

  /// \param field_attribute_types The set of unique field attribute types. Set by the template parameter.
  using field_attribute_types = impl::unique_tuple<FieldAttributeTypes...>::type;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is not allowed.
  FieldRequirements() = delete;

  /// \brief Constructor with full fill.
  ///
  /// \param name [in] Name of the field.
  /// \param rank [in] Rank that the field will be assigned to.
  /// \param dimension [in] Dimension of the field. For example, a dimension of three would be a vector.
  /// \param min_number_of_states [in] Minimum number of rotating states that this field will have.
  FieldRequirements(const std::string &name, const stk::topology::rank_t &rank, const unsigned dimension,
                    const unsigned min_number_of_states);
  //@}

  //! \name Getters
  //@{

  /// \brief Return the field name.
  std::string get_name() const;

  /// \brief Return the field rank.
  stk::topology::rank_t get_rank() const;

  /// \brief Return the field dimension.
  unsigned get_dimension() const;

  /// \brief Return the minimum number of field states.
  unsigned get_min_number_of_states() const;
  //@}

  //! \name Actions
  //@{

  /// \brief Declare/create the field that this class defines and assign it to a part.
  void declare_field_on_part(stk::mesh::MetaData *const meta_data_ptr, const stk::mesh::Part &part) const;

  /// \brief Ensure that the current set of requirements is valid.
  ///
  /// Here, valid means that the rank, dimension, and number of states are > 0 and that the attributes have unique type.
  /// Currently, both of these conditions are satisfied naturally by the compiler, so this function does nothing.
  void check_if_valid() const;
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
};  // FieldRequirements

//! \name Template implementations
//@{

// \name Constructors and destructor
//{

template <typename FieldType, typename... FieldAttributeTypes,
          std::enable_if_t<impl::are_types_unique<FieldAttributeTypes>::value, bool>>
FieldRequirements<FieldType, FieldAttributeTypes...>::FieldRequirements(const std::string &name,
                                                                        const stk::topology::rank_t &rank,
                                                                        const unsigned dimension,
                                                                        const unsigned min_number_of_states) {
  field_name_ = name;
  field_rank_ = fank;
  field_dimension_ = dimension;
  field_min_number_of_states_ = min_number_of_states;
}
//}

// \name Getters
//@{

template <typename FieldType, typename... FieldAttributeTypes,
          std::enable_if_t<impl::are_types_unique<FieldAttributeTypes>::value, bool>>
std::string FieldRequirements<FieldType, ... FieldAttributeTypes>::get_name() const {
  return field_name_;
}

template <typename FieldType, typename... FieldAttributeTypes>
stk::topology::rank_t FieldRequirements<FieldType, FieldAttributeTypes...>::get_rank() const {
  return field_rank_;
}

template <typename FieldType, typename... FieldAttributeTypes,
          std::enable_if_t<impl::are_types_unique<FieldAttributeTypes>::value, bool>>
unsigned FieldRequirements<FieldType, FieldAttributeTypes...>::get_dimension() const {
  return field_dimension_;
}

template <typename FieldType, typename... FieldAttributeTypes,
          std::enable_if_t<impl::are_types_unique<FieldAttributeTypes>::value, bool>>
unsigned FieldRequirements<FieldType, FieldAttributeTypes...>::get_min_number_of_states() const {
  return field_min_number_of_states_;
}
//}

// \name Actions
//{

template <typename FieldType, typename... FieldAttributeTypes,
          std::enable_if_t<impl::are_types_unique<FieldAttributeTypes>::value, bool>>
void FieldRequirements<FieldType, FieldAttributeTypes...>::declare_field_on_part(
    stk::mesh::MetaData *const meta_data_ptr, const stk::mesh::Part &part) const {
  TEUCHOS_TEST_FOR_EXCEPTION(meta_data_ptr == nullptr, std::invalid_argument,
                             "FieldRequirements: MetaData pointer cannot be null).");

  // Declare the field and assign it to the given part.
  stk::mesh::Field<FieldType> &field =
      meta_data_ptr->declare_field<FieldType>(this->get_rank(), this->get_name(), this->get_min_number_of_states());
  stk::mesh::put_field_on_mesh(field, part, nullptr);

  // The following is a nifty one-liner that runs the declaration routine for each field attribute.
  // Note, each attribute is initialized as a shared_ptr to avoid memory leaks.
  // This is a workaround for a known issue with STK.
  (..., meta_data_ptr->declare_attribute_without_delete(part, std::shared_ptr<FieldAttributeTypes>(nullptr)));
}

template <typename FieldType, typename... FieldAttributeTypes,
          std::enable_if_t<impl::are_types_unique<FieldAttributeTypes>::value, bool>>
void FieldRequirements<FieldType, FieldAttributeTypes...>::declare_field_on_mesh(
    stk::mesh::MetaData *const meta_data_ptr) const {
  TEUCHOS_TEST_FOR_EXCEPTION(meta_data_ptr == nullptr, std::invalid_argument,
                             "FieldRequirements: MetaData pointer cannot be null).");

  // Declare the field and assign it to the given mesh.
  stk::mesh::Field<FieldType> &field =
      meta_data_ptr->declare_field<FieldType>(this->get_rank(), this->get_name(), this->get_min_number_of_states());
  stk::mesh::put_field_on_mesh(field, part, nullptr);

  // The following is a nifty one-liner that runs the declaration routine for each field attribute.
  // Note, each attribute is initialized as a shared_ptr to avoid memory leaks.
  // This is a workaround for a known issue with STK.
  (..., meta_data_ptr->declare_attribute_without_delete(part, std::shared_ptr<FieldAttributeTypes>(nullptr)));
}

template <typename FieldType, typename... FieldAttributeTypes,
          std::enable_if_t<impl::are_types_unique<FieldAttributeTypes>::value, bool>>
void FieldRequirements<FieldType, FieldAttributeTypes...>::check_if_valid() const {
}
//}
//@}

//! \name Non-member functions
//@{

/// \brief Merge any number of \c FieldRequirements together.
///
/// The resulting \c FieldRequirements will have the same FieldType, name, rank, and dimension as the provided
/// \c FieldRequirements (of course, this means an error will be thrown if the \c FieldType, name, rank or dimension of
/// the provided requirements differ). Finally, the min number of states of the output requirements will equal the
/// maximum over all the provided requirements and the FieldAttributeTypes will be the set union of all the provided
/// \c FieldAttributeTypes.
///
/// \param first_field_req [in] A \c FieldRequirements objects to merge with the other requirements.
/// \param other_field_req [in] Any number of other \c FieldRequirements objects to merge with the first.
template <typename FirstFieldReqType, typename... OtherFieldReqTypes>
auto merge_field_reqs(const FirstFieldReqType &first_field_req, const OtherFieldReqTypes... &other_field_req)
    -> decltype(FieldRequirements < FirstFieldReqType::FieldType,
                unique_tuple<tuple_cat_t<FirstFieldReqType::field_attribute_types,
                                         OtherFieldReqTypes::field_attribute_types...>>::type;) const {
  // The following uses a lambda fold expression to iterate over and verify each of the input requirements.
  (
      [] {
        // Check if the provided parameters are valid.
        other_field_req.check_if_valid();

        // Check for compatibility between both sets of requirements.
        TEUCHOS_TEST_FOR_EXCEPTION(first_field_req.get_name() == other_field_req.get_name(), std::invalid_argument,
                                   "merge_field_reqs: The input field requirements has incompatible name ("
                                       << other_field_req.get_name() << "). Expected name ("
                                       << first_field_req.get_name() << ").");

        TEUCHOS_TEST_FOR_EXCEPTION(first_field_req.get_rank() == other_field_req.get_rank(), std::invalid_argument,
                                   "merge_field_reqs: The input field requirements has incompatible rank ("
                                       << other_field_req.get_rank() << "). Expected rank ("
                                       << first_field_req.get_rank() << ").");

        TEUCHOS_TEST_FOR_EXCEPTION(first_field_req.get_dimension() == other_field_req.get_dimension(),
                                   std::invalid_argument,
                                   "merge_field_reqs: The input field requirements has incompatible dimension ("
                                       << other_field_req.get_dimension() << "). Expected dimension ("
                                       << first_field_req.get_dimension() << ").");
      }(),
      ...);

  // Merge the minimum number of states.
  const unsigned merged_min_num_states =
      std::max({first_field_req.get_min_number_of_states(), other_field_req.get_min_number_of_states()...});

  // Merge the field attributes.
  using MergedFieldAttributeTypes = unique_tuple<
      tuple_cat_t<FirstFieldReqType::field_attribute_types, OtherFieldReqTypes::field_attribute_types...>>::type;

  // Combine and return the new merged field requirements.
  return FieldRequirements<FieldType, MergedFieldAttributeTypes>(
      first_field_req.get_name(), first_field_req.get_rank(), first_field_req.get_dimension(), merged_min_num_states);
}
//@}

}  // namespace meta

}  // namespace mundy
