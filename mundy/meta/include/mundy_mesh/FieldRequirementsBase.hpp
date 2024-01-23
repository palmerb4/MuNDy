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

#ifndef MUNDY_META_FIELDREQUIREMENTSBASE_HPP_
#define MUNDY_META_FIELDREQUIREMENTSBASE_HPP_

/// \file FieldRequirementsBase.hpp
/// \brief Declaration of the FieldRequirementsBase class

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_mesh/MetaData.hpp>  // for mundy::mesh::MetaData

namespace mundy {

namespace meta {

/// \class FieldRequirementsBase
/// \brief A consistent interface for all \c FieldRequirementsBase.
class FieldRequirementsBase {
 public:
  //! \name Setters and Getters
  //@{

  /// \brief Set the required field name.
  /// \brief field_name [in] Required name of the field.
  virtual void set_field_name(const std::string &field_name) = 0;

  /// \brief Set the required field rank.
  /// \brief field_rank [in] Required rank of the field.
  virtual void set_field_rank(const stk::topology::rank_t &field_rank) = 0;

  /// \brief Set the required field rank.
  /// \brief field_rank [in] Required rank of the field.
  virtual void set_field_rank(const std::string &field_rank_string) = 0;

  /// \brief Set the required field dimension.
  /// \brief field_dimension [in] Required dimension of the field.
  virtual void set_field_dimension(const unsigned field_dimension) = 0;

  /// \brief Set the minimum required number of field states to the given value.
  /// \brief field_min_number_of_states [in] Minimum required number of states of the field.
  virtual void set_field_min_number_of_states(const unsigned field_min_number_of_states) = 0;

  /// \brief Set the minimum required number of field states UNLESS the current minimum number of states is larger.
  /// \brief field_min_number_of_states [in] Minimum required number of states of the field.
  virtual void set_field_min_number_of_states_if_larger(const unsigned field_min_number_of_states) = 0;

  /// \brief Get if the field name is constrained or not.
  virtual bool constrains_field_name() const = 0;

  /// \brief Get if the field rank is constrained or not.
  virtual bool constrains_field_rank() const = 0;

  /// \brief Get if the field dimension is constrained or not.
  virtual bool constrains_field_dimension() const = 0;

  /// \brief Get if the field minimum number of states is constrained or not.
  virtual bool constrains_field_min_number_of_states() const = 0;

  /// @brief Get if the field is fully specified.
  virtual bool is_fully_specified() const = 0;

  /// \brief Return the field name.
  /// Will throw an error if the field name is not constrained.
  virtual std::string get_field_name() const = 0;

  /// \brief Return the field rank.
  /// Will throw an error if the field rank is not constrained.
  virtual stk::topology::rank_t get_field_rank() const = 0;

  /// \brief Return the field dimension.
  /// Will throw an error if the field dimension is not constrained.
  virtual unsigned get_field_dimension() const = 0;

  /// \brief Return the minimum number of field states.
  /// Will throw an error if the minimum number of field states.
  virtual unsigned get_field_min_num_states() const = 0;

  /// \brief Return the typeinfo related to the field's type.
  virtual const std::type_info &get_field_type_info() const = 0;

  /// \brief Return the map from typeindex to field attribute.
  virtual std::map<std::type_index, std::any> get_field_attributes_map() = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Validate the given parameters and set the default values if not provided.
  static void validate_parameters_and_set_defaults(Teuchos::ParameterList *parameter_list_ptr) {
    if (parameter_list_ptr->isParameter("name")) {
      const bool valid_type = parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_ASSERT(
          valid_type, std::invalid_argument,
          "FieldRequirements: Type error. Given a parameter with name 'name' but with a type other than std::string.");
    } else {
      parameter_list_ptr->set("name", "INVALID", "Name of the field.");
    }

    if (parameter_list_ptr->isParameter("rank")) {
      const bool valid_type = ((parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("rank")) ||
                               (parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<stk::topology::rank_t>("rank")));
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "FieldRequirements: Type error. Given a parameter with name 'rank' but with a "
                             << "type other than std::string or stk::topology::rank_t.");
    } else {
      parameter_list_ptr->set("rank", stk::topology::INVALID_RANK, "Rank of the field, in string form.");
    }

    if (parameter_list_ptr->isParameter("dimension")) {
      const bool valid_type = parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<unsigned>("dimension");
      MUNDY_THROW_ASSERT(
          valid_type, std::invalid_argument,
          "FieldRequirements: Type error. Given a parameter with name 'dimension' but with a type other than unsigned.");
    } else {
      parameter_list_ptr->set("dimension", 0, "Dimension of the part.");
    }

    if (parameter_list_ptr->isParameter("min_number_of_states")) {
      const bool valid_type = parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<unsigned>("min_number_of_states");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "FieldRequirements: Type error. Given a parameter with name 'min_number_of_states' "
                         "but with a type other than unsigned.");
    } else {
      parameter_list_ptr->set("min_number_of_states", 1,
                              "Minimum number of rotating states that this field will have.");
    }
  }

  /// \brief Declare/create the field that this class defines.
  virtual void declare_field_on_part(mundy::mesh::MetaData *const meta_data_ptr, const stk::mesh::Part &part) const = 0;

  /// \brief Declare/create the field that this class defines and assign it to the entire mesh.
  virtual void declare_field_on_entire_mesh(mundy::mesh::MetaData *const meta_data_ptr) const = 0;

  /// \brief Delete the field name constraint (if it exists).
  virtual void delete_field_name() = 0;

  /// \brief Delete the field rank constraint (if it exists).
  virtual void delete_field_rank() = 0;

  /// \brief Delete the field dimension constraint (if it exists).
  virtual void delete_field_dimension() = 0;

  /// \brief Delete the field minimum number of states constraint (if it exists).
  virtual void delete_field_min_number_of_states() = 0;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means that the rank, dimension, and number of states are > 0, but as unsigned ints, this is always the
  /// case. We will however, leave this checker incase the class grows and the set of requirements is no longer
  /// automatically satisfied.
  virtual void check_if_valid() const = 0;

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
  virtual void add_field_attribute(const std::any &some_attribute) = 0;

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
  virtual void add_field_attribute(std::any &&some_attribute) = 0;

  /// \brief Merge the current parameters with any number of other \c FieldRequirements.
  ///
  /// Here, merging two a \c FieldRequirements object with this object amounts to setting the number of states to be the
  /// maximum over all the number of states over all the \c FieldRequirements. For this process to be valid, the given
  /// \c FieldRequirements must have the same rank, type, and dimension. The name of the other fields does not need to
  /// match the current name of this field.
  ///
  /// \param field_req_ptr [in] A \c FieldRequirements objects to merge with the current object.
  virtual void merge(const std::shared_ptr<FieldRequirementsBase> &field_req_ptr) = 0;

  /// \brief Merge the current parameters with any number of other \c FieldRequirements.
  ///
  /// Here, merging two a \c FieldRequirements object with this object amounts to setting the number of states to be the
  /// maximum over all the number of states over all the \c FieldRequirements. For this process to be valid, the given
  /// \c FieldRequirements must have the same rank, type, and dimension. The name of the other fields does not need to
  /// match the current name of this field.
  ///
  /// \param list_of_field_reqs [in] A list of other \c FieldRequirements objects to merge with the current object.
  virtual void merge(const std::vector<std::shared_ptr<FieldRequirementsBase>> &vector_of_field_req_ptrs) = 0;

  /// \brief Generate new instance of this class, constructed using the given parameter list.
  virtual std::shared_ptr<FieldRequirementsBase> create_new_instance(
      const Teuchos::ParameterList &parameter_list) const = 0;

  /// \brief Dump the contents of \c FieldRequirements to the given stream (defaults to std::cout).
  virtual void print_reqs(std::ostream &os = std::cout, int indent_level = 0) const = 0;

  /// \brief Return a string representation of the current set of requirements.
  virtual std::string get_reqs_as_a_string() const = 0;
  //@}
};  // FieldRequirementsBase

//}

//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQUIREMENTSBASE_HPP_
