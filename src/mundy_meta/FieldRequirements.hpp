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

// clang-format off
#include <gtest/gtest.h>                             // for AssertHelper, etc
#include <mpi.h>                                     // for MPI_COMM_WORLD, etc
#include <stddef.h>                                  // for size_t
#include <vector>                                    // for vector, etc
#include <random>                                    // for rand
#include <memory>                                    // for shared_ptr
#include <string>                                    // for string
#include <type_traits>                               // for static_assert
#include <stk_math/StkVector.hpp>                    // for Vec
#include <stk_mesh/base/BulkData.hpp>                // for BulkData
#include <stk_mesh/base/MetaData.hpp>                // for MetaData
#include <stk_mesh/base/GetEntities.hpp>             // for count_selected_entities
#include <stk_mesh/base/Types.hpp>                   // for EntityVector, etc
#include <stk_mesh/base/Ghosting.hpp>                // for create_ghosting
#include <stk_io/WriteMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_search/SearchMethod.hpp>               // for KDTREE
#include <stk_search/CoarseSearch.hpp>               // for coarse_search
#include <stk_search/BoundingBox.hpp>                // for Sphere, Box, tec.
#include <stk_balance/balance.hpp>                   // for balanceStkMesh
#include <stk_util/parallel/Parallel.hpp>            // for ParallelMachine
#include <stk_util/environment/WallTime.hpp>         // for wall_time
#include <stk_util/environment/perf_util.hpp>
#include <stk_unit_test_utils/BuildMesh.hpp>
#include "stk_util/util/ReportHandler.hpp"           // for ThrowAssert, etc
// clang-format on

namespace mundy {

namespace meta {

/// \class FieldRequirementsBase
/// \brief A consistant base for all \c FieldRequirements.
class FieldRequirementsBase {};  // FieldRequirementsBase

/// \class InvalidType
/// \brief An invalid typename to allow default templating \c FieldRequirements.
class InvalidType {};  // InvalidType

/// \class FieldRequirements
/// \brief A set of necessary parameters for declaring a new field.
///
/// \tparam FieldType Type for elements in the field.
template <typename FieldType = InvalidType>
class FieldRequirements {
 public:
  //! \name Typedefs
  //@{

  /// \tparam field_type Type for elements in the field. Set by the template parameter.
  typedef FieldType field_type;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is not allowed
  FieldRequirements() = delete;

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
                    const unsigned field_dimension, const unsigned field_min_number_of_states) {
    this.set_field_name(field_name);
    this.set_field_rank(field_rank);
    this.set_field_dimension(field_dimension);
    this.set_field_min_number_of_states(field_min_number_of_states);
  }

  /// \brief Construct from a parameter list.
  ///
  /// \param parameter_list [in] Optional list of parameters for specifying the part requirements. The set of valid
  /// parameters is accessible via \c get_valid_params.
  explicit FieldRequirements(const Teuchos::ParameterList &parameter_list) {
    // Validate the input params. Throws an error if a parameter is defined but not in the valid params.
    // This helps catch misspellings.
    parameter_list.validateParameters(this.get_valid_params());

    // Store the given parameters.
    if (parameter_list.isParameter("name")) {
      const std::string field_name = parameter_list.get<std::string>("name");
      this.set_field_name(field_name);
    }
    if (parameter_list.isParameter("rank")) {
      const std::string field_rank = parameter_list.get<std::string>("rank");
      this.set_field_rank(field_rank);
    }
    if (parameter_list.isParameter("dimension")) {
      const unsigned field_dimension = parameter_list.get<unsigned>("dimension");
      this.set_field_dimension(field_dimension);
    }
    if (parameter_list.isParameter("min_number_of_states")) {
      const unsigned field_min_number_of_states = parameter_list.get<unsigned>("min_number_of_states");
      this.set_field_min_number_of_states(field_min_number_of_states);
    }
  }
  //@}

  //! \name Setters and Getters
  //@{

  /// \brief Set the required field name.
  /// \brief field_name [in] Required name of the field.
  void set_field_name(const std::string &field_name) {
    field_name_ = field_name;
    field_name_is_set_ = true;
    this.check_if_valid();
  }

  /// \brief Set the required field rank.
  /// \brief field_rank [in] Required rank of the field.
  void set_field_rank(const stk::topology::rank_t &field_rank) {
    field_rank_ = field_rank;
    field_rank_is_set_ = true;
    this.check_if_valid();
  }

  /// \brief Set the required field rank.
  /// \brief field_rank [in] Required rank of the field.
  void set_field_rank(const std::string &field_rank_string) {
    const stk::topology::rank_t field_rank = mundy::meta::map_string_to_rank(field_rank_string);
    this.set_field_rank(field_rank)
  }

  /// \brief Set the required field dimension.
  /// \brief field_dimension [in] Required dimension of the field.
  void set_field_dimension(const unsigned field_dimension) {
    field_dimension_ = field_dimension;
    field_dimension_is_set_ = true;
    this.check_if_valid();
  }

  /// \brief Set the minimum required number of field states UNLESS the current minimum number of states is larger.
  /// \brief field_min_number_of_states [in] Minimum required number of states of the field.
  void set_field_min_number_of_states_if_larger(const unsigned field_min_number_of_states) {
    if (this->constrains_field_min_number_of_states()) {
      field_min_number_of_states_ = std::max(field_min_number_of_states, field_min_number_of_states_);
    } else {
      field_min_number_of_states_ = field_min_number_of_states;
    }
    field_min_number_of_states_is_set_ = true;
    this.check_if_valid();
  }

  /// \brief Get if the field name is constrained or not.
  bool constrains_field_name() {
    return field_name_is_set_;
  }

  /// \brief Get if the field rank is constrained or not.
  bool constrains_field_rank() {
    return field_rank_is_set_;
  }

  /// \brief Get if the field dimension is constrained or not.
  bool constrains_field_dimension() {
    return field_dimension_is_set_;
  }

  /// \brief Get if the field minimum number of states is constrained or not.
  bool constrains_field_min_number_of_states() {
    return field_min_number_of_states_is_set_;
  }

  /// \brief Return the field name.
  /// Will throw an error if the field name is not constrained.
  std::string get_field_name() {
    TEUCHOS_TEST_FOR_EXCEPTION(
        !this.constrains_field_name(), std::logic_error,
        "Attempting to access the field name requirement even though field name is unconstrained.");

    return field_name_;
  }

  /// \brief Return the field rank.
  /// Will throw an error if the field rank is not constrained.
  stk::topology::rank_t get_field_rank() {
    TEUCHOS_TEST_FOR_EXCEPTION(
        !this.constrains_field_rank(), std::logic_error,
        "Attempting to access the field rank requirement even though field rank is unconstrained.");

    return field_rank_;
  }

  /// \brief Return the field dimension.
  /// Will throw an error if the field dimension is not constrained.
  unsigned get_field_dimension() {
    TEUCHOS_TEST_FOR_EXCEPTION(
        !this.constrains_field_dimension(), std::logic_error,
        "Attempting to access the field dimension requirement even though field dimension is unconstrained.");

    return field_dimension_;
  }

  /// \brief Return the minimum number of field states.
  /// Will throw an error if the minimum number of field states.
  unsigned get_field_min_number_of_states() {
    TEUCHOS_TEST_FOR_EXCEPTION(!this.constrains_field_min_number_of_states(), std::logic_error,
                               "Attempting to access the field minimum number of states requirement even though field "
                               "min_number_of_states is unconstrained.");

    return field_min_number_of_states_;
  }

  /// \brief Get the default parameters for this class.
  static Teuchos::ParameterList get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("name", "INVALID", "Name of the field.");
    default_parameter_list.set("rank", stk::topology::INVALID_RANK, "Rank of the field.");
    default_parameter_list.set("dimension", stk::topology::INVALID_TOPOLOGY, "Dimension of the part.");
    default_parameter_list.set("min_number_of_states", stk::topology::INVALID_TOPOLOGY,
                               "Minimum number of rotating states that this field will have.");
    return default_parameter_list;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Declare/create the field that this class defines.
  stk::mesh::Field<FieldType> &declare_field_on_part(const stk::mesh::MetaData *meta_data_ptr,
                                                     const stk::mesh::Part &part) {
    TEUCHOS_TEST_FOR_EXCEPTION(meta_data_ptr == nullptr, std::invalid_argument,
                               "mundy::meta::FieldRequirements: MetaData pointer cannot be null).");

    TEUCHOS_TEST_FOR_EXCEPTION(this.constrains_field_name(), std::logic_error,
                               "mundy::meta::FieldRequirements: Field name must be set before calling declare_field.");
    TEUCHOS_TEST_FOR_EXCEPTION(this.constrains_field_rank(), std::logic_error,
                               "mundy::meta::FieldRequirements: Field rank must be set before calling declare_field.");
    TEUCHOS_TEST_FOR_EXCEPTION(
        this.constrains_field_dimension(), std::logic_error,
        "mundy::meta::FieldRequirements: Field dimension must be set before calling declare_field.");
    TEUCHOS_TEST_FOR_EXCEPTION(
        this.constrains_field_min_number_of_states(), std::logic_error,
        "mundy::meta::FieldRequirements: Field minimum number of states must be set before calling declare_field.");

    // Declare the field and assign it to the given part
    stk::mesh::Field<FieldType> &field =
        meta_data_ptr->declare_field<FieldType>(this.get_field_rank(), this.get_field_name());
    stk::mesh::put_field_on_mesh(field, part, nullptr);

    return field;
  }

  /// \brief Delete the field name constraint (if it exists).
  void delete_field_name_constraint() {
    field_name_is_set_ = false;
  }

  /// \brief Delete the field rank constraint (if it exists).
  void delete_field_rank_constraint() {
    field_rank_is_set_ = false;
  }

  /// \brief Delete the field dimension constraint (if it exists).
  void delete_field_dimension_constraint() {
    field_dimension_is_set_ = false;
  }

  /// \brief Delete the field minimum number of states constraint (if it exists).
  void delete_field_min_number_of_states_constraint() {
    field_min_number_of_states_is_set_ = false;
  }

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means that the rank, dimension, and number of states are > 0, but as unsigned ints, this is always the
  /// case. We will however, leave this checker incase the class grows and the set of requirements is no longer
  /// automatically satisfied.
  void check_if_valid() {
  }

  /// \brief Merge the current parameters with any number of other \c FieldRequirements.
  ///
  /// Here, merging two a \c FieldRequirements object with this object amounts to setting the number of states to be the
  /// maximum over all the number of states over all the \c FieldRequirements. For this process to be valid, the given
  /// \c FieldRequirements must have the same rank, type, and dimension. The name of the other fields does not need to
  /// match the current name of this field.
  ///
  /// \param list_of_field_reqs [in] A list of other \c FieldRequirements objects to merge with the current object.
  template <class... ArgTypes,
            typename std::enable_if<std::conjunction<std::is_convertible<Ts, FieldRequirements>...>::value>::type>
  void merge(const ArgTypes &...list_of_field_reqs) {
    for (const auto &field_reqs : list_of_field_reqs) {
      // Check if the provided parameters are valid.
      field_reqs.check_if_valid();

      // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
      if (field_reqs.constrains_field_name()) {
        if (this.constrains_field_name()) {
          TEUCHOS_TEST_FOR_EXCEPTION(this.get_field_name() == field_reqs.get_field_name(), std::invalid_argument,
                                     "mundy::meta::FieldRequirements: One of the inputs has incompatible name ("
                                         << field_reqs.get_field_name() << ").");
        } else {
          this.set_field_name(field_reqs.get_field_name());
        }
      }

      if (field_reqs.constrains_field_rank()) {
        if (this.constrains_field_rank()) {
          TEUCHOS_TEST_FOR_EXCEPTION(this.get_field_rank() == field_reqs.get_field_rank(), std::invalid_argument,
                                     "mundy::meta::FieldRequirements: One of the inputs has incompatible rank ("
                                         << field_reqs.get_field_rank() << ").");
        } else {
          this.set_field_rank(field_reqs.get_field_rank());
        }
      }

      if (field_reqs.constrains_field_dimension()) {
        if (this.constrains_field_dimension()) {
          TEUCHOS_TEST_FOR_EXCEPTION(this.get_field_dimension() == field_reqs.get_field_dimension(),
                                     std::invalid_argument,
                                     "mundy::meta::FieldRequirements: One of the inputs has incompatible dimension ("
                                         << field_reqs.get_field_dimension() << ").");
        } else {
          this.set_field_dimension(field_reqs.get_field_dimension());
        }
      }

      if (field_reqs.constrains_field_min_number_of_states()) {
        if (this.constrains_field_min_number_of_states()) {
          TEUCHOS_TEST_FOR_EXCEPTION(
              this.get_field_min_number_of_states() == field_reqs.get_field_min_number_of_states(),
              std::invalid_argument,
              "mundy::meta::FieldRequirements: One of the inputs has incompatible minimum number of states ("
                  << field_reqs.get_field_min_number_of_states() << ").");
        } else {
          this.set_field_min_number_of_states_if_larger(field_reqs.get_field_min_number_of_states());
        }
      }
    }
  }

  /// \brief Generate a new default constructed instance with the desired type.
  ///
  /// The current list of field type names and their corresponding types:
  ///  - FLOAT -> float
  ///  - DOUBLE -> double
  ///  - INT -> int
  ///  - INT64 -> int64_t
  ///  - UNSIGNED -> unsigned
  ///
  /// \param field_type_string [in] A string containing a valid field type.
  static std::shared_ptr<FieldRequirementsBase> create_new_instance(
      const std::string &field_type_string, const Teuchos::ParameterList &parameter_list) const {
    if (field_type_string == "FLOAT") {
      return std::make_shared<FieldRequirements<float>>(parameter_list);
    } else if (field_type_string == "DOUBLE") {
      return std::make_shared<FieldRequirements<double>>(parameter_list);
    } else if (field_type_string == "INT") {
      return std::make_shared<FieldRequirements<int>>(parameter_list);
    } else if (field_type_string == "INT64") {
      return std::make_shared<FieldRequirements<int64_t>>(parameter_list);
    } else if (field_type_string == "UNSIGNED") {
      return std::make_shared<FieldRequirements<unsigned>>(parameter_list);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                                 "The provided field type string " << field_type_string << " is not supported.");
    }
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
  unsigned field_min_min_number_of_states_;

  /// \brief If the name of the field is set or not.
  bool field_name_is_set_;

  /// \brief If the rank that the field will be assigned to is set or not.
  bool field_rank_is_set_;

  /// \brief If the dimension of the field. For example, a dimension of three would be a vector is set or not.
  bool field_dimension_is_set_;

  /// \brief If the minimum number of rotating states that this field will have is set or not.
  bool field_min_number_of_states_is_set_;
}

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_FIELDREQUIREMENTS_HPP_
