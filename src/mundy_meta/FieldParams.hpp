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

#ifndef MUNDY_META_FIELDPARAMS_HPP_
#define MUNDY_META_FIELDPARAMS_HPP_

/// \file FieldParams.hpp
/// \brief Declaration of the FieldParams class

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

/// \class FieldParamsBase
/// \brief A consistant base for all \c FieldParams.
class FieldParamsBase {};  // FieldParamsBase

/// \class FieldParams
/// \brief A set of necessary parameters for declaring a new field.
///
/// \tparam FieldType Type for elements in the field.
template <typename FieldType>
class FieldParams {
 public:
  //! \name Typedefs
  //@{

  /// \tparam field_type Type for elements in the field. Set by the template parameter.
  typedef FieldType field_type;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is not allowed
  FieldParams() = delete;

  /// \brief Constructor with full fill.
  ///
  /// \param field_name [in] Name of the field.
  ///
  /// \param field_rank [in] Rank that the field will be assigned to.
  ///
  /// \param field_dimension [in] Dimension of the field. For example, a dimension of three would be a vector.
  ///
  /// \param field_min_number_of_states [in] Number of rotating states that this field will have.
  FieldParams(const std::string &field_name, const unsigned field_rank, const unsigned field_dimension,
              const unsigned field_min_number_of_states)
      : field_name_(field_name),
        field_rank_(field_rank),
        field_dimension_(field_dimension),
        field_min_number_of_states_(field_min_number_of_states) {
    check_if_valid();
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Return the field name.
  std::string get_field_name() {
    return field_name_;
  }

  /// \brief Return the field rank.
  stk::topology::rank_t get_field_rank() {
    return field_rank_;
  }

  /// \brief Return the field dimension.
  unsigned get_field_dimension() {
    return field_dimension_;
  }

  /// \brief Return the field number of states.
  unsigned get_field_min_number_of_states() {
    return field_min_number_of_states_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means that the rank, dimension, and number of states are > 0, but as unsigned ints, this is always the
  /// case. We will however, leave this checker incase the class grows and the set of requirements is no longer
  /// automatically satisfied.
  void check_if_valid() {
  }

  /// \brief Merge the current parameters with any number of other \c FieldParams.
  ///
  /// Here, merging two a \c FieldParams object with this object amounts to setting the number of states to be the
  /// maximum over all the number of states over all the \c FieldParams. For this process to be valid, the given
  /// \c FieldParams must have the same rank, type, and dimension. The name of the other fields does not need to match
  /// the current name of this field.
  ///
  /// \param list_of_field_params [in] A list of other \c FieldParams objects to merge with the current object.
  template <class... ArgTypes,
            typename std::enable_if<std::conjunction<std::is_convertible<Ts, FieldParams>...>::value>::type>
  void merge(const ArgTypes &...list_of_field_params) {
    for (const auto &field_params : list_of_field_params) {
      // Check if the provided parameters are valid.
      field_params.check_if_valid();

      // Check if the provided rank, type, and dimension are the same.
      TEUCHOS_TEST_FOR_EXCEPTION(
          get_field_rank() == field_params.get_field_rank(), std::invalid_argument,
          "mundy::meta::FieldParams: Field " << field_params.get_field_name() << " has incompatible rank.");
      TEUCHOS_TEST_FOR_EXCEPTION(
          std::is_same<this ::field_type, field_params::field_type>, std::invalid_argument,
          "mundy::meta::FieldParams: Field " << field_params.get_field_name() << " has incompatible type.");
      TEUCHOS_TEST_FOR_EXCEPTION(
          get_field_dimension() == field_params.get_field_dimension(), std::invalid_argument,
          "mundy::meta::FieldParams: Field " << field_params.get_field_name() << " has incompatible dimension.");

      field_min_number_of_states_ =
          std::max(field_min_number_of_states_, field_params.get_field_min_number_of_states());
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
}

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_FIELDPARAMS_HPP_
