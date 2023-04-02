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

#ifndef MUNDY_META_HIERARCHYPARAMS_HPP_
#define MUNDY_META_HIERARCHYPARAMS_HPP_

/// \file FactoryRegistry.hpp
/// \brief Declaration of the FactoryRegistry class

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

/// \class FactoryRegistry
/// \brief An abstract factory that allow users to register factory methods and associate them with arbitrary keys.
///
/// \tparam KeyType The key type for factory method lookup. Must have a == operator.
/// \tparam BaseMethodType The base class from which all factory methods are derived.
template <typename KeyType, typename BaseMethodType>
class FactoryRegistry {
 public:
  //! \name Typedefs
  //@{

  /// \brief A function type of all create routines
  using CreateMethodFunction = std::unique_ptr<BaseMethodType>(*)();
  //@}

  //! \name Attributes
  //@{

  /// \brief Get the number of \c MultibodyManager classes this factory recognizes.
  static size_t get_number_of_subclasses() {
    return sizeof...(Managers);
  }

  /// \brief List of multibody types this factory recognizes.
  const std::vector<mundy::multibody>& get_valid_multibody_types() {
    return subclass_identifiers_;
  }

  /// \brief Get the requirements that the desired subclass imposes upon each particle and/or constraint.
  ///
  /// \param given_type [in] \c MultibodyManager multibody type specialization, for
  ///   which is_valid_multibody_type(given_type) returns true.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartParams
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::unique_ptr<PartParams> get_part_requirements(const mundy::multibody& multibody_type,
                                                           const stk::util::ParameterList& parameter_list) {
    ThrowAssertMsg(is_valid_multibody_type(given_type), "The provided type " << given_type << " is not valid.");

    // this is a templated switch statement
    // the manager whose multibody type matches multibody_type is returned by multibody_switch
    multibody_switch<multibody_type, ... Managers>::get_part_requirements(parameter_list);
  }

  //@}

  //! \name Actions
  //@{

  /// \brief Check whether this factory recognizes the given multibody type.
  bool is_valid_multibody_type(const mundy::multibody& given_type) const {
    return (std::find(subclass_identifiers_.begin(), subclass_identifiers_.end(), given_type) !=
            subclass_identifiers_.end());
  }

  /// \brief Return an instance of the specified \c MultibodyManager subclass.
  ///
  /// \param given_type [in] Multibody type specialization of the \c MultibodyManager subclass instance to return.  The
  /// \c get_valid_multibody_types() method returns a list of the supported types.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up the specific \c MultibodyManager subclass. A
  /// default parameter list is available for each \c MultibodyManager subclass that this factory knows how to make.
  std::unique_ptr<mundy::methods::MultibodyManager> make_subclass(const mundy::multibody& given_type,
                                                                  const stk::util::ParameterList& parameter_list) {
    ThrowAssertMsg(is_valid_multibody_type(given_type), "The provided type " << given_type << " is not valid.");

    // this is a templated switch statement
    // the manager whose multibody type matches multibody_type is returned by multibody_switch
    multibody_switch<multibody_type, ... Managers>(parameter_list);
  }

 private:
  /// \brief List of valid \c MultibodyManager subclasses, based on the multibody type they are specialized for.
  std::vector<mundy::multibody> subclass_identifiers_;
  std::map<KeyType, TCreateMethod> s_methods
};  // FactoryRegistry

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_HIERARCHYPARAMS_HPP_
