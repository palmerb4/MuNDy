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

#ifndef MUNDY_META_METAMETHOD_HPP_
#define MUNDY_META_METAMETHOD_HPP_

/// \file MetaMethodFactory.hpp
/// \brief Declaration of the MetaMethodFactory class

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

/// \class MetaMethodFactory
/// \brief A factory for generating a user-specified set of Mundy's \c MetaMethods.
///
/// The goal of \c MetaMethodFactory, as with most factories, is to provide an abstraction for case switches between
/// different methods. The switch type is defined by the \c KeyType template parameter and can be any type with a
/// comparison == operator and ostream operator <<.
///
/// \note This factory does not store an instance of \c MetaMethod; rather, it stores a map from \c KeyType to \c
/// MetaMethod's \c create_new_instance member function.
///
/// \tparam KeyType The key type for factory method lookup. Must have a == operator.
///
/// \tparam BaseMethodType The base class from which all factory methods are derived.
template <typename KeyType, typename MetaMethod>
class MetaMethodFactory {
 public:
  //! \name Typedefs
  //@{

  /// \brief A function pointer that takes a parameter list and produces a shared pointer to an object derived from
  /// \c MetaMethod.
  using NewMetaMethodGenerator = std::unique_ptr<BaseMethodType> (*)(stk::util::ParameterList&);

  /// \brief A function pointer that takes a parameter list and produces a PartParams instance.
  using NewRequirementsGenerator = PartParams (*)(stk::util::ParameterList&);

  /// \brief A function pointer that produces a stk::util::ParameterList instance.
  using NewDefaultParamsGenerator = stk::util::ParameterList (*)();
  //@}

  //! \name Attributes
  //@{

  /// \brief Get the number of \c MetaMethod classes this factory recognizes.
  size_t get_number_of_subclasses() {
    return generator_map_.size();
  }

  /// \brief Get the requirements that this a registered \c MetaMethod imposes upon each particle and/or constraint.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made a registered \c
  /// MetaMethod with respect to the parts, topology, and fields input into the \c run function. These assumptions may
  /// vary based parameters in the \c parameter_list.
  ///
  /// The registered \c MetaMethod accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a \c MetaMethod with this factory, use the
  /// provided \c register_new_method or \c register_new_methods function.
  ///
  /// \note This function does not cache its return value, so
  /// each time you call this function, a new \c PartParams will be created. You can save the result yourself if you
  /// wish to reuse it.
  ///
  /// \param key [in] A key corresponding to a registered \c MetaMethod.
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A default parameter list
  /// is accessible via \c get_default_params.
  std::unique_ptr<PartParams> get_part_requirements(const KeyType& key,
                                                    const stk::util::ParameterList& parameter_list) {
    return requirement_generator_map_[key](parameter_list);
  }

  /// \brief Get the default parameter list for a registered \c MetaMethod.
  ///
  /// The registered \c MetaMethod accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a \c MetaMethod with this factory, use the
  /// provided \c register_new_method or \c register_new_methods function.
  ///
  /// \note This function does not cache its return value, so
  /// each time you call this function, a new \c stk::util::ParameterList will be created. You can save the result
  /// yourself if you wish to reuse it.
  ///
  /// \param key [in] A key corresponding to a registered \c MetaMethod.
  stk::util::ParameterList get_default_params(const KeyType& key) {
    ThrowAssertMsg(is_valid_key(key), "The provided key " << key << " is not valid.");
    return default_params_generator_map_[key]();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  template <MetaMethod MethodToRegister>
  std::unique_ptr<MetaMethodFactory> register_new_method(const KeyType& key) {
    instance_generator_map_.insert(std::make_pair(key, MethodToRegister::create_new_instance));
    requirement_generator_map_.insert(std::make_pair(key, MethodToRegister::get_part_requirements));
    default_requirement_generator_map_.insert(std::make_pair(key, MethodToRegister::get_default_params));
  }

  /// \brief Generate a new instance of this class.
  virtual std::unique_ptr<MetaMethodFactory> create_new_instance(const stk::util::ParameterList& parameter_list) = 0;
  //@}
 private:
  std::map<KeyType, NewMetaMethodGenerator> instance_generator_map_;
  std::map<KeyType, NewRequirementsGenerator> requirement_generator_map_;
  std::map<KeyType, NewDefaultParamsGenerator> default_params_generator_map_;
};  // MetaMethodFactory

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_METAMETHOD_HPP_
