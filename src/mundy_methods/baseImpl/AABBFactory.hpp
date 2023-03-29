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

#ifndef MUNDY_METHODS_AABBFACTORY_HPP_
#define MUNDY_METHODS_AABBFACTORY_HPP_

/// \file AABBFactory.hpp
/// \brief Declaration of the AABBFactory class

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

namespace methods {

/// \class AABBFactory
/// \brief Enumeration of all valid \c AABBManager classes.
///
/// This factory class knows how to initialize any of Mundy's \c AABBManager subclasses, given the
/// mundy::multibody type that the subclass acts upon.
class AABBFactory {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  AABBFactory();

  //@}
  //! \name Attributes
  //@{

  /// \brief Get the number of \c AABBManager classes this factory recognizes.
  static int get_number_of_subclasses() {
    return 9;
  }

  /// \brief List of multibody types this factory recognizes.
  const std::vector<mundy::multibody>& get_valid_multibody_types() {
    return subclass_identifiers_;
  }

  /// \brief Get the requirements that the desired subclass imposes upon each particle and/or constraint.
  ///
  /// \param given_type [in] \c AABBManager multibody type specialization, for
  ///   which is_valid_multibody_type(given_type) returns true.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartParams
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::unique_ptr<PartParams> get_part_requirements(const mundy::multibody& multibody_type,
                                                           const stk::util::ParameterList& parameter_list) {
    ThrowAssertMsg(is_valid_multibody_type(given_type), "The provided type " << given_type << " is not valid.");

    switch (multibody_type) {
      case mundy::multibody::SPHERE:
        return AABBSphereManager.get_part_requirements(parameter_list);
      case mundy::multibody::SPHEROCYLINDER:
        return AABBSphereocylinderManager.get_part_requirements(parameter_list);
      case mundy::multibody::SUPERELLIPSOID:
        return AABBSuperellipsoidManager.get_part_requirements(parameter_list);
      case mundy::multibody::POLYTOPE:
        return AABBPolytopeManager.get_part_requirements(parameter_list);
      case mundy::multibody::COLLISION:
        return AABBCollisionManager.get_part_requirements(parameter_list);
      case mundy::multibody::SPRING:
        return AABBSpringManager.get_part_requirements(parameter_list);
      case mundy::multibody::TORSIONALSPRING:
        return AABBTorsionalSpringManager.get_part_requirements(parameter_list);
      case mundy::multibody::JOINT:
        return AABBJointManager.get_part_requirements(parameter_list);
      case mundy::multibody::HINGE:
        return AABBHingeManager.get_part_requirements(parameter_list);
    }
  }

  //@}

  //! \name Actions
  //@{

  /// \brief Check whether this factory recognizes the given multibody type.
  bool is_valid_multibody_type(const mundy::multibody& given_type) const {
    return (std::find(subclass_identifiers_.begin(), subclass_identifiers_.end(), given_type) !=
            subclass_identifiers_.end());
  }

  /// \brief Return an instance of the specified \c AABBManager subclass.
  ///
  /// \param given_type [in] AABB type specialization of the \c AABBManager subclass instance to return.  The
  /// \c get_valid_multibody_types() method returns a list of the supported types.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up the specific \c AABBManager subclass. A
  /// default parameter list is available for each \c AABBManager subclass that this factory knows how to make.
  std::unique_ptr<mundy::methods::AABBManager> make_subclass(const mundy::multibody& given_type,
                                                                  const stk::util::ParameterList& parameter_list) {
    ThrowAssertMsg(is_valid_multibody_type(given_type), "The provided type " << given_type << " is not valid.");

    switch (given_type) {
      case mundy::multibody::SPHERE:
        return std::make_unique<AABBSphereManager>(parameter_list);
      case mundy::multibody::SPHEROCYLINDER:
        return std::make_unique<AABBSphereocylinderManager>(parameter_list);
      case mundy::multibody::SUPERELLIPSOID:
        return std::make_unique<AABBSuperellipsoidManager>(parameter_list);
      case mundy::multibody::POLYTOPE:
        return std::make_unique<AABBPolytopeManager>(parameter_list);
      case mundy::multibody::COLLISION:
        return std::make_unique<AABBCollisionManager>(parameter_list);
      case mundy::multibody::SPRING:
        return std::make_unique<AABBSpringManager>(parameter_list);
      case mundy::multibody::TORSIONALSPRING:
        return std::make_unique<AABBTorsionalSpringManager>(parameter_list);
      case mundy::multibody::JOINT:
        return std::make_unique<AABBJointManager>(parameter_list);
      case mundy::multibody::HINGE:
        return std::make_unique<AABBHingeManager>(parameter_list);
    }
  }

 private:
  /// \brief List of valid \c AABBManager subclasses, based on the multibody type they are specialized for.
  std::vector<mundy::multibody> subclass_identifiers_;
}

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_AABBFACTORY_HPP_
