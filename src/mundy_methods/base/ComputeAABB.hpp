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

#ifndef MUNDY_METHODS_COMPUTEAABB_HPP_
#define MUNDY_METHODS_COMPUTEAABB_HPP_

/// \file ComputeAABB.hpp
/// \brief Declaration of the ComputeAABB class

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

/// \class ComputeAABB
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeAABB : public MetaMethod<ComputeAABB>, public MetaMethodRegistry<ComputeAABB> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeAABB() = delete;

  /// \brief Constructor
  ComputeAABB(const stk::mesh::BulkData *bulk_data_ptr, const Teuchos::RCP<Teuchos::ParameterList> &parameter_list)
      : bulk_data_ptr_(bulk_data_ptr), enabled_multibody_names_(enabled_multibody_names) {
    // TODO(palmerb4): ideally we store the fields instead of having to look them up every time we want to call execute
    // The problem I have with storing the fields is that we would need a copy of this class for each multibody
    // type. We may have a different parameter list per multibody type. We may also have the same parameter list per
    // multibody type. We simply allow the user to pass in all the parameters for all varients in this constructor!!!
    // The execute command will then take in the multibody name and use the correct params.

    // This means that compute AABB should store instances for each multibody name rather than calling
    // create_new_instance within execute. This also means that the constructor for multibody method needs to be (const
    // stk::mesh::BulkData *bulk_data_ptr, const Teuchos::RCP<Teuchos::ParameterList> &parameter_list)

    TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                               "mundy::methods::ComputeAABB: bulk_data_ptr cannot be a nullptr.");

    // Create the internal parameter list
    // Each variant has a different set of valid parameters. We could, in theory, collect all of these requirements, and
    // output it when get_part_requirements is called but doing so could lead to outputting more requirements than
    // necessary. We need to better tie together how the methods are initialized and how the methods should be
    // constructed. Currently, we access requirements based on multibody name but only want to initialize one
    // ComputeAABB. As well, in the current parameter list paradigm, method requirements are hidden within each part.

    // Some methods, like ComputeAABB, take in one part and act on it in isolation. Technically, these methods could
    // take in a vector of parts and loop over each of them. Methods like ResolveConstraints, on the other hand, are
    // unique in that they need to take in a vector of parts and have those parts all interact in unison. Sure, lets
    // just change the interface to be a map of named parts.

    // If we consider that Methods assigned to Parts like Fields, then a Method can take in a vector of subparts... No
    // it cant. If this thought were correct, then how would ResolveConstraints know how to specialize for each
    // multibody type? I think this shows the flaw, we are trying to consider Methods like Fields, but unlike fields, a
    // method assigned to a parent, may need to specialize for each of its children. The response may seem to be
    // assigning the specializations to each subpart, but what about the parent method that needs to control their
    // collective interaction?
    // Well, how does ComputeAABB for a sphere differ from ComputeAABB for a Spring? They have different kernels. In the
    // PointForceFMM technique of ComputeMobility, each part passed into ComputeMobility will differ based on the kernel
    // they use for computing their imposed force.

    // The multibody class stores named kernels. We provide some prebuilt kernels.

    // Every single part that we are given should also provide its own kernel. But what if the method requires many

    // (Cannot wrap parts in their methods because it isolates methods from their children) Ok, why not wrap a part in
    // its methods? Allow multiple ComputeAABBs to be created, one for each part that requests that method. That creates
    // an issue because how would subparts access the methods generated by their parents!

    // store the set of enabled_multibody_names as a parameter!
    parameter_list_ = Teuchos::rcp(new Teuchos::ParameterList(*getValidParameters()));

    // TODO: access the vector of multibody varient params
    const size_t num_valid_multibodies = len(parameter_list[something]);
    variant_map_.emplace_back(bulk_data_ptr, parameter_list)
  }
  //@}

  execute(const stk::mesh::Part &part, const std::string &multibody_name) {
    // create and run a ComputeAABB variant corresponding to the provided multibody type name
    MetaMethodFactory<ComputeAABB>::create_new_instance(multibody_name, parameter_list).execute(bulk_data_ptr, part);
  }

  static std::unique_ptr<PartParams> get_part_requirements(const std::string &multibody_name,
                                                           const Teuchos::RCP<Teuchos::ParameterList> &parameter_list) {
    return MetaMethodFactory<ComputeAABB>::get_part_requirements(multibody_name, parameter_list);
  }

 private:
  Teuchos::RCP<Teuchos::ParameterList> parameter_list_;
  std::map<std::string, std::unique_ptr<MultibodyMethod>> variant_map_;
  std::vector<std::string> enabled_multibody_names_;
}

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTEAABB_HPP_
