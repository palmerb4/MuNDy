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

#ifndef MUNDY_META_METAKERNEL_HPP_
#define MUNDY_META_METAKERNEL_HPP_

/// \file MetaKernel.hpp
/// \brief Declaration of the MetaKernel class

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

/// \class MetaKernel
/// \brief An abstract interface for all of Mundy's methods.
///
/// A \c MetaKernel represents an atomic unit of computation applied to a \b single multibody object in isolation
/// (sphere, spring, hinge, etc.). Through STK and Kokkos looping constructs, a \c MetaKernel can be efficiently applied
/// to large groups of multibody objects.
///
/// While \c MetaKernel only accepts a single multibody object, but Mundy offers two pairwise specializations:
///  - \c MetaKernelPairwise for pairwise anti-symmetric interaction between two multibody objects.
///  - \c MetaKernelPairwiseSymmetric for pairwise symmetric interaction between two multibody objects.
///
/// The goal of \c MetaKernel is to wrap a kernel that acts on an STK Element with a known multibody type.
/// The wrapper can output the assumptions of the wrapped kernel with respect to the fields and topology associated with
/// the provided element. Note, this element is part of some STK Part, so the output requirements are \c PartParams for
/// that part. Requirements cannot be applied at the element-level.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c MetaKernel must
/// implement the following static member functions
///   - \c details_get_part_requirements implementation of the \c get_part_requirements interface.
///   class.
///   - \c details_get_valid_params implementation of the \c get_valid_params interface.
///   - \c details_get_class_identifier implementation of the \c get_class_identifier interface.
///   - \c details_create_new_instance implementation of the \c create_new_instance interface.
///
/// \tparam DerivedMetaKernel A class derived from \c MetaKernel that implements the desired interface.
template <class DerivedMetaKernel,
          typename std::enable_if<std::is_base_of<MetaKernel, DerivedMetaKernel>::value, void>::type>
class MetaKernel : public Teuchos::Describable {
 public:
  //! \name Attributes
  //@{

  /// \brief Get the requirements that this \c MetaKernel imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this \c MetaKernel
  /// with respect to the parts, topology, and fields input into the \c run function. These assumptions may vary
  /// based parameters in the \c parameter_list.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartParams
  /// will be created. You can save the result yourself if you wish to reuse it.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::unique_ptr<PartParams> get_part_requirements(const Teuchos::ParameterList& parameter_list) const {
    return DerivedMetaKernel::details_get_part_requirements(parameter_list);
  }

  /// \brief Get the valid parameters and their default parameter list for this \c MetaKernel.
  static Teuchos::ParameterList get_valid_params() const {
    return DerivedMetaKernel::details_get_valid_params();
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaKernel.
  static std::string get_class_identifier() const {
    return DerivedMetaKernel::details_get_class_identifier();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::unique_ptr<MetaKernelFactory> create_new_instance(const Teuchos::ParameterList& parameter_list) const {
    return DerivedMetaKernel::details_create_new_instance(parameter_list);
  }

  /// \brief Run the kernel's core calculation.
  virtual void execute(const stk::mesh::Entity multibody_entity) = 0;
  //@}

  //! @name Implementation of Teuchos::Describable interface
  //@{

  //! A string description of this object.
  virtual std::string description() const;

  /// \brief Describe this object.
  ///
  /// At higher verbosity levels, this method will print out the list
  /// of names of supported solvers.  You can also get this list
  /// directly by using the supportedSolverNames() method.
  virtual void describe(Teuchos::FancyOStream& out,
                        const Teuchos::EVerbosityLevel verbLevel = Teuchos::Describable::verbLevel_default) const;
  //@}
};  // MetaKernel

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_METAKERNEL_HPP_
