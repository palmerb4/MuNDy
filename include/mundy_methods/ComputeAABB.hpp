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

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/KernelDispatcher.hpp>  // for mundy::meta::KernelDispatcher
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for MUNDY_REGISTER_METACLASS

namespace mundy {

namespace methods {

/// \class ComputeAABB
/// \brief Method for computing the axis aligned boundary box of different parts.
///
/// The methodology behind the design choices in this class is as follows:
/// The \c ComputeAABB class is a \c MetaMethod that is responsible for computing the axis aligned bounding box of
/// different parts. Originally, this class was designed as a "MetaMultibodyMethod" that assigned a \c MetaKernel to
/// each enabled multibody part. However, this design was not flexible enough to handle the case where a multiple
/// \c MetaKernels needed to be assigned to the same multibody type but has the advantage of allowing for default
/// kernels. The alternative design is to allow users the freedom to directly specify the \c MetaKernels that they want
/// to use for each part. This design is more flexible but requires more work on the user's part since they must
/// specify the \c MetaKernels themselves.
class ComputeAABB : public mundy::meta::KernelDispatcher<mundy::meta::MetaKernel<void>,
                                                         mundy::meta::MetaKernelFactory<void, ComputeAABB>> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeAABB() = delete;

  /// \brief Constructor
  ComputeAABB(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

  /// \brief Get the unique registration identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaMethod.
  static RegistrationType get_registration_id() {
    return registration_id_;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethod<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<ComputeAABB>(bulk_data_ptr, fixed_params);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "COMPUTE_AABB";
  //@}
};  // ComputeAABB

}  // namespace methods

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register ComputeAABB with the global MetaMethodFactory.
MUNDY_REGISTER_METACLASS(mundy::methods::ComputeAABB, mundy::meta::GlobalMetaMethodFactory<void>)
//}

#endif  // MUNDY_METHODS_COMPUTEAABB_HPP_
