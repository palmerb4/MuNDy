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

#ifndef MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCALDRAG_HPP_
#define MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCALDRAG_HPP_

/// \file LocalDrag.hpp
/// \brief Declaration of the LocalDrag class

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <string>   // for std::string
#include <utility>  // for std::pair
#include <vector>   // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>               // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>              // for mundy::mesh::MetaData
#include <mundy_meta/MetaFactory.hpp>           // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>            // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethod.hpp>            // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>          // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>      // for mundy::meta::PartRequirements
#include <mundy_methods/ComputeMobility.hpp>    // for mundy::methods::ComputeMobility

namespace mundy {

namespace methods {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

namespace map_rigid_body_force_to_rigid_body_velocity {

namespace techniques {

/// \class LocalDrag
/// \brief Method for computing the rigid body force to rigid body velocity using the local drag of different parts.
class LocalDrag : public mundy::meta::MetaKernelDispatcher<LocalDrag> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  LocalDrag() = delete;

  /// \brief Constructor
  LocalDrag(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    return std::make_shared<LocalDrag>(bulk_data_ptr, fixed_params);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "LOCAL_DRAG";
  //@}
};  // LocalDrag

}  // namespace techniques

}  // namespace map_rigid_body_force_to_rigid_body_velocity

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace methods

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register LocalDrag with the ComputeMobility's method factory.
MUNDY_REGISTER_METACLASS(mundy::methods::compute_mobility::techniques::rigid_body_motion::
                             map_rigid_body_force_to_rigid_body_velocity::techniques::LocalDrag,
                         mundy::methods::ComputeMobility::OurMethodFactory)
//}

#endif  // MUNDY_METHODS_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCALDRAG_HPP_
