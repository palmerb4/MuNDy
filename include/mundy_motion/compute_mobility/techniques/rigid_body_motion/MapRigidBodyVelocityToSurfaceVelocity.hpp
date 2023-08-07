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

#ifndef MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYVELOCITYTOSURFACEVELOCITY_HPP_
#define MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYVELOCITYTOSURFACEVELOCITY_HPP_

/// \file MapRigidBodyVelocityToSurfaceVelocity.hpp
/// \brief Declaration of the MapRigidBodyVelocityToSurfaceVelocity class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                                        // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>                            // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>                                    // for MUNDY_REGISTER_METACLASS
#include <mundy_motion/compute_mobility/techniques/RigidBodyMotion.hpp>  // for mundy::motion::...::RigidBodyMotion

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

/// \class MapRigidBodyVelocityToSurfaceVelocity
/// \brief Method for using rigid body motion about a known body point to compute the velocity at all surface points.
class MapRigidBodyVelocityToSurfaceVelocity
    : public mundy::meta::MetaKernelDispatcher<MapRigidBodyVelocityToSurfaceVelocity> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MapRigidBodyVelocityToSurfaceVelocity() = delete;

  /// \brief Constructor
  MapRigidBodyVelocityToSurfaceVelocity(mundy::mesh::BulkData *const bulk_data_ptr,
                                        const Teuchos::ParameterList &fixed_params);
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
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list) {
    return std::make_shared<MapRigidBodyVelocityToSurfaceVelocity>(bulk_data_ptr, parameter_list);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "MAP_RIGID_BODY_VELOCITY_TO_SURFACE_VELOCITY";
  //@}
};  // MapRigidBodyVelocityToSurfaceVelocity

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace motion

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register MapRigidBodyForceToRigidBodyVelocity with RigidBodyMotion's method factory.
MUNDY_REGISTER_METACLASS(
    mundy::motion::compute_mobility::techniques::rigid_body_motion::MapRigidBodyVelocityToSurfaceVelocity,
    mundy::motion::compute_mobility::techniques::RigidBodyMotion::OurMethodFactory)
//}

#endif  // MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPRIGIDBODYVELOCITYTOSURFACEVELOCITY_HPP_
