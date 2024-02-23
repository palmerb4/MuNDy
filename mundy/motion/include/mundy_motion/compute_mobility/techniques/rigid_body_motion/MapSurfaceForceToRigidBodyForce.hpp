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

#ifndef MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPSURFACEFORCETORIGIDBODYFORCE_HPP_
#define MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPSURFACEFORCETORIGIDBODYFORCE_HPP_

/// \file MapSurfaceForceToRigidBodyForce.hpp
/// \brief Declaration of the MapSurfaceForceToRigidBodyForce class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                                       // for mundy::mesh::BulkData
#include <mundy_meta/MetaFactory.hpp>                                    // for mundy::meta::GlobalMetaMethodFactory
#include <mundy_meta/MetaKernelDispatcher.hpp>                           // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>                                   // for MUNDY_REGISTER_METACLASS
#include <mundy_motion/compute_mobility/techniques/rigid_body_motion/map_surface_force_to_rigid_body_force/kernels/Sphere.hpp>  // for mundy::motion::...::kernels::Sphere

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

/// \class MapSurfaceForceToRigidBodyForce
/// \brief Method for mapping the surface forces on a rigid body to get the total force and torque at a known location.
class MapSurfaceForceToRigidBodyForce : public mundy::meta::MetaKernelDispatcher<MapSurfaceForceToRigidBodyForce> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MapSurfaceForceToRigidBodyForce() = delete;

  /// \brief Constructor
  MapSurfaceForceToRigidBodyForce(mundy::mesh::BulkData *const bulk_data_ptr,
                                  const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

  /// \brief Get the unique registration identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaMethodSubsetExecutionInterface.
  static RegistrationType get_registration_id() {
    return registration_id_;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list) {
    return std::make_shared<MapSurfaceForceToRigidBodyForce>(bulk_data_ptr, parameter_list);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "MAP_SURFACE_FORCE_TO_RIGID_BODY_FORCE";
  //@}
};  // MapSurfaceForceToRigidBodyForce

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace motion

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register our default kernels
MUNDY_REGISTER_METACLASS(
    mundy::motion::compute_mobility::techniques::rigid_body_motion::map_surface_force_to_rigid_body_force::kernels::
        Sphere,
    mundy::motion::compute_mobility::techniques::rigid_body_motion::MapSurfaceForceToRigidBodyForce::OurKernelFactory)
//}

#endif  // MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAPSURFACEFORCETORIGIDBODYFORCE_HPP_
