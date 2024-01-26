#
# A) Define the subpackage
#
tribits_subpackage(Motion)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use 
# find include/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort

SET(HEADERS 
  include/mundy_motion/ResolveConstraints.hpp
  include/mundy_motion/compute_time_integration/techniques/NodeEuler.hpp
  include/mundy_motion/compute_mobility/techniques/rigid_body_motion/MapSurfaceForceToRigidBodyForce.hpp
  include/mundy_motion/compute_mobility/techniques/rigid_body_motion/map_rigid_body_velocity_to_surface_velocity/kernels/Sphere.hpp
  include/mundy_motion/compute_mobility/techniques/rigid_body_motion/map_surface_force_to_rigid_body_force/kernels/Sphere.hpp
  include/mundy_motion/compute_mobility/techniques/rigid_body_motion/map_rigid_body_force_to_rigid_body_velocity/techniques/LocalDrag.hpp
  include/mundy_motion/compute_mobility/techniques/rigid_body_motion/map_rigid_body_force_to_rigid_body_velocity/techniques/local_drag/kernels/Sphere.hpp
  include/mundy_motion/compute_mobility/techniques/rigid_body_motion/MapRigidBodyVelocityToSurfaceVelocity.hpp
  include/mundy_motion/compute_mobility/techniques/rigid_body_motion/MapRigidBodyForceToRigidBodyVelocity.hpp
  include/mundy_motion/compute_mobility/techniques/RigidBodyMotion.hpp
  include/mundy_motion/utils/Quaternion.hpp
  # include/mundy_motion/resolve_constraints/techniques/PairwisePotential.hpp
  # include/mundy_motion/resolve_constraints/techniques/NonSmoothLCP.hpp
  include/mundy_motion/ComputeMobility.hpp
  include/mundy_motion/ComputeTimeIntegration.hpp
  )

SET(SOURCES 
  src/mundy_motion/ResolveConstraints.cpp
  src/mundy_motion/ComputeMobility.cpp
  src/mundy_motion/compute_time_integration/techniques/NodeEuler.cpp
  src/mundy_motion/compute_mobility/techniques/rigid_body_motion/MapRigidBodyVelocityToSurfaceVelocity.cpp
  src/mundy_motion/compute_mobility/techniques/rigid_body_motion/map_rigid_body_velocity_to_surface_velocity/kernels/Sphere.cpp
  src/mundy_motion/compute_mobility/techniques/rigid_body_motion/MapRigidBodyForceToRigidBodyVelocity.cpp
  src/mundy_motion/compute_mobility/techniques/rigid_body_motion/map_surface_force_to_rigid_body_force/kernels/Sphere.cpp
  src/mundy_motion/compute_mobility/techniques/rigid_body_motion/map_rigid_body_force_to_rigid_body_velocity/techniques/LocalDrag.cpp
  src/mundy_motion/compute_mobility/techniques/rigid_body_motion/map_rigid_body_force_to_rigid_body_velocity/techniques/local_drag/kernels/Sphere.cpp
  src/mundy_motion/compute_mobility/techniques/rigid_body_motion/MapSurfaceForceToRigidBodyForce.cpp
  src/mundy_motion/compute_mobility/techniques/RigidBodyMotion.cpp
  src/mundy_motion/ComputeTimeIntegration.cpp
  # src/mundy_motion/resolve_constraints/techniques/PairwisePotential.cpp
  # src/mundy_motion/resolve_constraints/techniques/NonSmoothLCP.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
tribits_add_library(
  mundy_motion
  HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()