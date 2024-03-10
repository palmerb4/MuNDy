#
# A) Define the subpackage
#
tribits_subpackage(Constraint)

# By default we enable all optional packages. They can be disabled by the user if they are not needed.
GLOBAL_SET(${PACKAGE_NAME}_ENABLE_ALL_OPTIONAL_PACKAGES ON)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use 
# find include/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort

SET(HEADERS 
  # include/mundy_constraint/DeleteCollisionConstraints.hpp
  include/mundy_constraint/ComputeConstraintForcing.hpp
  # include/mundy_constraint/generate_collision_constraints/kernels/CollisionSphereSphere.hpp
  include/mundy_constraint/compute_constraint_projection/kernels/Collision.hpp
  include/mundy_constraint/ComputeConstraintProjection.hpp
  include/mundy_constraint/compute_constraint_violation/kernels/Collision.hpp
  include/mundy_constraint/ComputeConstraintViolation.hpp
  include/mundy_constraint/compute_constraint_forcing/kernels/Collision.hpp
  include/mundy_constraint/ComputeConstraintResidual.hpp
  #include/mundy_constraint/GenerateCollisionConstraints.hpp
  )

SET(SOURCES 
  # src/mundy_constraint/GenerateCollisionConstraints.cpp
  # src/mundy_constraint/generate_collision_constraints/kernels/CollisionSphereSphere.cpp
  src/mundy_constraint/ComputeConstraintViolation.cpp
  src/mundy_constraint/ComputeConstraintResidual.cpp
  src/mundy_constraint/compute_constraint_projection/kernels/Collision.cpp
  src/mundy_constraint/compute_constraint_violation/kernels/Collision.cpp
  src/mundy_constraint/ComputeConstraintProjection.cpp
  src/mundy_constraint/compute_constraint_forcing/kernels/Collision.cpp
  src/mundy_constraint/ComputeConstraintForcing.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
tribits_add_library(
  mundy_constraint
  HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()