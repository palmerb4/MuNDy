#
# A) Define the subpackage
#
tribits_subpackage(Shape)

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
  include/mundy_shape/ComputeAABB.hpp
  include/mundy_shape/compute_aabb/kernels/Sphere.hpp
  include/mundy_shape/compute_aabb/kernels/Spherocylinder.hpp
  include/mundy_shape/ComputeBoundingRadius.hpp
  include/mundy_shape/compute_bounding_radius/kernels/Sphere.hpp
  include/mundy_shape/ComputeOBB.hpp
  include/mundy_shape/compute_obb/kernels/Sphere.hpp
  include/mundy_shape/PerformRegistration.hpp
  include/mundy_shape/Shapes.hpp
  include/mundy_shape/shapes/Spheres.hpp
  include/mundy_shape/shapes/Spherocylinders.hpp
  )

SET(SOURCES 
  src/mundy_shape/compute_aabb/kernels/Sphere.cpp
  src/mundy_shape/compute_aabb/kernels/Spherocylinder.cpp
  src/mundy_shape/compute_bounding_radius/kernels/Sphere.cpp
  src/mundy_shape/compute_obb/kernels/Sphere.cpp
  src/mundy_shape/PerformRegistration.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
tribits_add_library(
  mundy_shape
  HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()