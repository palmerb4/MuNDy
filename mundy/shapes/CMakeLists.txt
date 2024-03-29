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
SET(NOINSTALLHEADERS 
  include/mundy_shapes/ComputeAABB.hpp
  include/mundy_shapes/compute_aabb/kernels/Sphere.hpp
  include/mundy_shapes/compute_aabb/kernels/Spherocylinder.hpp
  include/mundy_shapes/compute_aabb/kernels/SpherocylinderSegment.hpp
  include/mundy_shapes/ComputeBoundingRadius.hpp
  include/mundy_shapes/compute_bounding_radius/kernels/Sphere.hpp
  include/mundy_shapes/compute_bounding_radius/kernels/Spherocylinder.hpp
  include/mundy_shapes/compute_bounding_radius/kernels/SpherocylinderSegment.hpp
  include/mundy_shapes/ComputeOBB.hpp
  include/mundy_shapes/compute_obb/kernels/Sphere.hpp
  include/mundy_shapes/compute_obb/kernels/Spherocylinder.hpp
  include/mundy_shapes/compute_obb/kernels/SpherocylinderSegment.hpp
  include/mundy_shapes/declare_and_initialize_shapes/techniques/GridCoordinateMapping.hpp
  include/mundy_shapes/declare_and_initialize_shapes/techniques/GridOfSpheres.hpp
  include/mundy_shapes/DeclareAndInitShapes.hpp
  include/mundy_shapes/Shapes.hpp
  include/mundy_shapes/Spheres.hpp
  include/mundy_shapes/SpherocylinderSegments.hpp
  include/mundy_shapes/Spherocylinders.hpp
  )

SET(SOURCES 
  src/mundy_shapes/compute_aabb/kernels/Sphere.cpp
  src/mundy_shapes/compute_aabb/kernels/Spherocylinder.cpp
  src/mundy_shapes/compute_aabb/kernels/SpherocylinderSegment.cpp
  src/mundy_shapes/compute_bounding_radius/kernels/Sphere.cpp
  src/mundy_shapes/compute_bounding_radius/kernels/Spherocylinder.cpp
  src/mundy_shapes/compute_bounding_radius/kernels/SpherocylinderSegment.cpp
  src/mundy_shapes/compute_obb/kernels/Sphere.cpp
  src/mundy_shapes/compute_obb/kernels/Spherocylinder.cpp
  src/mundy_shapes/compute_obb/kernels/SpherocylinderSegment.cpp
  src/mundy_shapes/declare_and_initialize_shapes/techniques/GridOfSpheres.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_shapes
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES include/mundy_shapes
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()