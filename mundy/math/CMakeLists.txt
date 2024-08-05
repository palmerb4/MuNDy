#
# A) Define the subpackage
#
tribits_subpackage(Math)

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
  include/mundy_math/Accessor.hpp
  include/mundy_math/Array.hpp
  include/mundy_math/distance/EllipsoidEllipsoid.hpp
  include/mundy_math/distance/SegmentSegment.hpp
  include/mundy_math/MaskedView.hpp
  include/mundy_math/Matrix3.hpp
  include/mundy_math/Matrix.hpp
  include/mundy_math/minimize.hpp
  include/mundy_math/minimize_kokkos.hpp
  include/mundy_math/minimize_scalar_vector_make.hpp
  include/mundy_math/Quaternion.hpp
  include/mundy_math/ShiftedView.hpp
  include/mundy_math/StridedView.hpp
  include/mundy_math/Tolerance.hpp
  include/mundy_math/TransposedView.hpp
  include/mundy_math/Vector3.hpp
  include/mundy_math/Vector.hpp
  )

SET(SOURCES 
  src/mundy_math/a_necessary_empty_file.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_math
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES include/mundy_math
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()