
#
# A) Define the subpackage
#
tribits_subpackage(Alens)

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
  include/mundy_alens/actions_crosslinkers.hpp
  include/mundy_alens/ComputeBrownianVelocity.hpp
  include/mundy_alens/compute_brownian_velocity/kernels/SpheresKernel.hpp
  include/mundy_alens/ComputeMobility.hpp
  include/mundy_alens/compute_mobility/techniques/LocalDragNonOrientableSpheres.hpp
  include/mundy_alens/compute_mobility/techniques/RPYSpheres.hpp
  include/mundy_alens/compute_mobility/techniques/RPYSpheresPerf.hpp
  include/mundy_alens/periphery/Gauss_Legendre_Nodes_and_Weights.hpp
  include/mundy_alens/periphery/Periphery.hpp
  )

SET(SOURCES 
  src/compute_brownian_velocity/kernels/SpheresKernel.cpp
  # src/compute_mobility/techniques/periphery_utils/cnpy.cpp
  # src/compute_mobility/techniques/periphery_utils/kernels.cpp
  # src/compute_mobility/techniques/periphery_utils/Periphery.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_alens
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES include/mundy_alens
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()
