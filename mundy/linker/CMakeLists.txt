#
# A) Define the subpackage
#
tribits_subpackage(Agent)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use 
# find include/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort

SET(HEADERS
  include/mundy_linker/Linkers.hpp
  include/mundy_linker/linkers/NeighborLinkers.hpp
  include/mundy_linker/linkers/neighbor_linkers/SphereSpheres.hpp
  include/mundy_linker/PerformRegistration.hpp
  )

SET(SOURCES 
  src/mundy_linker/PerformRegistration.cpp
)

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
tribits_add_library(
  mundy_linker
  HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  SOURCES ${SOURCES}
  )


tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()