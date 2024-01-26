#
# A) Define the subpackage
#
tribits_subpackage(Mesh)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use 
# find include/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort

SET(HEADERS 
  include/mundy_mesh/MetaData.hpp
  include/mundy_mesh/MeshBuilder.hpp
  include/mundy_mesh/BulkData.hpp
  )

SET(SOURCES 
  src/mundy_mesh/MeshBuilder.cpp
  src/mundy_mesh/MetaData.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
tribits_add_library(
  mundy_mesh
  HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  SOURCES ${SOURCES}
  HEADERS_INSTALL_SUBDIR mundy_mesh
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()