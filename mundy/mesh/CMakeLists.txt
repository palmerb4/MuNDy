#
# A) Define the subpackage
#
tribits_subpackage(Mesh)

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
  include/mundy_mesh/BulkData.hpp
  include/mundy_mesh/MeshBuilder.hpp
  include/mundy_mesh/MetaData.hpp
  include/mundy_mesh/StringToSelector.hpp
  include/mundy_mesh/StringToTopology.hpp
  include/mundy_mesh/utils/DestroyFlaggedEntities.hpp
  include/mundy_mesh/utils/FillFieldWithValue.hpp
  include/mundy_mesh/utils/SelectorEval.hpp
  include/mundy_mesh/utils/SelectorLexem.hpp
  include/mundy_mesh/utils/SelectorNode.hpp
  include/mundy_mesh/utils/SelectorParser.hpp
  )

SET(SOURCES 
  src/mundy_mesh/MeshBuilder.cpp
  src/mundy_mesh/MetaData.cpp
  src/mundy_mesh/StringToSelector.cpp
  src/mundy_mesh/StringToTopology.cpp
  src/mundy_mesh/utils/DestroyFlaggedEntities.cpp
  src/mundy_mesh/utils/SelectorEval.cpp
  src/mundy_mesh/utils/SelectorLexem.cpp
  src/mundy_mesh/utils/SelectorNode.cpp
  src/mundy_mesh/utils/SelectorParser.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_mesh
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES include/mundy_mesh
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()