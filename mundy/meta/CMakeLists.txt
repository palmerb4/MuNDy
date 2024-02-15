#
# A) Define the subpackage
#
tribits_subpackage(Meta)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use 
# find include/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort

SET(HEADERS 
  include/mundy_meta/PartRequirements.hpp
  include/mundy_meta/HasMeshRequirementsAndIsRegisterable.hpp
  include/mundy_meta/MetaKernelDispatcher.hpp
  include/mundy_meta/MetaRegistry.hpp
  include/mundy_meta/FieldRequirementsFactory.hpp
  include/mundy_meta/MetaTechniqueDispatcher.hpp
  include/mundy_meta/FieldRequirementsRegistry.hpp
  include/mundy_meta/MetaFactory.hpp
  include/mundy_meta/utils/MeshGeneration.hpp
  include/mundy_meta/MetaMethod.hpp
  include/mundy_meta/MetaKernel.hpp
  include/mundy_meta/FieldRequirementsBase.hpp
  include/mundy_meta/FieldRequirements.hpp
  include/mundy_meta/MeshRequirements.hpp
  )

SET(SOURCES 
  src/mundy_meta/FieldRequirements.cpp
  src/mundy_meta/PartRequirements.cpp
  src/mundy_meta/MeshRequirements.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
tribits_add_library(
  mundy_meta
  HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()