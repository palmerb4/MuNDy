#
# A) Define the subpackage
#
tribits_subpackage(Meta)

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
  include/mundy_meta/FieldReqsBase.hpp
  include/mundy_meta/FieldReqsFactory.hpp
  include/mundy_meta/FieldReqs.hpp
  include/mundy_meta/FieldReqsRegistry.hpp
  include/mundy_meta/HasMeshReqsAndIsRegisterable.hpp
  include/mundy_meta/MeshReqs.hpp
  include/mundy_meta/MetaFactory.hpp
  include/mundy_meta/MetaKernelDispatcher.hpp
  include/mundy_meta/MetaKernel.hpp
  include/mundy_meta/MetaMethodExecutionInterface.hpp
  include/mundy_meta/MetaMethodPairwiseSubsetExecutionInterface.hpp
  include/mundy_meta/MetaMethodSubsetExecutionInterface.hpp
  include/mundy_meta/MetaRegistry.hpp
  include/mundy_meta/MetaTechniqueDispatcher.hpp
  include/mundy_meta/ParameterValidationHelpers.hpp
  include/mundy_meta/PartReqs.hpp
  include/mundy_meta/utils/MeshGeneration.hpp
  )

SET(SOURCES 
  src/mundy_meta/PartReqs.cpp
  src/mundy_meta/MeshReqs.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_meta
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES include/mundy_meta
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()