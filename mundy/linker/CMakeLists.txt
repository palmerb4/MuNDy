#
# A) Define the subpackage
#
tribits_subpackage(Agent)

#
# B) Define the header and source files 
#
SET(HEADERS "")
SET(SOURCES "")

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
SET(HEADERS ${HEADERS}
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  )

APPEND_GLOB(HEADERS ${CMAKE_CURRENT_BINARY_DIR}/include/mundy_linker/*.hpp)
APPEND_GLOB(SOURCES ${CMAKE_CURRENT_BINARY_DIR}/src/mundy_linker/*.cpp)

#
# C) Add the libraries, tests, and examples
#
tribits_add_library(
  mundy_linker
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()