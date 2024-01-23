#
# A) Define the subpackage
#
tribits_subpackage(Meta)

#
# B) Define the header and source files 
#
SET(HEADERS "")
SET(SOURCES "")

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

APPEND_GLOB(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/mundy_meta/*.hpp)
APPEND_GLOB(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/mundy_meta/*.cpp)

#
# C) Add the libraries, tests, and examples
#
list(LENGTH HEADERS HEADERS_LEN)
list(LENGTH SOURCES SOURCES_LEN)

if (HEADERS_LEN AND SOURCES_LEN)
  # Add a standard library
  tribits_add_library(
    mundy_meta
    HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
    SOURCES ${SOURCES}
    )
elseif (HEADERS_LEN)
  # Directly install the headers
  tribits_install_headers(HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp)
else()
  # Empty package. Do nothing.
endif()

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()