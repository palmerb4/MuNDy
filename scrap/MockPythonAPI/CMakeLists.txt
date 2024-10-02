cmake_minimum_required(VERSION 3.12)
project(mock)

set(CMAKE_CXX_STANDARD 14)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})

# Create the core library
add_library(mock_core
    mock.cpp
    mock.hpp
)

# Create the API library
add_library(mock_api
    mock_api.cpp
    mock_api.hpp
)

# Link the API library with the core library
target_link_libraries(mock_api PRIVATE mock_core)

# Build the Python module
pybind11_add_module(mock mock_pybind.cpp)
target_link_libraries(mock PRIVATE mock_api mock_core)

# Install the Python module to the site-packages directory
message(STATUS "PYTHON_SITE_PACKAGES: ${PYTHON_SITE_PACKAGES}")
install(TARGETS mock
        COMPONENT python
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin")

# Create our performance test executables
add_executable(delayed delayed.cpp)
target_link_libraries(delayed PRIVATE mock_api mock_core)
add_executable(direct direct.cpp)
target_link_libraries(direct PRIVATE mock_api mock_core)
