# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2023 Flatiron Institute
# Author: Bryce Palmer
#
# Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Mundy. If not, see
# <https://www.gnu.org/licenses/>.
#
# **********************************************************************************************************************
# @HEADER

# ################################################## BASE #############################################################
cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)
project(Mundy CXX)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ################################################ OPTIONS ############################################################
option(ENABLE_CUDA "Enable build targeting GPU." OFF)
option(ENABLE_ALL_WARNINGS "Show most warnings for most compilers." ON)
option(ENABLE_WERROR "Warnings are errors." ON)
option(ENABLE_OPENMP "Enable OpenMP flags." OFF)
option(ENABLE_UNIT_TESTS "Enable unit testing." ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED)

# Create targets
set(mundy_exe_name "mundy.X")
add_executable(${mundy_exe_name} ${CMAKE_CURRENT_SOURCE_DIR}/mundy.cpp)
add_library(mundy "")

if(ENABLE_UNIT_TESTS)
  set(utest_ex_name "mundy_unit_tests.X")
  add_executable(${utest_ex_name} ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.cpp)
endif()

# ################################################### MPI ##############################################################
find_package(MPI REQUIRED)
target_link_libraries(mundy PUBLIC $<$<BOOL:${MPI_CXX_FOUND}>:MPI::MPI_CXX>)

if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    target_link_libraries(mundy PUBLIC
        ${CUDA_cusparse_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES})
endif()

# ################################################# TRILINOS ############################################################
set(CMAKE_PREFIX_PATH ${Trilinos_DIR} ${CMAKE_PREFIX_PATH})
find_package(Trilinos QUIET REQUIRED)
message(STATUS "Found Trilinos = ${Trilinos_DIR}")
target_link_libraries(mundy PUBLIC ${Trilinos_LIBRARIES})
target_include_directories(mundy SYSTEM PUBLIC ${Trilinos_INCLUDE_DIRS})
target_include_directories(mundy SYSTEM PUBLIC ${Trilinos_TPL_INCLUDE_DIRS})
if(Trilinos_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

# ################################################## OpenMP #############################################################
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(mundy PUBLIC $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
endif()

# ################################################### CUDA ##############################################################
if(ENABLE_CUDA)
    target_compile_definitions(mundy PUBLIC USE_STK_SIMD_NONE)
    separate_arguments(Trilinos_CXX_COMPILER_FLAGS)
    target_compile_options(mundy PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${Trilinos_CXX_COMPILER_FLAGS}>)
    target_compile_options(mundy PUBLIC $<$<COMPILE_LANGUAGE:CXX>:--expt-relaxed-constexpr>)
endif()

# ############################################### MUNDY ##########################################################
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# Add -restrict to Intel CXX compiler
target_compile_options(mundy PUBLIC $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Intel>>:-restrict>)

# Logic for handling warnings
if(ENABLE_ALL_WARNINGS)
    # GCC, Clang, and Intel seem to accept these
    list(APPEND MUNDY_CXX_FLAGS "-Wall" "-Wextra" "-pedantic")

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # Intel always reports some diagnostics we don't necessarily care about
        list(APPEND MUNDY_CXX_FLAGS "-diag-disable:11074,11076")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
        # Avoid notes about -faligned-new with GCC > 7
        list(APPEND MUNDY_CXX_FLAGS "-faligned-new")
    endif()
endif()

# Add our extra flags according to language
separate_arguments(MUNDY_CXX_FLAGS)
target_compile_options(mundy PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${MUNDY_CXX_FLAGS}>)
target_compile_options(mundy PRIVATE $<$<BOOL:${ENABLE_WERROR}>:-Werror>)

target_include_directories(mundy PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Most linking, etc, is set to PUBLIC for libmundy, so we merely link to libmundy for the exes
target_link_libraries(${mundy_exe_name} PRIVATE mundy)
if(ENABLE_UNIT_TESTS)
  # Unit tests are a separate executable and depend on libmundy and GoogleTest/GoogleMock
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
  )

  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # Link against gtest_main and mundy
  target_link_libraries(${utest_ex_name} PRIVATE 
    mundy 
    GTest::gtest 
    GTest::gtest_main 
    GTest::gmock 
    GTest::gmock_main)
  target_include_directories(${utest_ex_name} PRIVATE 
    "${CMAKE_SOURCE_DIR}/unit_tests"
    "${googletest_SOURCE_DIR}/googlemock/include"
    "${googletest_SOURCE_DIR}/googletest/include")
endif()

add_subdirectory(src)
add_subdirectory(include)
if(ENABLE_UNIT_TESTS)
  add_subdirectory(unit_tests)
  
  # Register our unit test executable with CTest
  enable_testing()
  add_test(NAME unit_tests COMMAND ${utest_ex_name})
endif()

if(ENABLE_UNIT_TESTS)
  install(TARGETS ${utest_ex_name}
          EXPORT "${PROJECT_NAME}Targets"
          RUNTIME DESTINATION bin
          ARCHIVE DESTINATION lib
          LIBRARY DESTINATION lib)
endif()
install(TARGETS ${mundy_exe_name} mundy
        EXPORT "${PROJECT_NAME}Targets"
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include DESTINATION include)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

export(
    TARGETS mundy
    NAMESPACE ${PROJECT_NAME}::
    FILE ${PROJECT_NAME}Targets.cmake)
install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME})
configure_package_config_file(
    cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME})
    install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    DESTINATION lib/cmake/${PROJECT_NAME})
    
