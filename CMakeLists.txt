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
cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(Mundy CXX)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ################################################ OPTIONS ############################################################
option(ENABLE_CUDA "Enable build targeting GPU" OFF)
option(ENABLE_ALL_WARNINGS "Show most warnings for most compilers" ON)
option(ENABLE_WERROR "Warnings are errors" ON)
option(ENABLE_OPENMP "Enable OpenMP flags" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED)

# Create targets
set(exe_name "mundy_test")
add_library(test_mundy_lib "")
add_executable(${exe_name} ${CMAKE_CURRENT_SOURCE_DIR}/test_mundy_app.cpp)

# ################################################### MPI ##############################################################
find_package(MPI REQUIRED)
target_link_libraries(test_mundy_lib PUBLIC $<$<BOOL:${MPI_CXX_FOUND}>:MPI::MPI_CXX>)

if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    target_link_libraries(test_mundy_lib PUBLIC
        ${CUDA_cusparse_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES})
endif()

# ################################################# TRILINOS ############################################################
set(CMAKE_PREFIX_PATH ${Trilinos_DIR} ${CMAKE_PREFIX_PATH})
find_package(Trilinos QUIET REQUIRED)

if(Trilinos_FOUND)
    message(STATUS "Found Trilinos = ${Trilinos_LIBRARY_DIRS}")
    target_link_libraries(test_mundy_lib PUBLIC ${Trilinos_LIBRARIES})
    target_include_directories(test_mundy_lib SYSTEM PUBLIC ${Trilinos_INCLUDE_DIRS})
    target_include_directories(test_mundy_lib SYSTEM PUBLIC ${Trilinos_TPL_INCLUDE_DIRS})

    if(Trilinos_BUILD_SHARED_LIBS)
        set(BUILD_SHARED_LIBS ON)
    endif()
endif()

# ################################################## OpenMP #############################################################
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(test_mundy_lib PUBLIC $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
endif()

# ################################################### CUDA ##############################################################
if(ENABLE_CUDA)
    target_compile_definitions(test_mundy_app PUBLIC USE_STK_SIMD_NONE)
    separate_arguments(Trilinos_CXX_COMPILER_FLAGS)
    target_compile_options(test_mundy_app PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${Trilinos_CXX_COMPILER_FLAGS}>)
    target_compile_options(test_mundy_app PUBLIC $<$<COMPILE_LANGUAGE:CXX>:--expt-relaxed-constexpr>)
endif()

# ############################################### TEST_STK_APP ##########################################################
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# Add -restrict to Intel CXX compiler
target_compile_options(test_mundy_app PUBLIC $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Intel>>:-restrict>)

# Logic for handling warnings
if(ENABLE_ALL_WARNINGS)
    # GCC, Clang, and Intel seem to accept these
    list(APPEND TEST_STK_APP_CXX_FLAGS "-Wall" "-Wextra" "-pedantic")

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # Intel always reports some diagnostics we don't necessarily care about
        list(APPEND TEST_STK_APP_CXX_FLAGS "-diag-disable:11074,11076")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
        # Avoid notes about -faligned-new with GCC > 7
        list(APPEND TEST_STK_APP_CXX_FLAGS "-faligned-new")
    endif()
endif()

# Add our extra flags according to language
separate_arguments(TEST_STK_APP_CXX_FLAGS)
target_compile_options(test_mundy_app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TEST_STK_APP_CXX_FLAGS}>)
target_compile_options(test_mundy_app PRIVATE $<$<BOOL:${ENABLE_WERROR}>:-Werror>)

target_include_directories(test_mundy_app PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(${exe_name} PRIVATE test_mundy_lib)

add_subdirectory(src)

export(
    TARGETS test_mundy_app
    NAMESPACE ${PROJECT_NAME}::
    FILE ${PROJECT_NAME}Targets.cmake)
