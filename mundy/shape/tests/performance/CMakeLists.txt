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

# # Fetch OpenRand
# include(FetchContent)
# FetchContent_Declare(
#   open_rand
#   GIT_REPOSITORY https://github.com/msu-sparta/OpenRAND.git
#   GIT_TAG        main
# )
# FetchContent_GetProperties(open_rand)
# if(NOT open_rand_POPULATED)
#   FetchContent_Populate(open_rand)

#   # Modify the CMakeLists.txt file
#   file(READ ${open_rand_SOURCE_DIR}/CMakeLists.txt OPENRAND_CMAKELISTS)
#   string(REPLACE "cmake_minimum_required(VERSION 3.23)" "cmake_minimum_required(VERSION 3.11)" MODIFIED_OPENRAND_CMAKELISTS "${OPENRAND_CMAKELISTS}")
#   file(WRITE ${open_rand_SOURCE_DIR}/CMakeLists.txt "${MODIFIED_OPENRAND_CMAKELISTS}")
# endif()
# FetchContent_MakeAvailable(open_rand)

# include_directories(${open_rand_SOURCE_DIR}/include)

# message(STATUS "open_rand_SOURCE_DIR: ${open_rand_SOURCE_DIR}/include")

tribits_add_executable(
    PerformanceTestBrownianMotion
    SOURCES
    # ${open_rand_SOURCE_DIR}/include/openrand/base_state.h
    # ${open_rand_SOURCE_DIR}/include/openrand/philox.h
    # ${open_rand_SOURCE_DIR}/include/openrand/squares.h
    # ${open_rand_SOURCE_DIR}/include/openrand/threefry.h
    # ${open_rand_SOURCE_DIR}/include/openrand/tyche.h
    # ${open_rand_SOURCE_DIR}/include/openrand/util.h
    BrownianMotion.cpp
    CATEGORIES PERFORMANCE
    )