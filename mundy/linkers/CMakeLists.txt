#
# A) Define the subpackage
#
tribits_subpackage(Linker)

# By default we enable all optional packages. They can be disabled by the user if they are not needed.
GLOBAL_SET(${PACKAGE_NAME}_ENABLE_ALL_OPTIONAL_PACKAGES ON)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use 
# find include/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort

SET(HEADERS
  include/mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp
  include/mundy_linkers/compute_signed_separation_distance_and_contact_normal/kernels/SphereSphereLinker.hpp
  include/mundy_linkers/DestroyNeighborLinkers.hpp
  include/mundy_linkers/destroy_neighbor_linkers/techniques/DestroyDistantNeighbors.hpp
  include/mundy_linkers/EvaluateLinkerPotentials.hpp
  include/mundy_linkers/evaluate_linker_potentials/kernels/SphereSphereHertzianContact.hpp
  include/mundy_linkers/GenerateNeighborLinkers.hpp
  include/mundy_linkers/generate_neighbor_linkers/techniques/STKSearch.hpp
  include/mundy_linkers/LinkerPotentialForceMagnitudeReduction.hpp
  include/mundy_linkers/linker_potential_force_magnitude_reduction/kernels/Sphere.hpp
  include/mundy_linkers/Linkers.hpp
  include/mundy_linkers/NeighborLinkers.hpp
  include/mundy_linkers/neighbor_linkers/SphereSphereLinkers.hpp
  )

SET(SOURCES 
  src/mundy_linkers/compute_signed_separation_distance_and_contact_normal/kernels/SphereSphereLinker.cpp
  src/mundy_linkers/destroy_neighbor_linkers/techniques/DestroyDistantNeighbors.cpp
  src/mundy_linkers/evaluate_linker_potentials/kernels/SphereSphereHertzianContact.cpp
  src/mundy_linkers/generate_neighbor_linkers/techniques/STKSearch.cpp
  src/mundy_linkers/linker_potential_force_magnitude_reduction/kernels/Sphere.cpp
)

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
tribits_add_library(
  mundy_linkers
  HEADERS ${HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  SOURCES ${SOURCES}
  )


tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()