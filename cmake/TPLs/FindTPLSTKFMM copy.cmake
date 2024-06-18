# Find the STKFMM Library
#
#  STKFMM_FOUND - System has MKL
#  STKFMM_INCLUDE_DIRS - MKL include files directories
#  STKFMM_LIBRARIES - The MKL libraries
#  STKFMM_INTERFACE_LIBRARY - MKL interface library
#  STKFMM_SEQUENTIAL_LAYER_LIBRARY - MKL sequential layer library
#  STKFMM_CORE_LIBRARY - MKL core library
#
#  Example usage:
#
#  find_package(STKFMM)
#  if(STKFMM_FOUND)
#    target_link_libraries(TARGET ${STKFMM_LIBRARIES})
#  endif()

# Perform the find package and wrap the results in an imported target
find_package(STKFMM)
tribits_extpkg_create_imported_all_libs_target_and_config_file(
  STKFMM
  INNER_FIND_PACKAGE_NAME STKFMM
  IMPORTED_TARGETS_FOR_ALL_LIBS  ${STKFMM_LIBRARIES})