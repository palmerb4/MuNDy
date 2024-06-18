# The purpose of this file is to reorganize pvfmm's non-standard cmake files

# Use the existing find_package to find pvfmm
find_package(pvfmm QUIET)

if(pvfmm_FOUND)
    # Augment the include directories to include both the base and the pvfmm sub-directory
    set(PVFMM_INCLUDE_DIRS ${PVFMM_INCLUDE_DIR} ${PVFMM_INCLUDE_DIR}/pvfmm)

    # Create an INTERFACE library that represents PVFMM and setup dependencies
    add_library(PVFMM::PVFMM INTERFACE IMPORTED)
    target_link_libraries(PVFMM::PVFMM
        INTERFACE ${PVFMM_LIB_DIR}/${PVFMM_STATIC_LIB} 
        INTERFACE ${PVFMM_DEP_LIB}
    )
    target_include_directories(PVFMM::PVFMM
        INTERFACE ${PVFMM_INCLUDE_DIRS}
    )
    target_compile_features(PVFMM::PVFMM INTERFACE cxx_std_17)
    target_compile_options(PVFMM::PVFMM INTERFACE -DSCTL_QUAD_T=__float128)

    # Set the PVFMM_LIBRARIES 
    set(PVFMM_LIBRARIES PVFMM::PVFMM)

    # Set PVFMM_FOUND to true explicitly to get the capitalization right
    set(PVFMM_FOUND TRUE)

    # Print what we found
    message(STATUS "Found PVFMM")
    message(STATUS "  Include Directories: ${PVFMM_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${PVFMM_LIBRARIES}")

else()
    # Explicitly set PVFMM_FOUND to false if not found
    set(PVFMM_FOUND FALSE)
endif()


# For the package to be considered found, we should handle the standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PVFMM DEFAULT_MSG PVFMM_LIBRARIES PVFMM_INCLUDE_DIRS PVFMM_FOUND)