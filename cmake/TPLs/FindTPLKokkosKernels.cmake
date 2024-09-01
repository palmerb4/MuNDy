FIND_PACKAGE(KokkosKernels REQUIRED
    CONFIG
    HINTS
      ${TPL_KokkosKernels_DIR}/lib/cmake/KokkosKernels
      ${TPL_KokkosKernels_DIR}
    COMPONENTS
      ${${PACKAGE_NAME}_KokkosKernels_REQUIRED_COMPONENTS}
    OPTIONAL_COMPONENTS
      ${${PACKAGE_NAME}_KokkosKernels_OPTIONAL_COMPONENTS}
)

# Create the TriBITS-compliant <tplName>Config.cmake wrapper file
# This appears to be the minimal requirement to load in a TriBITS-compliant TPL.
tribits_extpkgwit_create_package_config_file(
  KokkosKernels
  INNER_FIND_PACKAGE_NAME KokkosKernels
  IMPORTED_TARGETS_FOR_ALL_LIBS KokkosKernels::all_libs)