FIND_PACKAGE(Belos REQUIRED
    CONFIG
    HINTS
      ${TPL_Belos_DIR}/lib/cmake/Belos
      ${TPL_Belos_DIR}
    COMPONENTS
      ${${PACKAGE_NAME}_Belos_REQUIRED_COMPONENTS}
    OPTIONAL_COMPONENTS
      ${${PACKAGE_NAME}_Belos_OPTIONAL_COMPONENTS}
)

# Create the TriBITS-compliant <tplName>Config.cmake wrapper file
# This appears to be the minimal requirement to load in a TriBITS-compliant TPL.
tribits_extpkgwit_create_package_config_file(
  Belos
  INNER_FIND_PACKAGE_NAME Belos
  IMPORTED_TARGETS_FOR_ALL_LIBS Belos::all_libs)
