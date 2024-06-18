find_package(PVFMM)
tribits_extpkg_create_imported_all_libs_target_and_config_file(
  PVFMM
  INNER_FIND_PACKAGE_NAME PVFMM
  IMPORTED_TARGETS_FOR_ALL_LIBS PVFMM::PVFMM)
