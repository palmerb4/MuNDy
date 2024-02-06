FIND_PACKAGE(ArborX REQUIRED
    CONFIG
    HINTS
      ${TPL_ArborX_DIR}/lib/cmake/ArborX
      ${TPL_ArborX_DIR}
)

tribits_extpkg_create_imported_all_libs_target_and_config_file(
  ArborX
  INNER_FIND_PACKAGE_NAME ArborX
  IMPORTED_TARGETS_FOR_ALL_LIBS ArborX::ArborX )