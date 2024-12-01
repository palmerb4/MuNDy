FIND_PACKAGE(fmt REQUIRED
    CONFIG
    HINTS
      ${TPL_fmt_DIR}/lib/cmake/fmt
      ${TPL_fmt_DIR}
)

tribits_extpkg_create_imported_all_libs_target_and_config_file(
  fmt
  INNER_FIND_PACKAGE_NAME fmt
  IMPORTED_TARGETS_FOR_ALL_LIBS fmt::fmt )
