tribits_package_decl(Mundy)

tribits_add_debug_option()

# For some odd reason, our PACKAGE_SOURCE_DIR is not the current directory. It's ../ ??

# Global variables
GLOBAL_SET(MUNDY_DEFAULT_UNIT_TEST_MAIN
  "${PACKAGE_SOURCE_DIR}/mundy/tests/unit_tests/DefaultUnitTestMain.cpp")

tribits_add_test_directories(${PACKAGE_SOURCE_DIR}/mundy/tests)

tribits_process_subpackages()

tribits_package_def()

tribits_package_postprocess()