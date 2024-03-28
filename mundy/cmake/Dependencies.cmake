# The order of the packages below is important. Mundy subpackages can
# depend on subpackages listed above them, and not below them.
tribits_package_define_dependencies(
  LIB_REQUIRED_PACKAGES
  LIB_OPTIONAL_PACKAGES
  TEST_REQUIRED_PACKAGES
  TEST_OPTIONAL_PACKAGES
  LIB_REQUIRED_TPLS
  LIB_OPTIONAL_TPLS
  TEST_REQUIRED_TPLS GTest
  TEST_OPTIONAL_TPLS MPI
  SUBPACKAGES_DIRS_CLASSIFICATIONS_OPTREQS
    Core         core        PT  OPTIONAL
    Mesh         mesh        PT  OPTIONAL
    Math         math        PT  OPTIONAL
    Meta         meta        PT  OPTIONAL
    Agents       agents      PT  OPTIONAL
    Shapes       shapes      PT  OPTIONAL
    Linkers      linkers     PT  OPTIONAL
    # Balance      balance     PT  OPTIONAL
    Io           io          PT  OPTIONAL
    Constraints  constraints  PT  OPTIONAL
    # Motion       motion      PT  OPTIONAL
    Driver       driver      PT  OPTIONAL
  REGRESSION_EMAIL_LIST brycepalmer96@gmail.com
  )
