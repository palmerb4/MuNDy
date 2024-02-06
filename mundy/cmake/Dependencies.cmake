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
    Core        core        PT  REQUIRED
    Mesh        mesh        PT  REQUIRED
    Math        math        PT  REQUIRED
    Meta        meta        PT  REQUIRED
    #Balance     balance     PT  REQUIRED
    #Io          io          PT  REQUIRED
    Agent       agent       PT  REQUIRED
    Shape       shape       PT  REQUIRED
    Constraint  constraint  PT  REQUIRED
    Linker      linker      PT  REQUIRED
    Motion      motion      PT  REQUIRED
  REGRESSION_EMAIL_LIST brycepalmer96@gmail.com
  )
