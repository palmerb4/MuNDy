# The order of the packages below is important. Mundy subpackages can
# depend on subpackages listed above them, and not below them.
tribits_package_define_dependencies(
  SUBPACKAGES_DIRS_CLASSIFICATIONS_OPTREQS
    Core        core        PT  REQUIRED
    Mesh        mesh        PT  REQUIRED
    Math        math        PT  REQUIRED
    Meta        meta        PT  REQUIRED
    Balance     balance     PT  REQUIRED
    Io          io          PT  REQUIRED
    Agent       agent       PT  REQUIRED
    Shape       shape       PT  REQUIRED
    Constraint  constraint  PT  REQUIRED
    Linker      linker      PT  REQUIRED
    Motion      motion      PT  REQUIRED
  REGRESSION_EMAIL_LIST brycepalmer96@gmail.com
  )
