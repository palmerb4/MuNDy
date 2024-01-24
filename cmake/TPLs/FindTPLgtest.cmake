include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Set the global variables for TriBITS
GLOBAL_SET(TPL_gtest_INCLUDE_DIRS "${googletest_SOURCE_DIR}/googlemock/include;${googletest_SOURCE_DIR}/googletest/include")
GLOBAL_SET(TPL_gtest_LIBRARIES "GTest::gtest;GTest::gtest_main;GTest::gmock;GTest::gmock_main")