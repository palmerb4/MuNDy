TRILINOS_ROOT_DIR=$1
TPL_ROOT_DIR=$2
MUNDY_SOURCE_DIR=$3

# bash ../do-cmake.sh /mnt/sw/nix/store/ajfmwdjwipp5rrpkq8dj4aff23ar4cix-trilinos-14.2.0 ~/envs/MundyScratch/ ../
echo "Using Trilinos dir: $TRILINOS_ROOT_DIR"
echo "Using TPL dir: $TPL_ROOT_DIR"
echo "Using STK test-app dir: $MUNDY_SOURCE_DIR"

cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE:-DEBUG} \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_CXX_FLAGS="-O0 -g -march=native" \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR:-$HOME/envs/MundyScratch} \
-DTPL_ENABLE_MPI=ON \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_OPENMP=OFF \
-DKokkos_ENABLE_CUDA=OFF \
-DMundy_ENABLE_TESTS=ON \
-DMundy_ENABLE_GTest=ON \
-DMundy_TEST_CATEGORIES="BASIC;CONTINUOUS;NIGHTLY;HEAVY;PERFORMANCE" \
-DTPL_GTest_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_Kokkos_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_STK_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_Teuchos_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DOpenRAND_INCLUDE_DIRS:PATH=${TPL_ROOT_DIR}/include \
${ccache_args} \
${compiler_flags} \
${install_dir} \
${extra_args} \
${MUNDY_SOURCE_DIR}
