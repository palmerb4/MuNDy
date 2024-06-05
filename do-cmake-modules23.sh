TRILINOS_ROOT_DIR=$1
TPL_ROOT_DIR=$2
MUNDY_SOURCE_DIR=$3

# bash ../do-cmake-cje.sh /mnt/sw/nix/store/ajfmwdjwipp5rrpkq8dj4aff23ar4cix-trilinos-14.2.0 ~/mundyscratch/ ../
# If you want to build MPI-aware CUDA into this, you have to change the wrappings for the MPI compiler.
# OMPI_CXX=nvcc_wrapper make -j1 --trace
echo "Using Trilinos dir: $TRILINOS_ROOT_DIR"
echo "Using TPL dir: $TPL_ROOT_DIR"
echo "Using STK test-app dir: $MUNDY_SOURCE_DIR"

cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE:-DEBUG} \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR:-$HOME/mundyscratch} \
-DTPL_ENABLE_MPI=ON \
-DTPL_ENABLE_CUDA=ON \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_OPENMP=ON \
-DKokkos_ENABLE_CUDA=ON \
-DMundy_ENABLE_MundyAgents=OFF \
-DMundy_ENABLE_MundyAlens=OFF \
-DMundy_ENABLE_MundyBalance=OFF \
-DMundy_ENABLE_MundyDriver=OFF \
-DMundy_ENABLE_MundyConstraints=OFF \
-DMundy_ENABLE_MundyCore=ON \
-DMundy_ENABLE_MundyIo=OFF \
-DMundy_ENABLE_MundyLinkers=OFF \
-DMundy_ENABLE_MundyMath=OFF \
-DMundy_ENABLE_MundyMesh=OFF \
-DMundy_ENABLE_MundyMeta=OFF \
-DMundy_ENABLE_MundyMotion=OFF \
-DMundy_ENABLE_MundyShapes=OFF \
-DMundy_ENABLE_TESTS=ON \
-DMundy_ENABLE_GTest=ON \
-DMundy_TEST_CATEGORIES="BASIC;CONTINUOUS;NIGHTLY;HEAVY;PERFORMANCE" \
-DTPL_GTest_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_OpenRAND_DIR:PATH=${TPL_ROOT_DIR} \
-DTPL_Kokkos_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_STK_DIR:PATH=${TRILINOS_ROOT_DIR} \
-DTPL_Teuchos_DIR:PATH=${TRILINOS_ROOT_DIR} \
${ccache_args} \
${compiler_flags} \
${install_dir} \
${extra_args} \
${MUNDY_SOURCE_DIR}

