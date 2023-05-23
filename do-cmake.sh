TRILINOS_ROOT_DIR=$1
MUNDY_SOURCE_DIR=$2

echo "Using Trilinos dir: $TRILINOS_ROOT_DIR"
echo "Using STK test-app dir: $MUNDY_SOURCE_DIR"

cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE:-DEBUG} \
-DCMAKE_CXX_COMPILER=mpicxx \
-DENABLE_OPENMP=${ENABLE_OPENMP:-OFF} \
-DENABLE_CUDA=${ENABLE_CUDA:-OFF} \
-DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS:-OFF} \
-DTrilinos_DIR:PATH=${TRILINOS_ROOT_DIR} \
${ccache_args} \
${compiler_flags} \
${install_dir} \
${extra_args} \
${MUNDY_SOURCE_DIR}
