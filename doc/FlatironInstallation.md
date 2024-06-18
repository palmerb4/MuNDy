# Building MuNDyScratch

## Basic install

You need to install some dependencies as part of this, which we are going to assume go into `~/mundylib`. Take a look
at Chris Edelmaier's `.bashrc` file for what modules to load for Mundy. For instance, I put the following as a function
into my `.bashrc`.

```bash
modulealens() {
module -q purge
module load modules/2.2-20230808
module load gcc/11.4.0
module load openmpi/4.0.7
module load hdf5/mpi-1.14.1-2
module load intel-oneapi-mkl/2023.1.0
module load netcdf-c/4.9.2
module load flexiblas/3.3.0
module load cmake/3.26.3
#module load trilinos/mpi-14.2.0
module load python/3.10.10
module load llvm
}
```

Then you can just type `modulealens` in a new shell to setup the environment.

```bash
git clone --recursive git@github.com:palmerb4/MuNDyScratch.git
git checkout benchmarks
cd dep
bash install_gtest.sh ~/mundylib
bash install_openrand.sh ~/mundylib
cd ..
mkdir -p build
cd build
bash ../do-cmake.sh /mnt/sw/nix/store/ajfmwdjwipp5rrpkq8dj4aff23ar4cix-trilinos-14.2.0/lib/cmake/Trilinos ~/mundylib ../
make -j12
ctest
```

For now, this builds the executables in the build folder at `mundy/alens/tests/performance_tests/MundyAlens_StickySettings.exe`.

## Profiling

To use the Kokkos profiling tooks, you need to take a few steps.

### Build the Kokkos tools

Luckily, you don't need to know the Kokkos path to build the tools!

```bash
git clone --recursive git@github.com:kokkos/kokkos-tools.git
mkdir ~/libkokkos
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/libkokkos ..
make -j12
make install
```

Then you can run the profiled version of the code by specifying the Kokkos tool (look online for details, there are a lot of these to choose from).

```bash
/mnt/home/cedelmaier/Projects/Biophysics/MuNDyScratch/build/mundy/alens/tests/performance_tests/MundyAlens_StickySettings.exe --kokkos-tools-libs=/mnt/home/cedelmaier/libkokkos/lib64/libkp_space_time_stack.so --no_use_input_file --num_spheres=100 --sphere_radius=0.5 --initial_sphere_separation=0.5 --backbone_spring_constant=1.0 --backbone_spring_rest_length=1.0 --crosslinker_spring_constant=0.3 --crosslinker_rest_length=1.5 --crosslinker_left_binding_rate=0.1 --crosslinker_right_binding_rate=0.1 --num_time_steps=4001 --timestep_size=0.00025 --io_frequency=4000 --kt=0.0 --initial_loadbalance --use_mundy_io
```