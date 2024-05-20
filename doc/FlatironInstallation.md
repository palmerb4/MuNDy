# Building MuNDyScratch

## Basic install

You need to install some dependencies as part of this, which we are going to assume go into `~/mundylib`.

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