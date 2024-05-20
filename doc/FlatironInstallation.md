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
```

For now, this builds the executables in the build folder at `mundy/alens/tests/performance_tests/MundyAlens_StickySettings.exe`.