#!/bin/bash

#./install_gtest.sh /path/to/install/directory

# Check if an install directory was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <install_directory>"
    exit 1
fi

# The directory where Google Test will be installed
INSTALL_DIR=$1

# Temporary directory for building Google Test
BUILD_DIR=$(mktemp -d)

# Google Test's GitHub repository
GTEST_REPO="https://github.com/google/googletest.git"

# Clone Google Test
git clone --depth=1 $GTEST_REPO $BUILD_DIR

# Proceed to the build directory
cd $BUILD_DIR

# Create a build directory
mkdir build && cd build

# Configure the project with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR

# Build Google Test
make -j$(nproc)

# Install Google Test to the specified directory
make install

# Cleanup
cd $OLDPWD
rm -rf $BUILD_DIR

echo "Google Test has been installed to $INSTALL_DIR"
