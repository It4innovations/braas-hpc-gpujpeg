#!/bin/bash

# Build script for SYCL test

set -e

echo "Building SYCL preprocessor kernel test..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source and output paths
SOURCE_FILE="$SCRIPT_DIR/test_sycl.cpp"
OUTPUT_BINARY="$SCRIPT_DIR/test_sycl"

# Include directories
INCLUDE_DIRS="-I$PROJECT_ROOT -I$PROJECT_ROOT/libgpujpeg -I$PROJECT_ROOT/src"

# Compiler flags
SYCL_FLAGS="-fsycl"
CXX_FLAGS="-std=c++17 -O2 -Wall"
DEFINES="-DGPUJPEG_USE_SYCL -DGPUJPEG_INTERNAL_BUILD"

# Build with icpx (Intel oneAPI DPC++/SYCL compiler)
echo "Compiler: icpx"
echo "Source: $SOURCE_FILE"
echo "Output: $OUTPUT_BINARY"
echo ""

icpx $SYCL_FLAGS $CXX_FLAGS $DEFINES $INCLUDE_DIRS \
    "$SOURCE_FILE" \
    -o "$OUTPUT_BINARY"

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo "Executable: $OUTPUT_BINARY"
    echo ""
    echo "To run the test, execute:"
    echo "  $OUTPUT_BINARY"
else
    echo "Build failed!"
    exit 1
fi

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
#export ONEAPI_DEVICE_SELECTOR=*:cpu

echo "Running SYCL preprocessor kernel test..."
$OUTPUT_BINARY