# GPU Device Compatibility Layer Documentation

This document describes the CUDA/HIP/SYCL compatibility layer for GPUJPEG.

## Overview

The compatibility layer allows GPUJPEG to compile and run on multiple GPU backends:
- **CUDA** (NVIDIA GPUs)
- **HIP** (AMD GPUs via ROCm)
- **SYCL** (Intel GPUs and others via oneAPI)

## Files Added

### Core Compatibility Files
1. **src/gpujpeg_device_compat.h** - Main compatibility header with macros
2. **src/gpujpeg_device_compat_sycl.cpp** - SYCL implementation for runtime functions

### C++ Wrappers for CUDA Files
These files allow .cu files to be compiled with HIP or as regular C++:
1. **src/gpujpeg_dct_gpu.cpp** - Wraps gpujpeg_dct_gpu.cu
2. **src/gpujpeg_huffman_gpu_decoder.cpp** - Wraps gpujpeg_huffman_gpu_decoder.cu
3. **src/gpujpeg_huffman_gpu_encoder.cpp** - Wraps gpujpeg_huffman_gpu_encoder.cu
4. **src/gpujpeg_postprocessor.cpp** - Wraps gpujpeg_postprocessor.cu
5. **src/gpujpeg_preprocessor.cpp** - Wraps gpujpeg_preprocessor.cu

## API Mapping

The compatibility layer provides the following macro mappings:

### Types
| Original CUDA | Compatibility Macro | HIP Equivalent | SYCL Equivalent |
|---------------|---------------------|----------------|-----------------|
| cudaStream_t | gpuStream_t | hipStream_t | sycl::queue* |
| cudaError_t | gpuError_t | hipError_t | int |
| cudaEvent_t | gpuEvent_t | hipEvent_t | void* |
| cudaGraphicsResource | gpuGraphicsResource | hipGraphicsResource | void* |

### Memory Management
| Original CUDA | Compatibility Macro |
|---------------|---------------------|
| cudaMalloc | gpuMalloc |
| cudaMallocHost | gpuMallocHost |
| cudaFree | gpuFree |
| cudaFreeHost | gpuFreeHost |
| cudaMemcpy | gpuMemcpy |
| cudaMemcpyAsync | gpuMemcpyAsync |
| cudaMemcpy2DAsync | gpuMemcpy2DAsync |
| cudaMemcpyToSymbol | gpuMemcpyToSymbol |
| cudaMemcpyToSymbolAsync | gpuMemcpyToSymbolAsync |

### Stream Management
| Original CUDA | Compatibility Macro |
|---------------|---------------------|
| cudaStreamDefault | gpuStreamDefault |
| cudaStreamSynchronize | gpuStreamSynchronize |

### Event Management
| Original CUDA | Compatibility Macro |
|---------------|---------------------|
| cudaEventCreate | gpuEventCreate |
| cudaEventDestroy | gpuEventDestroy |
| cudaEventRecord | gpuEventRecord |
| cudaEventSynchronize | gpuEventSynchronize |
| cudaEventElapsedTime | gpuEventElapsedTime |

### Device Management
| Original CUDA | Compatibility Macro |
|---------------|---------------------|
| cudaSetDevice | gpuSetDevice |
| cudaGetLastError | gpuGetLastError |
| cudaGetErrorString | gpuGetErrorString |

### Function Configuration
| Original CUDA | Compatibility Macro |
|---------------|---------------------|
| cudaFuncSetCacheConfig | gpuFuncSetCacheConfig |
| cudaFuncCachePreferShared | gpuFuncCachePreferShared |

## Building for Different Backends

### Building for CUDA (Default)
```bash
cmake -B build -S .
cmake --build build
```

### Building for HIP (AMD GPUs)
```bash
# Set HIP compiler
export CC=hipcc
export CXX=hipcc

cmake -B build -S . \
  -DCMAKE_CUDA_COMPILER=hipcc \
  -DGPUJPEG_USE_HIP=ON

cmake --build build
```

### Building for SYCL (Intel GPUs)
```bash
# Use Intel oneAPI DPC++ compiler
source /opt/intel/oneapi/setvars.sh

cmake -B build -S . \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGPUJPEG_USE_SYCL=ON

cmake --build build
```

## Integration Steps Performed

1. **Created compatibility header** (gpujpeg_device_compat.h)
   - Defines macros for all CUDA API calls
   - Automatically detects backend (CUDA/HIP/SYCL)
   - Provides unified interface

2. **Created SYCL runtime implementation** (gpujpeg_device_compat_sycl.cpp)
   - Implements SYCL-specific runtime functions
   - Wraps SYCL queue operations
   - Provides basic timing support

3. **Updated core headers to use compatibility layer**
   - libgpujpeg/gpujpeg_common.h
   - src/gpujpeg_common_internal.h
   - src/gpujpeg_util.h
   - src/gpujpeg_postprocessor.h

4. **Created C++ wrappers for .cu files**
   - Allows compilation with HIP or standard C++ compilers
   - Simply includes the .cu file after including compatibility header

## Modified Files

### Headers Updated
- `libgpujpeg/gpujpeg_common.h` - Added compatibility layer inclusion
- `src/gpujpeg_common_internal.h` - Replaced cuda_runtime.h with gpujpeg_device_compat.h
- `src/gpujpeg_util.h` - Updated error checking macros
- `src/gpujpeg_postprocessor.h` - Changed cudaStream_t to gpuStream_t

### Internal Structure Changes
- `struct gpujpeg_timer` - Uses `gpuEvent_t` instead of `cudaEvent_t`
- `struct gpujpeg_coder` - Uses `gpuStream_t` instead of `cudaStream_t`
- Timer macros - Use `gpu*` functions instead of `cuda*`
- Error checking macros - Use `gpu*` functions

## CMakeLists.txt Example Updates

Add this to your CMakeLists.txt to support all backends:

```cmake
# Detect GPU backend
if(GPUJPEG_USE_HIP)
    set(GPU_BACKEND "HIP")
    enable_language(HIP)
    add_compile_definitions(GPUJPEG_USE_HIP)
elseif(GPUJPEG_USE_SYCL)
    set(GPU_BACKEND "SYCL")
    add_compile_definitions(GPUJPEG_USE_SYCL)
    find_package(IntelDPCPP REQUIRED)
else()
    set(GPU_BACKEND "CUDA")
    enable_language(CUDA)
    add_compile_definitions(GPUJPEG_USE_CUDA)
endif()

message(STATUS "Building GPUJPEG with ${GPU_BACKEND} backend")

# Source files
set(GPUJPEG_SOURCES
    src/gpujpeg_common.c
    src/gpujpeg_encoder.c
    src/gpujpeg_decoder.c
    # ... other .c files ...
)

# GPU kernel sources
if(GPU_BACKEND STREQUAL "CUDA")
    set(GPUJPEG_GPU_SOURCES
        src/gpujpeg_dct_gpu.cu
        src/gpujpeg_huffman_gpu_decoder.cu
        src/gpujpeg_huffman_gpu_encoder.cu
        src/gpujpeg_postprocessor.cu
        src/gpujpeg_preprocessor.cu
    )
elseif(GPU_BACKEND STREQUAL "HIP")
    set(GPUJPEG_GPU_SOURCES
        src/gpujpeg_dct_gpu.cpp
        src/gpujpeg_huffman_gpu_decoder.cpp
        src/gpujpeg_huffman_gpu_encoder.cpp
        src/gpujpeg_postprocessor.cpp
        src/gpujpeg_preprocessor.cpp
    )
elseif(GPU_BACKEND STREQUAL "SYCL")
    set(GPUJPEG_GPU_SOURCES
        src/gpujpeg_device_compat_sycl.cpp
        # SYCL would need kernel rewrite
    )
endif()

add_library(gpujpeg ${GPUJPEG_SOURCES} ${GPUJPEG_GPU_SOURCES})

# Link GPU runtime
if(GPU_BACKEND STREQUAL "CUDA")
    target_link_libraries(gpujpeg PRIVATE CUDA::cudart)
elseif(GPU_BACKEND STREQUAL "HIP")
    target_link_libraries(gpujpeg PRIVATE hip::host)
elseif(GPU_BACKEND STREQUAL "SYCL")
    target_link_libraries(gpujpeg PRIVATE IntelDPCPP::IntelDPCPP)
endif()
```

## Limitations and Notes

### SYCL Support
- **Current status**: Infrastructure in place, but kernels need refactoring
- SYCL uses a different programming model (command groups, kernels as lambdas)
- Direct .cu file inclusion won't work for SYCL
- Requires rewriting CUDA kernels to SYCL kernels
- The current SYCL implementation provides runtime API compatibility only

### HIP Support
- **Current status**: Should work with minimal changes
- HIP is designed to be compatible with CUDA syntax
- The .cpp wrapper approach allows hipcc to compile .cu files
- Most CUDA code translates directly to HIP

### CUDA Support
- **Current status**: Fully functional
- No changes to existing CUDA functionality
- Backward compatible with existing code

## Testing

To verify the compatibility layer works:

1. **For CUDA**:
   ```bash
   cmake -B build-cuda -S .
   cmake --build build-cuda
   ./build-cuda/gpujpeg_test
   ```

2. **For HIP** (if you have AMD GPU):
   ```bash
   export CC=hipcc CXX=hipcc
   cmake -B build-hip -S . -DGPUJPEG_USE_HIP=ON
   cmake --build build-hip
   ./build-hip/gpujpeg_test
   ```

3. **For SYCL** (requires kernel rewrites):
   ```bash
   source /opt/intel/oneapi/setvars.sh
   cmake -B build-sycl -S . -DGPUJPEG_USE_SYCL=ON
   cmake --build build-sycl
   # Note: Will need kernel implementations
   ```

## Future Work

1. **Complete SYCL kernel implementations**
   - Rewrite DCT kernels in SYCL
   - Rewrite Huffman encoder/decoder kernels
   - Rewrite preprocessor/postprocessor kernels

2. **Add runtime backend selection**
   - Allow choosing backend at runtime if multiple are available

3. **Performance optimization**
   - Tune HIP kernels for AMD GPUs
   - Optimize SYCL kernels for Intel GPUs

4. **Extended testing**
   - Add CI/CD for all three backends
   - Performance benchmarking across backends

## Contact and Support

For issues or questions about the compatibility layer:
- Check existing issues on GitHub
- Create a new issue with [GPU-COMPAT] tag
- Include backend type (CUDA/HIP/SYCL) in bug reports
