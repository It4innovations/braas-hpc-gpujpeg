/**
 * @file
 * Copyright (c) 2025, CESNET
 *
 * GPU Device Compatibility Layer for CUDA, HIP, and SYCL
 *
 * This header provides a unified interface for GPU operations across
 * CUDA, HIP, and SYCL backends.
 */

#ifndef GPUJPEG_DEVICE_COMPAT_H
#define GPUJPEG_DEVICE_COMPAT_H

#include <stddef.h>  // for size_t

// Determine which GPU backend to use
// Check CMake-defined macros first, then fall back to compiler detection
#if defined(GPUJPEG_USE_HIP)
    // HIP backend explicitly requested via CMake
#elif defined(GPUJPEG_USE_SYCL)
    // SYCL backend explicitly requested via CMake
#elif defined(GPUJPEG_USE_CUDA)
    // CUDA backend explicitly requested via CMake
#elif defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
    // Auto-detect HIP from compiler
    #define GPUJPEG_USE_HIP
#elif defined(__SYCL_DEVICE_ONLY__) || defined(SYCL_LANGUAGE_VERSION)
    // Auto-detect SYCL from compiler
    #define GPUJPEG_USE_SYCL
#else
    // Default to CUDA
    #define GPUJPEG_USE_CUDA
#endif

// ==============================================================================
// CUDA Backend
// ==============================================================================
#ifdef GPUJPEG_USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel function qualifiers
#define GPU_DEVICE                          __device__
#define GPU_HOST                            __host__
#define GPU_GLOBAL                          __global__
#define GPU_SHARED                          __shared__
#define GPU_CONSTANT                        __constant__

// CUDA doesn't need item parameter
#define GPU_KERNEL_ITEM_PARAM               /* empty */
#define GPU_KERNEL_ITEM_ARG                 /* empty */
#define GPU_ITEM_COMMA                      /* empty */

// Built-in variables
#define GPU_THREAD_IDX_X                    threadIdx.x
#define GPU_THREAD_IDX_Y                    threadIdx.y
#define GPU_THREAD_IDX_Z                    threadIdx.z
#define GPU_BLOCK_IDX_X                     blockIdx.x
#define GPU_BLOCK_IDX_Y                     blockIdx.y
#define GPU_BLOCK_IDX_Z                     blockIdx.z
#define GPU_BLOCK_DIM_X                     blockDim.x
#define GPU_BLOCK_DIM_Y                     blockDim.y
#define GPU_BLOCK_DIM_Z                     blockDim.z
#define GPU_GRID_DIM_X                      gridDim.x
#define GPU_GRID_DIM_Y                      gridDim.y
#define GPU_GRID_DIM_Z                      gridDim.z

// Synchronization
#define GPU_SYNCTHREADS()                   __syncthreads()

// Atomic operations
#define GPU_ATOMIC_ADD(ptr, val)            atomicAdd(ptr, val)

// Math functions
#define GPU_RINTF(x)                        rintf(x)

// Vector types
#define GPU_MAKE_UINT4(x, y, z, w)          make_uint4(x, y, z, w)

// Type definitions
#define gpuStream_t                         cudaStream_t
#define gpuError_t                          cudaError_t
#define gpuEvent_t                          cudaEvent_t

// Error codes
#define gpuSuccess                          cudaSuccess

// Memory management
#define gpuMalloc                           cudaMalloc
#define gpuMallocHost                       cudaMallocHost
#define gpuFree                             cudaFree
#define gpuFreeHost                         cudaFreeHost
#define gpuMemcpy                           cudaMemcpy
#define gpuMemcpyAsync                      cudaMemcpyAsync
#define gpuMemcpy2DAsync                    cudaMemcpy2DAsync
#define gpuMemset                           cudaMemset
#define gpuMemsetAsync                      cudaMemsetAsync
#define gpuHostRegister                     cudaHostRegister
#define gpuHostUnregister                   cudaHostUnregister
#define gpuMemcpyHostToDevice               cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost               cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice             cudaMemcpyDeviceToDevice
#define gpuMemcpyToSymbol                   cudaMemcpyToSymbol
#define gpuMemcpyToSymbolAsync              cudaMemcpyToSymbolAsync

// Host register flags
#define gpuHostRegisterDefault              cudaHostRegisterDefault

// Stream management
#define gpuStreamDefault                    cudaStreamDefault
#define gpuStreamSynchronize                cudaStreamSynchronize

// Event management
#define gpuEventCreate                      cudaEventCreate
#define gpuEventDestroy                     cudaEventDestroy
#define gpuEventRecord                      cudaEventRecord
#define gpuEventSynchronize                 cudaEventSynchronize
#define gpuEventElapsedTime                 cudaEventElapsedTime

// Device management
#define gpuSetDevice                        cudaSetDevice
#define gpuGetDevice                        cudaGetDevice
#define gpuGetDeviceCount                   cudaGetDeviceCount
#define gpuGetDeviceProperties              cudaGetDeviceProperties
#define gpuGetLastError                     cudaGetLastError
#define gpuGetErrorString                   cudaGetErrorString
#define gpuDeviceReset                      cudaDeviceReset
#define gpuDriverGetVersion                 cudaDriverGetVersion
#define gpuRuntimeGetVersion                cudaRuntimeGetVersion

// Device properties
#define gpuDeviceProp                       cudaDeviceProp

// Memory copy kinds
#define gpuMemcpyKind                       cudaMemcpyKind

// Function attributes
#define gpuFuncSetCacheConfig               cudaFuncSetCacheConfig
#define gpuFuncCachePreferShared            cudaFuncCachePreferShared

// Runtime version
#ifndef GPUART_VERSION
#define GPUART_VERSION                      CUDART_VERSION
#endif

// Kernel launch syntax (CUDA uses <<<>>> syntax natively)
#define GPU_KERNEL_LAUNCH(kernel, grid, block, smem, stream, ...) \
    kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)

// ==============================================================================
// HIP Backend
// ==============================================================================
#elif defined(GPUJPEG_USE_HIP)

// Define HIP platform for AMD GPUs
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

// Use C-compatible HIP API header for C files
#ifdef __cplusplus
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>
    #ifndef half
    typedef __half half;
    #endif
#else
    #include <hip/hip_runtime_api.h>
#endif

// Kernel function qualifiers
#define GPU_DEVICE                          __device__
#define GPU_HOST                            __host__
#define GPU_GLOBAL                          __global__
#define GPU_SHARED                          __shared__
#define GPU_CONSTANT                        __constant__

// HIP doesn't need item parameter
#define GPU_KERNEL_ITEM_PARAM               /* empty */
#define GPU_KERNEL_ITEM_ARG                 /* empty */
#define GPU_ITEM_COMMA                      /* empty */

// Built-in variables
#define GPU_THREAD_IDX_X                    hipThreadIdx_x
#define GPU_THREAD_IDX_Y                    hipThreadIdx_y
#define GPU_THREAD_IDX_Z                    hipThreadIdx_z
#define GPU_BLOCK_IDX_X                     hipBlockIdx_x
#define GPU_BLOCK_IDX_Y                     hipBlockIdx_y
#define GPU_BLOCK_IDX_Z                     hipBlockIdx_z
#define GPU_BLOCK_DIM_X                     hipBlockDim_x
#define GPU_BLOCK_DIM_Y                     hipBlockDim_y
#define GPU_BLOCK_DIM_Z                     hipBlockDim_z
#define GPU_GRID_DIM_X                      hipGridDim_x
#define GPU_GRID_DIM_Y                      hipGridDim_y
#define GPU_GRID_DIM_Z                      hipGridDim_z

// Synchronization
#define GPU_SYNCTHREADS()                   __syncthreads()

// Atomic operations
#define GPU_ATOMIC_ADD(ptr, val)            atomicAdd(ptr, val)

// Math functions
#define GPU_RINTF(x)                        rintf(x)
#define GPU_ROUND(x)                        round(x)

// Vector types
#define GPU_MAKE_UINT4(x, y, z, w)          make_uint4(x, y, z, w)

// Type definitions
#ifdef __cplusplus
    #define gpuStream_t                         hipStream_t
    #define gpuError_t                          hipError_t
    #define gpuEvent_t                          hipEvent_t
    // Memory copy kinds
    #define gpuMemcpyKind                       hipMemcpyKind
#else
    #include <hip/hip_runtime_api.h>
    // For C files, use HIP's actual types (don't redefine)
    #define gpuStream_t                         hipStream_t
    #define gpuError_t                          hipError_t
    #define gpuEvent_t                          hipEvent_t
    
    // Memory copy kinds - use HIP's enum directly
    #define gpuMemcpyKind                       hipMemcpyKind
#endif

// Error codes
#define gpuSuccess                          hipSuccess

// Memory management
#define gpuMalloc                           hipMalloc
#define gpuMallocHost(ptr, size)            hipHostMalloc(ptr, size, hipHostMallocDefault)
#define gpuFree                             hipFree
#define gpuFreeHost                         hipHostFree
#define gpuMemcpy                           hipMemcpy
#define gpuMemcpyAsync                      hipMemcpyAsync
#define gpuMemcpy2DAsync                    hipMemcpy2DAsync
#define gpuMemset                           hipMemset
#define gpuMemsetAsync                      hipMemsetAsync
#define gpuHostRegister                     hipHostRegister
#define gpuHostUnregister                   hipHostUnregister

// Memory copy direction constants
#define gpuMemcpyHostToDevice               hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost               hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice             hipMemcpyDeviceToDevice
#define gpuMemcpyToSymbol                   hipMemcpyToSymbol
#define gpuMemcpyToSymbolAsync              hipMemcpyToSymbolAsync

// Host register flags
#define gpuHostRegisterDefault              hipHostRegisterDefault

// Stream management
#define gpuStreamDefault                    hipStreamDefault
#define gpuStreamSynchronize                hipStreamSynchronize

// Event management
#define gpuEventCreate                      hipEventCreate
#define gpuEventDestroy                     hipEventDestroy
#define gpuEventRecord                      hipEventRecord
#define gpuEventSynchronize                 hipEventSynchronize
#define gpuEventElapsedTime                 hipEventElapsedTime

// Device management
#define gpuSetDevice                        hipSetDevice
#define gpuGetDevice                        hipGetDevice
#define gpuGetDeviceCount                   hipGetDeviceCount
#define gpuGetDeviceProperties              hipGetDeviceProperties
#define gpuGetLastError                     hipGetLastError
#define gpuGetErrorString                   hipGetErrorString
#define gpuDeviceReset                      hipDeviceReset
#define gpuDriverGetVersion                 hipDriverGetVersion
#define gpuRuntimeGetVersion                hipRuntimeGetVersion

// Device properties
#define gpuDeviceProp                       hipDeviceProp_t

// Function attributes
#define gpuFuncSetCacheConfig               hipFuncSetCacheConfig
#define gpuFuncCachePreferShared            hipFuncCachePreferShared

// Runtime version
#ifndef GPUART_VERSION
#define GPUART_VERSION                      HIP_VERSION
#endif

// Kernel launch syntax (HIP also uses <<<>>> syntax)
#define GPU_KERNEL_LAUNCH(kernel, grid, block, smem, stream, ...) \
    kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)

// ==============================================================================
// SYCL Backend
// ==============================================================================
#elif defined(GPUJPEG_USE_SYCL)

#ifdef __cplusplus
#include <sycl/sycl.hpp>

// For SYCL, we need wrapper types since SYCL has a different architecture
// This is a simplified compatibility layer - full SYCL support would require
// more extensive refactoring

namespace gpujpeg_sycl {
    extern sycl::queue* default_queue;
}

#ifndef half
#define half sycl::half
#endif

#ifndef __float2half
#define __float2half(f) static_cast<half>(f)
#endif

#ifndef __half2float
#define __half2float(h) static_cast<float>(h)
#endif

// Kernel function qualifiers
#define GPU_DEVICE                          /* empty */
#define GPU_HOST                            /* empty */
#define GPU_GLOBAL                          /* empty */
#define GPU_SHARED                          /* empty */
#define GPU_CONSTANT                        /* empty */

// SYCL requires item parameter in kernels
#define GPU_KERNEL_ITEM_PARAM               sycl::nd_item<3> item
#define GPU_KERNEL_ITEM_ARG                 item
#define GPU_ITEM_COMMA                      ,

// Built-in variables - SYCL uses nd_item parameter
// Note: For SYCL kernels, 'item' parameter must be passed to kernel
#define GPU_THREAD_IDX_X                (item.get_local_id(2))
#define GPU_THREAD_IDX_Y                (item.get_local_id(1))
#define GPU_THREAD_IDX_Z                (item.get_local_id(0))
#define GPU_BLOCK_IDX_X                 (item.get_group(2))
#define GPU_BLOCK_IDX_Y                 (item.get_group(1))
#define GPU_BLOCK_IDX_Z                 (item.get_group(0))
#define GPU_BLOCK_DIM_X                 (item.get_local_range(2))
#define GPU_BLOCK_DIM_Y                 (item.get_local_range(1))
#define GPU_BLOCK_DIM_Z                 (item.get_local_range(0))
#define GPU_GRID_DIM_X                  (item.get_group_range(2))
#define GPU_GRID_DIM_Y                  (item.get_group_range(1))
#define GPU_GRID_DIM_Z                  (item.get_group_range(0))

// Synchronization
#define GPU_SYNCTHREADS()               item.barrier()

// Atomic operations
#define GPU_ATOMIC_ADD(ptr, val)            sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed, sycl::memory_scope::device>(*ptr).fetch_add(val)

// Math functions
#define GPU_RINTF(x)                    sycl::rint(x)
#define GPU_ROUND(x)                    sycl::round(x)

// Byte permutation function (CUDA __byte_perm equivalent)
// Permutes 4 bytes from x and y according to selector s
// Each nibble in s specifies which byte to select (0-3 from x, 4-7 from y-0x30)
inline uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t s) {
    uint32_t result = 0;
    for (int i = 0; i < 4; i++) {
        uint32_t selector = (s >> (i * 8)) & 0xF;
        uint32_t byte;
        if (selector <= 3) {
            byte = (x >> (selector * 8)) & 0xFF;
        } else {
            byte = (y >> ((selector - 4) * 8)) & 0xFF;
        }
        result |= (byte << (i * 8));
    }
    return result;
}

// Vector types - SYCL uses sycl::vec instead of CUDA vector types
#define GPU_MAKE_UINT4(x, y, z, w)      sycl::uint4(x, y, z, w)

// Define CUDA-style vector types for compatibility
struct uchar4 {
    uint8_t x, y, z, w;
    uchar4() = default;
    uchar4(uint8_t x_, uint8_t y_, uint8_t z_, uint8_t w_) : x(x_), y(y_), z(z_), w(w_) {}
};

struct int4 {
    int32_t x, y, z, w;
    int4() = default;
    int4(int32_t x_, int32_t y_, int32_t z_, int32_t w_) : x(x_), y(y_), z(z_), w(w_) {}
};

using uint4 = sycl::uint4;

// Kernel launch configuration types
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

// Item type for SYCL kernels
using sycl_item_t = sycl::nd_item<3>;

// Type definitions
typedef sycl::queue* gpuStream_t;
typedef int gpuError_t;
typedef void* gpuEvent_t;
#else
// C-compatible forward declarations
typedef void* gpuStream_t;
typedef int gpuError_t;
typedef void* gpuEvent_t;
#endif // __cplusplus

// Device properties structure
typedef struct gpuDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
} gpuDeviceProp;

// Error codes
#define gpuSuccess                          0

// Memory management functions - these need to be implemented as wrappers
#ifdef __cplusplus
extern "C" {
#endif

gpuError_t gpuMalloc(void** ptr, size_t size);
gpuError_t gpuMallocHost(void** ptr, size_t size);
gpuError_t gpuFree(void* ptr);
gpuError_t gpuFreeHost(void* ptr);
gpuError_t gpuMemcpy(void* dst, const void* src, size_t count, int kind);
gpuError_t gpuMemcpyAsync(void* dst, const void* src, size_t count, int kind, gpuStream_t stream);
gpuError_t gpuMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, 
                            size_t width, size_t height, int kind, gpuStream_t stream);
gpuError_t gpuMemcpyToSymbol(const void* symbol, const void* src, size_t count, 
                             size_t offset, int kind);
gpuError_t gpuMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, 
                                  size_t offset, int kind, gpuStream_t stream);
gpuError_t gpuMemset(void* ptr, int value, size_t count);
gpuError_t gpuMemsetAsync(void* ptr, int value, size_t count, gpuStream_t stream);
gpuError_t gpuHostRegister(void* ptr, size_t size, unsigned int flags);
gpuError_t gpuHostUnregister(void* ptr);

// Memory copy kinds
typedef enum gpuMemcpyKind {
    gpuMemcpyHostToDevice = 1,
    gpuMemcpyDeviceToHost = 2,
    gpuMemcpyDeviceToDevice = 3
} gpuMemcpyKind;

// Host register flags
enum {
    gpuHostRegisterDefault = 0
};

// Stream management
#ifdef __cplusplus
#define gpuStreamDefault                    (gpujpeg_sycl::default_queue)
#else
#define gpuStreamDefault                    ((gpuStream_t)0)
#endif
gpuError_t gpuStreamSynchronize(gpuStream_t stream);

// Event management
gpuError_t gpuEventCreate(gpuEvent_t* event);
gpuError_t gpuEventDestroy(gpuEvent_t event);
gpuError_t gpuEventRecord(gpuEvent_t event, gpuStream_t stream);
gpuError_t gpuEventSynchronize(gpuEvent_t event);
gpuError_t gpuEventElapsedTime(float* ms, gpuEvent_t start, gpuEvent_t end);

// Device management
gpuError_t gpuSetDevice(int device);
gpuError_t gpuGetDevice(int* device);
gpuError_t gpuGetDeviceCount(int* count);
gpuError_t gpuGetDeviceProperties(gpuDeviceProp* prop, int device);
gpuError_t gpuDeviceReset(void);
gpuError_t gpuDriverGetVersion(int* driverVersion);
gpuError_t gpuRuntimeGetVersion(int* runtimeVersion);
gpuError_t gpuGetLastError(void);
const char* gpuGetErrorString(gpuError_t error);

// Function attributes (no-op for SYCL)
#define gpuFuncSetCacheConfig(func, config)  (0)
#define gpuFuncCachePreferShared             0

// Runtime version
#ifndef GPUART_VERSION
#define GPUART_VERSION                      0
#endif

#ifdef __cplusplus
} // extern "C"

// Kernel launch - SYCL uses queue.submit with parallel_for
// Helper function for launching SYCL kernels
template<typename KernelFunc>
void sycl_launch_kernel(sycl::queue* q, dim3 grid, dim3 block, size_t smem, KernelFunc&& kernel) {
    q->submit([&](sycl::handler& cgh) {
        // Allocate local memory if needed
        if (smem > 0) {
            sycl::local_accessor<uint8_t, 1> local_mem(smem, cgh);
        }
        
        sycl::range<3> global_range(grid.z * block.z, grid.y * block.y, grid.x * block.x);
        sycl::range<3> local_range(block.z, block.y, block.x);
        
        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                        [=](sycl::nd_item<3> item) {
            kernel(item);
        });
    });
}

// Macro for kernel launch
#define GPU_KERNEL_LAUNCH(kernel, grid, block, smem, stream, ...) \
    sycl_launch_kernel(stream, grid, block, smem, \
        [=](sycl::nd_item<3> item) { \
            kernel(item, __VA_ARGS__); \
        })
#endif // __cplusplus

#endif // GPUJPEG_USE_SYCL

// ==============================================================================
// Common macros and utilities
// ==============================================================================

// Helper macro for checking GPU errors
#define GPUJPEG_CHECK_GPU(call, err_action) \
    do { \
        gpuError_t err = (call); \
        if (gpuSuccess != err) { \
            (void)fprintf(stderr, "[GPUJPEG] [Error] GPU call failed at %s:%d: %s\n", \
                    __FILE__, __LINE__, gpuGetErrorString(err)); \
            err_action; \
        } \
    } while (0)

// Stream type for public API (using void* for maximum compatibility)
#ifndef __DRIVER_TYPES_H__
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
#endif

#endif // GPUJPEG_DEVICE_COMPAT_H
