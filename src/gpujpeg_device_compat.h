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
    // Type definitions
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

#include <CL/sycl.hpp>

// For SYCL, we need wrapper types since SYCL has a different architecture
// This is a simplified compatibility layer - full SYCL support would require
// more extensive refactoring

namespace gpujpeg_sycl {
    extern sycl::queue* default_queue;
}

// Type definitions
typedef sycl::queue* gpuStream_t;
typedef int gpuError_t;
typedef void* gpuEvent_t;

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

// Memory copy kinds
enum {
    gpuMemcpyHostToDevice = 1,
    gpuMemcpyDeviceToHost = 2,
    gpuMemcpyDeviceToDevice = 3
};

// Stream management
#define gpuStreamDefault                    (gpujpeg_sycl::default_queue)
gpuError_t gpuStreamSynchronize(gpuStream_t stream);

// Event management
gpuError_t gpuEventCreate(gpuEvent_t* event);
gpuError_t gpuEventDestroy(gpuEvent_t event);
gpuError_t gpuEventRecord(gpuEvent_t event, gpuStream_t stream);
gpuError_t gpuEventSynchronize(gpuEvent_t event);
gpuError_t gpuEventElapsedTime(float* ms, gpuEvent_t start, gpuEvent_t end);

// Device management
gpuError_t gpuSetDevice(int device);
gpuError_t gpuGetLastError(void);
const char* gpuGetErrorString(gpuError_t error);

// Function attributes (no-op for SYCL)
#define gpuFuncSetCacheConfig(func, config)  (0)
#define gpuFuncCachePreferShared             0

// Runtime version
#ifndef GPUART_VERSION
#define GPUART_VERSION                      0
#endif

// Kernel launch - SYCL requires submit syntax, this is a placeholder
#define GPU_KERNEL_LAUNCH(kernel, grid, block, smem, stream, ...) \
    /* SYCL kernel launch would need significant refactoring */

#ifdef __cplusplus
}
#endif

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
