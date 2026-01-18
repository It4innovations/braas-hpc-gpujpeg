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

// (Un)comment to enable kernel launch logging
#define GPUJPEG_DEBUG_KERNEL_LAUNCH

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

// Warp/Sub-group operations
#define GPU_WARP_SIZE                       32
#define GPU_SUB_GROUP_BARRIER(item)         /* implicit in CUDA warps */
#define GPU_REQD_SUB_GROUP_SIZE(size)       /* not needed for CUDA */

// Atomic operations
#define GPU_ATOMIC_ADD(ptr, val)            atomicAdd(ptr, val)

// Math functions
#define GPU_RINTF(x)                        rintf(x)
#define GPU_ROUND(x)                        round(x)

// Vector types
#define GPU_MAKE_UINT4(x, y, z, w)          make_uint4(x, y, z, w)

// Warp collective functions
#ifndef FULL_MASK
#define FULL_MASK 0xFFFFFFFF
#endif

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

// Helper to get dim3 components (works with both dim3 and scalar)
// C++ inline functions instead of __builtin_types_compatible_p which doesn't work in C++
#ifdef __cplusplus
static inline unsigned int GPU_DIM_X(const dim3& d) { return d.x; }
static inline unsigned int GPU_DIM_X(unsigned int d) { return d; }
static inline unsigned int GPU_DIM_Y(const dim3& d) { return d.y; }
static inline unsigned int GPU_DIM_Y(unsigned int /*d*/) { return 1u; }
static inline unsigned int GPU_DIM_Z(const dim3& d) { return d.z; }
static inline unsigned int GPU_DIM_Z(unsigned int /*d*/) { return 1u; }
#else
#define GPU_DIM_X(d) (__builtin_types_compatible_p(__typeof__(d), dim3) ? (d).x : (unsigned int)(d))
#define GPU_DIM_Y(d) (__builtin_types_compatible_p(__typeof__(d), dim3) ? (d).y : 1u)
#define GPU_DIM_Z(d) (__builtin_types_compatible_p(__typeof__(d), dim3) ? (d).z : 1u)
#endif

#ifdef GPUJPEG_DEBUG_KERNEL_LAUNCH
#define GPU_KERNEL_LAUNCH(kernel, grid, block, smem, stream, ...) \
    do { \
        GPUJPEG_KERNEL_LAUNCH_LOG(kernel, grid, block, smem, stream); \
        gpuEvent_t _start_event, _stop_event; \
        gpuEventCreate(&_start_event); \
        gpuEventCreate(&_stop_event); \
        gpuEventRecord(_start_event, stream); \
        kernel<<<grid, block, smem, stream>>>(__VA_ARGS__); \
        gpuEventRecord(_stop_event, stream); \
        gpuStreamSynchronize(stream); \
        float _elapsed_ms = 0; \
        gpuEventElapsedTime(&_elapsed_ms, _start_event, _stop_event); \
        fprintf(stderr, "[CUDA] Kernel %s took %.3f ms\n", #kernel, _elapsed_ms); \
        gpuEventDestroy(_start_event); \
        gpuEventDestroy(_stop_event); \
    } while(0)
#else

// Kernel launch syntax (CUDA uses <<<>>> syntax natively)
#define GPU_KERNEL_LAUNCH(kernel, grid, block, smem, stream, ...) \
    do { \
        GPUJPEG_KERNEL_LAUNCH_LOG(kernel, grid, block, smem, stream); \
        kernel<<<grid, block, smem, stream>>>(__VA_ARGS__); \
    } while(0)

#endif

// For CUDA/HIP, shared memory parameter is not needed
#define GPU_SHARED_MEM_PARAM /* empty */
#define GPU_SHARED_MEM_ARG /* empty */
#define GPU_SHARED_PTR(type, name, offset) /* empty - name already declared as GPU_SHARED array */


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

// Warp/Sub-group operations
#define GPU_WARP_SIZE                       64  // AMD uses wavefront size 64 by default
#define GPU_SUB_GROUP_BARRIER(item)         /* implicit in HIP wavefronts */
#define GPU_REQD_SUB_GROUP_SIZE(size)       /* not needed for HIP */

// Atomic operations
#define GPU_ATOMIC_ADD(ptr, val)            atomicAdd(ptr, val)

// Math functions
#define GPU_RINTF(x)                        rintf(x)
#define GPU_ROUND(x)                        round(x)

// Vector types
#define GPU_MAKE_UINT4(x, y, z, w)          make_uint4(x, y, z, w)

// Warp collective functions
#ifndef FULL_MASK
#define FULL_MASK 0xFFFFFFFF
#endif

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

// Helper to get dim3 components (works with both dim3 and scalar)
// C++ inline functions instead of __builtin_types_compatible_p which doesn't work in C++
#ifdef __cplusplus
static inline unsigned int GPU_DIM_X(const dim3& d) { return d.x; }
static inline unsigned int GPU_DIM_X(unsigned int d) { return d; }
static inline unsigned int GPU_DIM_Y(const dim3& d) { return d.y; }
static inline unsigned int GPU_DIM_Y(unsigned int d) { return 1u; }
static inline unsigned int GPU_DIM_Z(const dim3& d) { return d.z; }
static inline unsigned int GPU_DIM_Z(unsigned int d) { return 1u; }
#else
#define GPU_DIM_X(d) (__builtin_types_compatible_p(__typeof__(d), dim3) ? (d).x : (unsigned int)(d))
#define GPU_DIM_Y(d) (__builtin_types_compatible_p(__typeof__(d), dim3) ? (d).y : 1u)
#define GPU_DIM_Z(d) (__builtin_types_compatible_p(__typeof__(d), dim3) ? (d).z : 1u)
#endif

#ifdef GPUJPEG_DEBUG_KERNEL_LAUNCH
// Kernel launch syntax (HIP also uses <<<>>> syntax)
#define GPU_KERNEL_LAUNCH(kernel, grid, block, smem, stream, ...) \
    do { \
        GPUJPEG_KERNEL_LAUNCH_LOG(kernel, grid, block, smem, stream); \
        gpuEvent_t _start_event, _stop_event; \
        gpuEventCreate(&_start_event); \
        gpuEventCreate(&_stop_event); \
        gpuEventRecord(_start_event, stream); \
        kernel<<<grid, block, smem, stream>>>(__VA_ARGS__); \
        gpuEventRecord(_stop_event, stream); \
        gpuStreamSynchronize(stream); \
        float _elapsed_ms = 0; \
        gpuEventElapsedTime(&_elapsed_ms, _start_event, _stop_event); \
        fprintf(stderr, "[HIP] Kernel %s took %.3f ms\n", #kernel, _elapsed_ms); \
        gpuEventDestroy(_start_event); \
        gpuEventDestroy(_stop_event); \
    } while(0)
#else

// Kernel launch syntax (HIP also uses <<<>>> syntax)
#define GPU_KERNEL_LAUNCH(kernel, grid, block, smem, stream, ...) \
    do { \
        GPUJPEG_KERNEL_LAUNCH_LOG(kernel, grid, block, smem, stream); \
        kernel<<<grid, block, smem, stream>>>(__VA_ARGS__); \
    } while(0)

#endif

// For CUDA/HIP, shared memory parameter is not needed
#define GPU_SHARED_MEM_PARAM /* empty */
#define GPU_SHARED_MEM_ARG /* empty */
#define GPU_SHARED_PTR(type, name, offset) /* empty - name already declared as GPU_SHARED array */


// ==============================================================================
// SYCL Backend
// ==============================================================================
#elif defined(GPUJPEG_USE_SYCL)

#ifdef __cplusplus
#include <sycl/sycl.hpp>

// For SYCL, we need wrapper types since SYCL has a different architecture
// This is a simplified compatibility layer - full SYCL support would require
// more extensive refactoring

#ifndef half
#define half sycl::half
#endif

#ifndef __umulhi
#define __umulhi sycl::mul_hi
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
#define GPU_SYNCTHREADS()               item.barrier(sycl::access::fence_space::local_space)

// Sub-group (warp) operations for SYCL
#define GPU_WARP_SIZE                       32  // Assume 32 for compatibility, may vary by device
#define GPU_SUB_GROUP_BARRIER(item)         do { \
                                                sycl::sub_group sg = item.get_sub_group(); \
                                                sycl::group_barrier(sg, sycl::memory_scope::sub_group); \
                                            } while(0)
#define GPU_REQD_SUB_GROUP_SIZE(size)       [[intel::reqd_sub_group_size(size)]]

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

#if 0
// Sub-group (warp) collective functions for SYCL
// __ballot_sync equivalent - returns bitmask of predicate across sub-group
inline uint32_t __ballot_sync(uint32_t mask, bool predicate) {
    sycl::sub_group sg = sycl::ext::oneapi::experimental::this_sub_group();
    sycl::vec<uint32_t, 1> ballot_result = sycl::group_ballot(sg, predicate);
    return ballot_result[0];
}
#endif

// __clz (count leading zeros) - SYCL equivalent
inline int __clz(uint32_t x) {
    return sycl::clz(x);
}

#ifndef FULL_MASK
#define FULL_MASK 0xFFFFFFFF
#endif


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

gpuError_t gpuMalloc(void** ptr, size_t size);
gpuError_t gpuMallocHost(void** ptr, size_t size);
gpuError_t gpuFree(void* ptr);
gpuError_t gpuFreeHost(void* ptr);
gpuError_t gpuMemcpy(void* dst, const void* src, size_t count, int kind);
gpuError_t gpuMemcpyAsync(void* dst, const void* src, size_t count, int kind, gpuStream_t stream);
gpuError_t gpuMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, 
                            size_t width, size_t height, int kind, gpuStream_t stream);

// gpuMemcpyToSymbol and gpuMemcpyToSymbolAsync are not supported in SYCL
// Use regular gpuMemcpy instead with device pointers
#define gpuMemcpyToSymbol(symbol, src, count, offset, kind) \
    _Pragma("GCC error \"gpuMemcpyToSymbol is not supported for SYCL. Use gpuMemcpy with device pointers instead.\"")
#define gpuMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream) \
    _Pragma("GCC error \"gpuMemcpyToSymbolAsync is not supported for SYCL. Use gpuMemcpyAsync with device pointers instead.\"")

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
#define gpuStreamDefault                    (gpujpeg_sycl::get_current_queue())
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

// Forward declare from gpujpeg_sycl namespace
namespace gpujpeg_sycl {
    // Async exception handler declaration
    extern sycl::async_handler async_handler;

    // Get the queue for the current device
    sycl::queue* get_current_queue();
}

// Kernel launch - SYCL uses queue.submit with parallel_for
// Helper to convert grid/block to dim3 (handles both dim3 and scalar)
inline dim3 to_dim3(const dim3& d) { return d; }
inline dim3 to_dim3(unsigned int v) { return dim3(v, 1, 1); }
inline dim3 to_dim3(int v) { return dim3(v, 1, 1); }


// Macro to declare shared memory parameter for SYCL kernels
#define GPU_SHARED_MEM_PARAM , sycl::local_accessor<uint8_t, 1> _sycl_shared_mem

// Macro to pass shared memory argument when calling device functions
#define GPU_SHARED_MEM_ARG , _sycl_shared_mem

// Macro to get typed pointer to shared memory for SYCL
#define GPU_SHARED_PTR(type, name, offset) \
    type* name = reinterpret_cast<type*>(_sycl_shared_mem.get_multi_ptr<sycl::access::decorated::no>().get() + (offset))

#endif // GPUJPEG_USE_SYCL

// ==============================================================================
// Common macros and utilities
// ==============================================================================

// Conditional kernel launch logging
#ifdef GPUJPEG_DEBUG_KERNEL_LAUNCH
    #if defined(GPUJPEG_USE_CUDA) || defined(GPUJPEG_USE_HIP)
        #define GPUJPEG_KERNEL_LAUNCH_LOG(kernel, grid, block, smem, stream) \
            fprintf(stderr, "[GPU_KERNEL_LAUNCH] %s: grid=(%u,%u,%u) block=(%u,%u,%u) smem=%zu stream=%p\n", \
                    #kernel, GPU_DIM_X(grid), GPU_DIM_Y(grid), GPU_DIM_Z(grid), \
                    GPU_DIM_X(block), GPU_DIM_Y(block), GPU_DIM_Z(block), (size_t)(smem), (void*)(stream))
    #elif defined(GPUJPEG_USE_SYCL)
        #define GPUJPEG_KERNEL_LAUNCH_LOG(kernel, grid, block, smem, stream) \
            do { \
                dim3 _log_grid = to_dim3(grid); \
                dim3 _log_block = to_dim3(block); \
                fprintf(stderr, "[GPU_KERNEL_LAUNCH] %s: grid=(%u,%u,%u) block=(%u,%u,%u) smem=%zu stream=%p\n", \
                        #kernel, _log_grid.x, _log_grid.y, _log_grid.z, _log_block.x, _log_block.y, _log_block.z, \
                        (size_t)(smem), (void*)(stream)); \
            } while(0)
    #endif
#else
    #define GPUJPEG_KERNEL_LAUNCH_LOG(kernel, grid, block, smem, stream) /* disabled */
#endif

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

// Debug macro to download device data and print it
// Usage: GPUJPEG_DEBUG_PRINT_DEVICE_DATA(device_ptr, float, 10);
#ifdef GPUJPEG_DEBUG_KERNEL_LAUNCH
    #define GPUJPEG_DEBUG_PRINT_DEVICE_DATA(ptr, type, count) \
        do { \
            type* _host_data = (type*)malloc((count) * sizeof(type)); \
            if (_host_data) { \
                gpuError_t _err = gpuMemcpy(_host_data, ptr, (count) * sizeof(type), gpuMemcpyDeviceToHost); \
                if (_err == gpuSuccess) { \
                    fprintf(stderr, "[GPUJPEG_DEBUG] Device data at %p (%s[%d]):\n", (void*)(ptr), #type, (int)(count)); \
                    const char* _type_str = #type; \
                    for (int _i = 0; _i < (int)(count); _i++) { \
                        fprintf(stderr, "  [%d] = ", _i); \
                        if (strcmp(_type_str, "float") == 0) { \
                            fprintf(stderr, "%f\n", (double)_host_data[_i]); \
                        } else if (strcmp(_type_str, "double") == 0) { \
                            fprintf(stderr, "%f\n", _host_data[_i]); \
                        } else if (strcmp(_type_str, "int") == 0) { \
                            fprintf(stderr, "%d\n", _host_data[_i]); \
                        } else if (strcmp(_type_str, "unsigned int") == 0 || strcmp(_type_str, "unsigned") == 0 || strcmp(_type_str, "uint32_t") == 0) { \
                            fprintf(stderr, "%u\n", _host_data[_i]); \
                        } else if (strcmp(_type_str, "char") == 0 || strcmp(_type_str, "int8_t") == 0) { \
                            fprintf(stderr, "%d\n", (int)_host_data[_i]); \
                        } else if (strcmp(_type_str, "unsigned char") == 0 || strcmp(_type_str, "uint8_t") == 0) { \
                            fprintf(stderr, "%u\n", (unsigned int)_host_data[_i]); \
                        } else if (strcmp(_type_str, "short") == 0 || strcmp(_type_str, "int16_t") == 0) { \
                            fprintf(stderr, "%d\n", (int)_host_data[_i]); \
                        } else if (strcmp(_type_str, "unsigned short") == 0 || strcmp(_type_str, "uint16_t") == 0) { \
                            fprintf(stderr, "%u\n", (unsigned int)_host_data[_i]); \
                        } else if (strcmp(_type_str, "long") == 0 || strcmp(_type_str, "int64_t") == 0) { \
                            fprintf(stderr, "%ld\n", (long)_host_data[_i]); \
                        } else if (strcmp(_type_str, "unsigned long") == 0 || strcmp(_type_str, "uint64_t") == 0) { \
                            fprintf(stderr, "%lu\n", (unsigned long)_host_data[_i]); \
                        } else { \
                            fprintf(stderr, "0x"); \
                            for (size_t _b = 0; _b < sizeof(type); _b++) { \
                                fprintf(stderr, "%02x", ((unsigned char*)&_host_data[_i])[_b]); \
                            } \
                            fprintf(stderr, "\n"); \
                        } \
                    } \
                } else { \
                    fprintf(stderr, "[GPUJPEG_DEBUG] Failed to copy device data: %s\n", gpuGetErrorString(_err)); \
                } \
                free(_host_data); \
            } else { \
                fprintf(stderr, "[GPUJPEG_DEBUG] Failed to allocate host memory for debug print\n"); \
            } \
        } while (0)
#else
    #define GPUJPEG_DEBUG_PRINT_DEVICE_DATA(ptr, type, count) /* disabled */
#endif

// SYCL kernel timing macro (only for SYCL backend)
#ifdef GPUJPEG_USE_SYCL
    #ifdef GPUJPEG_DEBUG_KERNEL_LAUNCH
        #define GPUJPEG_SYCL_KERNEL_WAIT_AND_PROFILE(sycl_event, start_time, kernel_name) \
            do { \
                (sycl_event).wait(); \
                auto _end_time = std::chrono::high_resolution_clock::now(); \
                auto _elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(_end_time - (start_time)).count(); \
                fprintf(stderr, "[SYCL] Kernel %s took %.3f ms\n", kernel_name, _elapsed_us / 1000.0); \
            } while(0)
    #else
        #define GPUJPEG_SYCL_KERNEL_WAIT_AND_PROFILE(sycl_event, start_time, kernel_name) /* disabled */
    #endif
#endif

// Stream type for public API (using void* for maximum compatibility)
#ifndef __DRIVER_TYPES_H__
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
#endif

#endif // GPUJPEG_DEVICE_COMPAT_H
