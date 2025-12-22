# Quick Reference: GPU Compatibility Layer

## For Developers Using GPUJPEG

### No Code Changes Required!
The compatibility layer is transparent. Your existing code using GPUJPEG will work with CUDA, HIP, or SYCL without modifications.

### Building for Different GPUs

**NVIDIA (CUDA) - Default:**
```bash
cmake -B build -S .
cmake --build build
```

**AMD (HIP):**
```bash
cmake -B build-hip -S . -DGPUJPEG_USE_HIP=ON
cmake --build build-hip
```

**Intel (SYCL):**
```bash
cmake -B build-sycl -S . -DGPUJPEG_USE_SYCL=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl
```

---

## For Developers Modifying GPUJPEG

### Use These Instead of CUDA Functions

| Don't Use | Use Instead | Why |
|-----------|-------------|-----|
| `cuda_runtime.h` | `gpujpeg_device_compat.h` | Backend abstraction |
| `cudaMalloc()` | `gpuMalloc()` | Multi-backend support |
| `cudaMemcpy()` | `gpuMemcpy()` | Multi-backend support |
| `cudaStream_t` | `gpuStream_t` | Type abstraction |
| `cudaEvent_t` | `gpuEvent_t` | Type abstraction |

### Quick API Reference

```c
// Include the compatibility header
#include "gpujpeg_device_compat.h"

// Memory allocation
gpuError_t err = gpuMalloc(&ptr, size);
gpuError_t err = gpuMallocHost(&ptr, size);

// Memory copy
gpuMemcpy(dst, src, size, gpuMemcpyHostToDevice);
gpuMemcpyAsync(dst, src, size, gpuMemcpyDeviceToHost, stream);

// Stream operations
gpuStream_t stream = gpuStreamDefault;
gpuStreamSynchronize(stream);

// Event timing
gpuEvent_t start, stop;
gpuEventCreate(&start);
gpuEventCreate(&stop);
gpuEventRecord(start, stream);
// ... do work ...
gpuEventRecord(stop, stream);
gpuEventSynchronize(stop);
float ms;
gpuEventElapsedTime(&ms, start, stop);

// Error checking
gpuError_t err = gpuGetLastError();
if (err != gpuSuccess) {
    printf("Error: %s\n", gpuGetErrorString(err));
}

// Free memory
gpuFree(ptr);
gpuFreeHost(host_ptr);
```

### Adding New GPU Code

**Option 1: Add to existing .cu file**
```cuda
// In src/my_kernel.cu
#include "gpujpeg_device_compat.h"

__global__ void my_kernel() {
    // Your kernel code
}
```

**Option 2: Create new .cu file**
1. Create `src/my_new_file.cu` with your CUDA code
2. Create `src/my_new_file.cpp`:
   ```cpp
   #include "gpujpeg_device_compat.h"
   #include "my_new_file.cu"
   ```
3. Add both to CMakeLists.txt

### Common Patterns

**Memory allocation with error checking:**
```c
uint8_t* d_buffer;
if (gpuMalloc(&d_buffer, size) != gpuSuccess) {
    fprintf(stderr, "GPU memory allocation failed\n");
    return -1;
}
```

**Timer pattern:**
```c
struct gpujpeg_timer {
    gpuEvent_t start;
    gpuEvent_t stop;
};

// Initialize
gpuEventCreate(&timer.start);
gpuEventCreate(&timer.stop);

// Use
gpuEventRecord(timer.start, stream);
// ... operations ...
gpuEventRecord(timer.stop, stream);

// Get duration
gpuEventSynchronize(timer.stop);
float ms;
gpuEventElapsedTime(&ms, timer.start, timer.stop);

// Cleanup
gpuEventDestroy(timer.start);
gpuEventDestroy(timer.stop);
```

---

## Files Overview

### Core Files (DO modify these)
- `src/*.c` - CPU code, uses compatibility layer
- `src/*.cu` - GPU kernels, uses compatibility layer
- `src/*.h` - Headers, use `gpu*` types

### Compatibility Layer (DON'T modify unless extending)
- `src/gpujpeg_device_compat.h` - Main header
- `src/gpujpeg_device_compat_sycl.cpp` - SYCL implementation
- `src/*_gpu.cpp` - Wrappers for .cu files

### Documentation (Read these)
- `GPU_COMPAT_README.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `CMakeLists.example.txt` - Build system example

---

## Troubleshooting

**Compiler errors about undefined CUDA types:**
```c
// Bad:
#include <cuda_runtime.h>

// Good:
#include "gpujpeg_device_compat.h"
```

**Linker errors with HIP:**
- Make sure you're using `hipcc` as compiler
- Check that .cpp wrappers are in CMakeLists.txt

**Runtime errors with SYCL:**
- SYCL kernels not implemented yet
- Only runtime API (memory, streams) works

---

## Need Help?

1. Check `GPU_COMPAT_README.md` for detailed docs
2. Check `IMPLEMENTATION_SUMMARY.md` for technical details
3. Look at existing code for examples
4. Create an issue with [GPU-COMPAT] tag

---

## Quick Checklist for New Code

- [ ] Include `gpujpeg_device_compat.h` instead of `cuda_runtime.h`
- [ ] Use `gpu*` functions instead of `cuda*`
- [ ] Use `gpuStream_t` instead of `cudaStream_t`
- [ ] Use `gpuEvent_t` instead of `cudaEvent_t`
- [ ] Test with CUDA backend first
- [ ] Create .cpp wrapper if adding new .cu file
- [ ] Update CMakeLists.txt with new files
- [ ] Document any backend-specific behavior

---

## Status at a Glance

| Feature | CUDA | HIP | SYCL |
|---------|:----:|:---:|:----:|
| Compiles | ✅ | ✅ | ✅ |
| Memory API | ✅ | ✅ | ✅ |
| Streams | ✅ | ✅ | ✅ |
| Events | ✅ | ✅ | ⚠️ |
| Kernels | ✅ | ✅ | ❌ |
| Tested | ✅ | ⚠️ | ❌ |

✅ = Fully working
⚠️ = Partially working / needs testing
❌ = Not yet implemented
