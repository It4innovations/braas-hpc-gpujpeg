/**
 * @file
 * SYCL compatibility layer implementation for GPUJPEG
 */

#include "gpujpeg_device_compat.h"

#ifdef GPUJPEG_USE_SYCL

#include <sycl/sycl.hpp>
#include <map>
#include <chrono>
#include <cstring>

namespace gpujpeg_sycl {
    sycl::queue* default_queue = nullptr;
    
    // Simple event tracking for timing
    struct EventData {
        std::chrono::high_resolution_clock::time_point timestamp;
    };
    
    std::map<gpuEvent_t, EventData> event_map;
}

extern "C" {

gpuError_t gpuMalloc(void** ptr, size_t size) {
    try {
        if (!gpujpeg_sycl::default_queue) {
            gpujpeg_sycl::default_queue = new sycl::queue(sycl::default_selector_v);
        }
        *ptr = sycl::malloc_device(size, *gpujpeg_sycl::default_queue);
        return *ptr ? gpuSuccess : -1;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMallocHost(void** ptr, size_t size) {
    try {
        if (!gpujpeg_sycl::default_queue) {
            gpujpeg_sycl::default_queue = new sycl::queue(sycl::default_selector_v);
        }
        *ptr = sycl::malloc_host(size, *gpujpeg_sycl::default_queue);
        return *ptr ? gpuSuccess : -1;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuFree(void* ptr) {
    try {
        if (gpujpeg_sycl::default_queue && ptr) {
            sycl::free(ptr, *gpujpeg_sycl::default_queue);
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuFreeHost(void* ptr) {
    try {
        if (gpujpeg_sycl::default_queue && ptr) {
            sycl::free(ptr, *gpujpeg_sycl::default_queue);
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMemcpy(void* dst, const void* src, size_t count, int kind) {
    try {
        if (!gpujpeg_sycl::default_queue) {
            gpujpeg_sycl::default_queue = new sycl::queue(sycl::default_selector_v);
        }
        gpujpeg_sycl::default_queue->memcpy(dst, src, count).wait();
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMemset(void* ptr, int value, size_t count) {
    try {
        if (!gpujpeg_sycl::default_queue) {
            gpujpeg_sycl::default_queue = new sycl::queue(sycl::default_selector_v);
        }
        gpujpeg_sycl::default_queue->memset(ptr, value, count).wait();
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMemsetAsync(void* ptr, int value, size_t count, gpuStream_t stream) {
    try {
        sycl::queue* q = stream ? stream : gpujpeg_sycl::default_queue;
        if (!q) {
            return -1;
        }
        q->memset(ptr, value, count);
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMemcpyAsync(void* dst, const void* src, size_t count, int kind, gpuStream_t stream) {
    try {
        sycl::queue* q = stream ? stream : gpujpeg_sycl::default_queue;
        if (!q) {
            return -1;
        }
        q->memcpy(dst, src, count);
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuHostRegister(void* ptr, size_t size, unsigned int flags) {
    // SYCL doesn't require explicit host memory registration
    // Host memory allocated with malloc can be used directly
    return gpuSuccess;
}

gpuError_t gpuHostUnregister(void* ptr) {
    // SYCL doesn't require explicit host memory unregistration
    return gpuSuccess;
}

gpuError_t gpuMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, 
                            size_t width, size_t height, int kind, gpuStream_t stream) {
    try {
        sycl::queue* q = stream ? stream : gpujpeg_sycl::default_queue;
        if (!q) {
            return -1;
        }
        
        // Perform 2D copy row by row
        // Need to capture offset values, not computed pointers, for async operations
        const uint8_t* src_base = static_cast<const uint8_t*>(src);
        uint8_t* dst_base = static_cast<uint8_t*>(dst);
        
        for (size_t row = 0; row < height; ++row) {
            size_t src_offset = row * spitch;
            size_t dst_offset = row * dpitch;
            q->memcpy(dst_base + dst_offset, src_base + src_offset, width);
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMemcpyToSymbol(const void* symbol, const void* src, size_t count, 
                             size_t offset, int kind) {
    try {
        if (!gpujpeg_sycl::default_queue) {
            gpujpeg_sycl::default_queue = new sycl::queue(sycl::default_selector_v);
        }
        
        // In SYCL, "symbols" are just pointers to device memory
        // Cast away const since we're initializing the memory
        void* dst = const_cast<void*>(static_cast<const void*>(static_cast<const char*>(symbol) + offset));
        
        // Perform the copy based on the kind
        if (kind == gpuMemcpyHostToDevice) {
            gpujpeg_sycl::default_queue->memcpy(dst, src, count).wait();
        } else if (kind == gpuMemcpyDeviceToDevice) {
            gpujpeg_sycl::default_queue->memcpy(dst, src, count).wait();
        } else {
            return -1;
        }
        
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, 
                                  size_t offset, int kind, gpuStream_t stream) {
    try {
        sycl::queue* q = stream ? stream : gpujpeg_sycl::default_queue;
        if (!q) {
            return -1;
        }
        
        // In SYCL, "symbols" are just pointers to device memory
        // Cast away const since we're initializing the memory
        void* dst = const_cast<void*>(static_cast<const void*>(static_cast<const char*>(symbol) + offset));
        
        // Perform async copy based on the kind (no wait)
        if (kind == gpuMemcpyHostToDevice || kind == gpuMemcpyDeviceToDevice) {
            q->memcpy(dst, src, count);
        } else {
            return -1;
        }
        
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuStreamSynchronize(gpuStream_t stream) {
    try {
        if (stream) {
            stream->wait();
        } else if (gpujpeg_sycl::default_queue) {
            gpujpeg_sycl::default_queue->wait();
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuEventCreate(gpuEvent_t* event) {
    try {
        *event = new gpujpeg_sycl::EventData();
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuEventDestroy(gpuEvent_t event) {
    try {
        if (event) {
            delete static_cast<gpujpeg_sycl::EventData*>(event);
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuEventRecord(gpuEvent_t event, gpuStream_t stream) {
    try {
        if (!event) {
            return -1;
        }
        
        // Record timestamp after synchronizing with stream
        sycl::queue* q = stream ? stream : gpujpeg_sycl::default_queue;
        if (q) {
            q->wait();
        }
        
        auto* evt_data = static_cast<gpujpeg_sycl::EventData*>(event);
        evt_data->timestamp = std::chrono::high_resolution_clock::now();
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuEventSynchronize(gpuEvent_t event) {
    // Events are recorded after synchronization, so nothing to do
    return gpuSuccess;
}

gpuError_t gpuEventElapsedTime(float* ms, gpuEvent_t start, gpuEvent_t end) {
    try {
        if (!ms || !start || !end) {
            return -1;
        }
        
        auto* start_data = static_cast<gpujpeg_sycl::EventData*>(start);
        auto* end_data = static_cast<gpujpeg_sycl::EventData*>(end);
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_data->timestamp - start_data->timestamp);
        *ms = duration.count() / 1000.0f;
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuSetDevice(int device) {
    try {
        // For SYCL, device selection is more complex and would need proper implementation
        // This is a simplified placeholder
        if (gpujpeg_sycl::default_queue) {
            delete gpujpeg_sycl::default_queue;
        }
        gpujpeg_sycl::default_queue = new sycl::queue(sycl::default_selector_v);
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuGetDevice(int* device) {
    try {
        if (!device) {
            return -1;
        }
        // Return 0 as default device
        *device = 0;
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuGetDeviceCount(int* count) {
    try {
        if (!count) {
            return -1;
        }
        // Count available GPU devices
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        *count = static_cast<int>(devices.size());
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuGetDeviceProperties(gpuDeviceProp* prop, int device) {
    try {
        if (!prop) {
            return -1;
        }
        
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (device < 0 || device >= static_cast<int>(devices.size())) {
            return -1;
        }
        
        auto& dev = devices[device];
        
        // Fill device properties
        std::string name = dev.get_info<sycl::info::device::name>();
        strncpy(prop->name, name.c_str(), sizeof(prop->name) - 1);
        prop->name[sizeof(prop->name) - 1] = '\0';
        
        prop->totalGlobalMem = dev.get_info<sycl::info::device::global_mem_size>();
        prop->sharedMemPerBlock = dev.get_info<sycl::info::device::local_mem_size>();
        prop->maxThreadsPerBlock = dev.get_info<sycl::info::device::max_work_group_size>();
        
        auto max_work_item_sizes = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
        for (int i = 0; i < 3; ++i) {
            prop->maxThreadsDim[i] = static_cast<int>(max_work_item_sizes[i]);
        }
        
        prop->clockRate = dev.get_info<sycl::info::device::max_clock_frequency>() * 1000; // Convert to kHz
        prop->multiProcessorCount = dev.get_info<sycl::info::device::max_compute_units>();
        
        // Set defaults for properties not directly available in SYCL
        prop->regsPerBlock = 65536; // Approximate default
        prop->warpSize = 32; // Typical warp size
        prop->memPitch = SIZE_MAX;
        prop->maxGridSize[0] = prop->maxGridSize[1] = prop->maxGridSize[2] = 65535;
        prop->totalConstMem = 64 * 1024; // 64KB typical
        prop->major = 1;
        prop->minor = 0;
        prop->textureAlignment = 512;
        prop->deviceOverlap = 1;
        
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuGetLastError(void) {
    // SYCL uses exceptions, not error codes
    return gpuSuccess;
}

gpuError_t gpuDeviceReset(void) {
    try {
        if (gpujpeg_sycl::default_queue) {
            gpujpeg_sycl::default_queue->wait();
            delete gpujpeg_sycl::default_queue;
            gpujpeg_sycl::default_queue = nullptr;
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuDriverGetVersion(int* driverVersion) {
    try {
        if (!driverVersion) {
            return -1;
        }
        // SYCL doesn't have a direct driver version query
        // Return a placeholder version
        *driverVersion = 1000; // 1.0
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuRuntimeGetVersion(int* runtimeVersion) {
    try {
        if (!runtimeVersion) {
            return -1;
        }
        // Return SYCL version as runtime version
        *runtimeVersion = SYCL_LANGUAGE_VERSION;
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

const char* gpuGetErrorString(gpuError_t error) {
    if (error == gpuSuccess) {
        return "success";
    }
    return "SYCL error";
}

} // extern "C"

#endif // GPUJPEG_USE_SYCL
