/**
 * @file
 * SYCL compatibility layer implementation for GPUJPEG
 */

#include "gpujpeg_device_compat.h"

#ifdef GPUJPEG_USE_SYCL

#include <CL/sycl.hpp>
#include <map>
#include <chrono>

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

gpuError_t gpuMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, 
                            size_t width, size_t height, int kind, gpuStream_t stream) {
    try {
        sycl::queue* q = stream ? stream : gpujpeg_sycl::default_queue;
        if (!q) {
            return -1;
        }
        
        // Perform 2D copy row by row
        for (size_t row = 0; row < height; ++row) {
            const uint8_t* src_row = static_cast<const uint8_t*>(src) + row * spitch;
            uint8_t* dst_row = static_cast<uint8_t*>(dst) + row * dpitch;
            q->memcpy(dst_row, src_row, width);
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMemcpyToSymbol(const void* symbol, const void* src, size_t count, 
                             size_t offset, int kind) {
    // SYCL doesn't have direct symbol support - would need refactoring
    return -1;
}

gpuError_t gpuMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, 
                                  size_t offset, int kind, gpuStream_t stream) {
    // SYCL doesn't have direct symbol support - would need refactoring
    return -1;
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

gpuError_t gpuGetLastError(void) {
    // SYCL uses exceptions, not error codes
    return gpuSuccess;
}

const char* gpuGetErrorString(gpuError_t error) {
    if (error == gpuSuccess) {
        return "success";
    }
    return "SYCL error";
}

} // extern "C"

#endif // GPUJPEG_USE_SYCL
