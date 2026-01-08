/**
 * @file
 * SYCL compatibility layer implementation for GPUJPEG
 */

#include "gpujpeg_device_compat.h"

#ifdef GPUJPEG_USE_SYCL

#include <sycl/sycl.hpp>
#include <map>
#include <vector>
#include <chrono>
#include <cstring>
#include <iostream>

namespace gpujpeg_sycl {
    std::vector<sycl::device> cached_devices;
    std::vector<sycl::queue*> device_queues;
    int current_device = 0;
    bool devices_cached = false;
    
    // Define async exception handler as a function (can be used as sycl::async_handler)
    void async_handler_impl(sycl::exception_list exceptions) {
        for (const auto& ex_ptr : exceptions) {
            try {
                std::rethrow_exception(ex_ptr);
            } catch (const sycl::exception& ex) {
                std::cerr << "\n[ASYNC SYCL EXCEPTION]\n"
                          << "  what(): " << ex.what() << "\n"
                          << "  code(): " << ex.code().value()
                          << std::endl;
            } catch (const std::exception& ex) {
                std::cerr << "\n[ASYNC std::exception]\n"
                          << ex.what() << std::endl;
            }
        }
    }
    
    // Export as sycl::async_handler
    sycl::async_handler async_handler = async_handler_impl;
    
    // Helper function to create a queue with appropriate properties
    // Returns the created queue or nullptr on error
    sycl::queue* create_queue(const sycl::device& device) {
        try {
            std::cout << "[SYCL] Creating queue for device: " 
                      << device.get_info<sycl::info::device::name>()
                      << " (" << (device.is_gpu() ? "GPU" : device.is_cpu() ? "CPU" : "Accelerator") << ")"
                      << std::endl;

            // Use in_order queue with profiling enabled for accurate event timing
            sycl::property_list props = {sycl::property::queue::in_order(), 
                                         sycl::property::queue::enable_profiling()};
            
            return new sycl::queue(device, gpujpeg_sycl::async_handler, props);
        } catch (const std::exception& e) {
            std::cerr << "[SYCL ERROR] Failed to create queue: " << e.what() << std::endl;
            return nullptr;
        } catch (...) {
            std::cerr << "[SYCL ERROR] Failed to create queue: unknown error" << std::endl;
            return nullptr;
        }
    }
    
    // Helper function to get all available devices (respects ONEAPI_DEVICE_SELECTOR)
    const std::vector<sycl::device>& get_devices() {
        if (!devices_cached) {
            try {
                // Get all devices (GPU and CPU) to support ONEAPI_DEVICE_SELECTOR
                // Try GPU devices first
                auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
                cached_devices.insert(cached_devices.end(), gpu_devices.begin(), gpu_devices.end());
                
                // Also try CPU devices
                auto cpu_devices = sycl::device::get_devices(sycl::info::device_type::cpu);
                cached_devices.insert(cached_devices.end(), cpu_devices.begin(), cpu_devices.end());
                
                // Also try accelerator devices
                auto accel_devices = sycl::device::get_devices(sycl::info::device_type::accelerator);
                cached_devices.insert(cached_devices.end(), accel_devices.begin(), accel_devices.end());
                
                devices_cached = true;
                
                std::cerr << "[SYCL] Found " << cached_devices.size() << " device(s):" << std::endl;
                
                // Create a queue for each device
                device_queues.resize(cached_devices.size(), nullptr);
                for (size_t i = 0; i < cached_devices.size(); ++i) {
                    std::cerr << "  [" << i << "] " << cached_devices[i].get_info<sycl::info::device::name>()
                              << " (" << (cached_devices[i].is_gpu() ? "GPU" : 
                                         cached_devices[i].is_cpu() ? "CPU" : "Accelerator") << ")" << std::endl;
                    
                    device_queues[i] = create_queue(cached_devices[i]);
                    if (!device_queues[i]) {
                        std::cerr << "[SYCL ERROR] Failed to create queue for device " << i << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[SYCL] Error enumerating devices: " << e.what() << std::endl;
            }
        }
        return cached_devices;
    }
    
    // Helper function to get the queue for the current device
    // Returns the queue or nullptr if not available
    sycl::queue* get_current_queue() {
        // Ensure devices are initialized
        get_devices();
        
        if (current_device >= 0 && current_device < static_cast<int>(device_queues.size())) {
            return device_queues[current_device];
        }
        return nullptr;
    }
}

gpuError_t gpuMalloc(void** ptr, size_t size) {
    try {
        sycl::queue* q = gpujpeg_sycl::get_current_queue();
        if (!q) {
            return -1;
        }
        
        // For CPU devices, use shared memory instead of device-only memory
        // This avoids memcpy issues with OpenCL CPU backend
        // auto dev = q->get_device();
        // if (dev.is_cpu()) {
        //     *ptr = sycl::malloc_shared(size, *q);
        // } else {
        *ptr = sycl::malloc_device(size, *q);
        // }
        return *ptr ? gpuSuccess : -1;
    } catch (const std::exception& e) {
        std::cerr << "[SYCL ERROR in gpuMalloc] " << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMallocHost(void** ptr, size_t size) {
    try {
        sycl::queue* q = gpujpeg_sycl::get_current_queue();
        if (!q) {
            return -1;
        }
        *ptr = sycl::malloc_host(size, *q);
        return *ptr ? gpuSuccess : -1;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuFree(void* ptr) {
    try {
        sycl::queue* q = gpujpeg_sycl::get_current_queue();
        if (q && ptr) {
            sycl::free(ptr, *q);
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuFreeHost(void* ptr) {
    try {
        sycl::queue* q = gpujpeg_sycl::get_current_queue();
        if (q && ptr) {
            sycl::free(ptr, *q);
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMemcpy(void* dst, const void* src, size_t count, int kind) {
    try {
        sycl::queue* q = gpujpeg_sycl::get_current_queue();
        if (!q) {
            return -1;
        }
        
        if (!dst || !src || count == 0) {
            std::cerr << "[gpuMemcpy ERROR] Invalid parameters: "
                      << "dst=" << dst << ", src=" << src << ", count=" << count << std::endl;
            return -1;
        }

        q->memcpy(dst, src, count).wait_and_throw();

        return gpuSuccess;
    } catch (const sycl::exception& e) {
        std::cerr << "[SYCL ERROR in gpuMemcpy] " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "[UNKNOWN ERROR in gpuMemcpy]" << std::endl;
        return -1;
    }
}

gpuError_t gpuMemset(void* ptr, int value, size_t count) {
    try {
        sycl::queue* q = gpujpeg_sycl::get_current_queue();
        if (!q) {
            return -1;
        }

        q->memset(ptr, value, count).wait_and_throw();        
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuMemsetAsync(void* ptr, int value, size_t count, gpuStream_t stream) {
    try {
        sycl::queue* q = stream ? stream : gpujpeg_sycl::get_current_queue();
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
        sycl::queue* q = stream ? stream : gpujpeg_sycl::get_current_queue();
        if (!q) {
            return -1;
        }
        
        if (!dst || !src || count == 0) {
            std::cerr << "[gpuMemcpyAsync ERROR] Invalid parameters: "
                      << "dst=" << dst << ", src=" << src << ", count=" << count << std::endl;
            return -1;
        }
        
        q->memcpy(dst, src, count);
        return gpuSuccess;
    } catch (const sycl::exception& e) {
        std::cerr << "[SYCL ERROR in gpuMemcpyAsync] " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "[UNKNOWN ERROR in gpuMemcpyAsync]" << std::endl;
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
        sycl::queue* q = stream ? stream : gpujpeg_sycl::get_current_queue();
        if (!q) {
            return -1;
        }
        
        // Validate parameters
        if (!dst || !src || width == 0 || height == 0) {
            std::cerr << "[gpuMemcpy2DAsync ERROR] Invalid parameters: "
                      << "dst=" << dst << ", src=" << src 
                      << ", width=" << width << ", height=" << height << std::endl;
            return -1;
        }

        // Perform 2D copy row by row
        // Submit all memcpy operations to the queue (they will be async)
        const uint8_t* src_base = static_cast<const uint8_t*>(src);
        uint8_t* dst_base = static_cast<uint8_t*>(dst);
        
        for (size_t row = 0; row < height; ++row) {
            size_t src_offset = row * spitch;
            size_t dst_offset = row * dpitch;
            
            // Each row copy is submitted asynchronously
            q->memcpy(dst_base + dst_offset, src_base + src_offset, width);
        }
        return gpuSuccess;
    } catch (const sycl::exception& e) {
        std::cerr << "[SYCL ERROR in gpuMemcpy2DAsync] " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "[UNKNOWN ERROR in gpuMemcpy2DAsync]" << std::endl;
        return -1;
    }
}

gpuError_t gpuStreamSynchronize(gpuStream_t stream) {
    try {
        if (stream) {
            stream->wait();
        } else {
            sycl::queue* q = gpujpeg_sycl::get_current_queue();
            if (q) {
                q->wait();
            }
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuEventCreate(gpuEvent_t* event) {
    try {
        // Allocate a new sycl::event
        *event = new sycl::event();
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuEventDestroy(gpuEvent_t event) {
    try {
        if (event) {
            delete static_cast<sycl::event*>(event);
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
        
        sycl::queue* q = stream ? stream : gpujpeg_sycl::get_current_queue();
        if (!q) {
            return -1;
        }
        
        // Submit an empty kernel to record the event at this point in the queue
        // This captures timing information via profiling
        auto* evt = static_cast<sycl::event*>(event);
        *evt = q->submit([&](sycl::handler& cgh) {
            cgh.host_task([]() { /* marker event */ });
        });
        
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuEventSynchronize(gpuEvent_t event) {
    try {
        if (event) {
            auto* evt = static_cast<sycl::event*>(event);
            evt->wait();
        }
        return gpuSuccess;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuEventElapsedTime(float* ms, gpuEvent_t start, gpuEvent_t end) {
    try {
        if (!ms || !start || !end) {
            return -1;
        }
        
        auto* start_evt = static_cast<sycl::event*>(start);
        auto* end_evt = static_cast<sycl::event*>(end);
        
        // Get profiling info in nanoseconds, convert to milliseconds like CUDA
        auto start_time = start_evt->get_profiling_info<sycl::info::event_profiling::command_end>();
        auto end_time = end_evt->get_profiling_info<sycl::info::event_profiling::command_end>();
        
        // Convert nanoseconds to milliseconds (1e6 ns = 1 ms)
        *ms = static_cast<float>(end_time - start_time) / 1e6f;
        return gpuSuccess;
    } catch (const sycl::exception& e) {
        std::cerr << "[SYCL ERROR in gpuEventElapsedTime] " << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuSetDevice(int device) {
    try {
        const auto& devices = gpujpeg_sycl::get_devices();
        
        if (device < 0 || device >= static_cast<int>(devices.size())) {
            std::cerr << "[SYCL ERROR] Invalid device index: " << device 
                      << " (available: " << devices.size() << ")" << std::endl;
            return -1;
        }
        
        if (device >= static_cast<int>(gpujpeg_sycl::device_queues.size()) || 
            !gpujpeg_sycl::device_queues[device]) {
            std::cerr << "[SYCL ERROR] Queue not available for device " << device << std::endl;
            return -1;
        }
        
        gpujpeg_sycl::current_device = device;
        
        auto& dev = devices[device];
        std::cerr << "[SYCL] Device set to [" << device << "]: " 
                  << dev.get_info<sycl::info::device::name>()
                  << " (" << (dev.is_gpu() ? "GPU" : dev.is_cpu() ? "CPU" : "Accelerator") << ")"
                  << " [in_order]" << std::endl;
        
        return gpuSuccess;
    } catch (const std::exception& e) {
        std::cerr << "[SYCL ERROR in gpuSetDevice] " << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuGetDevice(int* device) {
    try {
        if (!device) {
            return -1;
        }
        *device = gpujpeg_sycl::current_device;
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
        // Count all available devices (GPU and CPU)
        const auto& devices = gpujpeg_sycl::get_devices();
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
        
        // Initialize all fields to zero first
        memset(prop, 0, sizeof(gpuDeviceProp));
        
        const auto& devices = gpujpeg_sycl::get_devices();
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
        
        // Detect device type for appropriate defaults
        bool is_cpu = dev.is_cpu();
        // bool is_gpu = dev.is_gpu();
        
        // Set defaults for properties not directly available in SYCL
        // IMPORTANT: Set major/minor early so they're visible in initialization messages
        prop->major = 1;  // Simulate compute capability 3.5 (Kepler) for SYCL
        prop->minor = 0;
        prop->regsPerBlock = 65536; // Approximate default
        
        // Set warp size based on device type
        // Intel GPUs typically use 32 (SIMD-32), CPUs use 1 (no SIMD grouping at this level)
        if (is_cpu) {
            prop->warpSize = 1;  // CPUs don't have true warp/wavefront concept
        } else {
            prop->warpSize = 32; // GPUs typically use 32 (Intel) or 64 (AMD)
        }
        
        prop->memPitch = SIZE_MAX;
        prop->maxGridSize[0] = prop->maxGridSize[1] = prop->maxGridSize[2] = 65535;
        prop->totalConstMem = 64 * 1024; // 64KB typical
        prop->textureAlignment = 512;
        prop->deviceOverlap = 1;
        
        return gpuSuccess;
    } catch (const std::exception& e) {
        std::cerr << "[SYCL ERROR in gpuGetDeviceProperties] " << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }
}

gpuError_t gpuGetLastError(void) {
    // SYCL uses exceptions, not error codes
    // We need to wait on the queue to catch any async errors
    try {
        sycl::queue* q = gpujpeg_sycl::get_current_queue();
        if (q) {
            q->wait_and_throw();
        }
        return gpuSuccess;
    } catch (const sycl::exception& e) {
        std::cerr << "\n[SYCL ERROR in gpuGetLastError]\n"
                  << "  what(): " << e.what() << "\n"
                  << "  code(): " << e.code().value() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR in gpuGetLastError]\n"
                  << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "\n[UNKNOWN ERROR in gpuGetLastError]" << std::endl;
        return -1;
    }
}

gpuError_t gpuDeviceReset(void) {
    try {
        for (auto* q : gpujpeg_sycl::device_queues) {
            if (q) {
                q->wait();
                delete q;
            }
        }
        gpujpeg_sycl::device_queues.clear();
        gpujpeg_sycl::cached_devices.clear();
        gpujpeg_sycl::devices_cached = false;
        gpujpeg_sycl::current_device = 0;
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

#endif // GPUJPEG_USE_SYCL
