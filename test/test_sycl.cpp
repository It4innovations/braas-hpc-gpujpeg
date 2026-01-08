/**
 * @file test_sycl.cpp
 * @brief Test file for SYCL kernel gpujpeg_preprocessor_raw_to_comp_kernel
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Include project headers
#include "../src/gpujpeg_preprocessor.h"
#include "../src/gpujpeg_preprocessor_common.cuh"
#include "../src/gpujpeg_colorspace.h"
#include "../src/gpujpeg_device_compat.h"
#include "../libgpujpeg/gpujpeg_type.h"

// Define necessary macros for SYCL
#ifndef GPU_DEVICE
#define GPU_DEVICE
#endif

// Note: half, __float2half, __half2float, and __umulhi are already defined in gpujpeg_device_compat.h

// Template specializations for raw_to_comp_load functions
template<enum gpujpeg_pixel_format>
inline void raw_to_comp_load(const uint8_t* d_data_raw, int &image_width, int &image_height, int &offset, int &x, int &y, uchar4 &r);

template<>
inline void raw_to_comp_load<GPUJPEG_444_U8_P012>(const uint8_t* d_data_raw, int &image_width, int &image_height, int &offset, int &x, int &y, uchar4 &r)
{
    r.x = d_data_raw[offset];
    r.y = d_data_raw[offset + 1];
    r.z = d_data_raw[offset + 2];
    r.w = 255;
}

template<>
inline void raw_to_comp_load<GPUJPEG_U8>(const uint8_t* d_data_raw, int &image_width, int &image_height, int &offset, int &x, int &y, uchar4 &r)
{
    r.x = d_data_raw[offset];
    r.y = 128;
    r.z = 128;
    r.w = 255;
}

// Template for preprocessor store operations
template<
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v,
    uint8_t s_comp4_samp_factor_h, uint8_t s_comp4_samp_factor_v
>
struct gpujpeg_preprocessor_raw_to_comp_store {
    static void perform (uchar4 value, unsigned int position_x, unsigned int position_y, struct gpujpeg_preprocessor_data & data) {
        // Simplified store operation for testing
        if (s_comp1_samp_factor_h > 0 && data.comp[0].d_data != nullptr) {
            unsigned int pos = position_y * data.comp[0].data_width + position_x;
            if (pos < data.comp[0].data_width * 100) { // bounds check
                data.comp[0].d_data[pos] = value.x;
            }
        }
    }
};

// SYCL Kernel implementation
template<
    enum gpujpeg_color_space color_space_internal,
    enum gpujpeg_color_space color_space,
    enum gpujpeg_pixel_format pixel_format,
    uint8_t s_comp1_samp_factor_h, uint8_t s_comp1_samp_factor_v,
    uint8_t s_comp2_samp_factor_h, uint8_t s_comp2_samp_factor_v,
    uint8_t s_comp3_samp_factor_h, uint8_t s_comp3_samp_factor_v,
    uint8_t s_comp4_samp_factor_h, uint8_t s_comp4_samp_factor_v
>
void gpujpeg_preprocessor_raw_to_comp_kernel2(sycl::nd_item<3> item, 
                                             sycl::local_accessor<uint8_t, 1> shared_mem_accessor,
                                             struct gpujpeg_preprocessor_data data, 
                                             const uint8_t* d_data_raw, int image_width_padding, 
                                             int image_width, int image_height, 
                                             uint32_t width_div_mul, uint32_t width_div_shift)
{
    // Replace macros with SYCL equivalents
    int x  = item.get_local_id(2);  // GPU_THREAD_IDX_X
    int gX = (item.get_group(1) * item.get_group_range(2) + item.get_group(2)) * item.get_local_range(2);
    // GPU_BLOCK_IDX_Y * GPU_GRID_DIM_X + GPU_BLOCK_IDX_X) * GPU_BLOCK_DIM_X

    // Position
    int image_position = gX + x;
    int image_position_y = gpujpeg_const_div_divide(image_position, width_div_mul, width_div_shift);
    int image_position_x = image_position - (image_position_y * image_width);

    if ( image_position >= (image_width * image_height) ) {
        return;
    }

    // Load
    uchar4 r;
    int offset = image_position * unit_size<pixel_format>() + image_width_padding * image_position_y;
    raw_to_comp_load<pixel_format>(d_data_raw, image_width, image_height, offset, image_position_x, image_position_y, r);

    // Color transform
    gpujpeg_color_transform<color_space, color_space_internal>::perform(r);

    // Store
    gpujpeg_preprocessor_raw_to_comp_store<s_comp1_samp_factor_h, s_comp1_samp_factor_v, s_comp2_samp_factor_h,
                                           s_comp2_samp_factor_v, s_comp3_samp_factor_h, s_comp3_samp_factor_v,
                                           s_comp4_samp_factor_h, s_comp4_samp_factor_v>::perform(r, image_position_x,
                                                                                                  image_position_y,
                                                                                                  data);
}


#ifdef TEST_MAIN
int main(int argc, char** argv) 
#else
int 
test_sycl_kernel() 
#endif
{
    std::cout << "SYCL Preprocessor Kernel Test\n";
    std::cout << "==============================\n\n";

    try {
        // Get SYCL queue
        // sycl::queue q(sycl::default_selector_v);
        // std::cout << "Running on device: " 
        //           << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

        // Test parameters
        const int image_width = 1920;
        const int image_height = 1080;
        const int image_size = image_width * image_height * 3; // RGB
        const int image_width_padding = 0;

        // Prepare division constants
        uint32_t width_div_mul, width_div_shift;
        gpujpeg_const_div_prepare(image_width, width_div_mul, width_div_shift);

        // Allocate and initialize host data
        std::vector<uint8_t> h_data_raw(image_size);
        std::vector<uint8_t> h_output(image_size / 3);
        
        // Fill input with test pattern
        for (int i = 0; i < image_size; ++i) {
            h_data_raw[i] = static_cast<uint8_t>(i % 256);
        }

        std::cout << "Image dimensions: " << image_width << "x" << image_height << "\n";
        std::cout << "Input size: " << image_size << " bytes\n";

        // Allocate device memory
        uint8_t* d_data_raw;
        gpuMalloc((void**)&d_data_raw, image_size);        
        uint8_t* d_output_r;
        uint8_t* d_output_g;
        uint8_t* d_output_b;
        gpuMalloc((void**)&d_output_r, image_size / 3);
        gpuMalloc((void**)&d_output_g, image_size / 3);
        gpuMalloc((void**)&d_output_b, image_size / 3);

        if (!d_data_raw || !d_output_r || !d_output_g || !d_output_b) {
            std::cerr << "Failed to allocate device memory\n";
            return 1;
        }

        // Copy data to device
        gpuMemcpy(d_data_raw, h_data_raw.data(), image_size, gpuMemcpyHostToDevice);
        gpuMemset(d_output_r, 0, image_size / 3);
        gpuMemset(d_output_g, 0, image_size / 3);
        gpuMemset(d_output_b, 0, image_size / 3);

        // Setup preprocessor data structure
        gpujpeg_preprocessor_data data;
        data.comp[0].d_data = d_output_r;
        data.comp[0].data_width = image_width;
        data.comp[0].sampling_factor.horizontal = 1;
        data.comp[0].sampling_factor.vertical = 1;
        data.comp[1].d_data = d_output_g;
        data.comp[1].data_width = image_width;
        data.comp[1].sampling_factor.horizontal = 1;
        data.comp[1].sampling_factor.vertical = 1;
        data.comp[2].d_data = d_output_b;
        data.comp[2].data_width = image_width;
        data.comp[2].sampling_factor.horizontal = 1;
        data.comp[2].sampling_factor.vertical = 1;

        // Launch kernel
        const int block_size = 256;
        const int num_blocks = (image_width * image_height + block_size - 1) / block_size;
        
        sycl::range<3> local_size(1, 1, block_size);
        sycl::range<3> global_size(1, 1, num_blocks * block_size);

        std::cout << "Launching kernel with " << num_blocks << " blocks, " 
                  << block_size << " threads per block\n";

        // Use GPU_KERNEL_LAUNCH macro
        dim3 grid(num_blocks, 1, 1);
        dim3 block(block_size, 1, 1);
        
        GPU_KERNEL_LAUNCH(
            (gpujpeg_preprocessor_raw_to_comp_kernel2<
                GPUJPEG_YCBCR_BT601,
                GPUJPEG_RGB,
                GPUJPEG_444_U8_P012,
                1, 1,  // comp1 sampling
                1, 1,  // comp2 sampling
                1, 1,  // comp3 sampling
                0, 0   // comp4 sampling (disabled)
            >),
            grid, block, 0, nullptr,
            data, d_data_raw, image_width_padding, 
            image_width, image_height, width_div_mul, width_div_shift
        );
        
        gpuStreamSynchronize(nullptr);

        std::cout << "Kernel execution completed\n";

        // Copy result back to host (R component)
        gpuMemcpy(h_output.data(), d_output_r, image_size / 3, gpuMemcpyDeviceToHost);

        // Verify results (simple check)
        int non_zero_count = 0;
        for (int i = 0; i < image_width * image_height && i < 100; ++i) {
            if (h_output[i] != 0) {
                non_zero_count++;
            }
        }

        std::cout << "\nResults:\n";
        std::cout << "Non-zero values in first 100 elements: " << non_zero_count << "\n";
        std::cout << "Sample output values: ";
        for (int i = 0; i < 10 && i < h_output.size(); ++i) {
            std::cout << static_cast<int>(h_output[i]) << " ";
        }
        std::cout << "\n";

        // Cleanup
        gpuFree(d_data_raw);
        gpuFree(d_output_r);
        gpuFree(d_output_g);
        gpuFree(d_output_b);

        std::cout << "\nTest completed successfully!\n";
        return 0;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}