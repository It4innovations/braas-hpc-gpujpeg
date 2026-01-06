/**
 * @file test.cpp
 * @brief Simple test program to generate a gradient RGB image in BMP format
 * 
 * Creates a 1920x1080 BMP image with a horizontal gradient:
 * - Left side: Red
 * - Middle: Green
 * - Right side: Blue
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#pragma pack(push, 1)
typedef struct {
    uint16_t type;      // Magic identifier: 0x4d42 for BMP
    uint32_t size;      // File size in bytes
    uint16_t reserved1; // Reserved
    uint16_t reserved2; // Reserved
    uint32_t offset;    // Offset to image data in bytes
} BMPFileHeader;

typedef struct {
    uint32_t size;            // Header size in bytes
    int32_t width;            // Width of image
    int32_t height;           // Height of image
    uint16_t planes;          // Number of colour planes
    uint16_t bits;            // Bits per pixel
    uint32_t compression;     // Compression type
    uint32_t imagesize;       // Image size in bytes
    int32_t xresolution;      // Pixels per meter X
    int32_t yresolution;      // Pixels per meter Y
    uint32_t ncolors;         // Number of colours
    uint32_t importantcolors; // Important colours
} BMPInfoHeader;
#pragma pack(pop)

int main(int argc, char *argv[])
{
    const int width = 1920;
    const int height = 1080;
    const int channels = 3; // BGR for BMP
    
    // BMP rows must be padded to 4-byte boundary
    const int row_padded = (width * channels + 3) & (~3);
    const size_t image_size = row_padded * height;
    
    // Allocate memory for the image (use calloc to zero padding bytes)
    uint8_t *image = (uint8_t *)calloc(1, image_size);
    if (!image) {
        fprintf(stderr, "Failed to allocate memory for image\n");
        return 1;
    }
    
    // Generate gradient from Red -> Green -> Blue
    // BMP stores pixels bottom-up and in BGR format
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * row_padded + x * channels;
            
            // Calculate position ratio (0.0 to 1.0 from left to right)
            float t = (float)x / (float)(width - 1);
            
            uint8_t r, g, b;
            if (t < 0.5f) {
                // First half: Red to Green (0.0 to 0.5)
                float local_t = t * 2.0f; // Map to 0.0-1.0
                r = (uint8_t)(255 * (1.0f - local_t)); // Red decreases
                g = (uint8_t)(255 * local_t);          // Green increases
                b = 0;                                  // Blue stays 0
            } else {
                // Second half: Green to Blue (0.5 to 1.0)
                float local_t = (t - 0.5f) * 2.0f; // Map to 0.0-1.0
                r = 0;                                  // Red stays 0
                g = (uint8_t)(255 * (1.0f - local_t)); // Green decreases
                b = (uint8_t)(255 * local_t);          // Blue increases
            }
            
            // Store in BGR order for BMP
            image[idx + 0] = b; // Blue
            image[idx + 1] = g; // Green
            image[idx + 2] = r; // Red
        }
        // Padding is already zeroed from calloc
    }
    
    // Prepare BMP headers
    BMPFileHeader file_header;
    BMPInfoHeader info_header;
    
    file_header.type = 0x4D42; // 'BM'
    file_header.size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + image_size;
    file_header.reserved1 = 0;
    file_header.reserved2 = 0;
    file_header.offset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    
    info_header.size = sizeof(BMPInfoHeader);
    info_header.width = width;
    info_header.height = height;
    info_header.planes = 1;
    info_header.bits = 24;
    info_header.compression = 0; // No compression
    info_header.imagesize = image_size;
    info_header.xresolution = 2835; // 72 DPI
    info_header.yresolution = 2835; // 72 DPI
    info_header.ncolors = 0;
    info_header.importantcolors = 0;
    
    // Write the BMP file
    const char *filename = (argc > 1) ? argv[1] : "input.bmp";
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file '%s' for writing\n", filename);
        free(image);
        return 1;
    }
    
    // Write headers
    fwrite(&file_header, sizeof(BMPFileHeader), 1, f);
    fwrite(&info_header, sizeof(BMPInfoHeader), 1, f);
    
    // Write image data (bottom-up)
    for (int y = height - 1; y >= 0; y--) {
        fwrite(image + y * row_padded, 1, row_padded, f);
    }
    
    fclose(f);
    
    printf("Successfully created %dx%d BMP image: %s\n", width, height, filename);
    printf("File size: %u bytes\n", file_header.size);
    printf("Gradient: Red -> Green -> Blue (left to right)\n");
    
    free(image);
    return 0;
}
