/**
 * @file
 * Lossless GPU codec for 16-bit depth images (IEEE 754 half bit patterns), e.g.
 * the depth channel next to a JPEG colour image in remote rendering.
 *
 * The image is split into independent 32x32 tiles (edge tiles are smaller); each
 * tile is coded with the JPEG-LS MED predictor, zero runs and Rice codes, so the
 * encoder and the decoder run one GPU thread per tile. The stream format is
 * described in src/gpujpeg_depth.cu (identical to the "f16t" depth encoding of
 * Nexaris, which also has a CPU reference implementation).
 *
 * Backends: CUDA and HIP. The SYCL build returns an error for now.
 */

#ifndef GPUJPEG_DEPTH_H
#define GPUJPEG_DEPTH_H

#include <stddef.h>
#include <stdint.h>

#include "gpujpeg_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Element type of an encoder input image in device memory */
enum gpujpeg_depth_format {
    /// float (32 bit), converted to half with round-to-nearest-even (inf kept)
    GPUJPEG_DEPTH_F32 = 0,
    /// uint16 half bit patterns
    GPUJPEG_DEPTH_F16 = 1,
};

struct gpujpeg_depth_encoder;
struct gpujpeg_depth_decoder;

/**
 * Creates an encoder on the current device.
 * @param stream  stream for all work of the encoder (NULL = default stream)
 */
GPUJPEG_API struct gpujpeg_depth_encoder*
gpujpeg_depth_encoder_create(gpuStream_t stream);

GPUJPEG_API void
gpujpeg_depth_encoder_destroy(struct gpujpeg_depth_encoder* encoder);

/**
 * Encodes a depth image from device memory.
 *
 * @param d_image  width x height elements of `format`, rows of `pitch` elements
 * @param out      host buffer owned by the encoder, valid until the next call
 * @return 0 on success
 */
GPUJPEG_API int
gpujpeg_depth_encoder_encode(struct gpujpeg_depth_encoder* encoder, const void* d_image,
                             enum gpujpeg_depth_format format, int width, int height, int pitch,
                             uint8_t** out, size_t* out_size);

/**
 * Creates a decoder on the current device.
 * @param stream  stream for all work of the decoder (NULL = default stream)
 */
GPUJPEG_API struct gpujpeg_depth_decoder*
gpujpeg_depth_decoder_create(gpuStream_t stream);

GPUJPEG_API void
gpujpeg_depth_decoder_destroy(struct gpujpeg_depth_decoder* decoder);

/**
 * Decodes a stream (host memory) into device memory: width x height uint16 half
 * bit patterns, rows of `pitch` elements, row 0 first. The stream is validated;
 * a corrupt stream returns an error (the output is then undefined).
 *
 * @return 0 on success
 */
GPUJPEG_API int
gpujpeg_depth_decoder_decode(struct gpujpeg_depth_decoder* decoder, const uint8_t* data, size_t size,
                             int width, int height, uint16_t* d_output, int pitch);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_DEPTH_H
