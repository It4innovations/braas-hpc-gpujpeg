/**
 * @file
 * Lossless GPU codec for 16-bit depth images (see libgpujpeg/gpujpeg_depth.h).
 *
 * Stream format (all little-endian; identical to the Nexaris "f16t" encoding):
 *   u8 version (1), u8 log2 of the tile size (5 = 32x32), u16 0, u32 tile count,
 *   u32 offset x (tile count + 1): byte offsets of the tile streams relative to
 *   the end of this table, multiples of 4; the last one is the size of the tile
 *   data. Then the tile data.
 * Tiles cover the image in raster order (edge tiles are smaller) and are
 * independent. A tile stream is a bit stream, least significant bit first in
 * little-endian 32-bit words: 16 bits the half bits of the tile's first pixel,
 * 3 bits kr, 4 bits kv, then for the other pixels in raster order of the tile the
 * residuals of the JPEG-LS MED predictor on the 16-bit patterns (inside the
 * tile; first row: left neighbour, first column: pixel above), zigzag-mapped,
 * as: a run of zero residuals (Rice kr), then, unless the run reaches the end of
 * the tile, a non-zero residual minus 1 (Rice kv), repeated. Rice(v, k): q = v >> k;
 * q < 24: q one bits, a zero bit, the k low bits of v; else 24 one bits and the 16
 * bits of v. The encoder picks the kr (0..7) and kv (0..15) with the fewest bits
 * (the smallest k on ties).
 *
 * Encoder: one thread per tile writes the tile stream into its slot of a scratch
 * buffer; the host turns the stream sizes into the offset table; one block per
 * tile packs the streams. Decoder: one thread per tile.
 */

#include "../libgpujpeg/gpujpeg_depth.h"
#include "gpujpeg_device_compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define GPUJPEG_DEPTH_VERSION 1
#define GPUJPEG_DEPTH_TILE_LOG2 5
#define GPUJPEG_DEPTH_TILE (1 << GPUJPEG_DEPTH_TILE_LOG2)
#define GPUJPEG_DEPTH_RICE_ESCAPE 24
// worst case of a 32x32 tile: 23 header bits, 48 bits per pixel (run of 0 + escaped
// value), an escaped tail run -> 49167 bits
#define GPUJPEG_DEPTH_TILE_WORDS 1540
#define GPUJPEG_DEPTH_THREADS 64

#ifndef GPUJPEG_USE_SYCL

namespace {

GPU_DEVICE inline uint16_t
depth_float_to_half(float f)
{
    const uint32_t x = __float_as_uint(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t absx = x & 0x7FFFFFFFu;
    if ( absx >= 0x7F800000u ) // inf or nan
        return (uint16_t)(sign | 0x7C00u | (absx > 0x7F800000u ? 0x200u : 0u));
    if ( absx >= 0x477FF000u ) // overflow after rounding -> inf
        return (uint16_t)(sign | 0x7C00u);
    if ( absx < 0x38800000u ) { // subnormal or zero in half
        if ( absx < 0x33000000u )
            return (uint16_t)sign;
        const uint32_t mant = (absx & 0x007FFFFFu) | 0x00800000u;
        const int s = 126 - (int)(absx >> 23);
        uint32_t h = mant >> s;
        const uint32_t rem = mant & ((1u << s) - 1u);
        const uint32_t half = 1u << (s - 1);
        if ( rem > half || (rem == half && (h & 1u)) )
            ++h;
        return (uint16_t)(sign | h);
    }
    uint32_t h = (absx - 0x38000000u) >> 13;
    const uint32_t rem = absx & 0x1FFFu;
    if ( rem > 0x1000u || (rem == 0x1000u && (h & 1u)) )
        ++h;
    return (uint16_t)(sign | h);
}

GPU_DEVICE inline uint16_t
depth_med(uint16_t a, uint16_t b, uint16_t c) // a left, b up, c up-left
{
    const int ia = a, ib = b, ic = c;
    const int mx = ia > ib ? ia : ib, mn = ia < ib ? ia : ib;
    if ( ic >= mx )
        return (uint16_t)mn;
    if ( ic <= mn )
        return (uint16_t)mx;
    return (uint16_t)(ia + ib - ic);
}

// MED prediction of tile pixel (x, y) (tile-local) of the image `img` (rows of `pitch`)
GPU_DEVICE inline uint16_t
depth_predict(const uint16_t* img, int pitch, int x0, int y0, int x, int y)
{
    const uint16_t* row = img + (size_t)(y0 + y) * pitch + x0;
    if ( y == 0 )
        return x == 0 ? 0 : row[x - 1];
    const uint16_t* up = row - pitch;
    if ( x == 0 )
        return up[0];
    return depth_med(row[x - 1], up[x], up[x - 1]);
}

GPU_DEVICE inline uint32_t
depth_zigzag(uint16_t r)
{
    const int s = (int16_t)r;
    return s >= 0 ? (uint32_t)(2 * s) : (uint32_t)(-2 * s - 1);
}

GPU_DEVICE inline uint16_t
depth_unzigzag(uint32_t z)
{
    return (uint16_t)((z & 1u) ? -(int)((z + 1) / 2) : (int)(z / 2));
}

GPU_DEVICE inline uint32_t
depth_rice_bits(uint32_t v, uint32_t k)
{
    const uint32_t q = v >> k;
    return q < GPUJPEG_DEPTH_RICE_ESCAPE ? q + 1 + k : GPUJPEG_DEPTH_RICE_ESCAPE + 16;
}

struct depth_bit_writer
{
    uint32_t* out;
    uint32_t words;
    uint64_t acc;
    uint32_t nbits;

    GPU_DEVICE void put(uint32_t v, uint32_t n) // n <= 24
    {
        acc |= (uint64_t)(v & ((1u << n) - 1u)) << nbits;
        nbits += n;
        while ( nbits >= 32 ) {
            if ( words < GPUJPEG_DEPTH_TILE_WORDS )
                out[words] = (uint32_t)acc;
            ++words;
            acc >>= 32;
            nbits -= 32;
        }
    }
    GPU_DEVICE void rice(uint32_t v, uint32_t k)
    {
        const uint32_t q = v >> k;
        if ( q < GPUJPEG_DEPTH_RICE_ESCAPE ) {
            put((1u << q) - 1u, q);
            put(0, 1);
            put(v, k);
        }
        else {
            put((1u << GPUJPEG_DEPTH_RICE_ESCAPE) - 1u, GPUJPEG_DEPTH_RICE_ESCAPE);
            put(v, 16);
        }
    }
    GPU_DEVICE void flush()
    {
        if ( nbits > 0 ) {
            if ( words < GPUJPEG_DEPTH_TILE_WORDS )
                out[words] = (uint32_t)acc;
            ++words;
            acc = 0;
            nbits = 0;
        }
    }
};

struct depth_bit_reader
{
    const uint32_t* in;
    uint64_t bits;
    uint64_t pos;
    bool ok;

    GPU_DEVICE uint32_t get(uint32_t n) // n <= 24
    {
        if ( n == 0 )
            return 0;
        if ( pos + n > bits ) {
            ok = false;
            return 0;
        }
        const uint64_t word = pos >> 5;
        const uint32_t bit = (uint32_t)(pos & 31);
        uint64_t v = in[word] >> bit;
        if ( bit + n > 32 )
            v |= (uint64_t)in[word + 1] << (32 - bit);
        pos += n;
        return (uint32_t)(v & ((1u << n) - 1u));
    }
    GPU_DEVICE uint32_t rice(uint32_t k)
    {
        uint32_t q = 0;
        while ( q < GPUJPEG_DEPTH_RICE_ESCAPE && get(1) == 1 )
            ++q;
        if ( q == GPUJPEG_DEPTH_RICE_ESCAPE )
            return get(16);
        return (q << k) | get(k);
    }
};

GPU_GLOBAL void
depth_to_half_kernel(const float* in, int in_pitch, uint16_t* out, int width, int height)
{
    const int x = GPU_BLOCK_IDX_X * GPU_BLOCK_DIM_X + GPU_THREAD_IDX_X;
    const int y = GPU_BLOCK_IDX_Y;
    if ( x >= width || y >= height )
        return;
    out[(size_t)y * width + x] = depth_float_to_half(in[(size_t)y * in_pitch + x]);
}

GPU_GLOBAL void
depth_encode_kernel(const uint16_t* img, int width, int height, int pitch, int tiles_x, int tile_count,
                    uint32_t* scratch, uint32_t* sizes)
{
    const int tile = GPU_BLOCK_IDX_X * GPU_BLOCK_DIM_X + GPU_THREAD_IDX_X;
    if ( tile >= tile_count )
        return;
    const int x0 = (tile % tiles_x) * GPUJPEG_DEPTH_TILE, y0 = (tile / tiles_x) * GPUJPEG_DEPTH_TILE;
    const int tw = width - x0 < GPUJPEG_DEPTH_TILE ? width - x0 : GPUJPEG_DEPTH_TILE;
    const int th = height - y0 < GPUJPEG_DEPTH_TILE ? height - y0 : GPUJPEG_DEPTH_TILE;
    const int n = tw * th;

    // pass 1: bits of every Rice parameter
    uint64_t cost_run[8], cost_val[16];
    for ( int k = 0; k < 8; ++k )
        cost_run[k] = 0;
    for ( int k = 0; k < 16; ++k )
        cost_val[k] = 0;
    uint32_t run = 0;
    for ( int i = 1; i < n; ++i ) {
        const int x = i % tw, y = i / tw;
        const uint16_t v = img[(size_t)(y0 + y) * pitch + x0 + x];
        const uint32_t z = depth_zigzag((uint16_t)(v - depth_predict(img, pitch, x0, y0, x, y)));
        if ( z == 0 ) {
            ++run;
            continue;
        }
        for ( uint32_t k = 0; k < 8; ++k )
            cost_run[k] += depth_rice_bits(run, k);
        for ( uint32_t k = 0; k < 16; ++k )
            cost_val[k] += depth_rice_bits(z - 1, k);
        run = 0;
    }
    if ( run > 0 )
        for ( uint32_t k = 0; k < 8; ++k )
            cost_run[k] += depth_rice_bits(run, k);
    uint32_t kr = 0, kv = 0;
    for ( uint32_t k = 1; k < 8; ++k )
        if ( cost_run[k] < cost_run[kr] )
            kr = k;
    for ( uint32_t k = 1; k < 16; ++k )
        if ( cost_val[k] < cost_val[kv] )
            kv = k;

    // pass 2: the stream
    depth_bit_writer bw = {scratch + (size_t)tile * GPUJPEG_DEPTH_TILE_WORDS, 0, 0, 0};
    bw.put(img[(size_t)y0 * pitch + x0], 16);
    bw.put(kr, 3);
    bw.put(kv, 4);
    run = 0;
    for ( int i = 1; i < n; ++i ) {
        const int x = i % tw, y = i / tw;
        const uint16_t v = img[(size_t)(y0 + y) * pitch + x0 + x];
        const uint32_t z = depth_zigzag((uint16_t)(v - depth_predict(img, pitch, x0, y0, x, y)));
        if ( z == 0 ) {
            ++run;
            continue;
        }
        bw.rice(run, kr);
        bw.rice(z - 1, kv);
        run = 0;
    }
    if ( run > 0 )
        bw.rice(run, kr);
    bw.flush();
    sizes[tile] = bw.words;
}

GPU_GLOBAL void
depth_pack_kernel(const uint32_t* scratch, const uint32_t* word_offsets, uint32_t* out)
{
    const int tile = GPU_BLOCK_IDX_X;
    const uint32_t begin = word_offsets[tile], count = word_offsets[tile + 1] - begin;
    const uint32_t* src = scratch + (size_t)tile * GPUJPEG_DEPTH_TILE_WORDS;
    for ( uint32_t i = GPU_THREAD_IDX_X; i < count; i += GPU_BLOCK_DIM_X )
        out[begin + i] = src[i];
}

// Rows of a tile being decoded: the row above and the current one stay in the
// thread's local memory (the predictor never reads the output image), finished
// rows are written out.
struct depth_tile_rows
{
    uint16_t prev[GPUJPEG_DEPTH_TILE];
    uint16_t cur[GPUJPEG_DEPTH_TILE];
    int x, y, tw;
    uint16_t* out; // first pixel of the tile in the image
    int pitch;

    GPU_DEVICE void emit(uint16_t residual)
    {
        uint16_t pred;
        if ( y == 0 )
            pred = x == 0 ? 0 : cur[x - 1];
        else if ( x == 0 )
            pred = prev[0];
        else
            pred = depth_med(cur[x - 1], prev[x], prev[x - 1]);
        cur[x] = (uint16_t)(pred + residual);
        if ( ++x == tw ) {
            uint16_t* row = out + (size_t)y * pitch;
            for ( int i = 0; i < tw; ++i ) {
                row[i] = cur[i];
                prev[i] = cur[i];
            }
            x = 0;
            ++y;
        }
    }
};

GPU_GLOBAL void
depth_decode_kernel(const uint32_t* payload, const uint32_t* word_offsets, int width, int height, int tiles_x,
                    int tile_count, uint16_t* img, int pitch, int* error)
{
    const int tile = GPU_BLOCK_IDX_X * GPU_BLOCK_DIM_X + GPU_THREAD_IDX_X;
    if ( tile >= tile_count )
        return;
    const int x0 = (tile % tiles_x) * GPUJPEG_DEPTH_TILE, y0 = (tile / tiles_x) * GPUJPEG_DEPTH_TILE;
    const int tw = width - x0 < GPUJPEG_DEPTH_TILE ? width - x0 : GPUJPEG_DEPTH_TILE;
    const int th = height - y0 < GPUJPEG_DEPTH_TILE ? height - y0 : GPUJPEG_DEPTH_TILE;
    const int n = tw * th;
    const uint32_t begin = word_offsets[tile];
    depth_bit_reader br = {payload + begin, (uint64_t)(word_offsets[tile + 1] - begin) * 32, 0, true};
    depth_tile_rows rows;
    rows.x = 0;
    rows.y = 0;
    rows.tw = tw;
    rows.out = img + (size_t)y0 * pitch + x0;
    rows.pitch = pitch;
    // the first pixel is stored as is: a residual against the prediction 0
    rows.emit((uint16_t)br.get(16));
    const uint32_t kr = br.get(3), kv = br.get(4);
    int pos = 1;
    while ( pos < n && br.ok ) {
        const uint32_t run = br.rice(kr);
        if ( !br.ok || run > (uint32_t)(n - pos) ) {
            br.ok = false;
            break;
        }
        for ( const int end = pos + (int)run; pos < end; ++pos )
            rows.emit(0);
        if ( pos < n ) {
            const uint32_t z = br.rice(kv) + 1;
            if ( !br.ok || z > 0xFFFFu ) {
                br.ok = false;
                break;
            }
            rows.emit(depth_unzigzag(z));
            ++pos;
        }
    }
    if ( !br.ok || pos != n )
        *error = 1;
}

uint32_t
rd32(const uint8_t* p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

void
wr32(uint8_t* p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

// Grows a device buffer to at least `bytes`.
template <typename T>
bool
depth_reserve(T** ptr, size_t* capacity, size_t bytes)
{
    if ( *capacity >= bytes )
        return true;
    if ( *ptr != NULL )
        gpuFree(*ptr);
    *ptr = NULL;
    *capacity = 0;
    if ( gpuMalloc((void**)ptr, bytes) != gpuSuccess ) {
        fprintf(stderr, "[GPUJPEG] [Error] depth codec: device allocation of %zu bytes failed\n", bytes);
        return false;
    }
    *capacity = bytes;
    return true;
}

bool
depth_check(const char* what)
{
    const gpuError_t e = gpuGetLastError();
    if ( e == gpuSuccess )
        return true;
    fprintf(stderr, "[GPUJPEG] [Error] depth codec: %s: %s\n", what, gpuGetErrorString(e));
    return false;
}

} // namespace

struct gpujpeg_depth_encoder
{
    gpuStream_t stream;
    uint16_t* d_half;
    size_t half_capacity;
    uint32_t* d_scratch;
    size_t scratch_capacity;
    uint32_t* d_sizes;
    size_t sizes_capacity;
    uint32_t* d_offsets;
    size_t offsets_capacity;
    uint32_t* d_packed;
    size_t packed_capacity;
    std::vector<uint32_t> sizes;
    std::vector<uint32_t> offsets;
    std::vector<uint8_t> out;
};

struct gpujpeg_depth_decoder
{
    gpuStream_t stream;
    uint32_t* d_payload;
    size_t payload_capacity;
    uint32_t* d_offsets;
    size_t offsets_capacity;
    int* d_error;
    std::vector<uint32_t> offsets;
};

/* Documented at declaration */
struct gpujpeg_depth_encoder*
gpujpeg_depth_encoder_create(gpuStream_t stream)
{
    gpujpeg_depth_encoder* e = new gpujpeg_depth_encoder();
    e->stream = stream;
    return e;
}

/* Documented at declaration */
void
gpujpeg_depth_encoder_destroy(struct gpujpeg_depth_encoder* e)
{
    if ( e == NULL )
        return;
    for ( void* p : {(void*)e->d_half, (void*)e->d_scratch, (void*)e->d_sizes, (void*)e->d_offsets, (void*)e->d_packed} )
        if ( p != NULL )
            gpuFree(p);
    delete e;
}

/* Documented at declaration */
int
gpujpeg_depth_encoder_encode(struct gpujpeg_depth_encoder* e, const void* d_image, enum gpujpeg_depth_format format,
                             int width, int height, int pitch, uint8_t** out, size_t* out_size)
{
    if ( e == NULL || d_image == NULL || width <= 0 || height <= 0 || pitch < width || out == NULL || out_size == NULL )
        return -1;
    const int tiles_x = (width + GPUJPEG_DEPTH_TILE - 1) / GPUJPEG_DEPTH_TILE;
    const int tiles_y = (height + GPUJPEG_DEPTH_TILE - 1) / GPUJPEG_DEPTH_TILE;
    const int tiles = tiles_x * tiles_y;

    const uint16_t* img = (const uint16_t*)d_image;
    int img_pitch = pitch;
    if ( format == GPUJPEG_DEPTH_F32 ) {
        if ( !depth_reserve(&e->d_half, &e->half_capacity, (size_t)width * height * sizeof(uint16_t)) )
            return -1;
        const dim3 block(256);
        const dim3 grid((width + 255) / 256, height);
        GPU_KERNEL_LAUNCH(depth_to_half_kernel, grid, block, 0, e->stream, (const float*)d_image, pitch, e->d_half,
                          width, height);
        if ( !depth_check("half conversion") )
            return -1;
        img = e->d_half;
        img_pitch = width;
    }
    else if ( format != GPUJPEG_DEPTH_F16 ) {
        return -1;
    }
    if ( !depth_reserve(&e->d_scratch, &e->scratch_capacity,
                        (size_t)tiles * GPUJPEG_DEPTH_TILE_WORDS * sizeof(uint32_t))
         || !depth_reserve(&e->d_sizes, &e->sizes_capacity, (size_t)tiles * sizeof(uint32_t))
         || !depth_reserve(&e->d_offsets, &e->offsets_capacity, (size_t)(tiles + 1) * sizeof(uint32_t)) )
        return -1;
    GPU_KERNEL_LAUNCH(depth_encode_kernel, dim3((tiles + GPUJPEG_DEPTH_THREADS - 1) / GPUJPEG_DEPTH_THREADS),
                      dim3(GPUJPEG_DEPTH_THREADS), 0, e->stream, img, width, height, img_pitch, tiles_x, tiles,
                      e->d_scratch, e->d_sizes);
    if ( !depth_check("tile encoder") )
        return -1;
    e->sizes.resize(tiles);
    gpuMemcpyAsync(e->sizes.data(), e->d_sizes, (size_t)tiles * sizeof(uint32_t), gpuMemcpyDeviceToHost, e->stream);
    if ( gpuStreamSynchronize(e->stream) != gpuSuccess || !depth_check("tile sizes") )
        return -1;

    // offset table (bytes in the stream, words on the device)
    const size_t header = 8 + 4 * ((size_t)tiles + 1);
    e->offsets.resize((size_t)tiles + 1);
    uint32_t words = 0;
    for ( int t = 0; t < tiles; ++t ) {
        if ( e->sizes[t] > GPUJPEG_DEPTH_TILE_WORDS )
            return -1; // cannot happen (worst case bound)
        e->offsets[t] = words;
        words += e->sizes[t];
    }
    e->offsets[tiles] = words;
    e->out.resize(header + (size_t)words * 4);
    uint8_t* o = e->out.data();
    o[0] = GPUJPEG_DEPTH_VERSION;
    o[1] = GPUJPEG_DEPTH_TILE_LOG2;
    o[2] = o[3] = 0;
    wr32(o + 4, (uint32_t)tiles);
    for ( int t = 0; t <= tiles; ++t )
        wr32(o + 8 + 4 * (size_t)t, e->offsets[t] * 4);

    if ( !depth_reserve(&e->d_packed, &e->packed_capacity, (size_t)words * 4 + 4) )
        return -1;
    gpuMemcpyAsync(e->d_offsets, e->offsets.data(), ((size_t)tiles + 1) * sizeof(uint32_t), gpuMemcpyHostToDevice,
                   e->stream);
    GPU_KERNEL_LAUNCH(depth_pack_kernel, dim3(tiles), dim3(128), 0, e->stream, e->d_scratch, e->d_offsets,
                      e->d_packed);
    if ( !depth_check("packing") )
        return -1;
    // the payload words land little-endian in the stream (x86-64, arm64 hosts)
    gpuMemcpyAsync(o + header, e->d_packed, (size_t)words * 4, gpuMemcpyDeviceToHost, e->stream);
    if ( gpuStreamSynchronize(e->stream) != gpuSuccess || !depth_check("download") )
        return -1;
    *out = e->out.data();
    *out_size = e->out.size();
    return 0;
}

/* Documented at declaration */
struct gpujpeg_depth_decoder*
gpujpeg_depth_decoder_create(gpuStream_t stream)
{
    gpujpeg_depth_decoder* d = new gpujpeg_depth_decoder();
    d->stream = stream;
    return d;
}

/* Documented at declaration */
void
gpujpeg_depth_decoder_destroy(struct gpujpeg_depth_decoder* d)
{
    if ( d == NULL )
        return;
    for ( void* p : {(void*)d->d_payload, (void*)d->d_offsets, (void*)d->d_error} )
        if ( p != NULL )
            gpuFree(p);
    delete d;
}

/* Documented at declaration */
int
gpujpeg_depth_decoder_decode(struct gpujpeg_depth_decoder* d, const uint8_t* data, size_t size, int width,
                             int height, uint16_t* d_output, int pitch)
{
    if ( d == NULL || data == NULL || d_output == NULL || width <= 0 || height <= 0 || pitch < width || size < 12 )
        return -1;
    const int tiles_x = (width + GPUJPEG_DEPTH_TILE - 1) / GPUJPEG_DEPTH_TILE;
    const int tiles_y = (height + GPUJPEG_DEPTH_TILE - 1) / GPUJPEG_DEPTH_TILE;
    const int tiles = tiles_x * tiles_y;
    const size_t header = 8 + 4 * ((size_t)tiles + 1);
    if ( data[0] != GPUJPEG_DEPTH_VERSION || data[1] != GPUJPEG_DEPTH_TILE_LOG2 || data[2] != 0 || data[3] != 0
         || rd32(data + 4) != (uint32_t)tiles || size < header )
        return -1;
    const size_t payload = size - header;
    if ( payload % 4 != 0 || rd32(data + 8 + 4 * (size_t)tiles) != payload )
        return -1;
    d->offsets.resize((size_t)tiles + 1);
    for ( int t = 0; t <= tiles; ++t ) {
        const uint32_t off = rd32(data + 8 + 4 * (size_t)t);
        if ( off % 4 != 0 || off > payload || (t > 0 && off < d->offsets[t - 1] * 4) )
            return -1;
        d->offsets[t] = off / 4;
    }
    // +1 word: the bit reader may load the word after a stream's last bit
    if ( !depth_reserve(&d->d_payload, &d->payload_capacity, payload + 4)
         || !depth_reserve(&d->d_offsets, &d->offsets_capacity, ((size_t)tiles + 1) * sizeof(uint32_t)) )
        return -1;
    if ( d->d_error == NULL && gpuMalloc((void**)&d->d_error, sizeof(int)) != gpuSuccess )
        return -1;
    gpuMemcpyAsync(d->d_payload, data + header, payload, gpuMemcpyHostToDevice, d->stream);
    gpuMemcpyAsync(d->d_offsets, d->offsets.data(), ((size_t)tiles + 1) * sizeof(uint32_t), gpuMemcpyHostToDevice,
                   d->stream);
    gpuMemsetAsync(d->d_error, 0, sizeof(int), d->stream);
    GPU_KERNEL_LAUNCH(depth_decode_kernel, dim3((tiles + GPUJPEG_DEPTH_THREADS - 1) / GPUJPEG_DEPTH_THREADS),
                      dim3(GPUJPEG_DEPTH_THREADS), 0, d->stream, d->d_payload, d->d_offsets, width, height, tiles_x,
                      tiles, d_output, pitch, d->d_error);
    if ( !depth_check("tile decoder") )
        return -1;
    int error = 0;
    gpuMemcpyAsync(&error, d->d_error, sizeof(int), gpuMemcpyDeviceToHost, d->stream);
    if ( gpuStreamSynchronize(d->stream) != gpuSuccess || !depth_check("tile decoder") )
        return -1;
    return error ? -1 : 0;
}

#else // GPUJPEG_USE_SYCL

struct gpujpeg_depth_encoder* gpujpeg_depth_encoder_create(gpuStream_t) { return NULL; }
void gpujpeg_depth_encoder_destroy(struct gpujpeg_depth_encoder*) {}
int gpujpeg_depth_encoder_encode(struct gpujpeg_depth_encoder*, const void*, enum gpujpeg_depth_format, int, int, int,
                                 uint8_t**, size_t*)
{
    fprintf(stderr, "[GPUJPEG] [Error] depth codec: not implemented for the SYCL backend yet\n");
    return -1;
}
struct gpujpeg_depth_decoder* gpujpeg_depth_decoder_create(gpuStream_t) { return NULL; }
void gpujpeg_depth_decoder_destroy(struct gpujpeg_depth_decoder*) {}
int gpujpeg_depth_decoder_decode(struct gpujpeg_depth_decoder*, const uint8_t*, size_t, int, int, uint16_t*, int)
{
    fprintf(stderr, "[GPUJPEG] [Error] depth codec: not implemented for the SYCL backend yet\n");
    return -1;
}

#endif // GPUJPEG_USE_SYCL
