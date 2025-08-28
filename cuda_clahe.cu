/*
    Copyright (C) 2022  MetaVi Labs Inc.

*/
#define CLAHE_PRIVATE
#include "cuda_clahe.h"
#include "stdio.h"
#include <math.h>
#include "malloc.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <npp.h>
#include <algorithm>
using std::min;
using std::max;


namespace MetaViLabs
{
namespace Cuda
{

//#define TILE_SIZE 8  // Size of the CLAHE tile (e.g., 8x8)
#define HIST_SIZE 256 // Number of histogram bins


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>  // For std::min and std::max

// Define histogram size (number of gray levels)
#define HIST_SIZE 256

// ----------------------------------------------------------------------------
// Device Kernels Declarations
// ----------------------------------------------------------------------------
/**
 * @brief Baseline CLAHE LUT builder for one tile per block (shared histogram + atomics).
 *
 * Builds a 256-bin histogram in shared memory for the block's tile using shared-memory atomics,
 * clips to @p clipLimit, redistributes excess uniformly, computes the CDF, and writes a 256-entry
 * LUT for the tile.
 *
 * @details
 * - One block == one tile. Threads cooperatively iterate tile pixels.
 * - Uses shared-memory atomics on a single histogram array -> can serialize on hot bins.
 * - CDF + LUT generation done by a single thread (cheap for 256 bins).
 * - NOTE: The original version used an inverted mapping (255 - mapped_value). Keep mapping
 *   consistent across variants to avoid “inverted” appearance differences.
 *
 * @param input     Pointer to grayscale image (width*height bytes).
 * @param width     Image width in pixels.
 * @param height    Image height in pixels.
 * @param tileSize  Tile size in pixels (tileSize x tileSize).
 * @param clipLimit Per-bin clip limit for CLAHE (counts).
 * @param tileLUT   Output LUT buffer [tilesX * tilesY * 256].
 * @param tilesX    Number of tiles along X = ceil(width / tileSize).
 * @param tilesY    Number of tiles along Y = ceil(height / tileSize).
 *
 * @note Launch: dim3 grid(tilesX, tilesY), dim3 block(16,16) (or similar).
 * @note Shared memory: ~256 * sizeof(int) for histogram (+ optional for CDF).
 */
__global__ void originalComputeTileLUT(const unsigned char* input, int width, int height,
                                       int tileSize, int clipLimit,
                                       unsigned char* tileLUT,
                                       int tilesX, int tilesY);

/**
 * @brief Baseline CLAHE LUT builder with aliasing hints (__restrict__) for better codegen.
 *
 * Same algorithm as originalComputeTileLUT but with __restrict__ on pointer parameters to help
 * the compiler use the read-only path and reorder memory ops safely.
 *
 * @details
 * - Often yields small speedups on embedded and desktop GPUs.
 * - Algorithmic behavior identical to baseline (be consistent with LUT inversion choice).
 *
 * @see originalComputeTileLUT
 */
__global__ void computeTileLUT(const unsigned char* __restrict__ input,
                               int width, int height,
                               int tileSize, int clipLimit,
                               unsigned char* __restrict__ tileLUT,
                               int tilesX, int tilesY);
        
/**
 * @brief Per-warp histograms in shared memory + reduction (reduces atomic contention).
 *
 * Each warp updates its own 256-bin histogram in shared memory (no inter-warp conflicts).
 * Afterward, all warps reduce into a final histogram, then clip/redistribute and build the LUT.
 *
 * @details
 * - Great when many threads hit the same bins (contentious histograms).
 * - Shared memory usage = WARPS_PER_BLOCK * 256 * 4B (+ 256 * 4B final) -> may reduce occupancy.
 * - Includes a strided zeroing pass and a shared-memory reduction step.
 *
 * @param ... (same as baseline)
 *
 * @note Launch: one tile per block; blockDim.x * blockDim.y must be a multiple of 32.
 * @note Dynamic shared memory: (WARPS * 256 + 256) * sizeof(unsigned int).
 * @warning Too much shared memory can lower active blocks/SM and hurt performance on Jetsons.
 */
__global__ void fasterComputeTileLUT_opt(const unsigned char* __restrict__ input,
                                         int width, int height,
                                         int tileSize, int clipLimit,
                                         unsigned char* __restrict__ tileLUT,
                                         int tilesX, int tilesY);

/**
 * @brief Warp-aggregated atomics histogram (aggregate identical values within a warp).
 *
 * Uses __match_any_sync to find threads in a warp that observed the same pixel value and performs
 * a single atomicAdd per unique value per warp, reducing atomic traffic to the shared histogram.
 * Then clips/redistributes and builds the LUT.
 *
 * @details
 * - Big wins when local texture causes many lanes in a warp to see the same value.
 * - Little benefit on random/noisy tiles (extra control/mask ops without fewer atomics).
 * - Histogram is __align__(128) to reduce bank conflicts.
 *
 * @param ... (same as baseline)
 *
 * @note Requires sm_70+ for __match_any_sync (Volta or newer).
 * @note Mapping in this kernel is the standard (non-inverted) CLAHE unless changed to match baseline.
 */
__global__ void test_A(const unsigned char* input,
                       int width, int height,
                       int tileSize, int clipLimit,
                       unsigned char* tileLUT,
                       int tilesX, int tilesY);

/**
 * @brief Simplified fast path with shared atomics + visual-quality tweaks.
 *
 * Keeps simple shared-memory atomics for the histogram (often efficient on Jetsons),
 * then applies two post-process adjustments when building the LUT:
 *  1) Border-tile area scaling: normalizes partial tiles so borders behave like full tiles.
 *  2) Brightness preservation & gentle identity blend: shifts output mean toward input mean
 *     (+ optional brighten offset) and blends slightly toward identity to reduce halos.
 *
 * @details
 * - Not “strict” CLAHE: adds controlled brightness/contrast tweaks for better visual consistency.
 * - Fast, robust choice for embedded targets; minimal control overhead.
 *
 * @param ... (same as baseline)
 *
 * @note Tune @c brighten_abs and @c alpha to taste; set both to zero for pure CLAHE behavior.
 * @note Mapping here is the standard (non-inverted) CLAHE unless deliberately inverted.
 */
__global__ void test_A_Modified_fast(const unsigned char*  input,
                                     int width, int height,
                                     int tileSize, int clipLimit,
                                     unsigned char*  tileLUT,
                                     int tilesX, int tilesY);

/**
 * @brief Vectorized global loads (uchar4) to improve memory throughput when aligned.
 *
 * Each thread processes 4 adjacent pixels along X using a single uchar4 load per iteration,
 * then updates the shared histogram with 4 atomics. Falls back to scalar loads for the tail.
 *
 * @details
 * - Can significantly reduce global load instructions and improve coalescing.
 * - Alignment-sensitive: reinterpret_cast<const uchar4*> prefers 4-byte aligned pointers.
 *   Consider adding an alignment prologue (scalar until aligned) for best results.
 *
 * @param ... (same as baseline)
 *
 * @note Combine with an alignment prologue like in hist_vec_warpagg for maximum benefit.
 * @note Mapping here is standard (non-inverted) unless changed.
 */
__global__ void test_C(const unsigned char*  input,
                       int width, int height,
                       int tileSize, int clipLimit,
                       unsigned char*  tileLUT,
                       int tilesX, int tilesY);

/**
 * @brief Apply-stage kernel: bilinear blend of four neighboring tile LUTs for each pixel.
 *
 * For each output pixel (x,y), finds the surrounding 2x2 tiles, looks up the mapped value
 * in each tile's LUT at input(x,y), and writes a bilinear interpolation of the four results.
 *
 * @details
 * - One thread == one pixel; memory-bound.
 * - Access pattern coalesces if blockDim.x is a multiple of 32 (row-major).
 * - Uses centered-tiles convention (x/tileSize - 0.5f) to compute fractional tile coords.
 * - Consider precomputing invTile = 1.0f/tileSize on host and replacing divisions.
 * - Fast path ideas: split interior vs border kernel; optionally stage 4 LUTs in shared
 *   when blocks align to tile cells; optionally vectorize I/O with uchar4 when width%4==0.
 *
 * @param input     Grayscale input image.
 * @param output    Output image (CLAHE-applied).
 * @param width     Image width.
 * @param height    Image height.
 * @param tileSize  Tile size used during LUT construction.
 * @param tilesX    Number of tiles along X.
 * @param tilesY    Number of tiles along Y.
 * @param tileLUT   All tiles’ LUTs [tilesX * tilesY * 256].
 *
 * @note Launch: dim3 block(32,8) or (16,16); grid = ceil(width/block.x) x ceil(height/block.y).
 * @note Border handling: clamps (tx,ty) so (tx+1,ty+1) is valid; an interior-only kernel can
 *       remove clamps/branches for speed and handle borders in a second pass.
 */
__global__ void applyCLAHE(const unsigned char* __restrict__ input,
                           unsigned char* __restrict__ output,
                           int width, int height, int tileSize,
                           int tilesX, int tilesY,
                           const unsigned char* __restrict__ tileLUT);

/**
 * @brief Vectorized loads + warp-aggregated atomics + alignment prologue (hybrid fast path).
 *
 * Hybrid approach that:
 *  - Scalars forward until 4-byte aligned,
 *  - Uses uchar4 vectorized loads across the row,
 *  - Applies warp aggregation (__match_any_sync) to reduce atomics for each byte of the vector.
 * Then clips/redistributes and builds the LUT.
 *
 * @details
 * - Often best on desktop GPUs (Ada/4090) where warp ops are cheap and bandwidth is high.
 * - May be neutral/negative on some Jetsons if register pressure reduces occupancy.
 *
 * @param ... (same as baseline)
 *
 * @note Requires sm_70+ for __match_any_sync.
 * @note Keep shared histogram __align__(128) to reduce bank conflicts.
 */
__global__ void hist_vec_warpagg(const unsigned char* __restrict__ input,
                                 int width, int height,
                                 int tileSize, int clipLimit,
                                 unsigned char* __restrict__ tileLUT,
                                 int tilesX, int tilesY);


// ----------------------------------------------------------------------------
// Host function: CLAHE_8u
// ----------------------------------------------------------------------------
int CLAHE_8u(unsigned char* __restrict__ input, unsigned char* __restrict__ output,
             int width, int height, int clipLimit, int tileSize)
{
    cudaError_t err;

    cudaEvent_t start, stop;

    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to create start event: %s\n", cudaGetErrorString(err));
    }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to create stop event: %s\n", cudaGetErrorString(err));
    }


    // Compute image size in bytes.
    const size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate device memory for input and output images.
    unsigned char *d_input = nullptr;
    unsigned char *d_output = nullptr;
    err = cudaMalloc((void**)&d_input, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_input: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc((void**)&d_output, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_output: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return -1;
    }

    // Copy the input image from host to device.
    err = cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for input: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }

    // Compute the number of tiles in X and Y directions.
    int tilesX = (width + tileSize - 1) / tileSize;
    int tilesY = (height + tileSize - 1) / tileSize;

    // Allocate device memory for the tile LUTs.
    // Each tile has a LUT of HIST_SIZE bytes.
    const size_t tileLUTSize = tilesX * tilesY * HIST_SIZE * sizeof(unsigned char);
    unsigned char *d_tileLUT = nullptr;
    err = cudaMalloc((void**)&d_tileLUT, tileLUTSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_tileLUT: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }

    int numPixels = tileSize * tileSize;
    //int effectiveClipLimit = max(clipLimit * numPixels / HIST_SIZE, 1);
    int effectiveClipLimit =clipLimit;

    cudaEventRecord(start, 0); // Record the start event on the default stream

    // ----------------------------------------------------------------------------
    // Launch Kernel 1: computeTileLUT
    // ----------------------------------------------------------------------------
    // For example, use a 16x16 block; each block processes one tile.
    dim3 blockTile(16,16);                 // 256 threads = 8 warps
    dim3 gridTile(tilesX, tilesY);

    originalComputeTileLUT<<<gridTile, blockTile>>>(d_input, width, height,
                                        tileSize, effectiveClipLimit,
                                        d_tileLUT, tilesX, tilesY);

    // ----------------------------------------------------------------------------
    // Histogram per warp kernal launch code

    // int tpb   = blockTile.x * blockTile.y;
    // int warps = tpb / 32;                  // requires multiple of 32
    // size_t shmem = (warps * HIST_SIZE + HIST_SIZE) * sizeof(unsigned int);
    // //            ^ per-warp hist + final hist

    // fasterComputeTileLUT_opt<<<gridTile, blockTile, shmem>>>(
    //     d_input, width, height, tileSize, effectiveClipLimit,
    //     d_tileLUT, tilesX, tilesY);
    // ----------------------------------------------------------------------------

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "computeTileLUT kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_tileLUT);
        return -1;
    }

    // Wait for the kernel to finish.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "computeTileLUT kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_tileLUT);
        return -1;
    }

    cudaEventRecord(stop, 0); // Record the stop event
    cudaEventSynchronize(stop); // Ensure the stop event has been recorded
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernal 1 Elapsed time: %f ms\n", elapsedTime);

    // ----------------------------------------------------------------------------
    // Launch Kernel 2: applyCLAHE
    // ----------------------------------------------------------------------------
    // Process the entire image with one thread per pixel.
    // dim3 block(32, 32);

    cudaEventRecord(start, 0); // Record the start event on the default stream

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    originalApplyCLAHE<<<grid, block>>>(d_input, d_output, width, height,
                                tileSize, tilesX, tilesY, d_tileLUT);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "applyCLAHE kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_tileLUT);
        return -1;
    }

    // Wait for the kernel to finish.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "applyCLAHE kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_tileLUT);
        return -1;
    }

    // Copy the output image from device to host.
    err = cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for output: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_tileLUT);
        return -1;
    }

    // Free device memory.
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_tileLUT);

    // Launch some kernel or perform asynchronous operations here...
    cudaEventRecord(stop, 0); // Record the stop event
    cudaEventSynchronize(stop); // Ensure the stop event has been recorded
    elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernal 2 Elapsed time: %f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Return 0 on success.
    return 0;
}

// This kernel is launched with one block per tile.
__global__ void originalComputeTileLUT(const unsigned char* input, int width, int height,
                               int tileSize, int clipLimit,
                               unsigned char* tileLUT,
                               int tilesX, int tilesY)
{
    // Determine which tile we are processing.
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tileIndex = tileY * tilesX + tileX;
    
    // Determine tile boundaries in the image.
    int startX = tileX * tileSize;
    int startY = tileY * tileSize;
    int endX   = min(startX + tileSize, width);
    int endY   = min(startY + tileSize, height);
    
    // Create a shared histogram.
    __shared__ int hist[HIST_SIZE];
    // Initialize histogram (first HIST_SIZE threads do the work)
    for (int i = threadIdx.x; i < HIST_SIZE; i += blockDim.x)
    {
        hist[i] = 0;
    }

    __syncthreads();
    
    // Accumulate histogram over the tile.
    // Use a 2D loop over the tile pixels. (Threads cooperate.)
    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y)
    {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x)
        {
            int idx = y * width + x;
            unsigned char pix = input[idx];
            atomicAdd(&hist[pix], 1);
        }
    }

    __syncthreads();
    
    // Clip histogram and redistribute excess.
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        int totalExcess = 0;
        for (int i = 0; i < HIST_SIZE; i++)
        {
            int excess = hist[i] - clipLimit;
            if (excess > 0)
            {
                hist[i] = clipLimit;
                totalExcess += excess;
            }
        }
        // Evenly redistribute the excess counts.
        int distribute = totalExcess / HIST_SIZE;
        for (int i = 0; i < HIST_SIZE; i++)
        {
            hist[i] += distribute;
        }
    }
    __syncthreads();
    
    // Compute CDF and build LUT.
    // We compute the CDF on one thread (or a few threads if you wish to parallelize the scan).
    __shared__ int cdf[HIST_SIZE];
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        cdf[0] = hist[0];
        for (int i = 1; i < HIST_SIZE; i++)
            cdf[i] = cdf[i - 1] + hist[i];
        
        // The minimum nonzero CDF value.
        int cdfMin = cdf[0];
        // Number of pixels in this tile.
        int numPixels = (endX - startX) * (endY - startY);
        
        // Build LUT: map each gray level into [0, 255].
        for (int i = 0; i < HIST_SIZE; i++)
        {
            // Avoid division by zero.
            tileLUT[tileIndex * HIST_SIZE + i] = 255- (unsigned char)(((cdf[i] - cdfMin) * 255) / max(numPixels - cdfMin, 1));
        }
    }
}


// This kernel is launched with one block per tile.
__global__ void computeTileLUT(const unsigned char* __restrict__ input, 
                               int width, int height,
                               int tileSize, int clipLimit,
                               unsigned char* __restrict__ tileLUT,
                               int tilesX, int tilesY)
{
    // Determine which tile we are processing.
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tileIndex = tileY * tilesX + tileX;
    
    // Determine tile boundaries in the image.
    int startX = tileX * tileSize;
    int startY = tileY * tileSize;
    int endX   = min(startX + tileSize, width);
    int endY   = min(startY + tileSize, height);
    
    // Create a shared histogram.
    __shared__ int hist[HIST_SIZE];
    // Initialize histogram (first HIST_SIZE threads do the work)
    for (int i = threadIdx.x; i < HIST_SIZE; i += blockDim.x)
    {
        hist[i] = 0;
    }

    __syncthreads();
    
    // Accumulate histogram over the tile.
    // Use a 2D loop over the tile pixels. (Threads cooperate.)
    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y)
    {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x)
        {
            int idx = y * width + x;
            unsigned char pix = input[idx];
            atomicAdd(&hist[pix], 1);
        }
    }

    __syncthreads();
    
    // Clip histogram and redistribute excess.
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        int totalExcess = 0;
        for (int i = 0; i < HIST_SIZE; i++)
        {
            int excess = hist[i] - clipLimit;
            if (excess > 0)
            {
                hist[i] = clipLimit;
                totalExcess += excess;
            }
        }
        // Evenly redistribute the excess counts.
        int distribute = totalExcess / HIST_SIZE;
        for (int i = 0; i < HIST_SIZE; i++)
        {
            hist[i] += distribute;
        }
    }
    __syncthreads();
    
    // Compute CDF and build LUT.
    // We compute the CDF on one thread (or a few threads if you wish to parallelize the scan).
    __shared__ int cdf[HIST_SIZE];
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        cdf[0] = hist[0];
        for (int i = 1; i < HIST_SIZE; i++)
            cdf[i] = cdf[i - 1] + hist[i];
        
        // The minimum nonzero CDF value.
        int cdfMin = cdf[0];
        // Number of pixels in this tile.
        int numPixels = (endX - startX) * (endY - startY);
        
        // Build LUT: map each gray level into [0, 255].
        for (int i = 0; i < HIST_SIZE; i++)
        {
            // Avoid division by zero.
            tileLUT[tileIndex * HIST_SIZE + i] = 255- (unsigned char)(((cdf[i] - cdfMin) * 255) / max(numPixels - cdfMin, 1));
        }
    }
}

__global__ void test_A(const unsigned char* input,
                       int width, int height,
                       int tileSize, int clipLimit,
                       unsigned char* tileLUT,
                       int tilesX, int tilesY)
{
    const int tileX = blockIdx.x;
    const int tileY = blockIdx.y;
    const int tileIndex = tileY * tilesX + tileX;

    const int startX = tileX * tileSize;
    const int startY = tileY * tileSize;
    const int endX   = min(startX + tileSize, width);
    const int endY   = min(startY + tileSize, height);

    __shared__ __align__(128) unsigned int hist[HIST_SIZE];

    // zero with ALL threads
    const int tpb = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < HIST_SIZE; i += tpb) {
        hist[i] = 0u;
    }
    __syncthreads();

    const int lane = tid & 31;

    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x) {
            const int idx = y * width + x;
            const unsigned int pix = input[idx];

            // Use only active lanes in this iteration
            unsigned m = __match_any_sync(__activemask(), pix);
            int leader = __ffs(m) - 1;  // 0..31
            if (lane == leader) {
                int votes = __popc(m);
                atomicAdd(&hist[pix], votes);
            }
        }
    }
    __syncthreads();

    // Clip + redistribute
    if (tid == 0) {
        int totalExcess = 0;
        for (int i = 0; i < HIST_SIZE; ++i) {
            int v = (int)hist[i];
            if (v > clipLimit) {
                totalExcess += (v - clipLimit);
                hist[i] = clipLimit;
            }
        }
        const int distribute = totalExcess / HIST_SIZE;
        for (int i = 0; i < HIST_SIZE; ++i) {
            hist[i] += distribute;
        }

        // CDF + LUT
        int cdf = 0, cdfMin = -1;
        const int numPixels = (endX - startX) * (endY - startY);
        for (int i = 0; i < HIST_SIZE; ++i) {
            cdf += hist[i];
            if (cdfMin < 0 && cdf) cdfMin = cdf;
            tileLUT[tileIndex * HIST_SIZE + i] =
                (unsigned char)(((cdf - cdfMin) * 255) / max(numPixels - cdfMin, 1));
        }
    }
}


// This kernel is launched with one block per tile.
#ifndef HIST_SIZE
#define HIST_SIZE 256
#endif

// Faster luminance-preserving CLAHE LUT builder.
// Keep the same launch config you already use (one block per tile, e.g., 16x16).
__global__ void test_A_Modified_fast(const unsigned char* __restrict__ input,
                                     int width, int height,
                                     int tileSize, int clipLimit,
                                     unsigned char* __restrict__ tileLUT,
                                     int tilesX, int tilesY)
{
    const int tileX = blockIdx.x, tileY = blockIdx.y;
    const int tileIndex = tileY * tilesX + tileX;

    const int startX = tileX * tileSize;
    const int startY = tileY * tileSize;
    const int endX   = min(startX + tileSize, width);
    const int endY   = min(startY + tileSize, height);

    __shared__ __align__(128) unsigned int hist[HIST_SIZE];

    // Zero histogram with all threads
    const int bx  = blockDim.x, by = blockDim.y;
    const int tpb = bx * by;
    const int tid = threadIdx.y * bx + threadIdx.x;
    for (int i = tid; i < HIST_SIZE; i += tpb) hist[i] = 0u;
    __syncthreads();

    // Accumulate histogram (simple shared atomics have been best on your GPUs)
    for (int y = startY + threadIdx.y; y < endY; y += by) {
        int idx = y * width + startX + threadIdx.x;
        for (int x = startX + threadIdx.x; x < endX; x += bx, ++idx) {
            atomicAdd(&hist[input[idx]], 1u);
        }
    }
    __syncthreads();

    // Single-thread post processing per tile
    if (tid == 0) {
        // 1) Clip & redistribute (standard CLAHE)
        int excess = 0;
        for (int i = 0; i < HIST_SIZE; ++i) {
            int v = (int)hist[i];
            if (v > clipLimit) { excess += (v - clipLimit); v = clipLimit; }
            hist[i] = (unsigned int)v;
        }
        const int distribute = excess / HIST_SIZE;
        if (distribute) {
            for (int i = 0; i < HIST_SIZE; ++i) hist[i] += (unsigned int)distribute;
        }

        // 2) Build CDF → provisional LUT (brightness-preserving, border fix)
        const int fullTile = tileSize * tileSize;
        const int tileArea = max((endX - startX) * (endY - startY), 1);
        const float scale  = (float)fullTile / (float)tileArea;  // treat border tiles as full tiles

        // meanIn = sum(i*hist[i]) / tileArea
        unsigned long long sumIn = 0ull;
        for (int i = 0; i < HIST_SIZE; ++i) sumIn += (unsigned long long)i * (unsigned long long)hist[i];
        const float meanIn = (float)sumIn / (float)tileArea;

        // Provisional LUT and meanOut in one pass
        unsigned int cdf = 0;
        int cdfMin = -1;
        float meanOutNum = 0.0f;
        unsigned char* dst = &tileLUT[tileIndex * HIST_SIZE];

        for (int i = 0; i < HIST_SIZE; ++i) {
            cdf += hist[i];
            if (cdfMin < 0 && cdf) cdfMin = (int)cdf;

            // Standard CLAHE mapping, with area scaling to avoid vignette
            const float num = ((float)cdf - (float)cdfMin) * scale;
            const float den = fmaxf((float)fullTile - (float)cdfMin * scale, 1.0f);
            float v = 255.0f * (num / den);
            v = fminf(255.0f, fmaxf(0.0f, v));
            const unsigned char u8 = (unsigned char)(v + 0.5f);
            dst[i] = u8;

            meanOutNum += v * (float)hist[i];
        }
        const float meanOut = meanOutNum / (float)tileArea;

        // 3) Brightness shift (+ a tiny identity blend to avoid halos)
        //    (Set brighten_abs=0 and alpha=0 for pure CLAHE behavior.)
        const float brighten_abs = 8.0f;  // try 5–12; or make this a parameter
        const float alpha        = 0.10f; // 0..0.25 small identity mix; 0 = none

        const float delta = (meanIn + brighten_abs) - meanOut;

        for (int i = 0; i < HIST_SIZE; ++i) {
            float v = (float)dst[i] + delta;              // brightness shift
            v = (1.0f - alpha)*v + alpha*(float)i;       // gentle identity blend
            v = fminf(255.0f, fmaxf(0.0f, v));           // clamp
            dst[i] = (unsigned char)(v + 0.5f);
        }
    }
}

__global__ void test_C(const unsigned char*  input, 
                       int width, int height,
                       int tileSize, int clipLimit,
                       unsigned char*  tileLUT,
                       int tilesX, int tilesY)
{
    const int tileX = blockIdx.x, tileY = blockIdx.y;
    const int tileIndex = tileY * tilesX + tileX;

    const int startX = tileX * tileSize;
    const int startY = tileY * tileSize;
    const int endX   = min(startX + tileSize, width);
    const int endY   = min(startY + tileSize, height);

    __shared__ __align__(128) unsigned int hist[HIST_SIZE];

    const int tpb = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < HIST_SIZE; i += tpb) hist[i] = 0;
    __syncthreads();

    // Vectorized over X: 4 pixels per thread
    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        int x = startX + (threadIdx.x << 2);  // *4

        // fast path: 4 at a time (ensure we stay inside)
        for (; x + 3 < endX; x += (blockDim.x << 2)) {
            const uchar4 v = *reinterpret_cast<const uchar4*>(input + y*width + x);
            atomicAdd(&hist[v.x], 1u);
            atomicAdd(&hist[v.y], 1u);
            atomicAdd(&hist[v.z], 1u);
            atomicAdd(&hist[v.w], 1u);
        }

        // tail (0..3 pixels)
        for (; x < endX; ++x) {
            unsigned char pix = input[y*width + x];
            atomicAdd(&hist[pix], 1u);
        }
    }
    __syncthreads();

    // Clip + redistribute + CDF + LUT (same as A)
    if (tid == 0) {
        int totalExcess = 0;
        for (int i = 0; i < HIST_SIZE; ++i) {
            int v = (int)hist[i];
            if (v > clipLimit) { totalExcess += (v - clipLimit); hist[i] = clipLimit; }
        }
        const int distribute = totalExcess / HIST_SIZE;
        for (int i = 0; i < HIST_SIZE; ++i) hist[i] += distribute;

        int cdf = 0, cdfMin = -1;
        const int numPixels = (endX - startX) * (endY - startY);
        for (int i = 0; i < HIST_SIZE; ++i) {
            cdf += hist[i];
            if (cdfMin < 0 && cdf) cdfMin = cdf;
            tileLUT[tileIndex * HIST_SIZE + i] =
                (unsigned char)(((cdf - cdfMin) * 255) / max(numPixels - cdfMin, 1));
        }
    }
}

#ifndef HIST_SIZE
#define HIST_SIZE 256
#endif

// One tile per block. Works for sm_70+ (uses __match_any_sync).
__global__ void hist_vec_warpagg(const unsigned char* __restrict__ input,
                                 int width, int height,
                                 int tileSize, int clipLimit,
                                 unsigned char* __restrict__ tileLUT,
                                 int tilesX, int tilesY)
{
    const int tileX = blockIdx.x, tileY = blockIdx.y;
    const int tileIndex = tileY * tilesX + tileX;

    const int startX = tileX * tileSize;
    const int startY = tileY * tileSize;
    const int endX   = min(startX + tileSize, width);
    const int endY   = min(startY + tileSize, height);

    // Shared histogram (aligned to reduce bank conflicts)
    __shared__ __align__(128) unsigned int hist[HIST_SIZE];

    const int bx = blockDim.x, by = blockDim.y;
    const int tpb = bx * by;
    const int tid = threadIdx.y * bx + threadIdx.x;
    const int lane = tid & 31;

    // Zero with all threads
    for (int i = tid; i < HIST_SIZE; i += tpb) hist[i] = 0;
    __syncthreads();

    // Prologue: walk to 4-byte alignment for vector loads
    for (int y = startY + threadIdx.y; y < endY; y += by) {
        int x = startX + threadIdx.x;

        // Scalar until pointer is 4-byte aligned or we run out
        for (; x < endX && ((y * width + x) & 3); x += bx) {
            const unsigned char pix = input[y * width + x];
            unsigned m = __match_any_sync(__activemask(), (unsigned)pix);
            int leader = __ffs(m) - 1;
            if (lane == leader) atomicAdd(&hist[pix], __popc(m));
        }

        // Vectorized body: uchar4 per iteration
        for (; x + 3 < endX; x += (bx << 2)) {
            const int base = y * width + x;
            // Safe on modern GPUs; if you want to be strict about alignment, guard with ((base & 3) == 0)
            const uchar4 v = *reinterpret_cast<const uchar4*>(input + base);

            // Warp-aggregated atomics: do it 4 times (one per byte)
            auto update = [&](unsigned p) {
                unsigned m = __match_any_sync(__activemask(), p);
                int leader = __ffs(m) - 1;
                if (lane == leader) atomicAdd(&hist[p], __popc(m));
            };
            update((unsigned)v.x);
            update((unsigned)v.y);
            update((unsigned)v.z);
            update((unsigned)v.w);
        }

        // Tail (0..3 pixels)
        for (; x < endX; x += bx) {
            const unsigned char pix = input[y * width + x];
            unsigned m = __match_any_sync(__activemask(), (unsigned)pix);
            int leader = __ffs(m) - 1;
            if (lane == leader) atomicAdd(&hist[pix], __popc(m));
        }
    }
    __syncthreads();

    // Clip + redistribute + CDF + LUT (single thread; cheap for 256 bins)
    if (tid == 0) {
        int totalExcess = 0;
        for (int i = 0; i < HIST_SIZE; ++i) {
            int v = (int)hist[i];
            if (v > clipLimit) { totalExcess += (v - clipLimit); hist[i] = clipLimit; }
        }
        const int distribute = totalExcess / HIST_SIZE;
        for (int i = 0; i < HIST_SIZE; ++i) hist[i] += distribute;

        int cdf = 0, cdfMin = -1;
        const int numPixels = (endX - startX) * (endY - startY);
        for (int i = 0; i < HIST_SIZE; ++i) {
            cdf += hist[i];
            if (cdfMin < 0 && cdf) cdfMin = cdf;
            tileLUT[tileIndex * HIST_SIZE + i] =
                (unsigned char)(((cdf - cdfMin) * 255) / max(numPixels - cdfMin, 1));
        }
    }
}


#ifndef HIST_SIZE
#define HIST_SIZE 256
#endif

// One tile per block; blockDim.x * blockDim.y must be a multiple of 32
__global__ void fasterComputeTileLUT_opt(const unsigned char* __restrict__ input,
                                   int width, int height,
                                   int tileSize, int clipLimit,
                                   unsigned char* __restrict__ tileLUT,
                                   int tilesX, int tilesY)
{
    // Which tile does this block handle?
    const int tileX = blockIdx.x, tileY = blockIdx.y;
    const int tileIndex = tileY * tilesX + tileX;

    // Tile bounds
    const int startX = tileX * tileSize;
    const int startY = tileY * tileSize;
    const int endX   = min(startX + tileSize, width);
    const int endY   = min(startY + tileSize, height);

    // Thread ids
    const int tpb  = blockDim.x * blockDim.y;             // threads per block
    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp = tid >> 5;                             // /32
    const int WARPS_PER_BLOCK = tpb >> 5;

    // Shared memory layout: [ warpHists (WARPS×256) | finalHist (256) ]
    extern __shared__ unsigned int s[];
    unsigned int* warpHists = s;                           // WARPS_PER_BLOCK * HIST_SIZE
    unsigned int* finalHist = warpHists + WARPS_PER_BLOCK * HIST_SIZE;

    // Zero all shared memory (use ALL threads, striding)
    for (int i = tid; i < WARPS_PER_BLOCK * HIST_SIZE + HIST_SIZE; i += tpb)
        s[i] = 0;
    __syncthreads();

    // Each warp updates its own histogram -> far fewer collisions
    unsigned int* myHist = warpHists + warp * HIST_SIZE;

    // Accumulate tile histogram
    for (int y = startY + threadIdx.y; y < endY; y += blockDim.y) {
        for (int x = startX + threadIdx.x; x < endX; x += blockDim.x) {
            int idx = y * width + x;                      
            unsigned char pix = input[idx];
            atomicAdd(&myHist[pix], 1u);
        }
    }
    __syncthreads();

    // Reduce per-warp histograms -> finalHist
    for (int bin = tid; bin < HIST_SIZE; bin += tpb) {
        unsigned int sum = 0;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w)
            sum += warpHists[w * HIST_SIZE + bin];
        finalHist[bin] = sum;
    }
    __syncthreads();

    // Clip + redistribute + CDF + LUT (single thread; cheap for 256 bins)
    if (tid == 0) {
        int totalExcess = 0;
        for (int i = 0; i < HIST_SIZE; ++i) {
            int v = (int)finalHist[i];
            if (v > clipLimit) {
                totalExcess += (v - clipLimit);
                finalHist[i] = clipLimit;
            }
        }
        const int distribute = totalExcess / HIST_SIZE;
        for (int i = 0; i < HIST_SIZE; ++i)
            finalHist[i] += distribute;

        // CDF
        int cdf = 0, cdfMin = -1;
        for (int i = 0; i < HIST_SIZE; ++i) {
            cdf += finalHist[i];
            if (cdfMin < 0 && cdf) cdfMin = cdf;
            int numPixels = (endX - startX) * (endY - startY);
            // map into [0,255]
            tileLUT[tileIndex * HIST_SIZE + i] =
                (unsigned char)(((cdf - cdfMin) * 255) / max(numPixels - cdfMin, 1));
        }
    }
}


__global__ void applyCLAHE(const unsigned char* __restrict__ input, 
                           unsigned char* __restrict__ output,
                           int width, int height, int tileSize,
                           int tilesX, int tilesY,
                           const unsigned char* __restrict__ tileLUT)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    
    // Determine the fractional tile coordinates.
    // We subtract 0.5 so that the centers of the tiles are used.
    float fx = ((float)x / tileSize) - 0.5f;
    float fy = ((float)y / tileSize) - 0.5f;
    
    // Find the top-left tile index.
    int tx = (int)floorf(fx);
    int ty = (int)floorf(fy);
    
    // Clamp indices so that (tx+1, ty+1) is valid.
    tx = max(0, min(tx, tilesX - 2));
    ty = max(0, min(ty, tilesY - 2));
    
    // Fractional distances.
    float dx = fx - tx;
    float dy = fy - ty;
    
    // Pointers to the LUTs for the four surrounding tiles.
    const unsigned char* lut11 = &tileLUT[(ty * tilesX + tx) * HIST_SIZE];
    const unsigned char* lut21 = &tileLUT[(ty * tilesX + (tx + 1)) * HIST_SIZE];
    const unsigned char* lut12 = &tileLUT[((ty + 1) * tilesX + tx) * HIST_SIZE];
    const unsigned char* lut22 = &tileLUT[((ty + 1) * tilesX + (tx + 1)) * HIST_SIZE];
    
    // Get the input pixel value.
    int pixelVal = input[y * width + x];
    
    // Look up the mapped values.
    float I11 = lut11[pixelVal];
    float I21 = lut21[pixelVal];
    float I12 = lut12[pixelVal];
    float I22 = lut22[pixelVal];
    
    // Bilinear interpolation.
    float w11 = (1.0f - dx) * (1.0f - dy);
    float w21 = dx * (1.0f - dy);
    float w12 = (1.0f - dx) * dy;
    float w22 = dx * dy;
    
    output[y * width + x] = (unsigned char)(I11 * w11 + I21 * w21 +
                                              I12 * w12 + I22 * w22);
}
        }
    }
