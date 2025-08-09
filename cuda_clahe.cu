/*
    Copyright (C) 2022  MetaVi Labs Inc.

    This source listing is licensed to ibidi GmbH for use in creating controlling
    software for ibidi produced microscopes. This license is a component of a broader
    agreement (not presented here) between ibidi GmbH and MetaVi Labs Inc.

    This license grants rights for derivative works and improvements limited to microscope components
    including Auto Focus and basic image enhancement. This license excludes ibidi from creating advanced image analysis
    based on this software. For example, ibidi may not use this source code to create its own image
    processing programs for cell segmentation, cell identification, or advanced features which rely on
    cell segmentation or identification.
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
// Device Kernels Declarations (assumed to be implemented elsewhere)
// ----------------------------------------------------------------------------
__global__ void computeTileLUT(const unsigned char* input, int width, int height,
                               int tileSize, int clipLimit,
                               unsigned char* tileLUT,
                               int tilesX, int tilesY);

__global__ void applyCLAHE(const unsigned char* input, unsigned char* output,
                           int width, int height, int tileSize,
                           int tilesX, int tilesY,
                           const unsigned char* tileLUT);

// ----------------------------------------------------------------------------
// Host function: CLAHE_8u
// ----------------------------------------------------------------------------
int CLAHE_8u(unsigned char *input, unsigned char *output,
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
    dim3 blockTile(16, 16);
    dim3 gridTile(tilesX, tilesY);
    computeTileLUT<<<gridTile, blockTile>>>(d_input, width, height,
                                            tileSize, effectiveClipLimit,
                                            d_tileLUT, tilesX, tilesY);
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

    
    // Launch some kernel or perform asynchronous operations here...
    cudaEventRecord(stop, 0); // Record the stop event
    cudaEventSynchronize(stop); // Ensure the stop event has been recorded
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CUDA Elapsed time: %f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ----------------------------------------------------------------------------
    // Launch Kernel 2: applyCLAHE
    // ----------------------------------------------------------------------------
    // Process the entire image with one thread per pixel.
    // dim3 block(32, 32);
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    applyCLAHE<<<grid, block>>>(d_input, d_output, width, height,
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

    // Return 0 on success.
    return 0;
}



// This kernel is launched with one block per tile.
__global__ void computeTileLUT(const unsigned char* input, int width, int height,
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


__global__ void applyCLAHE(const unsigned char* input, unsigned char* output,
                           int width, int height, int tileSize,
                           int tilesX, int tilesY,
                           const unsigned char* tileLUT)
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


}}
