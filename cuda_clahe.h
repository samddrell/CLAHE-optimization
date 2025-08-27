
/*
    Copyright (C) 2025  MetaVi Labs Inc.

*/

#ifndef CLAHE_H
#define CLAHE_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>


#ifdef __cplusplus
    extern "C" {
#endif

#ifdef _WIN32
#include <stdint.h>
#ifdef CUDA_CLIB_EXPORTS
#define CUDA_CLIB_API __declspec(dllexport) 
#else
#define CUDA_CLIB_API __declspec(dllimport) 
#endif
#else
#define CUDA_CLIB_API 
#endif

        namespace MetaViLabs
        {
            namespace Cuda
            {

// public
CUDA_CLIB_API int CLAHE_8u(unsigned char *input, unsigned char *output, int width, int height, int clipLimit, int tileSize);

//private
#ifdef CLAHE_PRIVATE

__global__ void CLAHE_8u_Kernel(unsigned char *input, unsigned char *output, int width, int height, int clipLimit, int tileSize);

#endif // if private

            }
        }//end namespace


#ifdef __cplusplus
            }
#endif
#endif