
/*
    Copyright (C) 2025  MetaVi Labs Inc.

    This source listing is licensed to ibidi GmbH for use in creating controlling
    software for ibidi produced microscopes. This license is a component of a broader
    agreement (not presented here) between ibidi GmbH and MetaVi Labs Inc.

    This license grants rights for derivative works and improvements limited to microscope components
    including Auto Focus and basic image enhancement. This license excludes ibidi from creating advanced image analysis
    based on this software. For example, ibidi may not use this source code to create its own image
    processing programs for cell segmentation, cell identification, or advanced features which rely on
    cell segmentation or identification.
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