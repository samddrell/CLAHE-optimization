// TODO: Build Test cases for CLAHE. Analyze speed and performance.
#include <fstream>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

#include "image.h"

int main()
{
    // Prepare CUDA CLAHE Images
    // Read in image data from a file
    std::string inFilePath = "p_00000009.png"; // Change this to your image file path
    std::string inFilePathSmall = "/DSC04441.jpg";
    Image smallImage;
    if (!(smallImage.OpenFile(inFilePathSmall))) 
    {
        printf("Failed to open image file: %s\n", inFilePathSmall.c_str());
        return 1; // Exit if the image cannot be opened
    }
    Image img;
    if (!(img.OpenFile(inFilePath))) 
    {
        printf("Failed to open image file: %s\n", inFilePath.c_str());
        return 1; // Exit if the image cannot be opened
    }
    Image bigImage(img);
    Image extraBigImage(bigImage);
    std::string ouFilePath1 = "normal_clahe_4_16.png";
    std::string ouFilePath2 = "big_clahe_4_16.png";
    std::string ouFilePath3 = "extra_big_clahe_4_16.png";
    std::string ouFilePath4 = "small_clahe_4_16.png";


    // Prepare CPU CLAHE Images
    // Read in image data from a file
    Image smallImageCPU;
    if (!(smallImageCPU.OpenFile(inFilePathSmall))) 
    {
        printf("Failed to open image file: %s\n", inFilePathSmall.c_str());
        return 1; // Exit if the image cannot be opened
    }
    Image imgCPU;
    if (!(imgCPU.OpenFile(inFilePath))) 
    {
        printf("Failed to open image file: %s\n", inFilePath.c_str());
        return 1; // Exit if the image cannot be opened
    }
    Image bigImageCPU(imgCPU);
    Image extraBigImageCPU(bigImageCPU);
    std::string ouFilePath1cpu = "CPU_normal_clahe_4_16.png";
    std::string ouFilePath2cpu = "CPU_big_clahe_4_16.png";
    std::string ouFilePath3cpu = "CPU_extra_big_clahe_4_16.png";
    std::string ouFilePath4cpu = "CPU_small_clahe_4_16.png";


    // Your kernel or image operation
    smallImage.CLAHE(4,16);
    img.CLAHE(4,16);
    bigImage.CLAHE(4,16);
    extraBigImage.CLAHE(4,16);

    smallImageCPU.cpuCLAHE(4,16);
    imgCPU.cpuCLAHE(4,16);
    bigImageCPU.cpuCLAHE(4,16);
    extraBigImageCPU.cpuCLAHE(4,16);
    // img.CLAHE(1,8);
    // img.CLAHE(1,8);
    // img.CLAHE(1,8);
    // img.CLAHE(4,8);
    // img.CLAHE(8,8);
    // img.CLAHE(2,8);
    // img.CLAHE(4,32);

    // CANNY Edge Detection
    std::string ouFilePathCANNY = "canny_applied.png";

    Image cannyImg;
    if (!(cannyImg.OpenFile(inFilePathSmall))) 
    {
        printf("Failed to open image file: %s\n", inFilePathSmall.c_str());
        return 1; // Exit if the image cannot be opened
    }
    cannyImg.cpuCANNY();
    // Save CUDA CLAHE Images
    if (!(cannyImg.SaveFile(ouFilePathCANNY))) 
    {
        printf("Failed to save image file: %s\n", ouFilePath4.c_str());
        return 1; // Exit if the image cannot be saved
    }

    // cudaEventRecord(stop);

    // cudaEventSynchronize(stop);
    float ms = 0;
    // cudaEventElapsedTime(&ms, start, stop);
    // printf("GPU CLAHE took %f ms to process %d x %d image\n", ms, img.m_width, img.m_height);

    // Save CUDA CLAHE Images
    // if (!(smallImage.SaveFile(ouFilePath4))) 
    // {
    //     printf("Failed to save image file: %s\n", ouFilePath4.c_str());
    //     return 1; // Exit if the image cannot be saved
    // }
    // if (!(img.SaveFile(ouFilePath1))) 
    // {
    //     printf("Failed to save image file: %s\n", ouFilePath1.c_str());
    //     return 1; // Exit if the image cannot be saved
    // }
    // if (!(bigImage.SaveFile(ouFilePath2))) 
    // {
    //     printf("Failed to save image file: %s\n", ouFilePath2.c_str());
    //     return 1; // Exit if the image cannot be saved
    // }
    // if (!(bigImage.SaveFile(ouFilePath3))) 
    // {
    //     printf("Failed to save image file: %s\n", ouFilePath3.c_str());
    //     return 1; // Exit if the image cannot be saved
    // }

    // Save CPU CLAHE Images
    // if (!(smallImageCPU.SaveFile(ouFilePath4cpu))) 
    // {
    //     printf("Failed to save image file: %s\n", ouFilePath4cpu.c_str());
    //     return 1; // Exit if the image cannot be saved
    // }
    // if (!(imgCPU.SaveFile(ouFilePath1cpu))) 
    // {
    //     printf("Failed to save image file: %s\n", ouFilePath1.c_str());
    //     return 1; // Exit if the image cannot be saved
    // }
    // if (!(bigImageCPU.SaveFile(ouFilePath2cpu))) 
    // {
    //     printf("Failed to save image file: %s\n", ouFilePath2.c_str());
    //     return 1; // Exit if the image cannot be saved
    // }
    // if (!(extraBigImageCPU.SaveFile(ouFilePath3cpu))) 
    // {
    //     printf("Failed to save image file: %s\n", ouFilePath3.c_str());
    //     return 1; // Exit if the image cannot be saved
    // }


    return 0;
}

