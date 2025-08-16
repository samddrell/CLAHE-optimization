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
    std::string inFilePathSmall = "DSC04441.jpg";
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
    // std::string ouFilePath1 = "normal_clahe_4_16.png";
    // std::string ouFilePath1 = "test_A_4_16.png";
    std::string ouFilePath1 = "restricted_4_16.png";
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
    // smallImage.SaveFile(ouFilePath4,100);
    img.CLAHE(4,16);
    img.SaveFile(ouFilePath1,100);
    bigImage.CLAHE(4,16);
    extraBigImage.CLAHE(4,16);
    // extraBigImage.SaveFile(ouFilePath3,100);

    smallImageCPU.cpuCLAHE(4,16);
    // smallImageCPU.SaveFile(ouFilePath4cpu,100);
    imgCPU.cpuCLAHE(4,16);
    // imgCPU.SaveFile(ouFilePath1cpu,100);
    bigImageCPU.cpuCLAHE(4,16);
    extraBigImageCPU.cpuCLAHE(4,16);
    // extraBigImageCPU.SaveFile(ouFilePath3cpu,100);
    // img.CLAHE(1,8);
    // img.CLAHE(1,8);
    // img.CLAHE(1,8);
    // img.CLAHE(4,8);
    // img.CLAHE(8,8);
    // img.CLAHE(2,8);
    // img.CLAHE(4,32);

    // CANNY Edge Detection
    // std::string ouFilePathCANNY = "canny_applied.png";

    // Image cannyImg;
    // if (!(cannyImg.OpenFile(inFilePathSmall))) 
    // {
    //     printf("Failed to open image file: %s\n", inFilePathSmall.c_str());
    //     return 1; // Exit if the image cannot be opened
    // }
    // cannyImg.cpuCANNY();
    // // Save CUDA CLAHE Images
    // if (!(cannyImg.SaveFile(ouFilePathCANNY))) 
    // {
    //     printf("Failed to save image file: %s\n", ouFilePath4.c_str());
    //     return 1; // Exit if the image cannot be saved
    // }

    return 0;
}

