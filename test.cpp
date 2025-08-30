// TODO: Build Test cases for CLAHE. Analyze speed and performance.
#include <fstream>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

#include "image.h"

#include <filesystem>
namespace fs = std::filesystem;

fs::path testDir =
#ifdef TEST_DATA_DIR
    fs::path(TEST_DATA_DIR);
#else
    fs::current_path(); // fallback
#endif

fs::path inFilePath      = testDir / "p_00000009.png";
fs::path inFilePathSmall = testDir / "DSC04441.jpg";


int main()
{
    // Prepare CUDA CLAHE Images
    // Read in image data from a file
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
    std::string outFilePath1 = "originalComputeTileLUT_with_originalApplyCLAHE.png";
    // std::string outFilePath1 = "new_normal_4_16.png";
    std::string outFilePath2 = "big_clahe_4_16.png";
    std::string outFilePath3 = "extra_big_clahe_4_16.png";
    std::string outFilePath4 = "small_clahe_4_16.png";


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
    std::string outFilePath1cpu = "CPU_normal_clahe_4_16.png";
    std::string outFilePath2cpu = "CPU_big_clahe_4_16.png";
    std::string outFilePath3cpu = "CPU_extra_big_clahe_4_16.png";
    std::string outFilePath4cpu = "CPU_small_clahe_4_16.png";

    // Prepare CPU CANNY Image
    std::string outFilePathCANNY = "canny_applied.png";
    Image imgCANNY = img;


    // Your kernel or image operation
    smallImage.CLAHE(4,16);
    // smallImage.SaveFile(ouFilePath4,100);
    img.CLAHE(4,16);
    // img.SaveFile(outFilePath1,100);
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
    imgCANNY.cpuCANNY();
    // imgCANNY.SaveFile(outFilePathCANNY,100);

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

