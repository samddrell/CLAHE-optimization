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
    // Prepare CUDA CLAHE Images - Read in image data from a file
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
    std::string outFilePath1 = "CUDA_CLAHE_EXAMPLE.png";


    // Prepare CPU CLAHE Images - Read in image data from a file
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
    std::string outFilePath1cpu = "CPU_CLAHE_EXAMPLE.png";

    // Apply CLAHE and save results

    smallImage.CLAHE(4,16);
    img.CLAHE(4,16);
    img.SaveFile(outFilePath1,100);
    bigImage.CLAHE(4,16);
    extraBigImage.CLAHE(4,16);

    smallImageCPU.cpuCLAHE(4,16);
    imgCPU.cpuCLAHE(4,16);
    imgCPU.SaveFile(outFilePath1cpu,100);
    bigImageCPU.cpuCLAHE(4,16);
    extraBigImageCPU.cpuCLAHE(4,16);

    return 0;
}

