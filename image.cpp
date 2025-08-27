// Includes
#include <cstdint>     // for uint8_t
#include <string.h>      // for std::string and std::memcpy
#include <iostream>     // for std::cout
#include <stdio.h>
#include <cstdio>
#include <stdlib.h>
#include <unordered_map>
#include <cctype>

#include <functional> // for std::function
#include <chrono>

#include "image.h" // for Image class

#include <png.h>
#include "jpeglib.h"
#include "jerror.h"
#include <setjmp.h> // May not be used
// #include <jpeglib12.h>

#include <cuda_runtime.h>

#include "cuda_clahe.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>


#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#ifdef _WIN32
#define strcasecmp  stricmp
#define strncasecmp  strnicmp
#endif


///////////////////////////////////////////////////////////////////////
// Custom error handler for JPEG
///////////////////////////////////////////////////////////////////////
struct custom_error_mgr {
    jpeg_error_mgr pub;       // "Inherit" base JPEG error manager
    jmp_buf setjmp_buffer;    // Jump buffer for error recovery
};

// Function alias
typedef struct custom_error_mgr* custom_error_ptr; // QUESTION: What and why?

void custom_error_exit(j_common_ptr cinfo) {
    custom_error_ptr myerr = (custom_error_ptr)cinfo->err;

    // Optional: print the default message
    (*cinfo->err->output_message)(cinfo);

    // Jump back to setjmp
    longjmp(myerr->setjmp_buffer, 1);
}

///////////////////////////////////////////////////////////////////////
// Custom error handler for JPEG - Reading
///////////////////////////////////////////////////////////////////////
struct my_error_mgr
{
    struct jpeg_error_mgr pub; /* "public" fields */

    jmp_buf setjmp_buffer; /* for return to caller */
};

typedef struct my_error_mgr *my_error_ptr; // QUESTION: What and why?

///////////////////////////////////////////////////////////////////////
// routine to replace the standard error_exit method - Reading
///////////////////////////////////////////////////////////////////////
void my_error_exit(j_common_ptr cinfo)
{
    /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
    my_error_ptr myerr = (my_error_ptr)cinfo->err;

    /* Always display the message. */
    /* We could postpone this until after returning, if we chose. */
    (*cinfo->err->output_message)(cinfo);

    /* Return control to the setjmp point */
    longjmp(myerr->setjmp_buffer, 1);
}

///////////////////////////////////////////////////////////////////////
// Image class constructor
///////////////////////////////////////////////////////////////////////
Image::Image() : m_width(0), m_height(0), m_data(nullptr) {}

///////////////////////////////////////////////////////////////////////
// Image class constructor
///////////////////////////////////////////////////////////////////////
Image::Image(int w, int h)
{
    m_width = w;
    m_height = h;
    m_buffSize = m_width * m_height * 3; // Calculate the resolution

    // Allocate memory for the pixel data Array
    // Initialize to 0
    if(w != 0 || h != 0) 
        m_data = new uint8_t[m_buffSize](); 
    else
        m_data = nullptr; // If width or height is 0, set m_data to nullptr
}    

///////////////////////////////////////////////////////////////////////
// Image copy constructor
// - Creates an Image object with double the resolution of the original
//   by reproducing the input image four times in a 2x2 grid.
///////////////////////////////////////////////////////////////////////
Image::Image(const Image& other)
{
    m_width = other.m_width * 2; // Double the width
    m_height = other.m_height * 2; // Double the height
    int newResolution = m_width * m_height; // Calculate the new resolution
    m_buffSize = newResolution * 3; // Calculate the new byte size

    if(m_width != 0 || m_height != 0) 
        m_data = new uint8_t[m_buffSize](); 
    else
        m_data = nullptr; // If width or height is 0, set m_data to nullptr

    int midwidth = other.m_width; // Original width
    int midheight = other.m_height; // Original height

    int originalRowSize = other.m_width * 3; // Size of one row in the original image
    int newRowSize = m_width * 3; // Size of one row in the new image

    for (int i = 0; i < m_height; i++)
    {
        for (int j = 0; j < originalRowSize; j++)
        {
            // print the current row on the right hand side of the image
            m_data[i * newRowSize + j] = other.m_data[(i % midheight) * originalRowSize + j];

            // print the current row, offset by original row dimension
            // - This prints the right hand side of the image
            m_data[originalRowSize + i * newRowSize + j] = other.m_data[(i % midheight) * originalRowSize + j];
        }
    }
}

///////////////////////////////////////////////////////////////////////
// Image Class Destructor
///////////////////////////////////////////////////////////////////////
Image::~Image() // Free memory
{
    if (m_data) {
        delete[] m_data;
        m_data = nullptr;
    }
}

///////////////////////////////////////////////////////////////////////
// Overloaded equality operator
///////////////////////////////////////////////////////////////////////
bool Image::operator==(const Image &other) const
{
    // Check if the dimensions of the images are equal
    if (m_width != other.m_width || m_height != other.m_height)
    {
        return false; // If dimensions are not equal, return false
    }    

    for (int i = 0; i < m_buffSize; i++)
    {
        // If any pixel data is not equal, return false
        if (m_data[i] != other.m_data[i])
        {
            return false;
        }
    }
    return true; // If all pixel data is equal, return true
}

///////////////////////////////////////////////////////////////////////
// Compare two images, within a certain percent error
///////////////////////////////////////////////////////////////////////
bool Image::compare(const Image &other, double maxPercentError) const
{
    if (m_width != other.m_width || m_height != other.m_height)
    {
        return false; 
    }    

    double mismatchCount = 0; 

    for (int i = 0; i < m_buffSize; i++)
    {
        if (m_data[i] != other.m_data[i])
        {
            mismatchCount += static_cast<double>(other.m_data[i]) - 
                static_cast<double>(m_data[i]); // Calculate the difference
        }
    }
    
    double percentError = mismatchCount / static_cast<double>(m_buffSize * 255);

    if( percentError > maxPercentError )
    {
        return false; 
    }
    else
    {
        return true;
    }
}  

///////////////////////////////////////////////////////////////////////
// GET the RED value of a pixel
///////////////////////////////////////////////////////////////////////
uint8_t Image::GetPixelRed(int x, int y) 
{
    if (x >= m_width || y >= m_height) 
    {
        return 0; // Return 0 if coordinates are out of bounds
    }
    return m_data[3* m_width *y+ 3 * x + 0]; 
}
///////////////////////////////////////////////////////////////////////
// GET the GREEN value of a pixel
///////////////////////////////////////////////////////////////////////
uint8_t Image::GetPixelGreen(int x, int y) 
{
    if (x >= m_width || y >= m_height) 
    {
        return 0; // Return 0 if coordinates are out of bounds
    }

    return m_data[3* m_width *y+ 3 * x + 1]; 
}
///////////////////////////////////////////////////////////////////////
// GET the BLUE value of a pixel
///////////////////////////////////////////////////////////////////////
uint8_t Image::GetPixelBlue(int x, int y) 
{
    if (x >= m_width || y >= m_height) 
    {
        return 0; // Return 0 if coordinates are out of bounds
    }

    return m_data[3* m_width *y+ 3 * x + 2]; 
}

///////////////////////////////////////////////////////////////////////
// SET the RED value of a pixel
///////////////////////////////////////////////////////////////////////
void Image::SetPixelRed(int x, int y,uint8_t r) 
{
    if (x >= m_width || y >= m_height) 
    {
        return; // Return if coordinates are out of bounds
    }

    m_data[3* m_width *y+ 3 * x + 0] = r; 
}
///////////////////////////////////////////////////////////////////////
// SET the GREEN value of a pixel
///////////////////////////////////////////////////////////////////////
void Image::SetPixelGreen(int x, int y, uint8_t g) 
{
    if (x >= m_width || y >= m_height) 
    {
        return; // Return if coordinates are out of bounds
    }

    m_data[3* m_width *y+ 3 * x + 1] = g; 
}
///////////////////////////////////////////////////////////////////////
// SET the BLUE value of a pixel
///////////////////////////////////////////////////////////////////////
void Image::SetPixelBlue(int x, int y, uint8_t b) 
{
    if (x >= m_width || y >= m_height) 
    {
        return; // Return if coordinates are out of bounds
    }

    m_data[3* m_width *y+ 3 * x + 2] = b; 
}

///////////////////////////////////////////////////////////////////////
// Save the image using libpng
///////////////////////////////////////////////////////////////////////
bool Image::SavePNG(std::string filePath) 
{   
    // This section opens the file for writing in binary mode ("wb")
    // If the file can't be opened, it returns false
    FILE* fp = fopen(filePath.c_str(), "wb");   
    if (!fp)
    {
        return false; 
    }

    // Create libpng structures
    // NOTE:
    // Maybe these shouldn't be stored on the stack if we want it to be 
    //      Faster? The documentation says that these are large. Maybe
    //      we should use a smart pointer or something like that? Or 
    //      Maybe having them on the stack is actually faster?

    // png_create_write_struct creates a write context for the PNG file
    // Parameters:
    // PNG_LIBPNG_VER_STRING is a string that contains the version number 
    //      of libpng
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
        nullptr,nullptr,nullptr); 
    if (!png)
    {
        fclose(fp); // Close the file if png_create_write_struct fails
        return false; // Return false if png_create_write_struct fails
    }

    // Create the info header (metadata), like width, height, format, etc.
    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        // Destroy the png struct if info creation fails
        // png_destroy_write_struct(&png, nullptr); 
        fclose(fp); // Close the file if png_create_info_struct fails
        return false; // Return false if png_create_info_struct fails
    }

    // Ensure image is not of size 0 before proceeding
    if (m_height == 0 || m_width == 0 || !m_data)
    {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }
        
    // Set up Error handling
    // setjmp(png_jmpbuf(png)) saves current stack context using the jump
    //      buffer, which is saved in png
    // If an error occurs, the program jumps back to this point and exits
    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }

    // Store the file pointer in the png struct
    // This is used to write the PNG data to the file
    png_init_io(png, fp);

    /*
    Set image metadata (header info);

    width - holds the width of the image in pixels (up to 2ˆ31).
    height - holds the height of the image in pixels (up to 2ˆ31).
    bit_depth - holds the bit depth of one of the image channels.
                (valid values are 1, 2, 4, 8, 16 and depend also on the
                color_type. See also significant bits (sBIT) below).
    color_type - describes which color/alpha channels are present.
                PNG_COLOR_TYPE_GRAY
                    (bit depths 1, 2, 4, 8, 16)
                PNG_COLOR_TYPE_GRAY_ALPHA
                    (bit depths 8, 16)
                PNG_COLOR_TYPE_PALETTE
                    (bit depths 1, 2, 4, 8)
                PNG_COLOR_TYPE_RGB
                    (bit_depths 8, 16)
                PNG_COLOR_TYPE_RGB_ALPHA
                    (bit_depths 8, 16)
                PNG_COLOR_MASK_PALETTE
                PNG_COLOR_MASK_COLOR
                PNG_COLOR_MASK_ALPHA
    interlace_type - PNG_INTERLACE_NONE or PNG_INTERLACE_ADAM7
    compression_type - (must be PNG_COMPRESSION_TYPE_DEFAULT)
    filter_method - (must be PNG_FILTER_TYPE_DEFAULT or, if you are writing
                a PNG to be embedded in a MNG datastream, can also be
                PNG_INTRAPIXEL_DIFFERENCING)
    */
    int bit_depth = 8;
    int color_type = PNG_COLOR_TYPE_RGB;
    int interlace_type = PNG_INTERLACE_NONE;
    int compression_type = PNG_COMPRESSION_TYPE_DEFAULT;
    int filter_method = PNG_FILTER_TYPE_DEFAULT;

    png_set_IHDR(png, info, m_width, m_height,
        bit_depth, color_type, interlace_type,
        compression_type, filter_method);

    // Create an array of pointers to each row of the image
    png_bytep* row_pointers = NULL;
    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * m_height);

    // png_bytep* row_pointers = new png_bytep[height];
    for (int y = 0; y < m_height; y++)
    {
        row_pointers[y] = m_data + y * m_width * 3; 
    }

    // Write the image data to the file
    png_set_rows(png, info, row_pointers);
    png_write_png(png, info, PNG_TRANSFORM_STRIP_ALPHA, nullptr);

    free(row_pointers);

    fclose(fp);
    
    png_destroy_write_struct(&png, &info);

    return true; // Return true if successful

}

///////////////////////////////////////////////////////////////////////
// Save the image using libpng
///////////////////////////////////////////////////////////////////////
bool Image::OpenPNG(std::string filePath)
{
    // Open the file for reading in binary mode ("rb")
    FILE *fp = fopen(filePath.c_str(), "rb");
    if (!fp) 
    {
        std::cout << "Failed at 1" << std::endl;
        return false; // Return false if file cannot be opened
    }

    // Check if the file is a PNG file by reading the first 8 bytes
    png_byte header[8];
    fread(header, 1, 8, fp);
    if(png_sig_cmp(header,0,8))
    {
        fclose(fp);
        std::cout << "Failed at 2" << std::endl;

        return false; // Return false if file is not a PNG
    }

    // Create libpng structures
    png_struct* png = png_create_read_struct
        (PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    // Return false if png_create_read_struct fails
    if (!png) 
    {
        std::cout << "Failed at 3" << std::endl;
        fclose(fp);
        return false; 
    }
    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        std::cout << "Failed at 4" << std::endl;
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        return false; // Return false if png_create_info_struct fails
    }
    png_infop end = png_create_info_struct(png);
    if (!end)
    {
        std::cout << "Failed at 5" << std::endl;
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return false; // Return false if png_create_info_struct fails
    }

    // Save stack context for error handling
    if (setjmp(png_jmpbuf(png)))
    {
        std::cout << "Failed at 6" << std::endl;
        png_destroy_read_struct(&png, &info, &end);
        fclose(fp);
        return false; // Return false if an error occurs
    }

    // Initialize the png struct with the file pointer
    png_init_io(png, fp);    

    // Inform libpng that the file pointer has been moved up 
    //      by 8 bytes to skip the PNG signature 
    png_set_sig_bytes(png, 8);

    // Read the PNG file
    png_read_png(png, info, PNG_TRANSFORM_STRIP_ALPHA, NULL);

    // Read in the image data

    png_bytep* row_pointers = png_get_rows(png, info);

    // Read in the image data into the Image object

    m_height = png_get_image_height(png, info);
    m_width = png_get_image_width(png, info);
    m_buffSize = m_width * m_height * 3; // Calculate the resolution
    m_data = new uint8_t[m_buffSize];

    for (int i = 0; i < m_height; i++)
    {
        memcpy(m_data + i * m_width * 3, row_pointers[i], m_width * 3);
    }

    // Destroy the png structures and close the file
    png_destroy_read_struct(&png, &info, &end);
    fclose(fp);
    return true; // Return true if successful
}

///////////////////////////////////////////////////////////////////////
// Save the image using turbo jpeg
///////////////////////////////////////////////////////////////////////
bool Image::SaveJPEG(std::string filename, int quality)
{
    // Create a jpeg compression object
    struct jpeg_compress_struct cinfo;

    struct custom_error_mgr jerr; // JPEG error handler.
    FILE *outfile;  // Target File

    // Pointer to array of pointers to image rows
    JSAMPARRAY  row_pointers = new JSAMPROW[m_height]; // Pointer to a single row of pixels
    for (int x = 0; x < m_height; x++)
    {
        row_pointers[x] = m_data + x * m_width * 3; // Set row pointers to the image data
    }

    // Step 1 Allocate and initialize JPEG compression object

    // Step 1.1 Set up the error handler
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = custom_error_exit;

    // Step 1.2 Initialize the JPEG compression object
    jpeg_create_compress(&cinfo);

    // Step 1.3 Set up the jump point
    if (setjmp(jerr.setjmp_buffer)) 
    {
        // We jumped here from a fatal JPEG error
        jpeg_destroy_compress(&cinfo);
        return false;
    }

    // Step 2 Specify data destination
    if ((outfile = fopen(filename.c_str(), "wb")) == NULL)
    {
        jpeg_destroy_compress(&cinfo);
        return false; // Exit if the file cannot be opened
    }

    jpeg_stdio_dest(&cinfo, outfile); // send compressed data to a stdio stream

    // Step 3 Set parameters for compression
    cinfo.image_width = m_width; // Image width in pixels
    cinfo.image_height = m_height; // Image height in pixels
    cinfo.input_components = 3; // Number of color components per pixel
    cinfo.in_color_space = JCS_RGB; // Color space of the input image
    cinfo.data_precision = 8; // data precision of input image. 

    jpeg_set_defaults(&cinfo); // Set default compression parameters
    jpeg_set_quality(&cinfo, quality, TRUE); // Set the quality of the compression
        // Uses 4:4:4 chroma subsampling by default
    cinfo.comp_info[0].h_samp_factor = cinfo.comp_info[0].v_samp_factor = 1;

    // Step 4 Start compressor
    jpeg_start_compress(&cinfo, TRUE);  // TRUE ensures that we will write a complete interchange-JPEG file
    
    // Step 6 Write scanlines
    while (cinfo.next_scanline < cinfo.image_height)
    {
        jpeg_write_scanlines(&cinfo, &row_pointers[cinfo.next_scanline], 1);
    }

    // Step 7 Finish Compression
    jpeg_finish_compress(&cinfo);
    fclose(outfile); // Close the output file

    // Step 8 Release JPEG compression object
    jpeg_destroy_compress(&cinfo); // Release the JPEG compression object
    delete[] row_pointers;

    return true; // Return true if successful
}

///////////////////////////////////////////////////////////////////////
// Public Encapsulation to Read the image using turbo jpeg
///////////////////////////////////////////////////////////////////////
int Image::OpenJPEG(std::string infilename)
{
    struct jpeg_decompress_struct cinfo; 

    return openJPEG(&cinfo, infilename);
}

///////////////////////////////////////////////////////////////////////
// Read the image using turbo jpeg
// NOTE:
//      We call the libjpeg API from within a separate function, because 
//      modifying the local non-volatile jpeg_decompress_struct instance 
//      below the setjmp() return point and then accessing the instance 
//      after setjmp() returns would result in undefined behavior that 
//      may potentially overwrite all or part of the structure.
//      (This note was quoted from the libjpeg example code)
///////////////////////////////////////////////////////////////////////
int Image::openJPEG(struct jpeg_decompress_struct *cinfo, std::string infilename)
{
    struct my_error_mgr jerr;   // Create an instance of our custom error manager
    FILE *infile;               // source file
    JSAMPARRAY buffer = NULL;   // Output row buffer 
    int col;
    int row_stride;             // physical row width in output buffer 

    // Open the input and output files so they can be closed if we long jump.
    if ((infile = fopen(infilename.c_str(), "rb")) == NULL)
    {
        fprintf(stderr, "can't open %s\n", infilename.c_str());
        return 0;
    }

    // Step 1: allocate and initialize JPEG decompression object

    cinfo->err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit; // then override error_exit.

    // Set up the jump point for error handling
    if (setjmp(jerr.setjmp_buffer))
    {
        //Clean up the JPEG object, close the input file, and return.
        jpeg_destroy_decompress(cinfo);
        fclose(infile);
        return 0;
    }

    // Initialize the JPEG decompression object.
    jpeg_create_decompress(cinfo);

    // Step 2: specify data source (eg, a file)

    jpeg_stdio_src(cinfo, infile);

    // Step 3: read file parameters with jpeg_read_header()

    // This is type-cast as void because we are only reading entire images
    (void)jpeg_read_header(cinfo, TRUE);

    m_width = cinfo->image_width; // Set the image width
    m_height = cinfo->image_height; // Set the image height
    m_buffSize = m_width * m_height * 3; // Calculate the resolution


    // Step 4: set parameters for decompression 
    //      Note: This step is optional, but it allows you to change
    //      the default parameters set by jpeg_read_header().
    // Step 5: Start decompressor 

    // This is type-cast as void because we are only reading entire images
    (void)jpeg_start_decompress(cinfo);

    row_stride = cinfo->output_width * cinfo->output_components;
    
    // This buffer is a stepping block for one row of output pixels in between the 
    //      input jpeg and the output ppm
    buffer = (*cinfo->mem->alloc_sarray)((j_common_ptr)cinfo, JPOOL_IMAGE, row_stride, 1);

    // Write to data member m_data adaptation
    m_data = new uint8_t[m_buffSize];


    // Step 6: Line by line, read jpeg to ppm
    while (cinfo->output_scanline < cinfo->output_height)
    {
        buffer[0] = m_data + (cinfo->output_scanline * row_stride);
        (void)jpeg_read_scanlines(cinfo, buffer, 1);
        // fwrite(buffer[0], 1, row_stride, outfile);
    }

    /* Step 7: Finish decompression */

    // This is type-cast as void because we are only reading entire images
    (void)jpeg_finish_decompress(cinfo);

    /* Step 8: Release JPEG decompression object */

    jpeg_destroy_decompress(cinfo);

    fclose(infile);

    return 1; // We want to return 1 on success, 0 on error.
}

///////////////////////////////////////////////////////////////////////
// Public Interface to Save the image, regardless of format.
// Note: This is currently only able to save at default quality.
///////////////////////////////////////////////////////////////////////
bool Image::SaveFile(std::string infilename,int quality)
{
    // Isolate the file extension from the filename
    int iLoc = infilename.find_last_of('.');
    if (iLoc == std::string::npos) return false; // No extension found
    std::string szExtention = infilename.substr(iLoc);
    int szExtentionLength = szExtention.length();
    for (int i = 0; i < szExtentionLength; i++)
    {
        std::cout << "I'm here!" << std::endl;
        szExtention[i] = std::tolower(szExtention[i]); // Convert to lowercase
    }

    std::unordered_map<std::string, 
        std::function<bool(const std::string filePath)>> saveFunctions;
    saveFunctions[".png"] = [this](std::string filePath)
    {
        return this->SavePNG(filePath);
    };
    saveFunctions[".jpg"] = [this,quality](std::string filePath)
    {
        return this->SaveJPEG(filePath,quality);
    };
    saveFunctions[".jpeg"] = [this,quality](std::string filePath)
    {
        return this->SaveJPEG(filePath,quality);
    };

    auto extensionFound = saveFunctions.find(szExtention);

    // Check if the extension is in the map
    // Return false if it fails to save or if the extension is not supported
    return (extensionFound != saveFunctions.end()) ? (saveFunctions[szExtention](infilename)) : false;
}

///////////////////////////////////////////////////////////////////////
// Public Interface to Open the image, regardless of format.
///////////////////////////////////////////////////////////////////////
bool Image::OpenFile(std::string infilename)
{

    // Isolate the file extension from the filename
    int iLoc = infilename.find_last_of('.');
    if (iLoc == std::string::npos) return false; // No extension found
    std::string szExtention = infilename.substr(iLoc);
    int szExtentionLength = szExtention.length();
    for (int i = 0; i < szExtentionLength; i++)
    {
        szExtention[i] = std::tolower(szExtention[i]); // Convert to lowercase
    }


    std::unordered_map<std::string, 
        std::function<bool(const std::string filePath)>> openFunctions;
    openFunctions[".png"] = [this](std::string filePath)
    {
        return this->OpenPNG(filePath);
    };
    openFunctions[".jpg"] = [this](std::string filePath)
    {
        return this->OpenJPEG(filePath);
    };
    openFunctions[".jpeg"] = [this](std::string filePath)
    {
        return this->OpenJPEG(filePath);
    };

    auto extensionFound = openFunctions.find(szExtention);

    // Check if the extension is in the map
    // Return false if it fails to save or if the extension is not supported
    return (extensionFound != openFunctions.end()) ? (openFunctions[szExtention](infilename)) : false;

}


void Image::CLAHE(int clipLimit, int tileGridSize)
{
    // Make temporary data buffer
    //  - This is 1D for grayscale
    uint8_t *inputData = new uint8_t[m_height * m_width];
    uint8_t *outputData = new uint8_t[m_height * m_width];
    // printf("Made processed Data of size %d\n", m_height * m_width);

    // combine all color channels into R
    for(int i = 0; i < m_width * m_height; i++)
    {   // Average the RGB values into R
        int offset = i * 3;
        inputData[i] = (m_data[offset + 0] + m_data[offset + 1] + m_data[offset + 2]) / 3; 
    }
    // printf("combined all color channels into R\n");

    // measure the time to execute the clahe
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);
    // Your kernel or image operation
    MetaViLabs::Cuda::CLAHE_8u(inputData,outputData, m_width, m_height, clipLimit, tileGridSize);

    // cudaEventRecord(stop);
    // printf("CLAHE_8u finished\n");

    // cudaEventSynchronize(stop);
    float ms = 0;
    // cudaEventElapsedTime(&ms, start, stop);
    // printf("GPU CLAHE took %f ms\n", ms);

    // Copy processed data back to m_data
    for(int i = 0; i < m_width * m_height; i++)
    {
        int offset = i * 3;
        m_data[offset + 0] = outputData[i]; // Set R
        m_data[offset + 1] = outputData[i]; // Set G
        m_data[offset + 2] = outputData[i]; // Set B
    }
    // printf("Copied processed data back to m_data\n");

    // Free the temporary data buffer
    delete[] inputData;
    delete[] outputData;
}

void Image::cpuCLAHE(int clipLimit, int tileGridSize)
{
    // Create cv::Mat from m_data
    cv::Mat img(m_height, m_width, CV_8UC3, m_data);
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // Create CLAHE object
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clipLimit);
    clahe->setTilesGridSize(cv::Size(tileGridSize, tileGridSize));

    // Apply CLAHE to the grayscale image
    cv::Mat claheImg;

    auto start = std::chrono::high_resolution_clock::now();
    clahe->apply(gray, claheImg);
    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate elapsed time in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "CPU Elapsed time:  " << duration.count() << " microseconds\n";

    // Convert back to BGR and copy to m_data
    cv::cvtColor(claheImg, img, cv::COLOR_GRAY2BGR);
    std::memcpy(m_data, img.data, m_buffSize);

}



void Image::cpuCANNY()
{
    // Create cv::Mat from m_data
    cv::Mat img(m_height, m_width, CV_8UC3, m_data);
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // Create CLAHE object

    // Apply Gaussian Blur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);

    // Apply Canny edge detection
    cv::Mat edges;
    double threshold1 = 100;
    double threshold2 = 200;
    cv::Canny(blurred, edges, threshold1, threshold2);

    // Convert back to BGR and copy to m_data
    cv::cvtColor(edges, img, cv::COLOR_GRAY2BGR);
    std::memcpy(m_data, img.data, m_buffSize);

}