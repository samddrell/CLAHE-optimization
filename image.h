// Includes
#include <cstdint>     // for uint8_t
#include <string>      // for std::string

#ifndef IMAGE_H
#define IMAGE_H

//Image Class
class Image
{
    public:
        uint8_t *Data;
        int Width;
        int Height;

        Image(); // Default constructor
        Image(int w, int h);    // Alocate memory for the Array
        Image(const Image& other);

        bool operator==(const Image &other) const;      
        bool compare(const Image &other, double maxPercentError = 0.0) const; // Compare two images      
        
        uint8_t GetPixelRed(int x, int y);          // Get the red value of a pixel
        uint8_t GetPixelGreen(int x, int y);        // Get the green value of a pixel
        uint8_t GetPixelBlue(int x, int y);         // Get the blue value of a pixel

        void SetPixelRed(int x, int y,uint8_t r);       // Set the red value of a pixel
        void SetPixelGreen(int x, int y, uint8_t g);    // Set the green value of a pixel
        void SetPixelBlue(int x, int y, uint8_t b);     // Set the blue value of a pixel

        bool SavePNG(std::string filePath);     // Save the image to a png file
        bool OpenPNG(std::string filePath);     // Read the image from a png file

        bool SaveJPEG(std::string filename, int quality = 100); // Save the image to a jpg file
        int OpenJPEG(std::string infilename);

        bool SaveFile(std::string infilename, int quality = 100);
        bool OpenFile(std::string infilename);

        void CLAHE(int clipLimit, int tileGridSize);
        void cpuCLAHE(int clipLimit, int tileGridSize);

        void cpuCANNY();

        ~Image(); // Free memory

    private:
        int m_buffSize;           // Resolution for JPEG compression
        int openJPEG(struct jpeg_decompress_struct *cinfo,
                        std::string infilename);

};

#endif // IMAGE_H