cmake_minimum_required(VERSION 3.18)
project(CLAHE_CUDA_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Manually specify OpenCV paths
set(OpenCV_INCLUDE_DIRS /usr/include/opencv4)
set(OpenCV_LIBRARIES
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_imgcodecs
    opencv_videoio
)



# Your external projects directory
# set(ENGINEERING_PROJECTS_DIR "/home/samddrell/sam-1/Engineering Projects")

# For libpng and jpeg
find_package(PkgConfig REQUIRED)
pkg_check_modules(PNG REQUIRED libpng)
pkg_check_modules(JPEG REQUIRED libjpeg)
# pkg_check_modules(OpenCV REQUIRED opencv4)

# Executable target (this is your main program)
add_executable(test
    test.cpp       # Replace this with your actual main .cpp file
    image.cpp      # Include source file for your image class
    cuda_clahe.cu  # Include your CUDA kernel file
)

# Optional CUDA properties
set_target_properties(test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)

# Find CUDA toolkit (includes cuda_runtime.h and friends)
find_package(CUDAToolkit REQUIRED)

# Include directories
target_include_directories(test PRIVATE
    # "${ENGINEERING_PROJECTS_DIR}/real_time_ml_streaming/src"  # for image.h
    "${CMAKE_CURRENT_SOURCE_DIR}"                             # for cuda_clahe.h
    ${PNG_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(test PRIVATE ${OpenCV_CFLAGS_OTHER})

# Link against the CUDA runtime
target_link_libraries(test PRIVATE 
    CUDA::cudart
    ${PNG_LIBRARIES}
    ${JPEG_LIBRARIES}
    ${OpenCV_LIBRARIES}

)

# print what CMake is seeing
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
