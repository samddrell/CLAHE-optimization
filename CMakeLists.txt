cmake_minimum_required(VERSION 3.16)

# Make sure nvcc is known BEFORE project()
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")

project(CLAHE_CUDA_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# CMake 3.16 doesn't know CUDA_ARCHITECTURES; pass -gencode directly.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_87,code=sm_87")

# --- OpenCV / PNG / JPEG ---
find_package(OpenCV QUIET COMPONENTS core imgproc highgui imgcodecs videoio)

find_package(PkgConfig REQUIRED)
if(NOT OpenCV_FOUND)
  pkg_check_modules(OpenCV REQUIRED opencv4)
endif()
pkg_check_modules(PNG  REQUIRED libpng)
pkg_check_modules(JPEG REQUIRED libjpeg)

add_executable(test
  test.cpp
  image.cpp
  cuda_clahe.cu
)

# Includes
target_include_directories(test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  /usr/local/cuda-11.4/include
  ${OpenCV_INCLUDE_DIRS}
  ${PNG_INCLUDE_DIRS}
  ${JPEG_INCLUDE_DIRS}
)

# Extra cflags (only if provided by pkg-config)
if(OpenCV_CFLAGS_OTHER)
  target_compile_options(test PRIVATE ${OpenCV_CFLAGS_OTHER})
endif()
if(PNG_CFLAGS_OTHER)
  target_compile_options(test PRIVATE ${PNG_CFLAGS_OTHER})
endif()
if(JPEG_CFLAGS_OTHER)
  target_compile_options(test PRIVATE ${JPEG_CFLAGS_OTHER})
endif()

# CUDA libs (manual, since we aren't using CUDAToolkit package)
target_link_directories(test PRIVATE
  /usr/local/cuda-11.4/lib64
)
target_link_libraries(test PRIVATE
  cudart
  ${OpenCV_LIBRARIES}
  ${PNG_LIBRARIES}
  ${JPEG_LIBRARIES}
)

message(STATUS "nvcc: ${CMAKE_CUDA_COMPILER}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBRARIES}")
