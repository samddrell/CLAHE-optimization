cmake_minimum_required(VERSION 3.16)

# -------- Locate nvcc BEFORE project() (portable) --------
# 1) Respect CUDACXX if set
if(DEFINED ENV{CUDACXX} AND NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "nvcc from CUDACXX" FORCE)
endif()

# 2) Otherwise try to find nvcc on PATH or in common locations
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(_NVCC nvcc
    HINTS
      /usr/local/cuda/bin
      /usr/local/cuda-12.6/bin /usr/local/cuda-12.5/bin /usr/local/cuda-12.4/bin
      /usr/local/cuda-12.3/bin /usr/local/cuda-12.2/bin /usr/local/cuda-12.1/bin
      /usr/local/cuda-11.8/bin /usr/local/cuda-11.7/bin /usr/local/cuda-11.6/bin
      /usr/local/cuda-11.5/bin /usr/local/cuda-11.4/bin
      /opt/cuda/bin
    PATHS ENV PATH)
  if(_NVCC)
    set(CMAKE_CUDA_COMPILER "${_NVCC}" CACHE FILEPATH "nvcc path" FORCE)
  endif()
endif()

# If still not found, stop with a helpful message
if(NOT DEFINED CMAKE_CUDA_COMPILER OR NOT EXISTS "${CMAKE_CUDA_COMPILER}")
  message(FATAL_ERROR
    "nvcc not found. Set env CUDACXX or pass -DCMAKE_CUDA_COMPILER=/full/path/to/nvcc")
endif()

project(CLAHE_CUDA_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 87 89)

# -------- GPU architecture selection (portable for 3.16) --------
# Let users override at configure time: -DCUDA_ARCH_LIST="87;89" etc.
# If empty, choose a sensible default for your supported devices.
set(CUDA_ARCH_LIST "" CACHE STRING "SM list (e.g. 87;89). Empty = auto.")

if(NOT CUDA_ARCH_LIST)
  # Support Orin (Ampere, sm_87) and Ada (sm_89) by default
  set(CUDA_ARCH_LIST "87;89")
endif()

# Apply arch flags: generate fatbin for each entry (CMake 3.16 path)
foreach(arch ${CUDA_ARCH_LIST})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

# -------- Dependencies: OpenCV, PNG, JPEG --------
find_package(OpenCV QUIET COMPONENTS core imgproc highgui imgcodecs videoio)

find_package(PkgConfig REQUIRED)
if(NOT OpenCV_FOUND)
  pkg_check_modules(OpenCV REQUIRED opencv4)
endif()
# PNG may appear as libpng or libpng16 depending on distro
pkg_check_modules(PNG QUIET libpng)
if(NOT PNG_FOUND)
  pkg_check_modules(PNG REQUIRED libpng16)
endif()
pkg_check_modules(JPEG REQUIRED libjpeg)

add_executable(test
  test.cpp
  image.cpp
  cuda_clahe.cu
)

# -------- Derive CUDA include/lib from nvcc path (no hard-coded version) --------
get_filename_component(_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_ROOT "${_CUDA_BIN_DIR}/.." REALPATH)

target_include_directories(test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDA_ROOT}/include
  ${OpenCV_INCLUDE_DIRS}
  ${PNG_INCLUDE_DIRS}
  ${JPEG_INCLUDE_DIRS}
)

# Extra cflags from pkg-config (if present)
if(OpenCV_CFLAGS_OTHER)
  target_compile_options(test PRIVATE ${OpenCV_CFLAGS_OTHER})
endif()
if(PNG_CFLAGS_OTHER)
  target_compile_options(test PRIVATE ${PNG_CFLAGS_OTHER})
endif()
if(JPEG_CFLAGS_OTHER)
  target_compile_options(test PRIVATE ${JPEG_CFLAGS_OTHER})
endif()

# Link cudart without hard-coded lib path (but add it if needed)
target_link_directories(test PRIVATE
  ${CUDA_ROOT}/lib64
)
target_link_libraries(test PRIVATE
  cudart
  ${OpenCV_LIBRARIES}
  ${PNG_LIBRARIES}
  ${JPEG_LIBRARIES}
)

message(STATUS "nvcc: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA root: ${CUDA_ROOT}")
message(STATUS "CUDA arch list: ${CUDA_ARCH_LIST}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBRARIES}")
