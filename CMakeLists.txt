cmake_minimum_required(VERSION 3.18)

project(PMPP LANGUAGES CXX C)


# --------- Optionally enable standards ---------
# ------------------ STANDARDS ------------------
enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ------------------ Path Settings ------------------
set(MAIN_PATH "${CMAKE_SOURCE_DIR}") # ------- Makes source dir
set(SRC "${MAIN_PATH}/src")

# ------------------ CUDA ------------------
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(Torch QUIET PATHS "${CMAKE_SOURCE_DIR}/external/libtorch")


add_executable(c02s06_vectorAddition "${SRC}/c02s06_vectorAddition.cu")
add_executable(rgb_to_grayscale "${SRC}/rgb_to_grayscale.cu")
target_link_libraries(rgb_to_grayscale "${TORCH_LIBRARIES}")