#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


__global__
void rgb_to_gray_kernel(unsigned char *rgb, unsigned char *gray, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        int gray_offset = row * width + col;
        int rgb_offset = gray_offset * 3;
        int r_offset = rgb_offset + 0;
        int g_offset = rgb_offset + 1;
        int b_offset = rgb_offset + 2;
        gray[gray_offset] = (unsigned char)(0.21f * rgb[r_offset] + 0.72f * rgb[g_offset] + 0.07f * rgb[b_offset]);
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    dim3 threads_per_block(16, 16);     // using 256 threads per block
    dim3 number_of_blocks(cdiv(width, threads_per_block.x),
                          cdiv(height, threads_per_block.y));

    rgb_to_gray_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        image.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width,
        height
    );

    return result;
}