#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

__global__
void mean_filter_kernel(unsigned char* result, unsigned char* image, int width, int height, int radius){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = threadIdx.z;
    if (row < height && col < width) {
        int base_offset = width * height * channel;
        int pixel_offset = row * width + col + base_offset;
        int sum = 0;
        int count = 0;
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int current_row = row + i;
                int current_col = col + j;
                if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width) {
                    int current_pixel_offset = current_row * width + current_col + base_offset;
                    sum += image[current_pixel_offset];
                    count+=1;
                }
            }
        }
        result[pixel_offset] = (unsigned char) (sum / count);
    }
}


inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}


torch::Tensor mean_filter(torch::Tensor image, int radius) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);
    assert(radius > 0);

    const auto channels = image.size(0);
    const auto height = image.size(1);
    const auto width = image.size(2);

    auto result = torch::empty_like(image);

    dim3 threads_per_block(16, 16, channels);
    dim3 number_of_blocks(
        cdiv(width, threads_per_block.x),
        cdiv(height, threads_per_block.y)
    );

    mean_filter_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height,
        radius
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}