#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


__global__ 
void matrix_multiplication_kernel(
    float* result, 
    float* matrix1, 
    float* matrix2, 
    int width, 
    int height, 
    int depth
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        float sum = 0;
        for(int i = 0; i < depth; i++) {
            sum += matrix1[row * depth + i] * matrix2[i * width + col];
        }
        result[row * height + col] = sum;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor matrix_multiplication(
    torch::Tensor matrix1, 
    torch::Tensor matrix2
) {

    const auto height = matrix1.size(0);
    const auto width = matrix2.size(1);
    const auto depth = matrix1.size(1);

    auto result = torch::zeros({height, width}, matrix1.options());

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks(
        cdiv(width, threads_per_block.x),
        cdiv(height, threads_per_block.y)
    );

    matrix_multiplication_kernel<<<number_of_blocks, threads_per_block>>>(
        result.data_ptr<float>(),
        matrix1.data_ptr<float>(),
        matrix2.data_ptr<float>(),
        width,
        height,
        depth
    );

    return result;
}