#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


__global__
void mat_vec_mul_kernel(float* result, float* input_matrix, float* input_vector, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n){
        for(int i=0; i < n; i++){
            result[idx] += input_matrix[idx * n + i];
        }
        result[idx] += input_vector[idx];
    }
}


inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor mat_vec_mul(torch::Tensor input_matrix, torch::Tensor input_vector){
    const auto n = input_matrix.size(0);
    auto result = torch::zeros({n}, input_matrix.options());

    dim3 thread_per_block(16);
    dim3 number_of_blocks(cdiv(n, thread_per_block.x));
    mat_vec_mul_kernel<<<number_of_blocks, thread_per_block>>>(
        result.data_ptr<float>(),
        input_matrix.data_ptr<float>(),
        input_vector.data_ptr<float>(),
        n
    );
    return result;
}