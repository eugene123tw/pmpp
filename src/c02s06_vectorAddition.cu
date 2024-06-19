#include <stdio.h>
#include <cuda_runtime.h>

__global__
void vecAddKernel(float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vecAdd(float *a_h, float *b_h, float *c_h, int n) {
    int size = n * sizeof(float);
    float *a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&b_d, size);
    cudaMalloc((void**)&c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n / 256.0), 256>>>(a_d, b_d, c_d, n);

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main() {
    int N = 1000;
    float x_h[N], y_h[N], res_h[N];
    for (int i = 0; i < N; i++) {
        x_h[i] = (float)i;
        y_h[i] = (i / 2.0) * (i / 2.0);
    }

    vecAdd(x_h, y_h, res_h, N);

    for (int i = 0; i < 100; i++) {
        printf("res_h[%2d]: %6f, delta: %4f\n", i, res_h[i], res_h[i] - (x_h[i] + y_h[i]));
    }
    printf("...\n");

    return 0;
}