from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("matrix_multiplication/matrix_multiplication.cu").read_text()
    cpp_source = "torch::Tensor matrix_multiplication(torch::Tensor matrix1, torch::Tensor matrix2);"

    # Load the CUDA kernel as a PyTorch extension
    matrix_multiplication_extension = load_inline(
        name="matrix_multiplication_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matrix_multiplication"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return matrix_multiplication_extension


def main():
    ext = compile_extension()

    a = torch.rand(3, 3).cuda()
    b = torch.rand(3, 3).cuda()

    y = ext.matrix_multiplication(a, b)

    assert torch.allclose(y, torch.matmul(a, b))


if __name__ == "__main__":
    main()
