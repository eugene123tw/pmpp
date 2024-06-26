from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("exercise_3-2/exercise_3-2.cu").read_text()
    cpp_source = "torch::Tensor mat_vec_mul(torch::Tensor input_matrix, torch::Tensor input_vector);"

    # Load the CUDA kernel as a PyTorch extension
    mat_vec_mul_extension = load_inline(
        name="mat_vec_mul_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["mat_vec_mul"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return mat_vec_mul_extension


def main():
    ext = compile_extension()

    n = 3
    b = torch.rand(n, n).contiguous().cuda()
    c = torch.rand(n).cuda()

    a = ext.mat_vec_mul(b, c)

    for i in range(n):
        assert torch.allclose(a[i], b[i].sum() + c[i])


if __name__ == "__main__":
    main()
