import torch
from timerit import Timerit
import numpy as np
from numpy.random import shuffle
from gpu_utils import poor_selective_matmul, create_sorted_weights_for_matmul

np.random.seed(42)


def benchmark_torch():
    device = torch.device("cuda:0")
    t1 = Timerit(num=200, verbose=2)
    total = 0
    for timer in t1:
        matrix = torch.rand((50000, 4000), device=device)
        srt_idx = np.arange(0, 4000)
        shuffle(srt_idx)
        sorting_idx = torch.tensor(srt_idx[:2000], device=device)
        vec = torch.rand((2000,), device=device)
        result = torch.zeros((50000,), device=device)
        with timer:
            result = matrix[:, sorting_idx].matmul(vec)
            total += result.sum().item()
    print("t1.total_time = %r" % (t1.total_time,))
    print(total)


def benchmark_cuda():
    device = torch.device("cuda:0")
    t1 = Timerit(num=200, verbose=2)
    total = 0
    for timer in t1:
        matrix = torch.rand((50000, 4000), device=device)
        srt_idx = np.arange(0, 4000)
        shuffle(srt_idx)
        sorting_idx = torch.tensor(srt_idx, dtype=torch.int32, device=device)
        vec = torch.rand((2000,), device=device)
        result = torch.zeros((50000,), device=device)
        with timer:
            result = poor_selective_matmul(
                matrix, sorting_idx.argsort().int(), vec, result, 2000
            )
            #  result = poor_selective_matmul(matrix, sorting_idx.int(), vec, result, 2000)
            total += result.sum().item()
    print("t1.total_time = %r" % (t1.total_time,))
    print(total)


def benchmark_cuda_weights():
    device = torch.device("cuda:0")
    t1 = Timerit(num=200, verbose=2)
    total = 0
    for timer in t1:
        matrix = torch.rand((50000, 4000), device=device)
        srt_idx = np.arange(0, 4000)
        shuffle(srt_idx)
        sorting_idx = torch.tensor(srt_idx, dtype=torch.int32, device=device)
        vec = torch.rand((2000,), device=device)
        result = torch.zeros((50000,), device=device)
        with timer:
            sorted_weights = torch.zeros((4000,), device=device)
            sorted_weights = create_sorted_weights_for_matmul(
                vec, sorting_idx, sorted_weights, 2000
            )
            result = matrix.matmul(sorted_weights)
            total += result.sum().item()
    print("t1.total_time = %r" % (t1.total_time,))
    print(total)


if __name__ == "__main__":
    print("CUDA weights")
    benchmark_cuda_weights()
    print("Poor CUDA")
    benchmark_cuda()
    print("Torch version")
    benchmark_torch()
