import torch
import numpy as np
from numpy.random import shuffle
from gpu_utils import poor_selective_matmul, create_sorted_weights_for_matmul
import pytest
import random


@pytest.fixture
def setup_data():
    np.random.seed(42)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42)

    device = torch.device("cuda:0")
    matrix = torch.rand((50000, 4000), device=device)
    srt_idx = np.arange(0, 4000)
    shuffle(srt_idx)
    sorting_idx = torch.tensor(srt_idx, dtype=torch.int32, device=device)
    sorting_idx_torch = sorting_idx[:2000].long()
    sorting_idx = sorting_idx.int()
    vec = torch.rand((2000,), device=device)
    return device, matrix, sorting_idx, sorting_idx_torch, vec


def test_poor_selective_matmul(setup_data):
    device, matrix, sorting_idx, sorting_idx_torch, vec = setup_data
    result_cuda = torch.zeros((50000,), device=device)
    result_cuda = poor_selective_matmul(
        matrix, sorting_idx.argsort().int(), vec, result_cuda, 2000
    )
    result_torch = matrix[:, sorting_idx_torch].matmul(vec)
    assert torch.allclose(result_torch, result_cuda)


def test_selective_weights_matmul(setup_data):
    device, matrix, sorting_idx, sorting_idx_torch, vec = setup_data

    sorted_weights = torch.zeros((4000,), device=device)
    sorted_weights = create_sorted_weights_for_matmul(
        vec, sorting_idx, sorted_weights, 2000
    )
    result_cuda = matrix.matmul(sorted_weights)

    result_torch = matrix[:, sorting_idx_torch].matmul(vec)
    assert torch.allclose(result_torch, result_cuda)
