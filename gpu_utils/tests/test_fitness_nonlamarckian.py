import torch
from gpu_utils import fitness_nonlamarckian


if __name__ == "__main__":
    device = torch.device("cuda:0")
    population = torch.zeros((50034, 4000), device=device)
    #  results = torch.zeros_like(population)
    results = torch.zeros((4000,), device=device)
    population[1, 0] = -7
    population[1, 2] = 3
    population[2, 3] = 18
    fixed = ((population - (2 - ((population - 2) % 4))) ** 2).sum(dim=0)
    results = fitness_nonlamarckian(population, -2.0, 2.0, 4.0, results)
    assert torch.allclose(fixed, results)
    __import__("pdb").set_trace()
