import torch
from torch.distributions import MultivariateNormal, Uniform

from utils import bounce_back_boundary_2d


class XavierMVNPopulationInitializer:
    def __init__(self, initial_value, lambda_, xavier_coeffs, device):
        self.initial_value = initial_value
        self.xavier_coeffs = xavier_coeffs.to(device)
        self.device = device

        sd = torch.eye(lambda_, device=self.device).cpu()
        mean = (
            torch.zeros_like(self.initial_value).unsqueeze(1).repeat(1, lambda_).cpu()
        )
        self.normal = MultivariateNormal(mean, sd)

    def get_new_population(self, lower, upper):
        population = self.normal.sample().to(self.device)
        population *= self.xavier_coeffs[:, None]
        population += self.initial_value[:, None]
        population[:, 0] = self.initial_value
        return bounce_back_boundary_2d(population, lower, upper)


class UniformPopulationInitializer:
    def __init__(self, initial_value_, lambda_, xavier_coeffs, device):
        self.xavier_coeffs = xavier_coeffs
        self.device = device
        self.lambda_ = lambda_

        self.uniform = Uniform(-self.xavier_coeffs.cpu(), self.xavier_coeffs.cpu())

    def get_new_population(self, _lower, _upper):
        return self.uniform.sample((self.lambda_,)).transpose(0, 1).to(self.device)


class StartFromUniformPopulationInitializer:
    def __init__(self, *args):
        self.args = args
        self.uniform = UniformPopulationInitializer(*args)
        self.xavier_mvn = None

    def get_new_population(self):
        # first iteration - start from uniform
        if self.uniform is not None:
            population = self.uniform.get_new_population()
            del self.uniform
            self.uniform = None
            return population
        # second iteration
        elif self.xavier_mvn is None:
            self.xavier_mvn = XavierMVNPopulationInitializer(*self.args)
        # consecutive iterations
        return self.xavier_mvn.get_new_population()
