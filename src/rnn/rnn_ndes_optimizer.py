import cma
import numpy as np
import torch

from src.classic.ndes import SecondaryMutation
from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.population_initializers import StartFromUniformPopulationInitializer


class RNNnDESOptimizer(BasenDESOptimizer):
    """nDES optimizer for RNNs, uses different initialization strategy than base
    optimizer."""

    def __init__(
        self,
        *args,
        population_initializer=StartFromUniformPopulationInitializer,
        secondary_mutation=SecondaryMutation.RandomNoise,
        **kwargs,
    ):
        super().__init__(
            *args,
            population_initializer=population_initializer,
            secondary_mutation=secondary_mutation,
            **kwargs,
        )

    def calculate_xavier_coefficients(self, layers_iter):
        return torch.ones_like(self.initial_value) * 0.4


class CMAESOptimizerRNN:
    def __init__(self, model, criterion, data_gen, restarts=None, **kwargs):
        self.model = model
        self.criterion = criterion
        self.data_gen = data_gen

    def _objective_function(self, weights):
        """Custom objective function for the DES optimizer."""
        self.model.set_weights(weights)
        _, (x_data, y_data) = next(self.data_gen)
        predicted = np.array([self.model.forward(x) for x in x_data])
        loss = self.criterion(y_data, predicted)
        return loss

    def run(self, test_func=None):
        """Optimize model's weights wrt. the given criterion.

        Returns:
            Optimized model.
        """
        es = cma.CMAEvolutionStrategy(11 * [0], 1.5, {"verb_disp": 1, "maxiter": 1000})
        es.optimize(self._objective_function)
        return es.best.get()[0]
