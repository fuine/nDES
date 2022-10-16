from population_initializers import XavierMVNPopulationInitializer
from math import sqrt

from timeit import default_timer as timer
import torch


class GanNDESOptimizer:
    """Base interface for the nDES optimizer for the neural networks optimization."""

    def __init__(
        self,
        model,
        criterion,
        use_fitness_ewma=False,
        population_initializer=XavierMVNPopulationInitializer,
        # lr=1e-3,
    ):
        """
        Args:
            model: ``pytorch``'s model
            criterion: Loss function, must be minimizable.
            data_gen: Data generator, should yield batches: (batch_idx, (x, y))
            x_val: Validation data
            y_val: Validation ground truth
            use_fitness_ewma: is ``True`` will use EWMA fitness loss tracker
            population_initializer: Class of the population initialization strategy
            lr: Learning rate, only used if secondary_mutation is set to gradient
            restarts: Optional number of NDES's restarts.
            **kwargs: Keyword arguments for NDES optimizer
        """
        self._layers_offsets_shapes = []
        self.model = model
        self.device = model.device
        self.criterion = criterion
        self.population_initializer = population_initializer
        self.use_fitness_ewma = use_fitness_ewma
        self.start = timer()
        self.initial_value = self.zip_layers(model.parameters())
        self.xavier_coeffs = self.calculate_xavier_coefficients(model.parameters())
        # self.secondary_mutation = kwargs.get("secondary_mutation", None)
        # self.lr = lr
        # if use_fitness_ewma:
        #     self.ewma_logger = FitnessEWMALogger(data_gen, model, criterion)
        #     self.kwargs["iter_callback"] = self.ewma_logger.update_after_iteration

    def zip_layers(self, layers_iter):
        """Concatenate flattened layers into a single 1-D tensor.
        This method also saves shapes of layers and their offsets in the final
        tensor, allowing for a fast unzip operation.

        Args:
            layers_iter: Iterator over model's layers.
        """
        self._layers_offsets_shapes = []
        tensors = []
        current_offset = 0
        for param in layers_iter:
            shape = param.shape
            tmp = param.flatten()
            current_offset += len(tmp)
            self._layers_offsets_shapes.append((current_offset, shape))
            tensors.append(tmp)
        return torch.cat(tensors, 0).contiguous()

    def unzip_layers(self, zipped_layers):
        """Iterator over 'unzipped' layers, with their proper shapes.

        Args:
            zipped_layers: Flattened representation of layers.
        """
        start = 0
        for offset, shape in self._layers_offsets_shapes:
            yield zipped_layers[start:offset].view(shape)
            start = offset

    @staticmethod
    def calculate_xavier_coefficients(layers_iter):
        xavier_coeffs = []
        for param in layers_iter:
            param_num_elements = param.numel()
            if len(param.shape) > 1:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(param)
                xavier_coeffs.extend(
                    [sqrt(6 / (fan_in + fan_out))] * param_num_elements
                )
            else:
                xavier_coeffs.extend([xavier_coeffs[-1]] * param_num_elements)
        return torch.tensor(xavier_coeffs)

    # @profile
    def _objective_function(self, weights):
        """Custom objective function for the DES optimizer."""
        # self._reweight_model(weights)
        batch_idx, (b_x, y) = next(self.data_gen)
        # if self.secondary_mutation == SecondaryMutation.Gradient:
        #     gradient = []
        #     with torch.enable_grad():
        #         self.model.zero_grad()
        #         out = self.model(b_x)
        #         loss = self.criterion(out, y)
        #         loss.backward()
        #         for param in self.model.parameters():
        #             gradient.append(param.grad.flatten())
        #         gradient = torch.cat(gradient, 0)
        #         # In-place mutation of the weights
        #         weights -= self.lr * gradient
        # else:
        out = self.model(b_x)
        loss = self.criterion(out, y)
        loss = loss.item()
        # if self.use_fitness_ewma:
        #     return self.ewma_logger.update_batch(batch_idx, loss)
        return loss

    # @profile
    def run(self, test_func=None):
        """Optimize model's weights wrt. the given criterion.

        Returns:
            Optimized model.
        """
        self.test_func = test_func
        best_value = self.initial_value
        with torch.no_grad():
            # requires_grad = self.secondary_mutation == SecondaryMutation.Gradient
            requires_grad = False
            for param in self.model.parameters():
                param.requires_grad = requires_grad
            population_initializer_args = [
                self.xavier_coeffs,
                self.device,
                None,
            ]
            population_initializer = self.population_initializer(
                best_value, *population_initializer_args
            )
            test_func = None
            ndes_config = {
                    'initial_value': best_value,
                    'fn': self._objective_function,
                    'xavier_coeffs': self.xavier_coeffs,
                    'population_initializer': population_initializer,
                    'test_func': test_func,
                }

            # restarty w obecnej konfiguracji są none
            # if self.restarts is not None:
            ndes = NDES(ndes_config)
            best_value = ndes.run()
            # self._reweight_model(best_value)
            return self.model