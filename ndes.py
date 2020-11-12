import gc

#  from pytorch_memlab import profile
from enum import Enum
from math import floor, log, sqrt
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch

from gpu_utils import create_sorted_weights_for_matmul, fitness_nonlamarckian
from utils import bounce_back_boundary_1d, bounce_back_boundary_2d, create_directory


class SecondaryMutation(Enum):
    RandomNoise = 1
    Gradient = 2


class NDES:

    """neural Differential Evolution Strategy (nDES) implementation."""

    def __init__(
        self, initial_value, fn, lower, upper, population_initializer, **kwargs
    ):
        self.initial_value = torch.empty_like(initial_value)
        self.initial_value.copy_(initial_value)
        self.problem_size = int(len(initial_value))
        self.fn = fn
        self.lower = lower
        self.upper = upper

        self.device = kwargs.get("device", torch.device("cpu"))
        self.dtype = kwargs.get("dtype", torch.float32)
        self.population_initializer = population_initializer

        if np.isscalar(lower):
            self.lower = torch.tensor(
                [lower] * self.problem_size, device=self.device, dtype=self.dtype
            )

        if np.isscalar(upper):
            self.upper = torch.tensor(
                [upper] * self.problem_size, device=self.device, dtype=self.dtype
            )

        # Scaling factor of difference vectors (a variable!)
        self.Ft = kwargs.get("Ft", 1)
        self.initFt = kwargs.get("initFt", 1)

        #  Fitness value after which the convergence is reached
        self.stopfitness = kwargs.get("stopfitness", -np.inf)

        # Strategy parameter setting:
        #  The maximum number of fitness function calls
        self.budget = kwargs.get("budget", 10000 * self.problem_size)
        #  Population size
        self.lambda_ = kwargs.get("lambda_", 4 * self.problem_size)
        #  Selection size
        self.mu = kwargs.get("mu", floor(self.lambda_ / 2))
        #  Weights to calculate mean from selected individuals
        self.weights = (
            log(self.mu + 1)
            - torch.arange(1.0, self.mu + 1, device=self.device, dtype=self.dtype).log()
        )
        #     \-> weights are normalized by the sum
        self.weights = self.weights / self.weights.sum()
        self.weights_pop = (
            log(self.lambda_ + 1)
            - torch.arange(
                1.0, self.lambda_ + 1, device=self.device, dtype=self.dtype
            ).log()
        )
        self.weights_pop = self.weights_pop / self.weights_pop.sum()
        #  Evolution Path decay factor
        self.cc = kwargs.get("ccum", self.mu / (self.mu + 2))
        #  Evolution Path decay factor
        self.cp = kwargs.get("cp", 1 / sqrt(self.problem_size))
        #  Maximum number of iterations after which algorithm stops
        self.max_iter = kwargs.get("maxit", floor(self.budget / (self.lambda_ + 1)))
        #  Size of the window of history - the step length history
        self.hist_size = kwargs.get("history", 5)
        self.tol = kwargs.get("tol", 1e-12)
        #  Number of function evaluations
        self.count_eval = 0
        self.sqrt_N = sqrt(self.problem_size)

        # nonLamarckian approach allows individuals to violate boundaries.
        # Fitness value is estimeted by fitness of repaired individual.
        self.lamarckism = kwargs.get("lamarckism", False)
        self.worst_fitness = kwargs.get("worst_fitness", torch.finfo(self.dtype).max)

        self.cpu = torch.device("cpu")
        self.start = timer()
        self.test_func = kwargs.get("test_func", None)
        self.iter_callback = kwargs.get("iter_callback", None)
        self.log_dir = kwargs.get("log_dir", ".")
        create_directory(self.log_dir)
        self.secondary_mutation = kwargs.get("secondary_mutation", None)

    # @profile
    def _fitness_wrapper(self, x):
        if (x >= self.lower).all() and (x <= self.upper).all():
            self.count_eval += 1
            return self.fn(x)
        return self.worst_fitness

    # @profile
    def _fitness_lamarckian(self, x):
        if np.isscalar(x):
            if self.count_eval < self.budget:
                return self._fitness_wrapper(x)
            return self.worst_fitness

        cols = 1 if len(x.shape) == 1 else x.shape[1]
        fitnesses = []
        if self.count_eval + cols <= self.budget:
            if cols > 1:
                for i in range(cols):
                    fitnesses.append(self._fitness_wrapper(x[:, i]))
                return torch.tensor(fitnesses, device=self.device, dtype=self.dtype)
            return self._fitness_wrapper(x)

        budget_left = self.budget - self.count_eval
        for i in range(budget_left):
            fitnesses.append(self._fitness_wrapper(x[:, i]))
        if not fitnesses and cols == 1:
            return self.worst_fitness
        return torch.tensor(
            fitnesses + [self.worst_fitness] * (cols - budget_left),
            device=self.device,
            dtype=self.dtype,
        )

    # @profile
    def _fitness_non_lamarckian(self, x, fitness):
        summed = torch.zeros_like(fitness)
        fitness_nonlamarckian(
            x, self.lower[0], self.upper[0], self.upper[0] - self.lower[0], summed
        )
        mask = summed > 0
        fitness[mask] = self.worst_fit + summed[mask]
        return fitness

    #  @profile
    def get_random_samples(self, limit):
        history_sample1 = torch.randint(0, limit, (self.lambda_,), device=self.cpu)
        history_sample2 = torch.randint(0, limit, (self.lambda_,), device=self.cpu)

        x1_sample = torch.randint(0, self.mu, (self.lambda_,), device=self.cpu)
        x2_sample = torch.randint(0, self.mu, (self.lambda_,), device=self.cpu)
        return history_sample1, history_sample2, x1_sample, x2_sample

    #  @profile
    def get_diffs(self, hist_head, history, d_mean, pc):
        limit = hist_head + 1 if self.iter_ <= self.hist_size else self.hist_size
        (
            history_sample1,
            history_sample2,
            x1_sample,
            x2_sample,
        ) = self.get_random_samples(limit)

        x1 = history[:, x1_sample, history_sample1]
        x2 = history[:, x2_sample, history_sample1]
        x_diff = x1 - x2
        diffs_cpu = (
            sqrt(self.cc)
            * (
                x_diff
                + torch.randn(self.lambda_, device=self.cpu, dtype=self.dtype)
                * d_mean[:, history_sample1]
            )
            + sqrt(1 - self.cc)
            * torch.randn(self.lambda_, device=self.cpu, dtype=self.dtype)
            * pc[:, history_sample2]
        )
        return diffs_cpu

    #  @profile
    def run(self):
        assert len(self.upper) == self.problem_size
        assert len(self.lower) == self.problem_size
        assert (self.lower < self.upper).all()

        print(f"Running nDES for a problem of size {self.problem_size}")

        # The best fitness found so far
        self.best_fitness = self.worst_fitness
        # The best solution found so far
        self.best_solution = None
        # The worst solution found so far
        self.worst_fit = None

        d_mean = torch.zeros(
            (self.problem_size, self.hist_size), device=self.cpu, dtype=self.dtype
        )
        pc = torch.zeros(
            (self.problem_size, self.hist_size), device=self.cpu, dtype=self.dtype
        )

        sorted_weights = torch.zeros_like(self.weights_pop)

        log_ = pd.DataFrame(
            columns=[
                "step",
                "pc",
                "mean_fitness",
                "best_fitness",
                "fn_cum",
                "best_found",
                "iter",
            ]
        )
        #  evaluation_times = []
        while self.count_eval < self.budget:  # and self.iter_ < self.max_iter:

            hist_head = -1
            self.iter_ = -1

            history = torch.zeros(
                (self.problem_size, self.mu, self.hist_size),
                dtype=self.dtype,
                device=self.cpu,
            )
            self.Ft = self.initFt
            population = None

            gc.collect()
            torch.cuda.empty_cache()
            cum_mean = (self.upper + self.lower) / 2

            population = self.population_initializer.get_new_population(
                lower=self.lower, upper=self.upper
            ).contiguous()

            if self.lamarckism:
                population = bounce_back_boundary_2d(population, self.lower, self.upper)

            #  start = timer()
            fitness = self._fitness_lamarckian(population)
            #  end = timer()
            #  evaluation_times.append(end - start)

            new_mean = torch.empty_like(self.initial_value)
            new_mean.copy_(self.initial_value)
            self.worst_fit = fitness.max().item()

            # Store population and selection means
            sorting_idx = fitness.argsort()
            sorted_weights_pop = self.weights_pop[sorting_idx]
            pop_mean = population.matmul(sorted_weights_pop)

            if self.secondary_mutation == SecondaryMutation.RandomNoise:
                chi_N = sqrt(self.problem_size)
            hist_norm = 1 / sqrt(2)

            stoptol = False
            old_mean = torch.empty_like(new_mean)
            while self.count_eval < self.budget and not stoptol:

                iter_log = {}
                torch.cuda.empty_cache()
                gc.collect()
                self.iter_ += 1
                hist_head = (hist_head + 1) % self.hist_size

                # Select best 'mu' individuals of population
                sorting_idx = fitness.argsort()
                selection = sorting_idx[: self.mu]

                # Save selected population in the history buffer
                #  history[:, :, hist_head] = (population[:, selection] * hist_norm / self.Ft).cpu()
                history[:, :, hist_head] = population.cpu()[:, selection]
                history[:, :, hist_head] *= hist_norm / self.Ft

                # Calculate weighted mean of selected points
                old_mean.copy_(new_mean)
                sorted_weights.zero_()
                sorted_weights = create_sorted_weights_for_matmul(
                    self.weights, sorting_idx.int(), sorted_weights, self.mu
                )
                new_mean = population.matmul(sorted_weights)

                # Write to buffers
                tmp = new_mean - pop_mean
                d_mean[:, hist_head] = (tmp / self.Ft).cpu()

                step = ((new_mean - old_mean) / self.Ft).cpu()

                # Update parameters
                if hist_head == 0:
                    pc[:, hist_head] = sqrt(self.mu * self.cp * (2 - self.cp)) * step
                else:
                    pc[:, hist_head] = (1 - self.cp) * pc[:, hist_head - 1] + sqrt(
                        self.mu * self.cp * (2 - self.cp)
                    ) * step

                print(f"|step|={(step**2).sum().item()}")
                print(f"|pc|={(pc**2).sum().item()}")
                iter_log["step"] = (step ** 2).sum().item()
                iter_log["pc"] = (pc ** 2).sum().item()

                # Sample from history with uniform distribution
                diffs_cpu = self.get_diffs(hist_head, history, d_mean, pc)
                population.copy_(diffs_cpu)
                diffs_shape = diffs_cpu.shape
                del diffs_cpu

                # New population
                population *= self.Ft
                population += new_mean.unsqueeze(1)

                if self.secondary_mutation == SecondaryMutation.RandomNoise:
                    population += (
                        self.tol
                        * (1 - 2 / sqrt(self.problem_size)) ** (self.iter_ / 2)
                        * torch.randn(diffs_shape, device=self.device, dtype=self.dtype)
                        / chi_N
                    )

                if self.lamarckism:
                    population = bounce_back_boundary_2d(
                        population, self.lower, self.upper
                    )

                sorted_weights_pop = self.weights_pop[sorting_idx]
                pop_mean = population.matmul(sorted_weights_pop)

                gc.collect()
                torch.cuda.empty_cache()

                # Evaluation
                #  start = timer()
                fitness = self._fitness_lamarckian(population)
                #  end = timer()
                #  evaluation_times.append(end - start)
                if not self.lamarckism:
                    fitness_non_lamarckian = self._fitness_non_lamarckian(
                        population, fitness
                    )

                wb = fitness.argmin()
                print(f"best fitness: {fitness[wb]}")
                print(f"mean fitness: {fitness.clamp(0, self.worst_fitness).mean()}")
                iter_log["best_fitness"] = fitness[wb].item()
                iter_log["mean_fitness"] = (
                    fitness.clamp(0, self.worst_fitness).mean().item()
                )
                iter_log["iter"] = self.iter_

                if self.test_func is None and fitness[wb] < self.best_fitness:
                    self.best_fitness = fitness[wb].item()
                    self.best_solution = population[:, wb]

                # Check worst fit
                ww = fitness.argmax()
                if fitness[ww] > self.worst_fit:
                    self.worst_fit = fitness[ww]

                # Fitness with penalty for non-lamarckian approach
                if not self.lamarckism:
                    fitness = fitness_non_lamarckian

                # Check if the middle point is the best found so far
                cum_mean = 0.8 * cum_mean + 0.2 * new_mean
                cum_mean_repaired = bounce_back_boundary_1d(
                    cum_mean, self.lower, self.upper
                )

                fn_cum = self._fitness_lamarckian(cum_mean_repaired)
                print(f"fn_cum: {fn_cum}")
                iter_log["fn_cum"] = fn_cum

                if self.test_func is None and fn_cum < self.best_fitness:
                    self.best_fitness = fn_cum
                    self.best_solution = cum_mean_repaired

                if fitness[0] <= self.stopfitness:
                    break

                if (
                    abs(fitness.max() - fitness.min()) < self.tol
                    and self.count_eval < 0.8 * self.budget
                ):
                    stoptol = True
                print(f"iter={self.iter_}")
                iter_log["best_found"] = self.best_fitness
                if self.iter_ % 50 == 0 and self.test_func is not None:
                    (test_loss, test_acc), self.best_solution = self.test_func(
                        population
                    )
                else:
                    test_loss, test_acc = None, None

                iter_log["test_loss"] = test_loss
                iter_log["test_acc"] = test_acc
                log_ = log_.append(iter_log, ignore_index=True)
                if self.iter_ % 50 == 0:
                    log_.to_csv(f"{self.log_dir}/ndes_log_{self.start}.csv")

                if self.iter_callback:
                    self.iter_callback()

        log_.to_csv(f"ndes_log_{self.start}.csv")
        #  np.save(f"times_{self.problem_size}.npy", np.array(evaluation_times))
        return self.best_solution  # , log_
