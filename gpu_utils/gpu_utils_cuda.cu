#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

namespace {


    template <typename scalar_t>
    __device__ __forceinline__ scalar_t bounce_back_lower(scalar_t x, scalar_t limit_up,
            scalar_t delta) {
        const scalar_t tmp = limit_up - fmodf((x - limit_up), delta);
        return tmp;
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t bounce_back_lower_diff(scalar_t x, scalar_t
            limit_up, scalar_t delta) {
        return pow(x - bounce_back_lower(x, limit_up, delta), 2);
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t bounce_back_upper(scalar_t x, scalar_t
            limit_down, scalar_t delta) {
        const scalar_t tmp = limit_down + fmodf((limit_down - x), delta);
        return tmp;
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t bounce_back_upper_diff(scalar_t x, scalar_t
            limit_down, scalar_t delta) {
        return pow(x - bounce_back_upper(x, limit_down, delta), 2);
    }

    template <typename scalar_t>
    __global__ void fitness_nonlamarckian_cuda_kernel(
            const scalar_t* __restrict__ population,
            scalar_t limit_down,
            scalar_t limit_up,
            scalar_t delta,
            scalar_t* __restrict__ results,
            size_t state_size) {

        const int column = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = blockIdx.y * state_size + column;

        if (column < state_size) {
            if (__builtin_expect(population[index] > limit_up, 0)) {
                const scalar_t result = bounce_back_lower_diff(population[index],
                        limit_up, delta);
                atomicAdd(results + column, result);
            } else if (__builtin_expect(population[index] < limit_down, 0)) {
                const scalar_t result = bounce_back_upper_diff(population[index],
                        limit_down, delta);
                atomicAdd(results + column, result);
            }
        }
    }

    template <typename scalar_t>
    __global__ void bounce_back_boundary_2d_cuda_kernel(
            scalar_t* __restrict__ population,
            scalar_t limit_down,
            scalar_t limit_up,
            scalar_t delta,
            size_t state_size) {

        const int column = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = blockIdx.y * state_size + column;

        if (column < state_size) {
            if (__builtin_expect(population[index] > limit_up, 0)) {
                const scalar_t result = bounce_back_lower(population[index], limit_up,
                        delta);
                population[index] = result;
            } else if (__builtin_expect(population[index] < limit_down, 0)) {
                const scalar_t result = bounce_back_upper(population[index], limit_down,
                        delta);
                population[index] = result;
            }
        }
    }



    template <typename scalar_t>
    __global__ void poor_selective_matmul_cuda_kernel(
            const scalar_t* __restrict__ matrix,
            const int* __restrict__ sorting,
            const scalar_t* __restrict__ vec,
            scalar_t* __restrict__ results,
            int limit,
            size_t state_size) {

        const int column = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = blockIdx.y * state_size + column;
        const int row = blockIdx.y;

        if (column < state_size) {
            if (sorting[column] < limit) {
                const scalar_t result = matrix[index] * vec[sorting[column]];
                atomicAdd(results + row, result);
            }
        }
    }

    template <typename scalar_t>
    __global__ void create_sorted_weights_for_matmul_cuda_kernel(
            const scalar_t* __restrict__ weights,
            const int* __restrict__ sorting,
            scalar_t* __restrict__ sorted_weights,
            int limit,
            size_t weights_size) {

        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int i = index; i < weights_size; i += stride) {
            const int sorted_weights_index = sorting[i];
            if (i >= limit) {
                sorted_weights[sorted_weights_index] = 0;
            } else {
                sorted_weights[sorted_weights_index] = weights[i];
            }
        }
    }

    // template <typename scalar_t>
    // __global__ void poor_selective_matmul_cuda_kernel(
            // const scalar_t* __restrict__ matrix,
            // const int* __restrict__ sorting,
            // const scalar_t* __restrict__ vec,
            // scalar_t* __restrict__ results,
            // int limit,
            // size_t state_size) {

        // const int column = blockIdx.x * blockDim.x + threadIdx.x;
        // const int index = blockIdx.y * state_size + column;
        // const int row = blockIdx.y;

        // if (column < state_size) {
            // int i = 0;
            // // check if the current column (invididual) is in the top-limit individuals
            // for ( ; i < limit; ++i ) {
                // if (sorting[i] == column) {
                    // break;
                // }
            // }
            // // if so then we need to include it in the result
            // if (i < limit) {
                // const scalar_t result = matrix[index] * vec[i];
                // atomicAdd(results + row, result);
            // }
        // }
    // }
}

torch::Tensor fitness_cuda_nonlamarckian(
        torch::Tensor population,
        double limit_down,
        double limit_up,
        double delta,
        torch::Tensor res) {

    const auto batch_size = population.size(0);
    const auto state_size = population.size(1);

    const int threads = 128;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(population.scalar_type(), "fitness_nonlamarckian_cuda", ([&] {
                fitness_nonlamarckian_cuda_kernel<scalar_t><<<blocks, threads>>>(
                        population.data_ptr<scalar_t>(),
                        static_cast<scalar_t>(limit_down),
                        static_cast<scalar_t>(limit_up),
                        static_cast<scalar_t>(delta),
                res.data_ptr<scalar_t>(),
                state_size
                );
            }));

    return res;
}

torch::Tensor bounce_back_boundary_2d_cuda(
        torch::Tensor population,
        double limit_down,
        double limit_up,
        double delta) {

    const auto batch_size = population.size(0);
    const auto state_size = population.size(1);

    const int threads = 128;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(population.scalar_type(), "bounce_back_boundary_2d_cuda", ([&] {
        bounce_back_boundary_2d_cuda_kernel<scalar_t><<<blocks, threads>>>(
            population.data_ptr<scalar_t>(),
            static_cast<scalar_t>(limit_down),
            static_cast<scalar_t>(limit_up),
            static_cast<scalar_t>(delta),
            state_size
        );
    }));

    return population;
}

torch::Tensor poor_selective_matmul_cuda(
        torch::Tensor matrix,
        torch::Tensor sorting,
        torch::Tensor vec,
        torch::Tensor result,
        int limit) {

    const auto batch_size = matrix.size(0);
    const auto state_size = matrix.size(1);

    const int threads = 128;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(matrix.scalar_type(), "poor_selective_matmul_cuda", ([&] {
        poor_selective_matmul_cuda_kernel<scalar_t><<<blocks, threads>>>(
            matrix.data_ptr<scalar_t>(),
            sorting.data_ptr<int>(),
            vec.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            limit,
            state_size
        );
    }));

    return result;
}

torch::Tensor create_sorted_weights_for_matmul_cuda(
        torch::Tensor weights,
        torch::Tensor sorting,
        torch::Tensor sorted_weights,
        int limit) {

    const auto weights_size = sorting.size(0);

    const int threads = 128;
    const int blocks = (weights_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "create_sorted_weights_for_matmul_cuda", ([&] {
        create_sorted_weights_for_matmul_cuda_kernel<scalar_t><<<blocks, threads>>>(
            weights.data_ptr<scalar_t>(),
            sorting.data_ptr<int>(),
            sorted_weights.data_ptr<scalar_t>(),
            limit,
            weights_size
        );
    }));

    return sorted_weights;
}
