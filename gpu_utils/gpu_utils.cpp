#include <torch/extension.h>

#include <iostream>


// CUDA forward declarations

torch::Tensor fitness_cuda_nonlamarckian(
    torch::Tensor population,
    double limit_down,
    double limit_up,
    double delta,
    torch::Tensor results);

torch::Tensor bounce_back_boundary_2d_cuda(
        torch::Tensor population,
        double limit_down,
        double limit_up,
        double delta);

torch::Tensor poor_selective_matmul_cuda(
        torch::Tensor matrix,
        torch::Tensor sorting,
        torch::Tensor vec,
        torch::Tensor result,
        int limit);

torch::Tensor create_sorted_weights_for_matmul_cuda(
        torch::Tensor weights,
        torch::Tensor sorting,
        torch::Tensor sorted_weights,
        int limit);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fitness_nonlamarckian(
    torch::Tensor population,
    double limit_down,
    double limit_up,
    double delta,
    torch::Tensor results) {
  CHECK_INPUT(population);
  CHECK_INPUT(results);

  return fitness_cuda_nonlamarckian(population, limit_down, limit_up, delta, results);
}

torch::Tensor bounce_back_boundary_2d(
    torch::Tensor population,
    double limit_down,
    double limit_up,
    double delta) {
  CHECK_INPUT(population);

  return bounce_back_boundary_2d_cuda(population, limit_down, limit_up, delta);
}

torch::Tensor poor_selective_matmul(
        torch::Tensor matrix,
        torch::Tensor sorting,
        torch::Tensor vec,
        torch::Tensor result,
        int limit) {
    CHECK_INPUT(matrix);
    CHECK_INPUT(sorting);
    CHECK_INPUT(vec);
    CHECK_INPUT(result);
    return poor_selective_matmul_cuda(matrix, sorting, vec, result, limit);
}

torch::Tensor create_sorted_weights_for_matmul(
        torch::Tensor weights,
        torch::Tensor sorting,
        torch::Tensor sorted_weights,
        int limit) {
    CHECK_INPUT(weights);
    CHECK_INPUT(sorting);
    CHECK_INPUT(sorted_weights);
    return create_sorted_weights_for_matmul_cuda(weights, sorting, sorted_weights, limit);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fitness_nonlamarckian", &fitness_nonlamarckian, "Fitness Non-lamarckian (CUDA)");
    m.def("bounce_back_boundary_2d", &bounce_back_boundary_2d, "Bounce back boundary 2d (CUDA)");
    m.def("poor_selective_matmul", &poor_selective_matmul, "Poor selective matmul (CUDA)");
    m.def("create_sorted_weights_for_matmul", &create_sorted_weights_for_matmul, "Sorted weights (CUDA)");
}

