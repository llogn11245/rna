#include <tuple>
#include <string>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>
#include <torch/extension.h>

#include "core.h"

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be Float")
#define CHECK_INT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Int, #x " must be Int")

std::tuple<at::Tensor, at::Tensor> rna_loss(
        const at::Tensor& xs, const at::Tensor& ys,
        const at::Tensor& xn, const at::Tensor& yn,
        const int blank) {
    
    CHECK_CONTIGUOUS(xs);
    CHECK_CONTIGUOUS(ys);
    CHECK_CONTIGUOUS(xn);
    CHECK_CONTIGUOUS(yn);

    CHECK_FLOAT(xs);
    CHECK_INT(ys);
    CHECK_INT(xn);
    CHECK_INT(yn);

    CHECK_CUDA(xs);
    CHECK_CUDA(ys);
    CHECK_CUDA(xn);
    CHECK_CUDA(yn);

    TORCH_CHECK(xs.dim() == 4, "xs must have 4 dimensions");
    TORCH_CHECK(xn.numel() == xs.size(0), "xn shape must be equal to (N,)");
    TORCH_CHECK(yn.numel() == xs.size(0), "yn shape must be equal to (N,)");
    TORCH_CHECK(xs.size(2) == ys.size(1) + 1, "ys shape (N, U-1) mismatched with xs (N, T, U, V)");

    const auto N = xs.size(0);
    const auto T = xs.size(1);
    const auto U = xs.size(2);
    const auto V = xs.size(3);

    const auto S = T - yn.min().item().toInt() + 1;

    at::Tensor grads = at::zeros_like(xs);

    auto buffer_opts = xs.options().dtype(at::kFloat);
    auto counts_opts = xs.options().dtype(at::kInt);
    auto costs_opts = xs.options().dtype(at::kFloat);

    auto counts_shape = std::vector<int64_t>{N, U * 2};
    auto buffer_shape = std::vector<int64_t>{N, S, U};
    auto costs_shape = std::vector<int64_t>{N};

    at::Tensor costs = at::empty(costs_shape, costs_opts);
    at::Tensor counts = at::zeros(counts_shape, counts_opts);
    at::Tensor alphas = at::empty(buffer_shape, buffer_opts);
    at::Tensor betas = at::empty(buffer_shape, buffer_opts);

    auto stream = at::cuda::getCurrentCUDAStream(xs.device().index());

    auto status = run_warp_rna(
        stream,
        reinterpret_cast<unsigned int*>(counts.data_ptr<int>()),
        alphas.data_ptr<float>(), betas.data_ptr<float>(),
        ys.data_ptr<int>(), xs.data_ptr<float>(),
        grads.data_ptr<float>(), costs.data_ptr<float>(),
        xn.data_ptr<int>(), yn.data_ptr<int>(),
        N, T, S, U, V, blank
    );

    TORCH_CHECK(status == RNA_STATUS_SUCCESS, "rna_loss failed with status: ", status);

    return std::make_tuple(costs, grads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rna_loss", &rna_loss,
          "CUDA-Warp Recurrent Neural Aligner loss (forward and backward)",
          pybind11::arg("xs"), pybind11::arg("ys"),
          pybind11::arg("xn"), pybind11::arg("yn"),
          pybind11::arg("blank") = 0);
}
