#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

namespace {
template <typename scalar_t>
__global__ void quantize_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const bool training,
    const float momentum,
    const float eps,
    scalar_t* __restrict__ output) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = (input[idx] < minv) ? clampv : fminf(fmaxf(delta*floor((input[idx] / delta) + 0.5), minv), maxv);
  }
}

template <typename scalar_t>
__global__ void quantize_cuda_backward_kernel(
    scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ input,
    const float minv,
    const float maxv,
    const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
      const float x = input[idx];
      grad_h[idx] *= (x > minv)*(x < maxv);
  }
}
} // namespace

at::Tensor qbatch_norm_cuda_forward(
    const at::Tensor input,
    const at::Tensor running_mean,
    const at::Tensor running_var,
    const at::Tensor weight,
    const at::Tensor bias,
    const bool training,
    const float momentum,
    const float eps) {
  const auto size = input.numel();
  auto output = at::zeros_like(input);

  const int32_t threads = 128;
  const int32_t blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "qbatch_norm_forward_cuda", ([&] {
    qbatch_norm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        running_mean.data<scalar_t>(),
        running_var.data<scalar_t>(),
        running_weight.data<scalar_t>(),
        running_bias.data<scalar_t>(),
        training,
        momentum,
        eps,
        output.data<scalar_t>());

  }));

  return output;
}

std::vector<at::Tensor> qbatch_norm_cuda_backward(
    const at::Tensor grad_h,
    const at::Tensor input,
    const at::Tensor running_mean,
    const at::Tensor running_var,
    const at::Tensor weight,
    const at::Tensor bias) {

  const auto size = input.numel();
  auto dinput = at::zeros_like(input);
  auto dweight = at::zeros_like(weight);
  auto dbias = at::zeros_like(bias);
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "qbatch_norm_backward_cuda", ([&] {
    quantize_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_h.data<scalar_t>(),
        input.data<scalar_t>(),
        running_mean.data<scalar_t>(),
        running_var.data<scalar_t>(),
        running_weight.data<scalar_t>(),
        running_bias.data<scalar_t>(),
        dinput.data<scalar_t>(),
        dweight.data<scalar_t>(),
        dbias.data<scalar_t>());
  }));

  return {dinput, d_weight, d_bias};
}
