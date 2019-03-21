#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

namespace {
template <typename scalar_t>
__global__ void log_quantize_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const float base,
    const float min_exp,
    const float max_exp,
    const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    const float x = input[idx];
    const float x_exp = roundf(__log2f(abs(x)));
    const float x_round = x_exp < min_exp ? 0 :  __powf(base, fminf(x_exp, max_exp));
    output[idx] = (signbit(x) ? -1 : 1) * x_round;
  }
}

template <typename scalar_t>
__global__ void log_quantize_cuda_backward_kernel(
    scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ input,
    const float base,
    const float max_exp,
    const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const float max_val = __powf(base, max_exp);
  if (idx < size) {
      const float x = input[idx];
      grad_h[idx] *= (x > -max_val)*(x < max_val);
  }
}
} // namespace

at::Tensor log_quantize_cuda_forward(
    const at::Tensor input,
    const float base,
    const float min_exp,
    const float max_exp) {
  const auto size = input.numel();
  auto output = at::zeros_like(input);

  const int32_t threads = 1024;
  const int32_t blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "log_quantize_forward_cuda", ([&] {
    log_quantize_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        output.data<scalar_t>(),
        base,
        min_exp,
        max_exp,
        size);
  }));

  return output;
}

at::Tensor log_quantize_cuda_backward(
    at::Tensor grad_h,
    const at::Tensor input,
    const float base,
    const float max_exp) {

  const auto size = input.numel();
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "log_quantize_backward_cuda", ([&] {
    log_quantize_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_h.data<scalar_t>(),
        input.data<scalar_t>(),
        base,
        max_exp,
        size);
  }));

  return grad_h;
}
