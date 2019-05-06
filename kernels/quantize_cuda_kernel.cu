#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

namespace {
template <typename scalar_t>
__global__ void quantize_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const float delta,
    const float minv,
    const float maxv,
    const float clampv,
    const int32_t size) {
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

at::Tensor quantize_cuda_forward(
    const at::Tensor input,
    const float delta,
    const float minv,
    const float maxv,
    const float clampv) {
  // const auto size = input.size(0)*input.size(1)*input.size(2)*input.size(3);
  const auto size = input.numel();
  auto output = at::zeros_like(input);

  const int32_t threads = 128;
  const int32_t blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "quantize_forward_cuda", ([&] {
    quantize_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        output.data<scalar_t>(),
        delta,
        minv,
        maxv,
        clampv,
        size);
  }));

  return output;
}

at::Tensor quantize_cuda_backward(
    at::Tensor grad_h,
    const at::Tensor input,
    const float minv,
    const float maxv) {

  // const auto size = input.size(0)*input.size(1)*input.size(2)*input.size(3);
  const auto size = input.numel();
  const int threads = 128;
  const int blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "quantize_backward_cuda", ([&] {
    quantize_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_h.data<scalar_t>(),
        input.data<scalar_t>(),
        minv,
        maxv,
        size);
  }));

  return grad_h;
}
