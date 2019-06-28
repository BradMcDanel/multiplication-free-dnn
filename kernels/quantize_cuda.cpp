#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor quantize_cuda_forward(
    const at::Tensor input,
    const float delta,
    const float minv,
    const float maxv,
    const float clampv);

at::Tensor quantize_cuda_backward(
    at::Tensor grad_h,
    const at::Tensor input,
    const float minv,
    const float maxv);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor quantize_forward(
    const at::Tensor input,
    const float delta,
    const float minv,
    const float maxv,
    const float clampv) {
  CHECK_INPUT(input);

  return quantize_cuda_forward(input, delta, minv, maxv, clampv);
}

at::Tensor quantize_backward(
    at::Tensor grad_h,
    const at::Tensor input,
    const float minv,
    const float maxv) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input);
  return quantize_cuda_backward(grad_h, input, minv, maxv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &quantize_forward, "quantize forward (CUDA)");
  m.def("backward", &quantize_backward, "quantize backward (CUDA)");
}
