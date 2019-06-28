#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor log_quantize_cuda_forward(
    const at::Tensor input,
    const float base,
    const float min_exp,
    const float max_exp);

at::Tensor log_quantize_cuda_backward(
    at::Tensor grad_h,
    const at::Tensor input,
    const float base,
    const float max_exp);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor log_quantize_forward(
    const at::Tensor input,
    const float base,
    const float min_exp,
    const float max_exp) {
  CHECK_INPUT(input);

  return log_quantize_cuda_forward(input, base, min_exp, max_exp);
}

at::Tensor log_quantize_backward(
    at::Tensor grad_h,
    const at::Tensor input,
    const float base,
    const float max_exp) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input);
  return log_quantize_cuda_backward(grad_h, input, base, max_exp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &log_quantize_forward, "log quantize forward (CUDA)");
  m.def("backward", &log_quantize_backward, "log quantize backward (CUDA)");
}
