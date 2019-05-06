#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor qbatch_norm_cuda_forward(
    const at::Tensor input,
    const at::Tensor running_mean,
    const at::Tensor running_var,
    const at::Tensor weight,
    const at::Tensor bias,
    const bool training,
    const float momentum,
    const float eps);


at::Tensor qbatch_norm_cuda_backward(
    const at::Tensor grad_h,
    const at::Tensor input,
    const at::Tensor running_mean,
    const at::Tensor running_var,
    const at::Tensor weight,
    const at::Tensor bias);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor qbatch_norm_forward(
    const at::Tensor input,
    const at::Tensor running_mean,
    const at::Tensor running_var,
    const at::Tensor weight,
    const at::Tensor bias,
    const bool training,
    const float momentum,
    const float eps) {
  CHECK_INPUT(input);
  CHECK_INPUT(running_mean);
  CHECK_INPUT(running_var);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);

  return qbatch_norm_cuda_forward(input, running_mean, running_var, weight,
                                  bias, training, momentum, eps);
}

at::Tensor qbatch_norm_backward(
    const at::Tensor grad_h,
    const at::Tensor input,
    const at::Tensor running_mean,
    const at::Tensor running_var,
    const at::Tensor weight,
    const at::Tensor bias) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input);
  CHECK_INPUT(running_mean);
  CHECK_INPUT(running_var);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);

  return qbatch_norm_cuda_backward(grad_h, input, running_mean, running_var,
                                   weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &qbatch_norm_forward, "qbatch_norm forward (CUDA)");
  m.def("backward", &qbatch_norm_backward, "qbatch_norm backward (CUDA)");
}
