#include <ATen/ATen.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <vector>

#include "utils/checks.h"
#include "utils/cuda.cuh"
#include "quant_bn.h"

#include <ATen/cuda/CUDAContext.h>

// Operations for reduce
template<typename T>
struct SumOp {
  __device__ SumOp(const T *t, int c, int s)
      : tensor(t), chn(c), sp(s) {}
  __device__ __forceinline__ T operator()(int batch, int plane, int n) {
    return tensor[(batch * chn + plane) * sp + n];
  }
  const T *tensor;
  const int chn;
  const int sp;
};

template<typename T>
struct VarOp {
  __device__ VarOp(T m, const T *t, int c, int s)
      : mean(m), tensor(t), chn(c), sp(s) {}
  __device__ __forceinline__ T operator()(int batch, int plane, int n) {
    T val = tensor[(batch * chn + plane) * sp + n];
    return (val - mean) * (val - mean);
  }
  const T mean;
  const T *tensor;
  const int chn;
  const int sp;
};

template<typename T>
struct GradOp {
  __device__ GradOp(T _weight, T _bias, const T *_z, const T *_dz, int c, int s)
      : weight(_weight), bias(_bias), z(_z), dz(_dz), chn(c), sp(s) {}
  __device__ __forceinline__ Pair<T> operator()(int batch, int plane, int n) {
    T _y = (z[(batch * chn + plane) * sp + n] - bias) / weight;
    T _dz = dz[(batch * chn + plane) * sp + n];
    return Pair<T>(_dz, _y * _dz);
  }
  const T weight;
  const T bias;
  const T *z;
  const T *dz;
  const int chn;
  const int sp;
};

/***********
 * mean_var
 ***********/

template<typename T>
__global__ void mean_var_kernel(const T *x, T *mean, T *var, int num, int chn, int sp) {
  int plane = blockIdx.x;
  T norm = T(1) / T(num * sp);

  T _mean = reduce<T, SumOp<T>>(SumOp<T>(x, chn, sp), plane, num, sp) * norm;
  __syncthreads();
  T _var = reduce<T, VarOp<T>>(VarOp<T>(_mean, x, chn, sp), plane, num, sp) * norm;

  if (threadIdx.x == 0) {
    mean[plane] = _mean;
    var[plane] = _var;
  }
}

std::vector<at::Tensor> mean_var_cuda(at::Tensor x) {
  CHECK_CUDA_INPUT(x);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  // Prepare output tensors
  auto mean = at::empty({chn}, x.options());
  auto var = at::empty({chn}, x.options());

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.type(), "mean_var_cuda", ([&] {
    mean_var_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x.data<scalar_t>(),
        mean.data<scalar_t>(),
        var.data<scalar_t>(),
        num, chn, sp);
  }));

  return {mean, var};
}

/**********
 * forward
 **********/

template<typename T>
__global__ void forward_kernel(T *x, const T *mean, const T *var, const T *weight, const T *bias,
                               bool affine, float eps, int num, int chn, int sp, 
                               int log_min_exp, int log_max_exp, float delta, float maxv) {
  int plane = blockIdx.x;

  float minv = -maxv;
  float clampv = minv;
  T _mean = mean[plane];
  T _var = var[plane];
  T _weight = affine ? abs(weight[plane]) + eps : T(1);
  T _bias = affine ? bias[plane] : T(0);
  T mul = rsqrt(_var + eps) * _weight;

  //log quantize mul
  T mul_exp = roundf(__log2f(abs(mul)));
  T mul_round = mul_exp < log_min_exp ? 0 :  __powf(2, fminf(mul_exp, log_max_exp));
  mul = (signbit(mul) ? -1 : 1) * mul_round;

  //linear quant _shift
  T _shift = -_mean*mul + _bias;
  _shift = (_shift < minv) ?
          clampv :
          fminf(fmaxf(delta*floor((_shift / delta) + 0.5), minv), maxv);

  for (int batch = 0; batch < num; ++batch) {
    for (int n = threadIdx.x; n < sp; n += blockDim.x) {
      T _x = x[(batch * chn + plane) * sp + n];
      T _y = _x * mul + _shift;
      x[(batch * chn + plane) * sp + n] = _y;
    }
  }
}

at::Tensor forward_cuda(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                        bool affine, float eps, int log_min_exp, int log_max_exp, float delta, float maxv) {
  CHECK_CUDA_INPUT(x);
  CHECK_CUDA_INPUT(mean);
  CHECK_CUDA_INPUT(var);
  CHECK_CUDA_INPUT(weight);
  CHECK_CUDA_INPUT(bias);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.type(), "forward_cuda", ([&] {
    forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x.data<scalar_t>(),
        mean.data<scalar_t>(),
        var.data<scalar_t>(),
        weight.data<scalar_t>(),
        bias.data<scalar_t>(),
        affine, eps, num, chn, sp,
        log_min_exp, log_max_exp, delta, maxv);
  }));

  return x;
}

/***********
 * edz_eydz
 ***********/

template<typename T>
__global__ void edz_eydz_kernel(const T *z, const T *dz, const T *weight, const T *bias,
                                T *edz, T *eydz, bool affine, float eps, int num, int chn, int sp) {
  int plane = blockIdx.x;

  T _weight = affine ? abs(weight[plane]) + eps : 1.f;
  T _bias = affine ? bias[plane] : 0.f;

  Pair<T> res = reduce<Pair<T>, GradOp<T>>(GradOp<T>(_weight, _bias, z, dz, chn, sp), plane, num, sp);
  __syncthreads();

  if (threadIdx.x == 0) {
    edz[plane] = res.v1;
    eydz[plane] = res.v2;
  }
}

std::vector<at::Tensor> edz_eydz_cuda(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                      bool affine, float eps) {
  CHECK_CUDA_INPUT(z);
  CHECK_CUDA_INPUT(dz);
  CHECK_CUDA_INPUT(weight);
  CHECK_CUDA_INPUT(bias);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(z, num, chn, sp);

  auto edz = at::empty({chn}, z.options());
  auto eydz = at::empty({chn}, z.options());

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(z.type(), "edz_eydz_cuda", ([&] {
    edz_eydz_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        z.data<scalar_t>(),
        dz.data<scalar_t>(),
        weight.data<scalar_t>(),
        bias.data<scalar_t>(),
        edz.data<scalar_t>(),
        eydz.data<scalar_t>(),
        affine, eps, num, chn, sp);
  }));

  return {edz, eydz};
}

/***********
 * backward
 ***********/

template<typename T>
__global__ void backward_kernel(const T *z, const T *dz, const T *var, const T *weight, const T *bias, const T *edz,
	                        const T *eydz, T *dx, bool affine, float eps, int num, int chn, int sp) {
  int plane = blockIdx.x;

  T _weight = affine ? abs(weight[plane]) + eps : 1.f;
  T _bias = affine ? bias[plane] : 0.f;
  T _var = var[plane];
  T _edz = edz[plane];
  T _eydz = eydz[plane];

  T _mul = _weight * rsqrt(_var + eps);
  T count = T(num * sp);

  for (int batch = 0; batch < num; ++batch) {
    for (int n = threadIdx.x; n < sp; n += blockDim.x) {
      T _dz = dz[(batch * chn + plane) * sp + n];
      T _y = (z[(batch * chn + plane) * sp + n] - _bias) / _weight;

      dx[(batch * chn + plane) * sp + n] = (_dz - _edz / count - _y * _eydz / count) * _mul;
    }
  }
}

at::Tensor backward_cuda(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                                      at::Tensor edz, at::Tensor eydz, bool affine, float eps) {
  CHECK_CUDA_INPUT(z);
  CHECK_CUDA_INPUT(dz);
  CHECK_CUDA_INPUT(var);
  CHECK_CUDA_INPUT(weight);
  CHECK_CUDA_INPUT(bias);
  CHECK_CUDA_INPUT(edz);
  CHECK_CUDA_INPUT(eydz);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(z, num, chn, sp);

  auto dx = at::zeros_like(z);

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(z.type(), "backward_cuda", ([&] {
    backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        z.data<scalar_t>(),
        dz.data<scalar_t>(),
        var.data<scalar_t>(),
        weight.data<scalar_t>(),
        bias.data<scalar_t>(),
        edz.data<scalar_t>(),
        eydz.data<scalar_t>(),
        dx.data<scalar_t>(),
        affine, eps, num, chn, sp);
  }));

  return dx;
}