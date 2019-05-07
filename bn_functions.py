'''
This code is a modifed version of https://github.com/mapillary/inplace_abn.git to
support quantization of batch normalization parameters during the forward pass.
We are very thankful to the authors of this codebase for their clear and clean
implementation -- making it easy to modify for our purposes.
'''

from os import path
import torch
import torch.distributed as dist
import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load

_src_path = path.join(path.dirname(path.abspath(__file__)), "kernels")
_backend = load(name="quant_bn",
                extra_cflags=["-O3"],
                sources=[path.join(_src_path, f) for f in [
                    "quant_bn.cpp",
                    "quant_bn_cpu.cpp",
                    "quant_bn_cuda.cu",
                    "quant_bn_cuda_half.cu",
                ]],
                extra_cuda_cflags=["--expt-extended-lambda"])

def _check(fn, *args, **kwargs):
    success = fn(*args, **kwargs)
    if not success:
        raise RuntimeError("CUDA Error encountered in {}".format(fn))


def _broadcast_shape(x):
    out_size = []
    for i, s in enumerate(x.size()):
        if i != 1:
            out_size.append(1)
        else:
            out_size.append(s)
    return out_size


def _reduce(x):
    if len(x.size()) == 2:
        return x.sum(dim=0)
    else:
        n, c = x.size()[0:2]
        return x.contiguous().view((n, c, -1)).sum(2).sum(0)


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count

class QuantBN(autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, log_min_exp=-8,
                log_max_exp=8, delta=2**-4, maxv = 127*2**-4):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.affine = weight is not None and bias is not None

        # Prepare inputs
        count = _count_samples(x)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0, dtype=torch.float32)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0, dtype=torch.float32)

        if ctx.training:
            mean, var = _backend.mean_var(x)

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * count / (count - 1))

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)

        # BN forward 
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps, log_min_exp, log_max_exp,
                         delta, maxv)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()

        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
        else:
            # TODO: implement simplified CUDA backward for inference mode
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))

        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = eydz if ctx.affine else None
        if dweight is not None:
            dweight[weight < 0] *= -1
        dbias = edz if ctx.affine else None

        return dx, dweight, dbias, None, None, None, None, None, None, None, None, None


quant_bn = QuantBN.apply

__all__ = ["quant_bn"]
