import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.cpp_extension import load
from torch.distributions import categorical

from itertools import product
import util
from bn import QuantBN

quantize_cuda = load(
    'quantize_cuda', ['kernels/quantize_cuda.cpp', 'kernels/quantize_cuda_kernel.cu'], extra_cflags=['-O3'])
log_quantize_cuda = load(
    'log_quantize_cuda', ['kernels/log_quantize_cuda.cpp', 'kernels/log_quantize_cuda_kernel.cu'], extra_cflags=['-O3'])
shift_cuda = load(
    'shift_cuda', ['kernels/shift_cuda.cpp', 'kernels/shift_cuda_kernel.cu'], extra_cflags=['-O3'])


def _make_pair(x):
    if hasattr(x, '__len__'):
        return x
    else:
        return (x, x)

class log_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base, min_exp, max_exp, backward_clip=False, neg_clip=False):
        ctx.save_for_backward(x)
        ctx.base, ctx.max_exp, ctx.backward_clip = base, max_exp, backward_clip
        if neg_clip:
            x.clamp_(0)
        if x.is_cuda:
            return log_quantize_cuda.forward(x, base, min_exp, max_exp)
        else:
            return x.sign()*base**x.abs().log2().round().clamp(min_exp, max_exp)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.backward_clip:
            return grad_output, None, None, None, None, None
        x, = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if x.is_cuda:
            grad_output = log_quantize_cuda.backward(grad_output, x, ctx.base, ctx.max_exp)
        else:
            max_val = ctx.base**ctx.max_exp
            grad_output = grad_output * ((x > max_val) * (x < -max_val)).float()

        return grad_output, None, None, None, None, None

class LogQuantize(nn.Module):
    def __init__(self, base, min_exp, max_exp, backward_clip=False, neg_clip=False):
        super(LogQuantize, self).__init__()
        self.base = base
        self.min_exp = min_exp
        self.max_exp = max_exp
        self.backward_clip = backward_clip
        self.neg_clip = neg_clip

    def forward(self, x):
        return log_quantize.apply(x, self.base, self.min_exp, self.max_exp,
                                  self.backward_clip, self.neg_clip)

    def extra_repr(self):
        return 'base={base}, min_exp={min_exp}, max_exp={max_exp}'.format(**self.__dict__)


class quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, minv, maxv, clampv):
        ctx.save_for_backward(x)
        ctx.minv, ctx.maxv = minv, maxv
        if x.is_cuda:
            return quantize_cuda.forward(x, delta, minv, maxv, clampv)
        else:
            return x.div(delta).add_(0.5).floor_().mul_(delta).clamp_(minv, maxv)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if x.is_cuda:
            grad_output = quantize_cuda.backward(grad_output, x, ctx.minv, ctx.maxv)
        else:
            grad_output = grad_output * ((x > ctx.minv) * (x < ctx.maxv)).float()

        return grad_output, None, None, None, None


class Quantize(nn.Module):
    def __init__(self, delta, minv, maxv, clampv=None):
        super(Quantize, self).__init__()
        if clampv is None:
            clampv = minv
        self.delta = delta
        self.minv = minv
        self.maxv = maxv
        self.clampv = clampv
        self._data = [None]*8

    def forward(self, x):
        h = quantize.apply(x, self.delta, self.minv, self.maxv, self.clampv)
        self._data[h.device.index] = h

        return h
    
    @property
    def data(self):
        d = []
        for _d in self._data:
            if _d is not None:
                d.append(_d)
        
        return d

    def extra_repr(self):
        return 'delta={delta}, minv={minv}, maxv={maxv}'.format(**self.__dict__)


class shift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift):
        ctx.save_for_backward(shift)
        return shift_cuda.forward(x, shift)

    @staticmethod
    def backward(ctx, grad_output):
        shift, = ctx.saved_tensors
        grad_output = shift_cuda.backward(grad_output, shift)

        return grad_output, None


class Shift(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(Shift, self).__init__()
        self.channels = in_channels
        self.kernel_size = kernel_size
        if kernel_size == 3:
            p = torch.Tensor([0.3, 0.4, 0.3])
        elif kernel_size == 5:
            p = torch.Tensor([0.1, 0.25, 0.3, 0.25, 0.1])
        elif kernel_size == 7:
            p = torch.Tensor([0.075, 0.1, 0.175, 0.3, 0.175, 0.1, 0.075])
        elif kernel_size == 9:
            p = torch.Tensor([0.05, 0.075, 0.1, 0.175, 0.2, 0.175, 0.1, 0.075, 0.05])
        else:
            raise RuntimeError('Unsupported kernel size')

        shift_t = categorical.Categorical(p).sample((in_channels, 2)) - (kernel_size // 2)
        self.register_buffer('shift_t', shift_t.int())
    
    def forward(self, x):
        if x.is_cuda:
            return shift.apply(x, self.shift_t)
        else:
            print('Shift only supports GPU for now..')
            assert False

    def extra_repr(self):
        s = ('{channels}, kernel_size={kernel_size}')
        return s.format(**self.__dict__)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super(Conv2d, self).__init__()
        self.stride = _make_pair(stride)
        self.padding = _make_pair(padding)
        self.dilation = _make_pair(dilation)
        self.groups = groups
        self.bias = None
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        N = out_channels*in_channels*kernel_size*kernel_size
        n = kernel_size * kernel_size * out_channels
        self._weight = nn.Parameter(torch.Tensor(N))
        self._weight.data.normal_(0, math.sqrt(2. / n))

        # pytorch v0.4, seems to error if register_buffer is called on empty Tensor
        self.register_buffer('_quant_idxs', torch.Tensor([0]).long())
        self.register_buffer('_quant_values', torch.Tensor([0]))
        self.register_buffer('_mask', torch.ones(N))

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
                    
    @property
    def weight(self):
        if self._quant_idxs.size()[0] > 1:
            self._weight.data[self._quant_idxs] = self._quant_values
        w = self.mask*self._weight
        return w.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

    @property
    def mask(self):
        return Variable(self._mask, requires_grad=False)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)

class LogConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, base=2, num_levels=2, max_exp=-1):
        super(LogConv2d, self).__init__()
        self.stride = _make_pair(stride)
        self.padding = _make_pair(padding)
        self.bias = None
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.base = base
        self.max_exp = max_exp
        self.min_exp = max_exp - num_levels + 1
        N = out_channels*in_channels*kernel_size*kernel_size
        self._weight = nn.Parameter(torch.Tensor(N))
        n = kernel_size * kernel_size * out_channels
        self._weight.data.normal_(0, math.sqrt(2. / n))
        self.bias = None
        self.register_buffer('_mask', torch.ones(N))

    def forward(self, x):
        return F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)

    @property
    def weight(self):
        w = self.mask * self._weight
        w = log_quantize.apply(w, self.base, self.min_exp, self.max_exp)
        return w.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
    
    @property
    def masked_weight(self):
        w = self.mask * self._weight
        return w.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

    @property
    def mask(self):
        return Variable(self._mask, requires_grad=False)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        s += ', groups={groups}, max_exp={max_exp}, '
        s += 'min_exp={min_exp}'
        return s.format(**self.__dict__)


class CheckerboardReshape(nn.Module):
    def __init__(self, s):
        super(CheckerboardReshape, self).__init__()
        self.s = s
        self.idxs = list(range(s))
 
    def __call__(self, x):
        B, C, W, H = x.shape
        h = torch.stack([x[:, :, i::self.s, j::self.s] for i, j in product(self.idxs, self.idxs)], 1)
        h = h.reshape(B, -1, W // self.s, H // self.s)
        return h

def make_quant_layer(data_exp, data_bins, weight_levels, max_exp, bn, reshape_stride=1):
    # quantization values
    delta = 2**data_exp
    maxv = data_bins * delta
    max_exp = int(max_exp)
    min_exp = int(max_exp - weight_levels + 1)
    bn_min_exp = -8
    bn_max_exp = 8
    bn_delta = 2**(data_exp - 8)
    bn_maxv = 2**16 * bn_delta

    def quant_layer(in_channels, out_channels, stride, groups, layer_idx, num_layers):
        layer = []
        first = layer_idx == 0
        last = layer_idx == num_layers - 1

        if first:
            if reshape_stride != 1:
                layer.append(CheckerboardReshape(reshape_stride))
            layer.append(Quantize(delta, -maxv, maxv))
        elif last:
            layer.append(nn.AdaptiveAvgPool2d(1))
        else:
            layer.append(Shift(in_channels, 3))

        layer.append(LogConv2d(in_channels, out_channels, 1, stride, 0, groups=groups,
                               num_levels=weight_levels, max_exp=max_exp))

        if not last:
            if bn == 'float-bn':
                layer.append(nn.BatchNorm2d(out_channels))
            elif bn == 'quant-bn':
                layer.append(QuantBN(out_channels, log_min_exp=bn_min_exp, log_max_exp=bn_max_exp,
                                     delta=bn_delta, maxv=bn_maxv))
            layer.append(Quantize(delta, 0, maxv))

        layer = nn.Sequential(*layer)
    
        return layer

    return quant_layer


def make_float_layer(reshape_stride=1):
    def float_layer(in_channels, out_channels, stride, groups, layer_idx, num_layers):
        layer = []
        first = layer_idx == 0
        last = layer_idx == num_layers - 1

        if first and reshape_stride != 1:
            layer.append(CheckerboardReshape(reshape_stride))
        elif last:
            pass
        else:
            layer.append(Shift(in_channels, 3))
        
        layer.append(Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups))

        if not last:
            layer.append(nn.BatchNorm2d(out_channels))
            layer.append(nn.ReLU(inplace=True))
        else:
            layer.append(nn.AdaptiveAvgPool2d(1))

        layer = nn.Sequential(*layer)

        return layer
    
    return float_layer


class ShiftNet(nn.Module):
    def __init__(self, settings, layer, in_channels=3, n_class=1000, dropout=False):
        super(ShiftNet, self).__init__()
        input_channel = settings[0][0]
        layer_idx = 0
        num_layers = sum([n for c, n, s, g in settings]) + 2
        self.num_layers = num_layers
        layers = [layer(in_channels, input_channel, 1, 1, layer_idx, num_layers)]
        layer_idx += 1

        prev_groups = settings[0][3]
        for k, (c, n, s, g) in enumerate(settings):
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                groups = prev_groups if i == 0 else g
                layers.append(layer(input_channel, output_channel, stride,
                                    groups, layer_idx, num_layers))
                input_channel = output_channel
                layer_idx += 1
                prev_groups = g

        if dropout:
            layers.append(nn.Dropout(0.5))

        layers.append(layer(input_channel, n_class, 1, 1, layer_idx, num_layers))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(2).squeeze(2)

    def __getitem__(self, i):
        return self.model[i]

    def __len__(self):
        return self.num_layers
