'''
This code is a modifed version of https://github.com/mapillary/inplace_abn.git to
support quantization of batch normalization parameters during the forward pass.
We are very thankful to the authors of this codebase for their clear and clean
implementation -- making it easy to modify for our purposes.
'''

import torch
import torch.nn as nn
import torch.nn.functional as functional

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from bn_functions import quant_bn


class QuantBN(nn.Module):
    """Quantized Batch Normalization

    Log quantization for scale, Linear quantization for mean and bias
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 log_min_exp=-8, log_max_exp=8, delta=2**-4, maxv = 127*2**-4):
        """
        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        """
        super(QuantBN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.log_min_exp = log_min_exp
        self.log_max_exp = log_max_exp
        self.delta = delta
        self.maxv = maxv
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.uniform_(self.weight, 0, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                  self.training, self.momentum, self.eps)

    def forward(self, x):
        return quant_bn(x, self.weight, self.bias, self.running_mean, self.running_var,
                        self.training, self.momentum, self.eps, self.log_min_exp,
                        self.log_max_exp, self.delta, self.maxv)

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
              'log_min_exp={log_min_exp}, log_max_exp={log_max_exp}, delta={delta}, maxv={maxv})'
        return rep.format(name=self.__class__.__name__, **self.__dict__)