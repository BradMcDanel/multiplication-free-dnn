from __future__ import print_function
from tqdm import tqdm

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
import torch.nn.functional as F
from torch.autograd import Variable
import net
import time
import gc
import os


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def build_model(args):
    settings = list(zip(args.filters, args.layers, args.strides, args.groups))
    if args.layer_type == 'float':
        layer = net.make_float_layer(args.reshape_stride)
    elif args.layer_type == 'quant':
        layer = net.make_quant_layer(args.data_exp, args.data_bins, args.weight_levels,
                                     args.max_weight_exp, bn=args.bn_type,
                                     reshape_stride=args.reshape_stride)
    model = net.ShiftNet(settings, layer=layer, in_channels=3*(args.reshape_stride**2),
                         n_class=args.n_class, dropout=False)
    
    if args.input_size != 224 and args.dataset == 'imagenet':
        model = nn.Sequential([net.Interpolate(args.input_size), model])
    
    return model


def train(model, train_loader, optimizer, epoch, loss):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    model.train()
    loss.train()
    end = time.time()
    model_loss = 0
    pbar = tqdm(train_loader, leave=False)
    for i, (data, target) in enumerate(pbar):
        batchsize = len(target)
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        prediction = data_parallel(model, data)
        loss_output = loss(prediction, target)

        if isinstance(loss_output, tuple):
            loss_value, outputs = loss_output
        else:
            loss_value = loss_output
        loss_value.backward()

        model_loss += batchsize*loss_value.item()
        loss_meter.update(loss_value.item(), batchsize)
        optimizer.step()

        pbar.set_postfix(loss='{loss_meter.avg:.4f}, comp={batch_time.avg:.4f}, data={data_time.avg:.4f}'.format(
                         loss_meter=loss_meter, batch_time=batch_time, data_time=data_time))

        batch_time.update(time.time() - end)
        end = time.time()

    optimizer.zero_grad()

    N = train_loader.num_samples
    model_loss = model_loss / N
    return model_loss

def test(model, test_loader, epoch, loss):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()
    loss.eval()
    end = time.time()
    num_correct, model_loss = 0, 0
    for i, (data, target) in enumerate(test_loader):
        batchsize = len(target)
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            prediction = data_parallel(model, data)
            loss_output = loss(prediction, target)

            if isinstance(loss_output, tuple):
                loss_value, outputs = loss_output
            else:
                loss_value = loss_output

            pred = prediction.data.max(1, keepdim=True)[1]
            correct = (pred.view(-1) == target.view(-1)).long().sum().item()
            num_correct += correct
            model_loss += batchsize * loss_value.sum()

        batch_time.update(time.time() - end)
        end = time.time()

    N = test_loader.num_samples
    model_loss = model_loss / N
    acc = 100. * (num_correct / N)

    return model_loss.item(), acc

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def num_nonzeros(model, top=True):
    '''
    Only considers conv layers for now
    '''
    if model == None:
        return 0

    non_zeros, total = 0, 0
    for layer in model.children():
        if isinstance(layer, net.LogConv2d) or isinstance(layer, net.Conv2d):
            flat_w = layer.mask.data.cpu().numpy().flatten()
            non_zeros += np.sum(flat_w != 0)
            total += len(flat_w)
        elif isinstance(layer, nn.Conv2d):
            if not layer.weight.requires_grad:
                continue
            B, C, W, H = layer.weight.shape
            total_W = B*C*W*H
            non_zeros += total_W
            total += total_W
        else:
            n, t = num_nonzeros(layer, False)
            non_zeros += n
            total += t
    
    return int(non_zeros), int(total)

def prune(model, prune_progress):
    model.cpu()
    layers = get_quantconv_layers(model)
    for layer_idx, layer in enumerate(layers):
        prune_pct = prune_progress * (1 - (1 / layer.groups))
        weight = layer._weight.data.abs().view(-1)
        num_weights = len(weight)
        num_prune = math.ceil(num_weights * prune_pct)
        prune_idxs = weight.sort()[1][:num_prune]
        mask = torch.ones(num_weights)
        mask[prune_idxs] = 0
        layer._weight.data[prune_idxs] = 0
        layer._mask = mask
    
    model.cuda()


def target_nonzeros(model):
    layers = get_quantconv_layers(model)
    total_weights = 0
    for layer_idx, layer in enumerate(layers):
        num_weights = len(layer._weight)
        total_weights +=  (1 / layer.groups) * num_weights
    
    return total_weights



def weight_quant(model, quant_rate, max_weight, levels):
    base = 2
    model.cpu()
    layers = get_quantconv_layers(model)
    #max_log = math.floor(math.log(max_weight, base))
    max_log = -1
    level_values = [base**(max_log - i) for i in range(levels)]
    level_values += [-l for l in level_values]
    level_values += [0.0]
    level_values = torch.Tensor(level_values)
    for layer in layers:
        weight = layer.weight.data.view(-1)
        num_nonzeros = (layer._mask == 1).long().sum().item()
        num_zeros = (layer._mask == 0).long().sum().item()
        num_weights = len(weight)
        num_quant = num_zeros + math.ceil(num_nonzeros * quant_rate)
        layer._quant_idxs = weight.abs().sort()[1][-num_quant:]
        wq = weight[layer._quant_idxs]
        assign_idxs = (wq.unsqueeze(0) - level_values.unsqueeze(1)).abs().min(dim=0)[1]
        layer._quant_values = level_values[assign_idxs]
    model.cuda()

def find_quantize(x, bins, sf_const):
    delta = select_delta(x, bins, sf_const)
    level_values = gen_levels(delta, bins)
    assign_idxs = (x.unsqueeze(0) - level_values.unsqueeze(1)).abs().min(dim=0)[1]
    return level_values[assign_idxs]

def gen_levels(delta, bins):
    level_values = (np.arange(bins) * delta).tolist()
    level_values += [-l for l in level_values] + [0]
    level_values = torch.Tensor(level_values)
    return level_values

def select_delta(w, bins, sf_const):
    deltas = [sf_const * 2**i for i in range(-14, 3)]
    min_loss = 1e10
    min_delta = 0
    for delta in deltas:
        level_values = gen_levels(delta, bins)
        loss = (w.unsqueeze(0) - level_values.unsqueeze(1)).abs().min(dim=0)[0].sum()
        if loss < min_loss:
            min_loss = loss
            min_delta = delta
    return min_delta


def batchnorm_quant(model, bins):
    model.cpu()
    layers = get_batchnorm_layers(model)
    for layer in layers:
        delta = select_delta(layer.running_mean, bins)
        level_values = gen_levels(delta, bins)
        assign_idxs = (layer.running_mean.unsqueeze(0) - level_values.unsqueeze(1)).abs().min(dim=0)[1]
        layer.running_mean = level_values[assign_idxs]

        delta = select_delta(layer.running_var, bins)
        level_values = gen_levels(delta, bins)
        assign_idxs = (layer.running_var.unsqueeze(0) - level_values.unsqueeze(1)).abs().min(dim=0)[1]
        layer.running_var = level_values[assign_idxs]

        if layer.affine:
            delta = select_delta(layer.bias, bins)
            level_values = gen_levels(delta, bins)
            assign_idxs = (layer.bias.unsqueeze(0) - level_values.unsqueeze(1)).abs().min(dim=0)[1]
            layer.bias.data = level_values[assign_idxs]

            delta = select_delta(layer.weight, bins)
            level_values = gen_levels(delta, bins)
            assign_idxs = (layer.weight.unsqueeze(0) - level_values.unsqueeze(1)).abs().min(dim=0)[1]
            layer.weight.data = level_values[assign_idxs]


    model.cuda()


def get_max_weight(model):
    max_w = -1e10
    for layer in get_quantconv_layers(model):
        max_w = max(max_w, layer.weight.abs().max().item())
    return max_w

def get_weights(model, float_weight=False):
    weights = []
    layers = get_quantconv_layers(model)
    for layer in layers:
        if float_weight:
            weights.extend(layer._weight.view(-1).data.cpu().tolist())
        else:
            weights.extend(layer.weight.view(-1).data.cpu().tolist())
    return np.array(weights)

def get_nonzero_layer_size(model):
    sizes = []
    layers = get_quantconv_layers(model)
    for i, layer in enumerate(layers):
        w = layer.weight.data.cpu().numpy()
        B, C, W, H = w.shape
        w = w.reshape(B, C*W*H)
        r, c = (w.sum(1) != 0).sum(), (w.sum(0) != 0).sum()
        if i > 0:
            c = min(c, sizes[i-1][0])
            sizes[i-1][0] = c
        sizes.append([r, c])
    
    return sizes

def get_nonzero_layer_ratio(model):
    ratios = []
    layers = get_quantconv_layers(model)
    for layer in layers:
        w = layer.weight.data.cpu().numpy().flatten()
        ratios.append((w != 0).sum() / float(len(w)))
    return ratios

def get_quantconv_layers(model):
    layers = []
    for layer in model.children():
        if isinstance(layer, net.LogConv2d) or isinstance(layer, net.Conv2d):
            layers.append(layer)
        else:
            layers.extend(get_quantconv_layers(layer))

    return layers

def set_batchnorm_alpha(model, alpha):
    for layer in get_batchnorm_layers(model):
        layer.alpha = alpha

def get_batchnorm_layers(model):
    layers = []
    for layer in model.children():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, net.BatchNorm2d):
            layers.append(layer)
        else:
            layers.extend(get_batchnorm_layers(layer))

    return layers

def get_shift_layers(model):
    layers = []
    for layer in model.children():
        if isinstance(layer, net.Shift):
            layers.append(layer)
        else:
            layers.extend(get_shift_layers(layer))

    return layers

def get_quant_layers(model):
    layers = []
    for layer in model.children():
        if isinstance(layer, net.Quantize):
            layers.append(layer)
        else:
            layers.extend(get_quant_layers(layer))

    return layers


def data_sparsity_loss(model):
    loss = 0
    for layer in get_quant_layers(model):
        for d in layer.data:
            loss += (d.norm(1).cuda(0) / d.shape[0])
        # loss += 1e-3*(1 / (channel_norm + 1e-9)).norm(2)
        # loss += ((channel_norm - channel_norm.mean()).abs().sum())
    return loss

def clear_quant_data(model):
    for layer in get_quant_layers(model):
        for i in range(len(layer._data)):
            layer._data[i] = None
