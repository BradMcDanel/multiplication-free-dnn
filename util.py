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

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, net.LogConv2d) or isinstance(m, net.Conv2d):
            group_decay.append(m._weight)
        elif isinstance(m, nn.BatchNorm2d):
            if m.bias is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, net.BatchNorm2d):
            group_no_decay.append(m._shift)
        elif isinstance(m, net.Bias):
            group_no_decay.append(m._shift)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.dataset in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate**2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            if epoch < 30:
                lr = args.lr
            elif epoch < 60:
                lr = 0.1 * args.lr
            else:
                lr = 0.01 * args.lr
            #lr = args.lr * (0.1 ** (epoch // 30))
    elif method == 'many-multistep':
        lr = args.lr * (0.97 ** (epoch // 3))
    else:
        assert False
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



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
    model = net.ShiftMobile(settings, layer=layer, in_channels=3*(args.reshape_stride**2),
                            n_class=args.n_class, dropout=args.dropout)
    
    if args.input_size != 224 and args.dataset == 'imagenet':
        model = nn.Sequential([net.Interpolate(args.input_size), model])
    
    return model

def convert_fp_to_quant_model(model, args, init=True):
    if init:
        model = model.model

    # delta = 2**data_exp
    # maxv = data_bins * delta
    # max_exp = int(max_exp)
    # min_exp = int(max_exp - weight_levels + 1)
    # bn_min_exp = -8
    # bn_max_exp = 8
    # bn_delta = 2**(data_exp - 8)
    # bn_maxv = 2**16 * bn_delta

    if type(model) == nn.Sequential:
        for i in range(len(model)):
            if isinstance(model[i], net.Conv2d):
                model[i] = convert_quant_conv(model[i], args)
            elif isinstance(model[i], nn.BatchNorm2d):
                model[i] = convert_quant_bn(model[i], args)
            elif isinstance(model[i], nn.ReLU) or isinstance(model[i], nn.ReLU6):
                model[i] = net.Quantize(2**args.data_exp, 0, args.data_bins*2**args.data_exp, 0)
            else:
                convert_fp_to_quant_model(model[i], args, False)

def convert_quant_conv(layer, args):
    qlayer = net.LogConv2d(in_channels=layer.in_channels, out_channels=layer.out_channels,
                           kernel_size=layer.kernel_size, stride=layer.stride,
                           padding=layer.padding, groups=layer.groups, num_levels=args.weight_levels,
                           max_exp=args.max_weight_exp)
    qlayer._weight.data = layer._weight.data.clone()
    qlayer._mask.data = layer._mask.data.clone()
    return qlayer

def convert_quant_bn(layer, args):
    min_exp = int(args.max_weight_exp - args.weight_levels + 1)
    max_exp = int(args.max_weight_exp)
    qlayer = bn.QuantBN(layer.num_features, eps=layer.eps, momentum=layer.momentum,
                        affine=layer.affine, log_min_exp=min_exp, log_max_exp=max_exp,
                        delta=2**args.data_exp, maxv=args.data_bins*2**args.data_exp)
    return qlayer

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args, batch=i,
                             nBatch=len(train_loader), method=args.lr_type)

        if args.cuda is not None:
            x = x.cuda()
        target = target.cuda()

        # compute output
        output = data_parallel(model, x)
        loss = criterion(output, target)

        if isinstance(loss, tuple):
            loss, _ = loss

        if isinstance(output, tuple):
            output, _ = output

        # if args.l1_penalty > 0:
        #     loss += args.l1_penalty*l1_weight_total(model)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # record stats in model for visualization
        model.model.stats['train_loss'].append(loss.item())
        # model.stats['train_loss'].append(loss.item())

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            print('Train:: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Data: {data_time.avg:.3f}\t'
                  'Loss: {loss.avg:.4f}\t'
                  'Acc@1: {top1.avg:.3f}\t'
                  'Acc@5: {top5.avg:.3f}'.format(
                   epoch, i, len(train_loader) - 1, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg


def validate(val_loader, model, criterion, epoch, args, no_print=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x, target) in enumerate(val_loader):
            if args.cuda is not None:
                x = x.cuda()
            target = target.cuda()

            # compute output
            output = data_parallel(model, x)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # record stats in model for visualization

        if not no_print:
            print('Test :: [{0}][{1}/{2}]\t'
                'Loss {loss.avg:.4f}\t'
                'Acc@1 {top1.avg:.3f}\t'
                'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(val_loader) - 1,
                loss=losses, top1=top1, top5=top5))

    model.stats['test_loss'].append(losses.avg)
    model.stats['test_acc'].append(top1.avg)
    return losses.avg, top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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

def prune_group(model, prune_progress):
    model.cpu()
    layers = get_quantconv_layers(model)
    for layer_idx, layer in enumerate(layers):
        prune_pct = prune_progress * (1 - (1 / layer.groups))
        weight = layer._mask * layer._weight
        weight = weight.data.abs().view(-1, layer.groups)

        # at least one entry per prune group must survive
        max_w = weight.max(1)[1]
        max_w += layer.groups*torch.arange(len(max_w))
        weight = weight.view(-1)
        weight[max_w] += 1e8

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


def l1_weight_total(model):
    l1_total = 0
    for layer in get_conv_layers(model):
        l1_total += layer._weight.norm(1)
    return l1_total


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

def get_conv_layers(model):
    layers = []
    for layer in model.children():
        if isinstance(layer, net.Conv2d) or isinstance(layer, net.LogConv2d):
            layers.append(layer)
        else:
            layers.extend(get_conv_layers(layer))

    return layers

def track_running_stats(model, active):
    for bn in get_batchnorm_layers(model):
        bn.track_running_stats = active

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
