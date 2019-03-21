import argparse
import math
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import net
import datasets

BRAM_WIDTH = 8

def ib(x, bits=0):
    return bin(x)[2:].zfill(bits)

def add_newlines(s, bits):
    return ''.join([s[i:i+bits] + '\n' for i in range(0, len(s), bits)])

def shift_index(y, x):
    if x == -1 and y == 1:
        return 7
    if x == -1 and y == 0:
        return 3
    if x == -1 and y == -1:
        return 8
    if x == 0 and y == 1:
        return 2
    if x == 0 and y == 0:
        return 0
    if x == 0 and y == -1:
        return 1
    if x == 1 and y == 1:
        return 6
    if x == 1 and y == 0:
        return 4
    if x == 1 and y == -1:
        return 5
    
#convert to n bits
def bias_to_bin(tensor, delta, bits=32):
    tensor = tensor.data.cpu().numpy()
    tensor = (tensor / delta).flatten()
    bin_str = ''
    for x in tensor:
        bin_str += str(bin(np.uint32(x)))[2:].zfill(bits)

    return bin_str

def uniform_to_bin(tensor, delta, maxv, bits=8):
    tensor = tensor.data.cpu().numpy()
    signs = np.sign(tensor).flatten()
    signs[signs==-1] = 0
    signs = np.uint8(signs)
    tensor = np.uint8(np.abs(tensor) / delta).flatten()
    bin_str = ''
    for sign, x in zip(signs, tensor):
        bin_str += ib(sign) + ib(x, bits-1)

    return bin_str

def shift_offset_to_bin(tensor):
    tensor = tensor.data.cpu().numpy()
    bin_str = ''
    for x, y in tensor:
        bin_str += ib(shift_index(x, y), 4)
    
    return bin_str

def logweight_to_bin(tensor, mask, groups, max_weight_exp, weight_levels, mask_bits=3, weight_bits=4):
    tensor = tensor.data.cpu().numpy().flatten()
    signs = np.sign(tensor).flatten()
    signs[signs==-1] = 0
    signs = np.uint8(signs)
    mask = mask.data.cpu().numpy().astype(np.uint8).reshape(-1, groups)
    idxs = np.argmax(mask != 0, 1)
    nonzero_idxs = groups*np.arange(len(idxs))+idxs
    tensor = np.uint8(np.ceil(np.log2(np.abs(tensor) + 1e-10)) + np.abs(max_weight_exp - weight_levels))

    bin_str = ''
    for i, (sign, x) in enumerate(zip(signs, tensor)):
        if i in nonzero_idxs:
            if x > weight_levels:
                x = 0
            bin_str += ib(idxs[i//groups], mask_bits) + ib(sign) + ib(x, weight_bits)

    return bin_str

def weight_vals(tensor, mask, groups, max_weight_exp, weight_levels, mask_bits=4, weight_bits=3):
    tensor = tensor.data.cpu().numpy().flatten()
    signs = np.sign(tensor).flatten()
    signs[signs==-1] = 0
    signs = np.uint8(signs)
    mask = mask.data.cpu().numpy().astype(np.uint8).reshape(-1, groups)
    idxs = np.argmax(mask != 0, 1)
    nonzero_idxs = groups*np.arange(len(idxs))+idxs
    weights = []
    for i, (sign, x) in enumerate(zip(signs, tensor)):
        if i in nonzero_idxs:
            weights.append(x)

    return weights

def num_bram_shift(bram_idx, groups):
    if groups not in [1, 2, 4, 8]:
        raise RuntimeError('Column groups of {} not supported'.format(groups))

    if bram_idx % 2 == 0:
        if groups == 1:
            return 8
        elif groups == 2:
            return 16
        else:
            return 32
    else:
        if groups < 8:
            return 0
        else:
            return 32

def gen_tiles(layer_weight, layer_mask, layer_bias, layer_scale, layer_shift,
              layer_groups, min_exp, max_exp, array_width,
              array_height, bram_columns=4, permute_rows=True):
    bias_idx = 0
    n_bias = 4
    N, C, H, W = layer_weight.shape
    layer_columns =  math.ceil(C / layer_groups)
    weight_levels = max_exp - min_exp + 1
    if layer_columns > array_width:
        raise RuntimeError('Systolic Array does not support vertical tiling')
    
    if layer_scale is not None:
        layer_weight = layer_weight * layer_scale.view(-1, 1, 1, 1)

    num_tiles = math.ceil(N / array_height)
    layer_weight = layer_weight.view(N, C)
    layer_mask = layer_mask.view(N, C)
    layer_bias = layer_bias.view(N)
    shift_C, _ = layer_shift.shape
    layer_shift = layer_shift.view(-1, BRAM_WIDTH, layer_groups, 2)
    layer_shift = layer_shift.transpose(1, 2).contiguous().view(shift_C, 2)
    bin_str = ''

    max_N = max(array_height, N)
    max_C = max(array_width, C)
    t = torch.zeros(max_N, max_C)
    t[:N, :C] = layer_weight.data
    layer_weight = t

    t = torch.zeros(max_N, max_C)
    t[:N, :C] = layer_mask.data
    layer_mask = t

    t = torch.zeros(max_N)
    t[:N] = layer_bias.data
    layer_bias = t

    for tile in range(num_tiles):
        tsi = array_height*tile
        tei = tsi + array_height
        shift_idx = 0
        for j, column_idx in enumerate(range(0, array_width, bram_columns)):
            if column_idx >= layer_columns:
                # zero padding
                bin_str += '0'*8*4*(array_height + 8)
                continue

            n_shift = num_bram_shift(j, layer_groups)
            shift = layer_shift[shift_idx:shift_idx+n_shift]
            shift_bin = shift_offset_to_bin(shift)
            shift_bin = shift_bin.ljust(128, '0')
            shift_idx += n_shift

            bias = layer_bias[bias_idx:bias_idx+n_bias]
            bias_bin = bias_to_bin(bias, 2**-11, bits=32)
            bias_idx += n_bias

            csi = layer_groups*column_idx
            cei = csi + bram_columns*layer_groups
            weight = layer_weight[tsi:tei, csi:cei]
            weight = weight.flip(0)
            mask = layer_mask[tsi:tei, csi:cei]
            mask = mask.flip(0)
            weight_bin = logweight_to_bin(weight, mask, layer_groups, max_exp, weight_levels)
            bin_str += bias_bin + shift_bin + weight_bin

    return bin_str



parser = argparse.ArgumentParser(description='Generates intructions/data format for FPGA')
parser.add_argument('--load-path', help='path to trained model')
parser.add_argument('--output-folder', help='path to generate output')
parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
parser.add_argument('--dataset', default='mnist', help='dataset name')
parser.add_argument('--batch-size', type=int, default=1,
                    help='input batch size for training (default: 64)')
parser.add_argument('--input-size', type=int, help='spatial width/height of input')
parser.add_argument('--array-width', type=int, default=128, help='systolic array width')
parser.add_argument('--array-height', type=int, default=64, help='systolic array height')
args = parser.parse_args()
args.cuda = True

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# get test data
data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size,
                            args.cuda, '-', in_memory=True, input_size=args.input_size, val_only=True)
train_dataset, train_loader, test_dataset, test_loader = data
for data, _ in test_loader: break
data = Variable(data.cuda())

# load model
model = torch.load(args.load_path)
layer_idx = 0
num_layers = len(model)
systolic_width = systolic_height = 32

model.cuda()
final_output = model(data)
init_input = model[0][0](data)
model.cpu()

delta = model[0][0].delta
maxv = model[0][0].maxv

# input data
for i in range(layer_idx, layer_idx + num_layers):
    groups = model[i][1].groups

    if i == 0:
        input_data = init_input
    else:
        input_data = model[i-1][3].data[0].contiguous()
        B, C, W, H = input_data.shape
        input_data = input_data.view(B, -1, BRAM_WIDTH, groups, W*H)
        input_data = input_data.transpose(2, 3).contiguous().view(B, C, W, H)

    input_bin = uniform_to_bin(input_data, delta, maxv)
    input_bin = add_newlines(input_bin, 8)
    input_path = os.path.join(args.output_folder, 'input-{}.txt'.format(i))
    with open(input_path, 'w') as fp:
        fp.write(input_bin)

#shift data
model.cuda()
for i in range(layer_idx, layer_idx + num_layers):
    groups = model[i][1].groups

    if i == 0 or i == len(model) - 1:
        continue

    shift_data = model[i][0](model[i-1][3].data[0])
    B, C, W, H = shift_data.shape
    shift_data = shift_data.view(B, -1, BRAM_WIDTH, groups, W*H)
    shift_data = shift_data.transpose(2, 3).contiguous().view(B, C, W, H)
    shift_bin = uniform_to_bin(shift_data, delta, maxv)
    shift_bin = add_newlines(shift_bin, 8)
    input_path = os.path.join(args.output_folder, 'shift-{}.txt'.format(i))
    with open(input_path, 'w') as fp:
        fp.write(shift_bin)

model.cpu()

# output data
for i in range(layer_idx, layer_idx + num_layers):
    groups = model[i][1].groups

    if i == len(model) - 1:
        output_data = final_output.unsqueeze(2).unsqueeze(2)
    else:
        output_data = model[i][3].data[0]
        B, C, W, H = output_data.shape

    output_bin = uniform_to_bin(output_data, delta, maxv)
    output_bin = add_newlines(output_bin, 8)
    output_path = os.path.join(args.output_folder, 'output-{}.txt'.format(i))
    with open(output_path, 'w') as fp:
        fp.write(output_bin)

# stride=2 output data
model.cuda()
for i in range(layer_idx, layer_idx + num_layers):
    stride = model[i][1].stride[0]
    
    if stride == 1:
        continue

    layer = model[i]
    shift_data = model[i][0](model[i-1][3].data[0])
    B, C, W, H = shift_data.shape
    output_data = F.conv2d(shift_data, layer[1].weight, layer[2].shift, stride=1, padding=0)
    output_data = model[i][3](output_data)

    output_bin = uniform_to_bin(output_data, delta, maxv)
    output_bin = add_newlines(output_bin, 8)
    output_path = os.path.join(args.output_folder, 'output-s=1-{}.txt'.format(i))
    with open(output_path, 'w') as fp:
        fp.write(output_bin)
model.cpu()



# weight data
for i in range(layer_idx, layer_idx + num_layers):
    layer = model[i]
    if i == 0:
        if hasattr(layer[2], 'scale'):
            scale = layer[2].scale
        else:
            scale = None
        weight_bin = gen_tiles(layer[1].weight, layer[1].mask, layer[2].shift, scale,
                               torch.zeros(64, 2), layer[1].groups,
                               layer[1].min_exp, layer[1].max_exp, args.array_width,
                               args.array_height)
    elif i == len(model) - 1:
        N, C, _, _ = layer[1].weight.shape
        weight_bin = gen_tiles(layer[1].weight, layer[1].mask, torch.ones(N), None,
                               torch.zeros(C, 2), layer[1].groups,
                               layer[1].min_exp, layer[1].max_exp, args.array_width,
                               args.array_height, permute_rows=False)
    else:
        if hasattr(layer[2], 'scale'):
            scale = layer[2].scale
        else:
            scale = None
        weight_bin = gen_tiles(layer[1].weight, layer[1].mask, layer[2].shift, scale,
                               layer[0].shift_t, layer[1].groups, 
                               layer[1].min_exp, layer[1].max_exp, args.array_width,
                               args.array_height)
    weight_bin = add_newlines(weight_bin, 32)

    weight_path = os.path.join(args.output_folder, 'weight-{}.txt'.format(i))
    with open(weight_path, 'w') as fp:
        fp.write(weight_bin)