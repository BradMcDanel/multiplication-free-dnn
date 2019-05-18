from __future__ import print_function

import argparse
import os
from pprint import pprint

import numpy as np
import torch
import math
import torch.backends.cudnn as cudnn
cudnn.benchmark = True 
import torch.optim as optim
import torch.nn as nn

import datasets
import util
import packing
import net

from resnet50 import ResNet50
from model_refinery_wrapper import ModelRefineryWrapper
import refinery_loss


def train_model(wrapped_model, model, model_path, train_loader, test_loader, init_lr, epochs, args):
    # train_loss_f = net.LabelSmoothing(0.1)
    # train_loss_f = nn.CrossEntropyLoss()
    train_loss_f = refinery_loss.RefineryLoss()
    val_loss_f = nn.CrossEntropyLoss()
    best_model_path = '.'.join(model_path.split('.')[:-1]) + '.best.pth'

    for bn in util.get_batchnorm_layers(model):
        bn.weight.requires_grad = False
        bn.weight.data.zero_().add_(1)

    # tracking stats
    if not hasattr(model, 'stats'):
        model.stats = {'train_loss': [], 'test_acc': [], 'test_loss': [],
                       'weight': [], 'lr': [], 'macs': [], 'efficiency': []}

    curr_weights, _ = util.num_nonzeros(model)
    if hasattr(model, 'packed_layer_size'):
        macs = np.sum([x*y for x, y in model.packed_layer_size])
    else:
        macs = curr_weights

    # compute pruning scheduele
    '''
    prune_epoch = 0
    max_p = 1.0
    prune_epochs = int(0.5*epochs)
    prune_rates = [max_p * (1 - (1 - (i / prune_epochs))**3) for i in range(prune_epochs)]
    prune_rates[-1] = max_p
    prune_epochs = np.arange(1, prune_epochs + 1)
    '''

    prune_epoch = 0
    max_prune_rate = 0.8
    final_prune_epoch = int(0.5*args.epochs)
    num_prune_epochs = 10
    prune_rates = [max_prune_rate*(1 - (1 - (i / num_prune_epochs))**3)
                   for i in range(num_prune_epochs)]
    prune_rates[-1] = max_prune_rate
    prune_epochs = np.linspace(0, final_prune_epoch, num_prune_epochs).astype('i').tolist()


    print("Pruning Epochs: {}".format(prune_epochs))
    print("Pruning Rates: {}".format(prune_rates))

    # optimizer
    optimizer = optim.SGD(util.group_weight(model), lr=init_lr, momentum=0.9,
                          nesterov=True, weight_decay=4e-5)
    print("Optimizer:")
    print(optimizer)

    best_acc = 0

    # pruning stage
    for epoch in range(1, epochs + 1):
        print('[Epoch {}]'.format(epoch))
        for g in optimizer.param_groups:     
            lr = g['lr']                    
            break        

        if epoch in prune_epochs:
            util.prune(model, prune_rates[prune_epoch])
            packing.pack_model(model, args.gamma)
            macs = np.sum([x*y for x, y in model.packed_layer_size])
            curr_weights, num_weights = util.num_nonzeros(model)
            prune_epoch += 1

        train_loss = util.train(train_loader, wrapped_model, train_loss_f, optimizer, epoch-1, args)
        test_loss, test_acc = util.validate(test_loader, model, val_loss_f, epoch-1, args)

        '''
        if epoch in prune_epochs:
            util.prune_group(model, prune_rates[prune_epoch])
            curr_weights, _ = util.num_nonzeros(model)
            prune_epoch += 1
        '''

        print('LR        :: {}'.format(lr))
        print('Train Loss:: {}'.format(train_loss))
        print('Test  Loss:: {}'.format(test_loss))
        print('Test  Acc.:: {}'.format(test_acc))
        print('Nonzeros  :: {}'.format(curr_weights))
        print('')
        print('')
        model.stats['lr'].append(lr)
        model.optimizer = optimizer.state_dict()

        model.cpu()
        torch.save(model, model_path)
        if test_acc > best_acc and epoch > prune_epochs[-1]:
            print('New best model found')
            torch.save(model, best_model_path)
            best_acc = test_acc

        model.cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic Training Script')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--filters', nargs='+', type=int, help='size of layers in each block')
    parser.add_argument('--layers', nargs='+',  type=int, help='number of layers for each block')
    parser.add_argument('--strides', nargs='+', type=int, help='stride for each block')
    parser.add_argument('--groups', nargs='+', type=int, help='number of sparse groups')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr-type', default='cosine', type=str, metavar='T',
                        help='learning rate strategy (default: cosine)',
                        choices=['cosine', 'multistep'])
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--print-freq', default=100, type=int, help='printing frequency')
    parser.add_argument('--aug', default='+', help='data augmentation level (`-`, `+`)')
    parser.add_argument('--data-exp', type=int, default=-4, help='fixed point exp')
    parser.add_argument('--gamma', type=float, default=1.75,
                        help='column combine gamma parameter (default: 1.75)')
    parser.add_argument('--data-bins', type=int, default=127,
                        help='max activation bin (default: 127)')
    parser.add_argument('--weight-levels', type=int, default=4,
                        help='number of weight levels')
    parser.add_argument('--max-weight-exp', type=float, default=1,
                        help='maximum weight exponent')
    parser.add_argument('--load-path', default=None,
                        help='path to load model - trains new model if None')
    parser.add_argument('--teacher-path', default=None,
                        help='path to teacher model')
    parser.add_argument('--in-memory', action='store_true',
                        help='ImageNet Dataloader setting (store in memory)')
    parser.add_argument('--input-size', type=int, help='spatial width/height of input')
    parser.add_argument('--reshape-stride', type=int, default=1, help='checkerboard reshape stride')
    parser.add_argument('--n-class', type=int, help='number of classes')
    parser.add_argument('--layer-type', default='float', choices=['float', 'quant'],
                        help='type of layer')
    parser.add_argument('--bn-type', default='float-bn',
                        choices=['float-bn', 'quant-bn'],
                        help='type of layer')
    parser.add_argument('--save-path', required=True, help='path to save model')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('Arguments:')
    pprint(args.__dict__, width=1)

    #set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load dataset
    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size,
                                args.cuda, args.aug, in_memory=args.in_memory,
                                input_size=args.input_size)
    train_dataset, train_loader, test_dataset, test_loader = data

    # load or create model
    if args.load_path == None:
        model = util.build_model(args)
    else:
        model = torch.load(args.load_path)

    # teacher model
    teacher = ResNet50()
    teacher.load_state_dict(torch.load(args.teacher_path))
    teacher.cuda()
    wrapped_model = ModelRefineryWrapper(model, teacher)

    if args.cuda:
        model = model.cuda()

    print(model)
    print(util.num_nonzeros(model))
    print('Target Nonzeros:', util.target_nonzeros(model))

    train_model(wrapped_model, model, args.save_path, train_loader, test_loader, args.lr, args.epochs, args)
    # train_model(model, model, args.save_path, train_loader, test_loader, args.lr, args.epochs, args)
