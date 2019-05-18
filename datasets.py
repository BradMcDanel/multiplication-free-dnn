import os
import pickle
from itertools import product
import warnings

import msgpack
import cv2
from torchvision import datasets, transforms 
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import io

import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

def imagenet_msgpack_loader(path, num_samples=1e10):
    samples = []
    f = open(path, "rb")
    for i, sample in enumerate(msgpack.Unpacker(f, use_list=False, raw=True)):
        if i == num_samples:
            break
        x, label = sample
        x = pickle.loads(x)
        samples.append((x, label))
        if i == num_samples - 1:
            break
    f.close()
    return samples

class ExternalInputIterator(object):
    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
        shuffle(self.samples)

    def __iter__(self):
        self.i = 0
        self.n = len(self.samples)
        return self

    def __next__(self):
        batch = []
        labels = []
        for b in range(self.batch_size):
            if b == self.n:
                self.i = 0
                break
            x, label = self.samples[self.i]
            #batch.append(np.frombuffer(pickle.loads(x), dtype=np.uint8))
            batch.append(np.frombuffer(x, dtype=np.uint8))
            labels.append(np.array([label], dtype=np.uint8))
            self.i += 1
        return (batch, labels)

    next = __next__

class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, iterator):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.iterator = iter(iterator)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.cast = ops.Cast(device = "gpu",
                             dtype = types.INT32)
        self.resize = ops.Resize(device = "gpu",
                                 image_type = types.RGB,
                                 interp_type = types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            crop = (227, 227),
                                            image_type = types.RGB,
                                            mean = [128., 128., 128.],
                                            std = [1., 1., 1.])
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.resize_rng = ops.Uniform(range = (256, 480))

    def define_graph(self):
        jpegs = self.input()
        labels = self.input_label()
        images = self.decode(jpegs)
        images = self.resize(images, resize_shorter = self.resize_rng())
        output = self.cmn(images, crop_pos_x = self.uniform(),
                          crop_pos_y = self.uniform())
        return (output, labels)

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, crop, iterator, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.iterator = iterator
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        # self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
                                                    random_aspect_ratio=[0.8, 1.25],
                                                    random_area=[0.1, 1.0],
                                                    num_attempts=100)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512,
                                                      random_aspect_ratio=[0.8, 1.25],
                                                      random_area=[0.1, 1.0],
                                                      num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs = self.input()
        self.labels = self.input_label()
        # self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, crop, size, iterator):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.iterator = iterator
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
 

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)

def get_imagenet_loaders(batch_size, num_workers=2, debug=False):
    train_path = '/home/jovyan/harvard-heavy/datasets/imagenet-msgpack/ILSVRC-train.bin'
    val_path = '/home/jovyan/harvard-heavy/datasets/imagenet-msgpack/ILSVRC-val.bin'

    if debug:
        train_samples = imagenet_msgpack_loader(train_path, 100000)
    else:
        train_samples = imagenet_msgpack_loader(train_path)

    val_samples = imagenet_msgpack_loader(val_path)


    train_iterator = ExternalInputIterator(train_samples, batch_size)
    train_iterator = iter(train_iterator)
    pipes = [HybridTrainPipe(batch_size, num_workers, i, 224, train_iterator) for i in range(8)]
    for pipe in pipes:
        pipe.build()
    train_loader = DALIClassificationIterator(pipes, size=len(train_samples)) 

    val_iterator = ExternalInputIterator(val_samples, batch_size)
    val_iterator = iter(val_iterator)
    pipes = [HybridValPipe(batch_size, num_workers, i, 224, 256, val_iterator) for i in range(8)]
    for pipe in pipes:
        pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=len(val_samples)) 

    return train_samples, train_loader, val_samples, val_loader, pipes

class InMemoryImageNet(Dataset):
    def __init__(self, path, num_samples, transforms):
        self.path = path
        self.num_samples = num_samples
        self.transforms = transforms
        self.samples = []
        f = open(self.path, "rb")
        for i, sample in enumerate(msgpack.Unpacker(f, use_list=False, raw=True)):
            # self.samples.append(sample)
            x, label = sample
            x = pickle.loads(x)
            self.samples.append((x, label))
            if i == self.num_samples - 1:
                break
        f.close()
        
    def __getitem__(self, index):
        x, y = self.samples[index]
        x = self.transforms(x)
        return (x, y)

    def __len__(self):
        return self.num_samples

class Fill(object):
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        img = np.array(img)
        red, green, blue = img.T
        areas = (red == 0) & (blue == 0) & (green == 0)
        img[areas.T] = (self.fill, self.fill, self.fill)
        img = Image.fromarray(img)
        return img

def get_dataset(dataset_root, dataset, batch_size, is_cuda=True, aug='+', sample_ratio=1,
                in_memory=False, val_only=False, input_size=224):
    if dataset == 'cifar10':
        train, train_loader, test, test_loader = get_cifar10(dataset_root, batch_size, is_cuda, aug)
                                                             
    elif dataset == 'imagenet':
        train, train_loader, test, test_loader = get_imagenet(dataset_root, batch_size,
                                                              is_cuda, in_memory=in_memory,
                                                              val_only=val_only,
                                                              input_size=input_size)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))

    return train, train_loader, test, test_loader


def get_cifar10(dataset_root, batch_size, is_cuda=True, aug='+'):
    kwargs = {'num_workers': 16, 'pin_memory': True} if is_cuda else {}
    stds = (0.247, 0.243, 0.261)

    if aug == '-':
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), stds),
        ] 
    elif aug == '+':
        transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), stds),
        ]
    else:
        raise ValueError('Invalid Augmentation setting `{}` not found'.format(aug))

    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=True, download=True,
                             transform=transforms.Compose(transform))
    test = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=False, download=True, 
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), stds),
                            ]))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                                shuffle=True, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, drop_last=False, **kwargs)
    train_loader.num_samples = len(train)
    test_loader.num_samples = len(test)
    return train, train_loader, test, test_loader

def get_imagenet(dataset_root, batch_size, is_cuda=True, num_workers=16,
                 cache_mul=16, in_memory=False, val_only=False, input_size=224):
    train_path = os.path.join(dataset_root, 'imagenet-msgpack', 'ILSVRC-train.bin')
    val_path = os.path.join(dataset_root, 'imagenet-msgpack', 'ILSVRC-val.bin')
    kwargs = {'num_workers': 32, 'pin_memory': True} if is_cuda else {}
    num_train = 1281167
    num_val = 50000
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    def msgpack_load(x):
        # x = pickle.loads(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = Image.open(io.BytesIO(x)).convert('RGB')
        # x[:, :, 0], x[:, :, 2] = x[:, :, 2], x[:, :, 0].copy()
        # x = Image.fromarray(x)
        return x

    if not val_only:
        train = InMemoryImageNet(train_path, num_train,
                                transforms=transforms.Compose([
                                    # pickle.loads,
                                    # lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR),
                                    # lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
                                    # transforms.ToPILImage(),
                                    msgpack_load,
                                    transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                                    shuffle=True, drop_last=False, **kwargs)
        train_loader.num_samples = num_train
    else:
        train, train_loader = None, None

    test = InMemoryImageNet(val_path, num_val,
                            transforms=transforms.Compose([
                                # pickle.loads,
                                # lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR),
                                # lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
                                # transforms.ToPILImage(),
                                msgpack_load,
                                transforms.Resize(int(input_size / 0.875)),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                                shuffle=False, drop_last=False, **kwargs)
    test_loader.num_samples = num_val
    return train, train_loader, test, test_loader
