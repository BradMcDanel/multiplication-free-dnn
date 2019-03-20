# Training Multiplication-free DNNs with Power of Two Weights

## Training Strategy and Example
Training can be performed in up to 2 stages. However, we can omit the first stage if we are confident in the parameters for pruning (and do not wish to start from pretrained model). Both training scripts `train.py` and `train_prune.py` can be used with a pretrained model (using the `--load-path` flag) or start training a new model if `--load-path` is omitted. In the case of a new model, several parameters control the width/depth of the model and activation quantization (see Stage 1 arguments below).


### Trains model without weight pruning
```
python train.py --dataset-root /hdd1/datasets --dataset cifar10 --batch-size 256 --epochs 150 --lr 0.2 --aug ++ --save-path models/cifar10-quant.pth --filters 32 64 128 --layers 4 4 4 --strides 1 2 2 --groups 1 2 8 --delta 0.0625 --maxv 7.9375 --max-weight-exp 1 --weight-levels 8 --sparsity-lambda 0 --layer-type quant --n-class 10
```

### Trains model with weight pruning
For pruning, the groups parameter determines how much a layer is pruned. If groups=8, then the layer will be reduced by a factor of 8.
```
python train_prune.py --dataset-root /hdd1/datasets --dataset cifar10 --batch-size 256 --epochs 150 --lr 0.2 --aug ++ --save-path models/cifar10-quant.pth --filters 32 64 128 --layers 4 4 4 --strides 1 2 2 --groups 1 2 8 --delta 0.0625 --maxv 7.9375 --max-weight-exp 1 --weight-levels 8 --sparsity-lambda 0 --layer-type quant --n-class 10
```
