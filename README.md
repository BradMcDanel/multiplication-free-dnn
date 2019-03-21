# Training Multiplication-free DNNs with Power of Two Weights

## Training Phase
Training can be performed in up to 2 stages. However, we can omit the first stage if we are confident in the parameters for pruning (and do not wish to start from pretrained model). Both training scripts `train.py` and `train_prune.py` can be used with a pretrained model (using the `--load-path` flag) or start training a new model if `--load-path` is omitted. In the case of a new model, several parameters control the width/depth of the model and activation quantization. For pruning, the groups parameter determines how much a layer is pruned. If groups=8, then the layer will be reduced by a factor of 8.

To train a CIFAR-10 model with column combining and power of two weights:
```
python train.py --dataset-root <dataset-root folder> --dataset cifar10 --batch-size 256 --epochs 50 --lr 0.2 --aug + --save-path cifar10-quant.pth --filters 128 256 512 --layers 6 6 6 --strides 1 2 2 --groups 1 2 8 --max-weight-exp 1 --weight-levels 8 --sparsity-lambda 0 --layer-type quant --bn-type quant-bn --n-class 10
```

## Packing for FPGA Deployment
Once trained, it can be converted into a packed binary format which can run on the FPGA:
```
python fpga_compiler.py --dataset-root /hdd1/datasets --dataset cifar10 --load-path cifar10-quant.pth --output-folder fpga-out
```

## FPGA Verilog code
See the [verilog/](https://github.com/BradMcDanel/multiplication-free-dnn/tree/master/verilog) for details on the verilog implementation of the multiplication-free systolic array.
