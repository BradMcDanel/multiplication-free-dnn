# Full-stack Optimization for Accelerating CNNs Using Powers-ofTwo Weights with FPGA Validation

This codebase requires PyTorch 1.0. Please refer to [net.py](https://github.com/BradMcDanel/multiplication-free-dnn/blob/master/net.py) for PyTorch layer definitions. Custom CUDA kernals for quantization (log and linear) and shift layers are in [kernels/](https://github.com/BradMcDanel/multiplication-free-dnn/tree/master/kernels). This codebase uses [Label Refinery](https://github.com/hessamb/label-refinery) for improved model performance.

We use a custom data loader which loads the entire ImageNet into memory (~200GB) before training (required due to network drives). This can be swapped out for the standard PyTorch dataloader and should lead to similar model performance.

## Quantized Powers-of-two training (smaller model ~1.5M weights)
```
python train.py --dataset-root <dataset folder> --dataset imagenet --batch-size 1024 \
                --epochs 140 --lr-type cosine --lr 0.256 --aug + --save-path <save path> \
                --teacher-path <pretrained resnet50 path> --filters 128 512 512 512 1024  \
                --layers 4 4 4 4 1 --strides 1 2 2 2 1 --groups 2 8 8 8 8 --max-weight-exp 0 \
                --weight-levels 8 --layer-type quant --bn-type quant-bn --n-class 1000 \
                --input-size 224 --reshape-stride 4 --gamma 0.4 --in-memory
```
