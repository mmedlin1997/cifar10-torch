# cifar10-torch
Machine learning tutorial application using model trained on CIFAR10 dataset.

Uses PyTorch framework, ONNX model format. 

## Usage
Select target processor with command line option -d. Options are: cpu, cuda, cuda:0
```bash
python prod.py -d cuda
```
Enable training and production inference modes with command line toggles: -t,--train; -p,--prod 
```bash
python prod.py -d cuda -t -p
```
