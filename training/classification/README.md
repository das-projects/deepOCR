# Character classification

The sample training script was made to train a character classification model with deepOCR.

## Setup

First, you need to install `deepOCR` (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r training/requirements.txt
```

## Usage

You can start your training in PyTorch:

```shell
python training/classification/train_pytorch.py mobilenet_v3_large --epochs 5 --device 0
```

For help with other available options

```shell
python training/classification/train_pytorch.py --help
```
