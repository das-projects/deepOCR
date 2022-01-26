# Text detection

The sample training script was made to train text detection model with docTR.

## Setup

First, you need to install `doctr` (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r training/requirements.txt
```

## Usage

You can start your training in TensorFlow:

```shell
python training/detection/train_tensorflow.py path/to/your/train_set path/to/your/val_set db_resnet50 --epochs 5
```
or PyTorch:

```shell
python training/detection/train_pytorch.py path/to/your/train_set path/to/your/val_set db_resnet50 --epochs 5 --device 0
```

## Data format

You need to provide both `train_path` and `val_path` arguments to start training. 
Each path must lead to folder with 1 subfolder and 1 file:

```shell
├── images
│   ├── sample_img_01.png
│   ├── sample_img_02.png
│   ├── sample_img_03.png   
│   └── ...
└── labels.json
```

Each JSON file must be a dictionary, where the keys are the image file names and the value is a dictionary with 3 entries: `img_dimensions` (spatial shape of the image), `img_hash` (SHA256 of the image file), `polygons` (the set of 2D points forming the localization polygon).
The order of the points does not matter inside a polygon. Points are (x, y) absolutes coordinates.

labels.json
```shell
{
    "sample_img_01.png" = {
        'img_dimensions': (900, 600),
        'img_hash': "theimagedumpmyhash",
        'polygons': [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
     },
     "sample_img_02.png" = {
        'img_dimensions': (900, 600),
        'img_hash': "thisisahash",
        'polygons': [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
     }
     ...
}
```

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

```shell
python training/detection/train_tensorflow.py --help
```
