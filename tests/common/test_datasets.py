from pathlib import Path

import numpy as np
import pytest

from deepocr import datasets


def test_visiondataset():
    url = 'https://data.deepai.org/mnist.zip'
    with pytest.raises(ValueError):
        datasets.datasets.VisionDataset(url, download=False)

    dataset = datasets.datasets.VisionDataset(url, download=True, extract_archive=True)
    assert len(dataset) == 0
    assert repr(dataset) == 'VisionDataset()'


def test_abstractdataset(mock_image_path):

    with pytest.raises(ValueError):
        datasets.datasets.AbstractDataset('my/fantasy/folder')

    # Check transforms
    path = Path(mock_image_path)
    ds = datasets.datasets.AbstractDataset(path.parent)
    # Patch some data
    ds.data = [(path.name, 0)]

    # Fetch the img
    img, target = ds[0]
    assert isinstance(target, int) and target == 0

    # Check img_transforms
    ds.img_transforms = lambda x: 1 - x
    img2, target2 = ds[0]
    assert np.all(img2.numpy() == 1 - img.numpy())
    assert target == target2

    # Check sample_transforms
    ds.img_transforms = None
    ds.sample_transforms = lambda x, y: (x, y + 1)
    img3, target3 = ds[0]
    assert np.all(img3.numpy() == img.numpy()) and (target3 == (target + 1))
