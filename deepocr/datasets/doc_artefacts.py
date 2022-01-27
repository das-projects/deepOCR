# Copyright (C) 2022, Arijit Das.
# Code adapted from doctr and huggingface
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from .datasets import VisionDataset

__all__ = ['DocArtefacts']


class DocArtefacts(VisionDataset):
    """Object detection dataset for non-textual elements in documents.
    The dataset includes a variety of synthetic document pages with non-textual elements.

    Example::
        >>> from deepocr.datasets import DocArtefacts
        >>> train_set = DocArtefacts(download=True)
        >>> img, target = train_set[0]

    Args:
        train: whether the subset should be the training one
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `VisionDataset`.
    """

    URL = 'https://github.com/mindee/doctr/releases/download/v0.4.0/artefact_detection-13fab8ce.zip'
    SHA256 = '13fab8ced7f84583d9dccd0c634f046c3417e62a11fe1dea6efbbaba5052471b'
    CLASSES = ["background", "qr_code", "bar_code", "logo", "photo"]

    def __init__(
        self,
        train: bool = True,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(self.URL, None, self.SHA256, True, **kwargs)
        self.train = train

        # Update root
        self.root = os.path.join(self.root, "train" if train else "val")
        # List images
        tmp_root = os.path.join(self.root, 'images')
        with open(os.path.join(self.root, "labels.json"), "rb") as f:
            labels = json.load(f)
        self.data: List[Tuple[str, Dict[str, Any]]] = []
        img_list = os.listdir(tmp_root)
        if len(labels) != len(img_list):
            raise AssertionError('the number of images and labels do not match')
        np_dtype = np.float32
        for img_name, label in labels.items():
            # File existence check
            if not os.path.exists(os.path.join(tmp_root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(tmp_root, img_name)}")

            # xmin, ymin, xmax, ymax
            boxes = np.asarray([obj['geometry'] for obj in label], dtype=np_dtype)
            classes = np.asarray([self.CLASSES.index(obj['label']) for obj in label], dtype=np.int64)
            if use_polygons:
                # (x, y) coordinates of top left, top right, bottom right, bottom left corners
                boxes = np.stack(
                    [
                        np.stack([boxes[:, 0], boxes[:, 1]], axis=-1),
                        np.stack([boxes[:, 2], boxes[:, 1]], axis=-1),
                        np.stack([boxes[:, 2], boxes[:, 3]], axis=-1),
                        np.stack([boxes[:, 0], boxes[:, 3]], axis=-1),
                    ], axis=1
                )
            self.data.append((img_name, dict(boxes=boxes, labels=classes)))
        self.root = tmp_root

    def extra_repr(self) -> str:
        return f"train={self.train}"
