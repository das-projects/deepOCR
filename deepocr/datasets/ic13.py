# Copyright (C) 2022, Arijit Das.
# Code adapted from doctr and huggingface
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .datasets import AbstractDataset
from .utils import convert_target_to_relative

__all__ = ["IC13"]


class IC13(AbstractDataset):
    """IC13 dataset from `"ICDAR 2013 Robust Reading Competition" <https://rrc.cvc.uab.es/>`_.

    Example::
        >>> # NOTE: You need to download both image and label parts from Focused Scene Text challenge Task2.1 2013-2015.
        >>> from deepocr.datasets import IC13
        >>> train_set = IC13(img_folder="/path/to/Challenge2_Training_Task12_Images",
        >>>                  label_folder="/path/to/Challenge2_Training_Task1_GT")
        >>> img, target = train_set[0]
        >>> test_set = IC13(img_folder="/path/to/Challenge2_Test_Task12_Images",
        >>>                 label_folder="/path/to/Challenge2_Test_Task1_GT")
        >>> img, target = test_set[0]

    Args:
        img_folder: folder with all the images of the dataset
        label_folder: folder with all annotation files for the images
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_folder: str,
        use_polygons: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, pre_transforms=convert_target_to_relative, **kwargs)

        # File existence check
        if not os.path.exists(label_folder) or not os.path.exists(img_folder):
            raise FileNotFoundError(
                f"unable to locate {label_folder if not os.path.exists(label_folder) else img_folder}")

        self.data: List[Tuple[Path, Dict[str, Any]]] = []
        np_dtype = np.float32

        img_names = os.listdir(img_folder)

        for img_name in img_names:

            img_path = Path(img_folder, img_name)
            label_path = Path(label_folder, "gt_" + Path(img_name).stem + ".txt")

            with open(label_path, newline='\n') as f:
                _lines = [
                    [val[:-1] if val.endswith(",") else val for val in row]
                    for row in csv.reader(f, delimiter=' ', quotechar="'")
                ]
            labels = [line[-1] for line in _lines]
            # xmin, ymin, xmax, ymax
            box_targets = np.array([list(map(int, line[:4])) for line in _lines], dtype=np_dtype)
            if use_polygons:
                # (x, y) coordinates of top left, top right, bottom right, bottom left corners
                box_targets = np.array(
                    [
                        [
                            [coords[0], coords[1]],
                            [coords[2], coords[1]],
                            [coords[2], coords[3]],
                            [coords[0], coords[3]],
                        ] for coords in box_targets
                    ], dtype=np_dtype
                )
            self.data.append((img_path, dict(boxes=box_targets, labels=labels)))
