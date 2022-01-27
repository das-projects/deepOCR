# Copyright (C) 2022, Arijit Das.
# Code adapted from doctr and huggingface
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from deepocr.utils.repr import NestedObject

from .. import functional as F

__all__ = ['SampleCompose', 'ImageTransform', 'ColorInversion', 'OneOf', 'RandomApply', 'RandomRotate', 'RandomCrop']


class SampleCompose(NestedObject):
    """Implements a wrapper that will apply transformations sequentially on both image and target
    Example::
        >>> from deepocr.transforms.modules import SampleCompose, ImageTransform, ColorInversion, RandomRotate
        >>> import torch
        >>> import numpy as np
        >>> transfos = SampleCompose([ImageTransform(ColorInversion((32, 32))), RandomRotate(30)])
        >>> out, out_boxes = transfos(torch.rand(shape=[64, 64, 3], minval=0, maxval=1), np.zeros((2, 4)))
    Args:
        transforms: list of transformation modules
    """

    _children_names: List[str] = ['sample_transforms']

    def __init__(self, transforms: List[Callable[[Any, Any], Tuple[Any, Any]]]) -> None:
        self.sample_transforms = transforms

    def __call__(self, x: Any, target: Any) -> Tuple[Any, Any]:
        for t in self.sample_transforms:
            x, target = t(x, target)

        return x, target


class ImageTransform(NestedObject):
    """Implements a transform wrapper to turn an image-only transformation into an image+target transform
    Example::
        >>> from deepocr.transforms import ImageTransform, ColorInversion
        >>> import torch
        >>> transfo = ImageTransform(ColorInversion((32, 32)))
        >>> out, _ = transfo(torch.rand(shape=[64, 64, 3], minval=0, maxval=1), None)
    Args:
        transform: the image transformation module to wrap
    """

    _children_names: List[str] = ['img_transform']

    def __init__(self, transform: Callable[[Any], Any]) -> None:
        self.img_transform = transform

    def __call__(self, img: Any, target: Any) -> Tuple[Any, Any]:
        img = self.img_transform(img)
        return img, target


class ColorInversion(NestedObject):
    """Applies the following tranformation to a tensor (image or batch of images):
    convert to grayscale, colorize (shift 0-values randomly), and then invert colors

    Example::
        >>> from deepocr.transforms.modules import ColorInversion
        >>> import torch
        >>> transfo = ColorInversion(min_val=0.6)
        >>> out = transfo(torch.rand(shape=[8, 64, 64, 3], minval=0, maxval=1))

    Args:
        min_val: range [min_val, 1] to colorize RGB pixels
    """
    def __init__(self, min_val: float = 0.5) -> None:
        self.min_val = min_val

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}"

    def __call__(self, img: Any) -> Any:
        return F.invert_colors(img, self.min_val)


class OneOf(NestedObject):
    """Randomly apply one of the input transformations

    Example::
        >>> from deepocr.transforms.modules import OneOf
        >>> import torch
        >>> transfo = OneOf([JpegQuality(), Gamma()])
        >>> out = transfo(torch.rand(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        transforms: list of transformations, one only will be picked
    """

    _children_names: List[str] = ['transforms']

    def __init__(self, transforms: List[Callable[[Any], Any]]) -> None:
        self.transforms = transforms

    def __call__(self, img: Any) -> Any:
        # Pick transformation
        transfo = self.transforms[int(random.random() * len(self.transforms))]
        # Apply
        return transfo(img)


class RandomApply(NestedObject):
    """Apply with a probability p the input transformation

    Example::
        >>> from deepocr.transforms.modules import RandomApply
        >>> import torch
        >>> transfo = RandomApply(Gamma(), p=.5)
        >>> out = transfo(torch.rand(shape=[64, 64, 3], minval=0, maxval=1))

    Args:
        transform: transformation to apply
        p: probability to apply
    """
    def __init__(self, transform: Callable[[Any], Any], p: float = .5) -> None:
        self.transform = transform
        self.p = p

    def extra_repr(self) -> str:
        return f"transform={self.transform}, p={self.p}"

    def __call__(self, img: Any) -> Any:
        if random.random() < self.p:
            return self.transform(img)
        return img


class RandomRotate(NestedObject):
    """Randomly rotate a tensor image and its boxes

    .. image:: https://github.com/mindee/doctr/releases/download/v0.4.0/rotation_illustration.png
        :align: center

    Args:
        max_angle: maximum angle for rotation, in degrees. Angles will be uniformly picked in
            [-max_angle, max_angle]
        expand: whether the image should be padded before the rotation
    """
    def __init__(self, max_angle: float = 5., expand: bool = False) -> None:
        self.max_angle = max_angle
        self.expand = expand

    def extra_repr(self) -> str:
        return f"max_angle={self.max_angle}, expand={self.expand}"

    def __call__(self, img: Any, target: np.ndarray) -> Tuple[Any, np.ndarray]:
        angle = random.uniform(-self.max_angle, self.max_angle)
        r_img, r_polys = F.rotate_sample(img, target, angle, self.expand)
        # Removes deleted boxes
        is_kept = (r_polys.max(1) > r_polys.min(1)).sum(1) == 2
        return r_img, r_polys[is_kept]


class RandomCrop(NestedObject):
    """Randomly crop a tensor image and its boxes

    Args:
        scale: tuple of floats, relative (min_area, max_area) of the crop
        ratio: tuple of float, relative (min_ratio, max_ratio) where ratio = h/w
    """
    def __init__(self, scale: Tuple[float, float] = (0.08, 1.), ratio: Tuple[float, float] = (0.75, 1.33)) -> None:
        self.scale = scale
        self.ratio = ratio

    def extra_repr(self) -> str:
        return f"scale={self.scale}, ratio={self.ratio}"

    def __call__(self, img: Any, target: Dict[str, np.ndarray]) -> Tuple[Any, Dict[str, np.ndarray]]:
        scale = random.uniform(self.scale[0], self.scale[1])
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        # Those might overflow
        crop_h = math.sqrt(scale * ratio)
        crop_w = math.sqrt(scale / ratio)
        xmin, ymin = random.uniform(0, 1 - crop_w), random.uniform(0, 1 - crop_h)
        xmax, ymax = xmin + crop_w, ymin + crop_h
        # Clip them
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(xmax, 1), min(ymax, 1)

        croped_img, crop_boxes = F.crop_detection(img, target["boxes"], (xmin, ymin, xmax, ymax))
        return croped_img, dict(boxes=crop_boxes)
