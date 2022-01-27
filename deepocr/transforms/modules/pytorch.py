# Copyright (C) 2022, Arijit Das.
# Code adapted from doctr and huggingface
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
from typing import Any, Dict, Tuple, Union

import torch
from PIL.Image import Image
from torch.nn.functional import pad
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

__all__ = ['Resize', 'GaussianNoise', 'ChannelShuffle', 'RandomHorizontalFlip']


class Resize(T.Resize):
    def __init__(
        self,
        size: Tuple[int, int],
        interpolation=F.InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = False,
    ) -> None:
        super().__init__(size, interpolation)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        target_ratio = self.size[0] / self.size[1]
        actual_ratio = img.shape[-2] / img.shape[-1]
        if not self.preserve_aspect_ratio or (target_ratio == actual_ratio):
            return super().forward(img)
        else:
            # Resize
            if actual_ratio > target_ratio:
                tmp_size = (self.size[0], max(int(self.size[0] / actual_ratio), 1))
            else:
                tmp_size = (max(int(self.size[1] * actual_ratio), 1), self.size[1])

            # Scale image
            img = F.resize(img, tmp_size, self.interpolation)
            # Pad (inverted in pytorch)
            _pad = (0, self.size[1] - img.shape[-1], 0, self.size[0] - img.shape[-2])
            if self.symmetric_pad:
                half_pad = (math.ceil(_pad[1] / 2), math.ceil(_pad[3] / 2))
                _pad = (half_pad[0], _pad[1] - half_pad[0], half_pad[1], _pad[3] - half_pad[1])
            return pad(img, _pad)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        _repr = f"output_size={self.size}, interpolation='{interpolate_str}'"
        if self.preserve_aspect_ratio:
            _repr += f", preserve_aspect_ratio={self.preserve_aspect_ratio}, symmetric_pad={self.symmetric_pad}"
        return f"{self.__class__.__name__}({_repr})"


class GaussianNoise(torch.nn.Module):
    """Adds Gaussian Noise to the input tensor

       Example::
           >>> from deepocr.transforms import GaussianNoise
           >>> import torch
           >>> transform = GaussianNoise(0., 1.)
           >>> out = transform(torch.rand((3, 224, 224)))

       Args:
           mean : mean of the gaussian distribution
           std : std of the gaussian distribution
       """
    def __init__(self, mean: float = 0., std: float = 1.) -> None:
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape the distribution
        noise = self.mean + 2 * self.std * torch.rand(x.shape, device=x.device) - self.std
        if x.dtype == torch.uint8:
            return (x + 255 * noise).round().clamp(0, 255).to(dtype=torch.uint8)
        else:
            return (x + noise.to(dtype=x.dtype)).clamp(0, 1)

    def extra_repr(self) -> str:
        return f"mean={self.mean}, std={self.std}"


class ChannelShuffle(torch.nn.Module):
    """Randomly shuffle channel order of a given image"""

    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Get a random order
        chan_order = torch.rand(img.shape[0]).argsort()
        return img[chan_order]


class RandomHorizontalFlip(T.RandomHorizontalFlip):

    def forward(
            self,
            img: Union[torch.Tensor, Image],
            target: Dict[str, Any]
    ) -> Tuple[Union[torch.Tensor, Image], Dict[str, Any]]:
        """
        Args:
            img: Image to be flipped.
            target: Dictionary with boxes (in relative coordinates of shape (N, 4)) and labels as keys
        Returns:
            Tuple of PIL Image or Tensor and target
        """
        if torch.rand(1) < self.p:
            _img = F.hflip(img)
            _target = target.copy()
            # Changing the relative bbox coordinates
            _target["boxes"][:, ::2] = 1 - target["boxes"][:, [2, 0]]
            return _img, _target
        return img, target
