# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torch import nn

from deepocr.models.preprocessor import PreProcessor

from ._utils import remap_preds, split_crops

__all__ = ['RecognitionPredictor']


class RecognitionPredictor(nn.Module):
    """Implements an object able to identify character sequences in images

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        split_wide_crops: wether to use crop splitting for high aspect ratio crops
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
        split_wide_crops: bool = True,
    ) -> None:

        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()
        self.split_wide_crops = split_wide_crops
        self.critical_ar = 8  # Critical aspect ratio
        self.dil_factor = 1.4  # Dilation factor to overlap the crops
        self.target_ar = 6  # Target aspect ratio

    @torch.no_grad()
    def forward(
        self,
        crops: List[Union[np.ndarray, torch.Tensor]],
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:

        if len(crops) == 0:
            return []
        # Dimension check
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError("incorrect input shape: all crops are expected to be multi-channel 2D images.")

        # Split crops that are too wide
        remapped = False
        if self.split_wide_crops:
            new_crops, crop_map, remapped = split_crops(
                crops,
                self.critical_ar,
                self.target_ar,
                self.dil_factor,
                isinstance(crops[0], np.ndarray)
            )
            if remapped:
                crops = new_crops

        # Resize & batch them
        processed_batches = self.pre_processor(crops)

        # Forward it
        _device = next(self.model.parameters()).device
        raw = [
            self.model(batch.to(device=_device), return_preds=True, **kwargs)['preds']  # type: ignore[operator]
            for batch in processed_batches
        ]

        # Process outputs
        out = [charseq for batch in raw for charseq in batch]

        # Remap crops
        if self.split_wide_crops and remapped:
            out = remap_preds(out, crop_map, self.dil_factor)

        return out
