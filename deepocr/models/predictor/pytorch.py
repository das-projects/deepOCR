# Copyright (C) 2022, Arijit Das.
# Code adapted from doctr and huggingface
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any, List, Union

import numpy as np
import torch
from torch import nn

from deepocr.io.elements import Document
from deepocr.models._utils import estimate_orientation
from deepocr.models.detection.predictor import DetectionPredictor
from deepocr.models.recognition.predictor import RecognitionPredictor
from deepocr.utils.geometry import rotate_boxes, rotate_image

from .base import _OCRPredictor

__all__ = ['OCRPredictor']


class OCRPredictor(nn.Module, _OCRPredictor):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        kwargs: keyword args of `DocumentBuilder`
    """

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        **kwargs: Any,
    ) -> None:

        nn.Module.__init__(self)
        self.det_predictor = det_predictor.eval()  # type: ignore[attr-defined]
        self.reco_predictor = reco_predictor.eval()  # type: ignore[attr-defined]
        _OCRPredictor.__init__(self, assume_straight_pages, straighten_pages, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        pages: List[Union[np.ndarray, torch.Tensor]],
        **kwargs: Any,
    ) -> Document:

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        # Detect document rotation and rotate pages
        if self.straighten_pages:
            origin_page_orientations = [estimate_orientation(page) for page in pages]
            pages = [rotate_image(page, -angle, expand=True) for page, angle in zip(pages, origin_page_orientations)]

        # Localize text elements
        loc_preds = self.det_predictor(pages, **kwargs)
        # Check whether crop mode should be switched to channels first
        channels_last = len(pages) == 0 or isinstance(pages[0], np.ndarray)

        # Crop images
        crops, loc_preds = self._prepare_crops(
            pages, loc_preds, channels_last=channels_last, assume_straight_pages=self.assume_straight_pages
        )
        # Rectify crop orientation
        if not self.assume_straight_pages:
            crops, loc_preds = self._rectify_crops(crops, loc_preds)
        # Identify character sequences
        word_preds = self.reco_predictor([crop for page_crops in crops for crop in page_crops], **kwargs)

        boxes, text_preds = self._process_predictions(loc_preds, word_preds)

        # Rotate back pages and boxes while keeping original image size
        if self.straighten_pages:
            boxes = [rotate_boxes(page_boxes,
                                  angle,
                                  orig_shape=page.shape[:2] if isinstance(page, np.ndarray) else page.shape[-2:]
                                  ) for page_boxes, page, angle in zip(boxes, pages, origin_page_orientations)]

        out = self.doc_builder(
            boxes,
            text_preds,
            [
                page.shape[:2] if channels_last else page.shape[-2:]  # type: ignore[misc]
                for page in pages
            ]
        )
        return out
