# Copyright (C) 2022, Arijit Das.
# Code adapted from doctr and huggingface
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List, Tuple

import cv2
import numpy as np

from deepocr.utils.repr import NestedObject

__all__ = ['FaceDetector']


class FaceDetector(NestedObject):

    """ Implements a face detector to detect profile pictures on resumes, IDS, driving licenses, passports...
    Based on open CV CascadeClassifier (haarcascades)

    Args:
        n_faces: maximal number of faces to detect on a single image, default = 1
    """

    def __init__(
        self,
        n_faces: int = 1,
    ) -> None:
        self.n_faces = n_faces
        # Instantiate classifier
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def extra_repr(self) -> str:
        return f"n_faces={self.n_faces}"

    def __call__(
        self,
        img: np.array,
    ) -> List[Tuple[float, float, float, float]]:
        """Detect n_faces on the img

        Args:
            img: image to detect faces on

        Returns:
            A list of size n_faces, each face is a tuple of relative xmin, ymin, xmax, ymax
        """
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(gray, 1.5, 3)
        # If faces are detected, keep only the biggest ones
        rel_faces = []
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda x: x[2] + x[3])[-min(self.n_faces, len(faces))]
            xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
            rel_faces.append((xmin, ymin, xmax, ymax))

        return rel_faces
