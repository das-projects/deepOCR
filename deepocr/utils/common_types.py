# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from pathlib import Path
from typing import List, Tuple, Union

__all__ = ['Point2D', 'BoundingBox', 'Polygon4P', 'Polygon', 'Bbox']


Point2D = Tuple[float, float]
BoundingBox = Tuple[Point2D, Point2D]
Polygon4P = Tuple[Point2D, Point2D, Point2D, Point2D]
Polygon = List[Point2D]
AbstractPath = Union[str, Path]
AbstractFile = Union[AbstractPath, bytes]
Bbox = Tuple[float, float, float, float]
