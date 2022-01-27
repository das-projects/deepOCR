# Copyright (C) 2022, Arijit Das.
# Code adapted from doctr and huggingface
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


from typing import Any, Dict, Optional

from deepocr.utils.repr import NestedObject

__all__ = ['BaseModel']


class BaseModel(NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = cfg
