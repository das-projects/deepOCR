from deepocr.file_utils import is_torch_available

from .base import *

if is_torch_available():
    from .pytorch import *  # type: ignore[misc]
