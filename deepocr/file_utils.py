# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py

import importlib.util
import logging
import os
import sys

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


__all__ = ['is_torch_available']

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()


if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logging.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False


if not _torch_available:
    raise ModuleNotFoundError("DocTR requires either TensorFlow or PyTorch to be installed. Please ensure one of them"
                              " is installed and that either USE_TF or USE_TORCH is enabled.")


def is_torch_available():
    return _torch_available