# Copyright (C) 2022, Raphael Kronberg.
# Code adapted from https://github.com/baudm/parseq and deepocr
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Optional, Tuple

import torch
from ..core import RecognitionModel, RecognitionPostProcessor
from deepocr.datasets import VOCABS
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)
_logger = logger
##########
__all__ = ["abinet", "crnn", "trba", "vitstr", "parseq_tiny", "parseq"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    'abinet': {
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'input_shape': (3, 32, 128),
        'model': 'abinet',
    },
    'crnn': {
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'input_shape': (3, 32, 384),
        'model': 'crnn',
    },
    'trba': {
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'input_shape': (3, 32, 128),
        'model': 'trba',
    },
    'vitstr': {
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'input_shape': (3, 32, 128),
        'model': 'vitstr',
    },
    'parseq_tiny': {
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'input_shape': (3, 384, 384),
        'model': 'parseq_tiny',
    },
    'parseq': {
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'input_shape': (3, 32, 128),
        'model': 'parseq',
    },
}


class TORCPostProcessor(RecognitionPostProcessor):
    """
    Postprocess raw prediction of the model (logits) to a list of words

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    @staticmethod
    def greedy_encoding(
            logits: torch.Tensor, model,
    ) -> List[Tuple[str, float]]:
        """
        Greedy incoding  adapted from https://github.com/baudm/parseq

        Args:
            logits: raw output of the model

        Returns:
            A tuple of 2 lists: a list of str (words) and a list of float (probs)

        """
        # Gather the most confident characters, and assign the smallest conf among those to the sequence prob
        logger.info(logits.shape)
        pred = logits.softmax(-1)
        labels, confidence = model.tokenizer.decode(pred)
        probs = [t.min() for t in confidence]
        return list(zip(labels, probs))

    def __call__(  # type: ignore[override]
            self,
            logits: torch.Tensor,
            model,
    ) -> List[Tuple[str, float]]:
        """
        Just return the predicted strings adapted form DeepOCR

        Args:
            logits: raw output of the model, shape (N, C + 1, seq_len)

        Returns:
            A tuple of 2 lists: a list of str (words) and a list of float (probs)

        """

        return self.greedy_encoding(logits=logits, model=model)


class TOCR(RecognitionModel, nn.Module):
    """
    Implements architectures as described in `"Scene Text Recognition with Permuted
     Autoregressive Sequence Models" <https://doi.org/10.1007/978-3-031-19815-1_11>`.

    Args:
        cfg: configuration dictionary
    """
    _children_names: List[str] = ['decoder', 'postprocessor']

    def __init__(
            self,
            cfg: Optional[Dict[str, Any]] = default_cfgs,
            pretrained: bool =True,
            **kwargs,
    ) -> None:
        super().__init__()
        self.vocab = VOCABS['english']
        self.cfg = cfg
        self.cfg_model = cfg['model']
        logger.info(("cfg_model", self.cfg_model))
        self.postprocessor = TORCPostProcessor(vocab=self.vocab)
        self.model = torch.hub.load('baudm/parseq', f'{self.cfg_model}', pretrained=pretrained)
        self.model = self.model.eval()

    def forward(
            self,
            x: torch.Tensor,
            target: Optional[List[str]] = None,
            return_model_output: bool = False,
            return_preds: bool = False,
    ) -> Dict[str, Any]:

        logits = self.model(x)

        out: Dict[str, Any] = {}
        if return_model_output:
            out["out_map"] = logits

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(logits, self.model)

        if target is not None:
            batch = x, target
            out['loss'] = self.model.training_step(batch, batch_idx=0)

        return out


def abinet(pretrained=True, **kwargs):
    """
    Implementation copied from Scene Text Recognition with Permuted Autoregressive Sequence Models
    Args:
        pretrained:
        **kwargs:

    Returns:

    """
    model = TOCR(cfg=default_cfgs['abinet'], pretrained=pretrained)
    return model


def crnn(pretrained=True, **kwargs):
    """
    Implementation copied from Scene Text Recognition with Permuted Autoregressive Sequence Models
    Args:
        pretrained (bool): If True, returns a model pre-trained

    Returns:
        text recognition architecture

    """
    model = TOCR(cfg=default_cfgs['crnn'], pretrained=pretrained)
    return model


def trba(pretrained=True, **kwargs):
    """
    Implementation copied from Scene Text Recognition with Permuted Autoregressive Sequence Models
    Args:
        pretrained (bool): If True, returns a model pre-trained

    Returns:
        text recognition architecture
    """
    model = TOCR(cfg=default_cfgs['trba'], pretrained=pretrained)
    return model


def vitstr(pretrained=True, **kwargs):
    """
    Implementation copied from Scene Text Recognition with Permuted Autoregressive Sequence Models
    Args:
        pretrained (bool): If True, returns a model pre-trained

    Returns:
        text recognition architecture

    """
    model = TOCR(cfg=default_cfgs['vitstr'], pretrained=pretrained)
    return model


def parseq_tiny(pretrained=True, **kwargs):
    """
    Implementation copied from Scene Text Recognition with Permuted Autoregressive Sequence Models
    Args:
        pretrained (bool): If True, returns a model pre-trained

    Returns:
        text recognition architecture

    """
    model = TOCR(cfg=default_cfgs['parseq_tiny'], pretrained=pretrained)
    return model


def parseq(pretrained=True, **kwargs):
    """
    Implementation copied from Scene Text Recognition with Permuted Autoregressive Sequence Models
    Args:
        pretrained (bool): If True, returns a model pre-trained

    Returns:
        text recognition architecture

    """
    model = TOCR(cfg=default_cfgs['parseq'], pretrained=pretrained)
    return model
