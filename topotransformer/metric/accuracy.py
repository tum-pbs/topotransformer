import os
from typing import Optional, List

from torch import Tensor
from torchmetrics import Metric
import numpy as np
import torch
import pandas as pd

from pdetransformer.metric import SimulationMSE
def filterObject(obj, threshold: float = 0.5):
    if torch.is_tensor(obj):
        return (obj > threshold).float()
    return (obj > threshold).astype(float)


class SimulationAccuracy(SimulationMSE):
    def __init__(self, **kwargs) -> None:
        #pop threshold from kwargs
        self.threshold = kwargs.pop("threshold", 0.5)
        super().__init__(**kwargs)
    def update(self, preds: Tensor, target: Tensor, class_labels: Tensor) -> None:

        shifted = 1 - torch.abs(filterObject(preds[:, :, :target.shape[2]], self.threshold) - filterObject(target, self.threshold))

        accuracy = shifted.mean(dim=(2, 3, 4))

        self._update(accuracy, class_labels)