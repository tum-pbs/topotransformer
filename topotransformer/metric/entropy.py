import torch
from torch import Tensor
from pdetransformer.metric import SimulationMSE

class TopologyEntropy(SimulationMSE):
    """
    Metric to calculate the Binary Cross Entropy (BCE) between predictions and targets.
    
    BCE measures how well the predicted probabilities match the binary ground truth.
    Lower BCE = better predictions, Higher BCE = worse predictions.
    
    BCE(p, y) = - [y * log(p) + (1-y) * log(1-p)]
    
    Expects:
        - preds: continuous predictions (probabilities in [0, 1])
        - target: binary ground truth (0 or 1)
    """
    def __init__(self, **kwargs) -> None:
        # Accept threshold for consistency with other metrics
        self.threshold = kwargs.pop("threshold", 0.5)
        super().__init__(**kwargs)

    def update(self, preds: Tensor, target: Tensor, class_labels: Tensor) -> None:
        # Slice to match target spatial dimensions (following the project's convention)
        # preds shape: [Batch, Channels, Depth/Time, Height, Width]
        logits = preds[:, :, :target.shape[2]].detach().clone()
        y = target[:, :, :target.shape[2]].detach().clone()
        
        # IMPORTANT: Model outputs are LOGITS, not probabilities!
        # We must apply sigmoid to convert logits to probabilities
        p = logits #torch.sigmoid(logits)
        
        # Ensure targets are binary (0 or 1)
        #y = (y > self.threshold).float()
        y = torch.clamp(y, 0,1)
        # Clamp predictions for log stability
        eps = 1e-7
        p = torch.clamp(p, eps, 1.0 - eps)
        
        # Binary Cross Entropy formula
        # BCE(p, y) = - [y * log(p) + (1-y) * log(1-p)]
        bce_field = - (y * torch.log(p) + (1 - y) * torch.log(1 - p))
        
        # Replace any NaN values with 0 (perfect prediction assumed)
        bce_field = torch.where(torch.isnan(bce_field), torch.zeros_like(bce_field), bce_field)
        
        # Calculate mean BCE over spatial/temporal dimensions (2, 3, 4)
        # Result shape should be [Batch, Channels]
        mean_bce = bce_field.mean(dim=(2, 3, 4))

        self._update(mean_bce, class_labels)
