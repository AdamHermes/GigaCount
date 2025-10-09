import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, Tuple, List
from .dmloss import DMLoss, _reshape_density

class AdaptiveHybridLoss(nn.Module):
    def __init__(self, bins: List[Tuple[float, float]], weight_count_loss=1.0,use_dm_loss=True, reduction=8):
        super().__init__()
        self.cross_entropy_fn = nn.CrossEntropyLoss(reduction="none")
        self.mse_loss_fn = nn.MSELoss()

        self.count_loss_fn = DMLoss(reduction=reduction, input_size=224)
        self.use_dm_loss = use_dm_loss
        self.weight_count_loss = weight_count_loss
        self.reduction = reduction
        self.bins = bins  

    def _bin_count(self, density_map: torch.Tensor) -> torch.Tensor:
        """Assigns ground-truth density maps to bin indices for classification."""
        bin_counts = torch.zeros_like(density_map, dtype=torch.long)
        for i, (low, high) in enumerate(self.bins):  
            mask = (density_map >= low) & (density_map <= high)
            bin_counts[mask] = i
        
        return bin_counts.squeeze(1)


    def forward(self, pred_class: Tensor, pred_density: Tensor, target_density: Tensor, target_points: List[Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        target_density = _reshape_density(target_density, reduction=self.reduction) if target_density.shape[-2:] != pred_density.shape[-2:] else target_density
        assert pred_density.shape == target_density.shape, f"Expected pred_density and target_density to have the same shape, got {pred_density.shape} and {target_density.shape}"
        target_class = self._bin_count(target_density)

        cross_entropy_loss = self.cross_entropy_fn(pred_class, target_class).sum(dim=(-1, -2)).mean()
        pred_prob = F.softmax(pred_class,dim=1)
        target_prob = F.one_hot(target_class, num_classes=len(self.bins)).permute(0, 3, 1, 2).float()
        euclidean_loss = ((pred_prob - target_prob) ** 2).sum(dim=(-1, -2, -3)).mean()



        if self.use_dm_loss:
            count_loss, loss_info = self.count_loss_fn(pred_density, target_density, target_points)
            loss_info["ce_loss"] = cross_entropy_loss.detach()
        else:
            count_loss = self.count_loss_fn(pred_density, target_density).sum(dim=(-1, -2, -3)).mean()
            loss_info = {
                "ce_loss": cross_entropy_loss.detach(),
                f"{self.count_loss}_loss": count_loss.detach(),
            }

        loss = cross_entropy_loss  + euclidean_loss + self.weight_count_loss * count_loss
        loss_info["loss"] = loss.detach()

        return loss, loss_info



