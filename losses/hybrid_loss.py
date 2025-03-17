import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, Tuple, List

class AdaptiveHybridLoss(nn.Module):
    def __init__(self, bins: List[Tuple[float, float]], weight_count_loss=1.0, use_dm_loss=False, reduction=32):
        super().__init__()
        self.cross_entropy_fn = nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()
        self.use_dm_loss = use_dm_loss
        self.weight_count_loss = weight_count_loss
        self.reduction = reduction
        self.bins = bins  # ✅ Store bins inside the loss function

    def _bin_count(self, density_map: torch.Tensor) -> torch.Tensor:
        """Assigns ground-truth density maps to bin indices for classification."""
        bin_counts = torch.zeros_like(density_map, dtype=torch.long)
        print("Bins Shape:", self.bins)
        print("DENSITY MAP: ", density_map.shape)
        for i, (low, high) in enumerate(self.bins):  
            print(f"Processing bin {i}: range ({low}, {high})")
            mask = (density_map >= low) & (density_map <= high)
            print(f"Assigned {mask.sum().item()} pixels to bin {i}")
            bin_counts[mask] = i
        
        print("Final bin counts unique values:", torch.unique(bin_counts))
        return bin_counts


    def forward(self, pred_class: Tensor, pred_density: Tensor, target_density: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        # ✅ Ensure target_density has correct shape
        target_density = target_density.unsqueeze(1) if target_density.dim() == 3 else target_density  # Ensure (N, 1, H, W)
        
        # ✅ Perform bilinear interpolation safely
        target_density = F.interpolate(target_density, size=pred_density.shape[-2:], mode="bilinear", align_corners=False)
        target_density = target_density.squeeze(1)  # Remove channel dim
        
        # ✅ Check for batch size mismatch
        assert target_density.shape[0] == pred_density.shape[0], "Batch size mismatch between target and prediction"

        # Compute classification loss
        target_class = self._bin_count(target_density)
        cross_entropy_loss = self.cross_entropy_fn(pred_class, target_class).mean()


        # Compute regression loss
        count_loss = self.mse_loss_fn(pred_density, target_density).mean()
        print("Shapes - pred_class:", pred_class.shape, "pred_density:", pred_density.shape, "target_density:", target_density.shape)
        print("target_class unique values:", torch.unique(target_class))  
        print("CrossEntropyLoss input stats: min:", pred_class.min().item(), "max:", pred_class.max().item())
        print("MSELoss input stats: min:", pred_density.min().item(), "max:", pred_density.max().item())

        loss = cross_entropy_loss + self.weight_count_loss * count_loss
        loss_info = {
            "loss": loss.detach(),
            "class_loss": cross_entropy_loss.detach(),
            "mse_loss": count_loss.detach(),
        }

        return loss, loss_info



