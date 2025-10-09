import torch
from torch import Tensor
from scipy.ndimage import gaussian_filter
from typing import Optional, List, Tuple
import os, sys, math

def get_id(x: str) -> int:
    return int(x.split(".")[0])


def generate_density_map(label: Tensor, height: int, width: int, sigma: Optional[float] = None) -> Tensor:
    """
    Generate the density map based on the dot annotations provided by the label.
    """
    density_map = torch.zeros((1, height, width), dtype=torch.float32)

    if len(label) > 0:
        assert len(label.shape) == 2 and label.shape[1] == 2, f"label should be a Nx2 tensor, got {label.shape}."
        label_ = label.long()
        label_[:, 0] = label_[:, 0].clamp(min=0, max=width - 1)
        label_[:, 1] = label_[:, 1].clamp(min=0, max=height - 1)
        density_map[0, label_[:, 1], label_[:, 0]] = 1.0

    if sigma is not None:
        assert sigma > 0, f"sigma should be positive if not None, got {sigma}."
        density_map = torch.from_numpy(gaussian_filter(density_map, sigma=sigma))

    return density_map



import torch
import torch.nn.functional as F
from typing import List, Tuple

def resize_density_map(x: Tensor, size: Tuple[int, int]) -> Tensor:
    x_sum = torch.sum(x, dim=(-1, -2))
    x = F.interpolate(x, size=size, mode="bilinear")
    scale_factor = torch.nan_to_num(torch.sum(x, dim=(-1, -2)) / x_sum, nan=0.0, posinf=0.0, neginf=0.0)
    return x * scale_factor

def collate_fn(batch: List[Tensor]) -> Tuple[Tensor, List[Tensor], Tensor]:
    batch = list(zip(*batch))
    images = batch[0]
    assert len(images[0].shape) == 4, f"images should be a 4D tensor, got {images[0].shape}."
    if len(batch) == 4:  # image, label, density_map, image_name
        images = torch.cat(images, 0)
        points = batch[1]  # list of lists of tensors, flatten it
        points = [p for points_ in points for p in points_]
        densities = torch.cat(batch[2], 0)
        image_names = batch[3]  # list of lists of strings, flatten it
        image_names = [name for names_ in image_names for name in names_]

        return images, points, densities, image_names

    elif len(batch) == 3:  # image, label, density_map
        images = torch.cat(images, 0)
        points = batch[1]
        points = [p for points_ in points for p in points_]
        densities = torch.cat(batch[2], 0)

        return images, points, densities
    
    elif len(batch) == 2:  # image, image_name. NWPU test dataset
        images = torch.cat(images, 0)
        image_names = batch[1]
        image_names = [name for names_ in image_names for name in names_]

        return images, image_names

    else:
        images = torch.cat(images, 0)

        return images
    
def cosine_annealing_warm_restarts(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    warmup_lr: float,
    T_0: int,
    T_mult: int,
    eta_min: float,
) -> float:
    """
    Learning rate scheduler.
    The learning rate will linearly increase from warmup_lr to lr in the first warmup_epochs epochs.
    Then, the learning rate will follow the cosine annealing with warm restarts strategy.
    """
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}."
    assert isinstance(warmup_epochs, int) and warmup_epochs >= 0, f"warmup_epochs must be non-negative, got {warmup_epochs}."
    assert isinstance(warmup_lr, float) and warmup_lr > 0, f"warmup_lr must be positive, got {warmup_lr}."
    assert isinstance(T_0, int) and T_0 >= 1, f"T_0 must be greater than or equal to 1, got {T_0}."
    assert isinstance(T_mult, int) and T_mult >= 1, f"T_mult must be greater than or equal to 1, got {T_mult}."
    assert isinstance(eta_min, float) and eta_min > 0, f"eta_min must be positive, got {eta_min}."
    assert isinstance(base_lr, float) and base_lr > 0, f"base_lr must be positive, got {base_lr}."
    assert base_lr > eta_min, f"base_lr must be greater than eta_min, got base_lr={base_lr} and eta_min={eta_min}."
    assert warmup_lr >= eta_min, f"warmup_lr must be greater than or equal to eta_min, got warmup_lr={warmup_lr} and eta_min={eta_min}."

    if epoch < warmup_epochs:
        lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
    else:
        epoch -= warmup_epochs
        if T_mult == 1:
            T_cur = epoch % T_0
            T_i = T_0
        else:
            n = int(math.log((epoch / T_0 * (T_mult - 1) + 1), T_mult))
            T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
            T_i = T_0 * T_mult ** (n)
        
        lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2

    return lr / base_lr
