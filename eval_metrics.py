import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor, nn
from typing import Dict, Tuple, Union
class CrowdCountingMetrics:
    @staticmethod
    def mean_absolute_error(pred_counts, gt_counts):
        """Computes Mean Absolute Error (MAE)."""
        return torch.abs(pred_counts - gt_counts).mean().item()

    @staticmethod
    def root_mean_squared_error(pred_counts, gt_counts):
        """Computes Root Mean Squared Error (RMSE)."""
        return torch.sqrt(((pred_counts - gt_counts) ** 2).mean()).item()

    @staticmethod
    def accuracy(pred_counts, gt_counts, threshold=0.2):
        """Computes Accuracy within a certain error threshold."""
        relative_error = torch.abs(pred_counts - gt_counts) / (gt_counts + 1e-6)
        correct = (relative_error < threshold).float()
        return correct.mean().item()

    @staticmethod
    def evaluate(predictions, ground_truths):
        """Computes all evaluation metrics."""
        mae = CrowdCountingMetrics.mean_absolute_error(predictions, ground_truths)
        rmse = CrowdCountingMetrics.root_mean_squared_error(predictions, ground_truths)
        acc = CrowdCountingMetrics.accuracy(predictions, ground_truths)
        
        return {
            "MAE": mae,
            "RMSE": rmse,
            "Accuracy": acc
        }
    

def sliding_window_predict(
    model: nn.Module,
    image: Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
) -> Tensor:
    """
    Generate the density map for an image using the sliding window method. Overlapping regions will be averaged.

    Args:
        model (nn.Module): The model to use.
        image (Tensor): The image (1, c, h, w) to generate the density map for. The batch size must be 1 due to varying image sizes.
        window_size (Union[int, Tuple[int, int]]): The size of the window.
        stride (Union[int, Tuple[int, int]]): The step size of the window.
    """
    assert len(image.shape) == 4, f"Image must be a 4D tensor (1, c, h, w), got {image.shape}"
    window_size = (int(window_size), int(window_size)) if isinstance(window_size, (int, float)) else window_size
    stride = (int(stride), int(stride)) if isinstance(stride, (int, float)) else stride
    window_size = tuple(window_size)
    stride = tuple(stride)
    assert isinstance(window_size, tuple) and len(window_size) == 2 and window_size[0] > 0 and window_size[1] > 0, f"Window size must be a positive integer tuple (h, w), got {window_size}"
    assert isinstance(stride, tuple) and len(stride) == 2 and stride[0] > 0 and stride[1] > 0, f"Stride must be a positive integer tuple (h, w), got {stride}"
    assert stride[0] <= window_size[0] and stride[1] <= window_size[1], f"Stride must be smaller than window size, got {stride} and {window_size}"

    image_height, image_width = image.shape[-2:]
    window_height, window_width = window_size
    stride_height, stride_width = stride

    num_rows = int(np.ceil((image_height - window_height) / stride_height) + 1)
    num_cols = int(np.ceil((image_width - window_width) / stride_width) + 1)

    reduction = model.reduction if hasattr(model, "reduction") else 1  # reduction factor of the model. For example, if reduction = 8, then the density map will be reduced by 8x.
    print("Reduction: ", reduction)
    windows = []
    for i in range(num_rows):
        for j in range(num_cols):
            x_start, y_start = i * stride_height, j * stride_width
            x_end, y_end = x_start + window_height, y_start + window_width
            if x_end > image_height:
                x_start, x_end = image_height - window_height, image_height
            if y_end > image_width:
                y_start, y_end = image_width - window_width, image_width

            window = image[:, :, x_start:x_end, y_start:y_end]
            windows.append(window)

    windows = torch.cat(windows, dim=0).to(image.device)  # batched windows, shape: (num_windows, c, h, w)

    model.eval()
    with torch.no_grad():
        _, preds = model(windows)
    preds = preds.cpu().detach().numpy()

    # assemble the density map

    pred_map = np.zeros((preds.shape[1], image_height // reduction, image_width // reduction), dtype=np.float32)
    count_map = np.zeros((preds.shape[1], image_height // reduction, image_width // reduction), dtype=np.float32)
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            x_start, y_start = i * stride_height, j * stride_width
            x_end, y_end = x_start + window_height, y_start + window_width
            if x_end > image_height:
                x_start, x_end = image_height - window_height, image_height
            if y_end > image_width:
                y_start, y_end = image_width - window_width, image_width

            pred_map[:, (x_start // reduction): (x_end // reduction), (y_start // reduction): (y_end // reduction)] += preds[idx, :, :, :]
            count_map[:, (x_start // reduction): (x_end // reduction), (y_start // reduction): (y_end // reduction)] += 1.
            idx += 1

    pred_map /= count_map  # average the overlapping regions
    return torch.tensor(pred_map).unsqueeze(0)  # shape: (1, 1, h // reduction, w // reduction)
