import torch
import torch.nn.functional as F
import numpy as np

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
