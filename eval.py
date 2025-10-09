import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np
from models.GigaCount import GigaCount
from losses.hybrid_loss import AdaptiveHybridLoss
from dataset import Crowd # Assume dataset module exists
from utils_custom import collate_fn, cosine_annealing_warm_restarts
from eval_metrics import CrowdCountingMetrics
from transforms import ColorJitter, GaussianBlur, RandomResizedCrop, RandomHorizontalFlip,RandomApply,PepperSaltNoise, NormalizeTensor, CombinedTrainTransform
from torchvision.transforms.v2 import Compose
from evaluate import evaluate_model  # Import evaluation function
from torch.optim.lr_scheduler import LambdaLR
from functools import partial

import torchvision.transforms as T

# Training Configurationsx  
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_TYPE = "adaptive_hybrid"
USE_AMP = True  
MAX_BIN_VALUE = 1e6  
CUSTOM_MEAN = [0.485, 0.456, 0.406]
CUSTOM_STD = [0.229, 0.224, 0.225]
transforms = Compose([
    RandomResizedCrop((224,224), scale=(0.8,1.0)),
    RandomHorizontalFlip()
])


original_bins = {
        "qnrf": {
            "bins": {
                "fine": [[0, 0], [1, 1], [2, 2], [3, 3], [4, "inf"]]
            },
            "anchor_points": {
                "fine": {
                    "middle": [0, 1, 2, 3, 4],
                    "average": [0, 1, 2, 3, 4.21937]
                }
            }
        }
    }

bins = original_bins["qnrf"]["bins"]["fine"]
anchor_points = original_bins["qnrf"]["anchor_points"]["fine"]["average"] 
bins = [(float(b[0]), float(b[1]) if b[1] != "inf" else float('inf')) for b in bins]
print("BINS SHAPE: ", bins)
anchor_points = [float(p) for p in anchor_points]
#bins = compute_dynamic_bins(original_bins["qnrf"]["bins"], num_bins=3)
#anchor_points = original_bins["qnrf"]["anchor_points"]["middle"]

model = GigaCount(bins=bins, anchor_points=anchor_points).to(DEVICE)
# model = _clip_ebc(
#     backbone="vit_b_16",
#     input_size=224,
#     reduction=8,
#     bins=bins,
#     anchor_points=anchor_points,
#     prompt_type="word",
#     num_vpt=32,
#     vpt_drop=0.0,
#     deep_vpt=True
# )
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

loss_fn = AdaptiveHybridLoss(bins=bins, reduction=8)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scaler = GradScaler(enabled=USE_AMP) 

# Scheduler 1: For fast convergence in the first 30 epochs
scheduler_high = LambdaLR(
    optimizer=optimizer,
    lr_lambda=partial(
        cosine_annealing_warm_restarts,
        warmup_epochs=20,
        warmup_lr=1e-6,
        T_0=5,
        T_mult=2,
        eta_min=1e-7,
        base_lr=1e-4
    )
)

# Scheduler 2: For fine-tuning from epoch 30+
scheduler_low = LambdaLR(
    optimizer=optimizer,
    lr_lambda=partial(
        cosine_annealing_warm_restarts,
        warmup_epochs=20,
        warmup_lr=1e-6,
        T_0=5,
        T_mult=2,
        eta_min=1e-7,
        base_lr=5e-5
    )
)

def reset_scheduler_low(optimizer):
    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=partial(
            cosine_annealing_warm_restarts,
            warmup_epochs=20,
            warmup_lr=1e-6,
            T_0=5,
            T_mult=2,
            eta_min=1e-7,
            base_lr=5e-5
        )
    )

# Load Dataset
def eval1():

    # best_mae = float('inf') 
    # best_rmse = float('inf') # Initialize best MAE to a high value
    # start_epoch = 0

    # Load checkpoint if available
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE, weights_only=False))
        # print(checkpoint.keys())

        # start_epoch = checkpoint["epoch"] + 1
        # best_mae = checkpoint["best_mae"]
        # if (checkpoint.get("best_rmse")):
        #   best_rmse = checkpoint["best_rmse"]
        # if (start_epoch+1)>=40:
        #   for param_group in optimizer.param_groups:
        #         param_group['lr'] = 5e-5
        # # If resuming from early epoch, restore optimizer/scheduler state
        # if (start_epoch + 1) % 40 == 0:
        #   print("Starting fresh at epoch, resetting learning rate.")
        # else:
        #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #     #if "scheduler_high_state_dict" in checkpoint:
        #         #scheduler_high.load_state_dict(checkpoint["scheduler_high_state_dict"])
        #     if "scheduler_low_state_dict" in checkpoint:
        #         scheduler_low = reset_scheduler_low(optimizer)
        #         #scheduler_low.load_state_dict(checkpoint["scheduler_low_state_dict"])
        #     if USE_AMP and checkpoint.get("scaler_state_dict"):
        #         scaler = GradScaler(enabled=USE_AMP)
        #         scaler.load_state_dict(checkpoint["scaler_state_dict"])



        #print(f"Resuming from epoch {start_epoch}, Best MAE: {best_mae:.2f}")


                
    mae, rmse = evaluate_model(model)  
    print(f"Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")  


    # if mae < best_mae:
    #     best_mae = mae
    #     save_best_model(model, best_mae)
    # if (rmse < best_rmse):
    #     best_rmse = rmse
    #     save_best_rmse_model(model,best_rmse)
    # # Log evaluation results

            # Save the best model based on MAE



def save_checkpoint(epoch, best_mae, best_rmse, model, optimizer, scaler, scheduler_high,scheduler_low):
    checkpoint = {
        "epoch": epoch,
        "best_mae": best_mae,
        "best_rmse": best_rmse,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if USE_AMP else None,
        "scheduler_high_state_dict": scheduler_high.state_dict(),
        "scheduler_low_state_dict": scheduler_low.state_dict(),
    }
    torch.save(checkpoint, "checkpoints/latest_checkpoint.pth")
    print(f"✅ Checkpoint saved at epoch {epoch+1}")


def save_best_model(model, best_mae):
    torch.save(model.state_dict(), "checkpoints/best_model.pth")
    print(f"Best model saved with MAE: {best_mae:.2f}")

def save_best_rmse_model(model, best_rmse):
    torch.save(model.state_dict(), "checkpoints/best_rmse_model.pth")
    print(f"Best model saved with RMSE: {best_rmse:.2f}")



if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    eval1()